import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.task_factories import CustomDatasetConf


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback


def build_custom_trajectory(npz_path, env):
    """Dynamically builds a Trajectory object from raw npz data."""
    import numpy as np
    import mujoco
    from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
    import jax.numpy as jnp

    print(f"\n⚙️ Dynamically building trajectory from {npz_path}...")
    custom_data = np.load(npz_path)
    qpos_traj = custom_data['qpos']
    N_steps = qpos_traj.shape[0]
    freq = float(custom_data.get('frequency', 60.0))
    dt = 1.0 / freq

    model = env.get_model()
    nq, nv, njnt = model.nq, model.nv, model.njnt

    # 1. Align DOFs (Prune or Pad)
    if qpos_traj.shape[1] > nq:
        qpos_traj = qpos_traj[:, :nq]
    elif qpos_traj.shape[1] < nq:
        qpos_traj = np.pad(qpos_traj, ((0, 0), (0, nq - qpos_traj.shape[1])), 'constant')

    # 2. Handle Velocity (qvel)
    qvel_traj = np.zeros((N_steps, nv))
    if 'qvel' in custom_data:
        raw_qvel = custom_data['qvel']
        if raw_qvel.shape[1] > nv:
            qvel_traj = raw_qvel[:, :nv]
        elif raw_qvel.shape[1] < nv:
            qvel_traj = np.pad(raw_qvel, ((0, 0), (0, nv - raw_qvel.shape[1])), 'constant')
        else:
            qvel_traj = raw_qvel
    else:
        for i in range(N_steps - 1):
            mujoco.mj_differentiatePos(model, qvel_traj[i], dt, qpos_traj[i], qpos_traj[i + 1])
        qvel_traj[-1] = qvel_traj[-2]

    # 3. Build LocoMuJoCo Trajectory
    jnt_type = model.jnt_type
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]

    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=freq)
    traj_data = TrajectoryData(jnp.array(qpos_traj), jnp.array(qvel_traj), split_points=jnp.array([0, N_steps]))

    print("✅ Trajectory built successfully in memory!")
    return Trajectory(traj_info, traj_data)






@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # ==========================================
        # --- THE INJECTION: HANDLE CUSTOM DATA ---
        # ==========================================
        factory_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)

        if "custom_dataset_conf" in factory_params:
            conf_dict = factory_params["custom_dataset_conf"]

            if "traj" in conf_dict and isinstance(conf_dict["traj"], str):
                npz_path = conf_dict["traj"]
                print(f"\n🔧 Intercepted string path in YAML: {npz_path}")

                # Load the Trajectory object (qpos and qvel)
                traj = Trajectory.load(npz_path)

                # --- START KINEMATICS FIX ---
                print("⚙️ Precomputing ALL missing physics data for JAX reward calculation...")
                import mujoco
                import numpy as np
                import jax.numpy as jnp
                from loco_mujoco.environments import SkeletonTorque

                # 1. Create a lightweight CPU environment directly to borrow the model
                temp_env = SkeletonTorque()
                mj_model = temp_env.get_model()
                mj_data = mujoco.MjData(mj_model)

                # 2. Setup arrays based on the environment model
                n_frames = traj.data.qpos.shape[0]
                nsite = mj_model.nsite
                nbody = mj_model.nbody

                site_xpos = np.zeros((n_frames, nsite, 3), dtype=np.float32)
                site_xmat = np.zeros((n_frames, nsite, 9), dtype=np.float32)
                subtree_com = np.zeros((n_frames, nbody, 3), dtype=np.float32)
                cvel = np.zeros((n_frames, nbody, 6), dtype=np.float32)
                xpos = np.zeros((n_frames, nbody, 3), dtype=np.float32)
                xquat = np.zeros((n_frames, nbody, 4), dtype=np.float32)

                # 3. Calculate full forward kinematics and velocities for every frame
                qpos_np = np.array(traj.data.qpos)
                qvel_np = np.array(traj.data.qvel)

                for i in range(n_frames):
                    mj_data.qpos[:] = qpos_np[i]
                    mj_data.qvel[:] = qvel_np[i]

                    # Compute kinematics AND center-of-mass velocities
                    mujoco.mj_kinematics(mj_model, mj_data)
                    mujoco.mj_comPos(mj_model, mj_data)
                    mujoco.mj_comVel(mj_model, mj_data)

                    site_xpos[i] = mj_data.site_xpos.copy()
                    site_xmat[i] = mj_data.site_xmat.copy()
                    subtree_com[i] = mj_data.subtree_com.copy()
                    cvel[i] = mj_data.cvel.copy()
                    xpos[i] = mj_data.xpos.copy()
                    xquat[i] = mj_data.xquat.copy()

                # 4. Inject ALL computed data back into the Trajectory
                traj.data = traj.data.replace(
                    site_xpos=jnp.array(site_xpos),
                    site_xmat=jnp.array(site_xmat),
                    subtree_com=jnp.array(subtree_com),
                    cvel=jnp.array(cvel),
                    xpos=jnp.array(xpos),
                    xquat=jnp.array(xquat)
                )
                print("✅ Full Physics profile perfectly baked into Trajectory!")
                # --- END KINEMATICS FIX ---

                # Replace the nested dictionary with the actual Class object
                factory_params["custom_dataset_conf"] = CustomDatasetConf(traj)
                print("✅ Custom dataset ready for training!\n")
        # ==========================================

        # Create the REAL environment for training
        env = factory.make(**config.experiment.env_params, **factory_params)

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config)

        # setup metric handler (optional)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # build training function
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        # save agent state
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i + 1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

                    # log all measures
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # metric for used for wandb sweep (optional)
                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                            step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        record_n_steps = config.experiment.record_n_steps if "record_n_steps" in config.experiment else 300
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True,
                           n_steps=record_n_steps, n_envs=20, record=True,
                           train_state_seed=0)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()