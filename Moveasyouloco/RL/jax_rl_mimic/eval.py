import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.task_factories import CustomDatasetConf

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

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
        print("✅ Custom dataset ready for evaluation!\n")

# create env - use factory_params dict instead of config
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic (now that physics is precomputed)

env = factory.make(**config.experiment.env_params, **factory_params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=True, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)
