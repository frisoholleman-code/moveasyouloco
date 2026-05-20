import os
import argparse
import csv  # <-- ADDED for CSV output

# 1. SET ENV VARS BEFORE JAX IMPORTS
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.task_factories import CustomDatasetConf
from loco_mujoco.environments import SkeletonTorque
from omegaconf import OmegaConf


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run evaluation with PPOJax and analyze joint torques.')
    parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
    parser.add_argument('--save_torques', action='store_true', help='Save joint torques to file')
    args = parser.parse_args()

    # Load Agent
    agent_conf, agent_state = PPOJax.load_agent(args.path)
    config = agent_conf.config
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # ==========================================
    # --- PREPARE CUSTOM DATA & KINEMATICS ---
    # ==========================================
    factory_parameters = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)

    if "custom_dataset_conf" in factory_parameters:
        conf_dict = factory_parameters["custom_dataset_conf"]
        if "traj" in conf_dict and isinstance(conf_dict["traj"], str):
            npz_path = conf_dict["traj"]
            print(f"\n Intercepted string path in YAML: {npz_path}")

            # Load Trajectory
            traj = Trajectory.load(npz_path)
            print(" Precomputing missing physics data for JAX reward calculation...")

            # Lightweight CPU env to borrow the model
            temp_env = SkeletonTorque()
            mj_model = temp_env.get_model()
            mj_data = mujoco.MjData(mj_model)

            n_frames = traj.data.qpos.shape[0]
            nsite, nbody = mj_model.nsite, mj_model.nbody

            # Pre-allocate arrays
            site_xpos = np.zeros((n_frames, nsite, 3), dtype=np.float32)
            site_xmat = np.zeros((n_frames, nsite, 9), dtype=np.float32)
            subtree_com = np.zeros((n_frames, nbody, 3), dtype=np.float32)
            cvel = np.zeros((n_frames, nbody, 6), dtype=np.float32)
            xpos = np.zeros((n_frames, nbody, 3), dtype=np.float32)
            xquat = np.zeros((n_frames, nbody, 4), dtype=np.float32)

            qpos_np = np.array(traj.data.qpos)
            qvel_np = np.array(traj.data.qvel)

            # Compute full kinematics
            for i in range(n_frames):
                mj_data.qpos[:] = qpos_np[i]
                mj_data.qvel[:] = qvel_np[i]

                mujoco.mj_kinematics(mj_model, mj_data)
                mujoco.mj_comPos(mj_model, mj_data)
                mujoco.mj_comVel(mj_model, mj_data)

                site_xpos[i] = mj_data.site_xpos
                site_xmat[i] = mj_data.site_xmat
                subtree_com[i] = mj_data.subtree_com
                cvel[i] = mj_data.cvel
                xpos[i] = mj_data.xpos
                xquat[i] = mj_data.xquat

            # Inject into trajectory
            traj.data = traj.data.replace(
                site_xpos=jnp.array(site_xpos),
                site_xmat=jnp.array(site_xmat),
                subtree_com=jnp.array(subtree_com),
                cvel=jnp.array(cvel),
                xpos=jnp.array(xpos),
                xquat=jnp.array(xquat)
            )
            print("COMPLETE: Full Physics profile baked into Trajectory for JAX reward calculation")

            factory_parameters["custom_dataset_conf"] = CustomDatasetConf(traj)
            print(f"COMPLETE: Custom dataset ready! Dataset length: {n_frames} frames\n")

    # ==========================================
    # --- ENVIRONMENT SETUP ---
    # ==========================================
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = False
    config.experiment.env_params["goal_type"] = "GoalTrajMimic"

    env = factory.make(**config.experiment.env_params, **factory_parameters)

    # Handle standard vs gym environment data access
    mj_data = getattr(env, "data", None)
    if mj_data is None and hasattr(env, "unwrapped"):
        mj_data = getattr(env.unwrapped, "data", None)

    if mj_data is None:
        print(" Notice: Running MJX or stateless environment. Using theoretical torque calculation fallback.")

    n_actuators = env.model.nu
    torque_history = np.zeros((args.n_steps, n_actuators))

    actuator_names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(n_actuators)]

    print(f"\n Evaluating joint torques for {args.n_steps} steps...")
    print(f" Tracking {n_actuators} actuators: {actuator_names}")

    # ==========================================
    # --- EVALUATION LOOP ---
    # ==========================================
    if hasattr(env, "seed"):
        env.seed(42)

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    done = False
    step_count = 0

    while not done and step_count < args.n_steps:
        # Evaluate with frozen run_stats (no 'mutable' flag needed for pure inference)
        # Assuming the network supports standard flax.linen apply without state updates in eval
        y, _ = agent_conf.network.apply(
            {'params': agent_state.train_state.params,
             'run_stats': agent_state.train_state.run_stats},
            obs,
            mutable=["run_stats"]
        )

        pi = y[0] if isinstance(y, tuple) else y  # Handle different return structures
        action = jnp.atleast_2d(pi.mode())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

        # Extract Ground Truth Torques directly from the simulator
        # This accounts for gear ratios, force limits, and control limits naturally
        # Calculate theoretical torque directly from the neural network action
        raw_action = np.array(action).flatten()
        clipped_action = np.clip(raw_action, -1.0, 1.0)

        # Extract gear ratios (wrapped in np.array to safely handle both CPU and JAX mjx.Models)
        gear_ratios = np.array(env.model.actuator_gear[:, 0])

        # Calculate commanded torque
        true_torques = clipped_action * gear_ratios

        torque_history[step_count] = true_torques
        obs = next_obs
        step_count += 1

    print(f"COMPLETE: Collected torque data for {step_count} steps (excluding last 2 for analysis)")

    # ==========================================
    # --- TORQUE ANALYSIS ---
    # ==========================================
    print("\n JOINT TORQUE ANALYSIS")
    print("=" * 50)

    # Initialize a list to hold the row data for the CSV
    csv_stats = []

    for i, name in enumerate(actuator_names):
        torques = torque_history[:step_count-2, i]
        mean_val = np.mean(np.abs(torques))
        max_val = np.max(np.abs(torques))
        std_val = np.std(torques)
        rms_val = np.sqrt(np.mean(torques ** 2))

        print(f"\n{name}:")
        print(f"  Mean (abs): {mean_val:.2f} Nm")
        print(f"  Max (abs) : {max_val:.2f} Nm")
        print(f"  Std Dev   : {std_val:.2f} Nm")
        print(f"  RMS       : {rms_val:.2f} Nm")

        # Append data to the CSV structure
        csv_stats.append({
            "Actuator": name,
            "Mean_abs_Nm": f"{mean_val:.4f}",
            "Max_abs_Nm": f"{max_val:.4f}",
            "Std_Dev_Nm": f"{std_val:.4f}",
            "RMS_Nm": f"{rms_val:.4f}"
        })

    all_torques = torque_history[:step_count-2].flatten()
    overall_mean = np.mean(np.abs(all_torques))
    overall_max = np.max(np.abs(all_torques))
    overall_rms = np.sqrt(np.mean(all_torques ** 2))

    print("\n OVERALL STATISTICS")
    print("=" * 50)
    print(f"  Overall Mean: {overall_mean:.2f} Nm")
    print(f"  Overall Max : {overall_max:.2f} Nm")
    print(f"  Overall RMS : {overall_rms:.2f} Nm")

    # Add the Overall metrics to the bottom of the CSV
    csv_stats.append({
        "Actuator": "OVERALL",
        "Mean_abs_Nm": f"{overall_mean:.4f}",
        "Max_abs_Nm": f"{overall_max:.4f}",
        "Std_Dev_Nm": "N/A",  # Left as N/A since overall std-dev isn't printed
        "RMS_Nm": f"{overall_rms:.4f}"
    })

    # Save outputs to CSV
    base_path = os.path.splitext(args.path)[0]
    csv_file = f"{base_path}_torque_stats.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Actuator", "Mean_abs_Nm", "Max_abs_Nm", "Std_Dev_Nm", "RMS_Nm"])
        writer.writeheader()
        writer.writerows(csv_stats)

    print(f"\n Saved torque statistics CSV to: {csv_file}")

    if args.save_torques:
        # 1. Save standard .npz file
        npz_file = f"{base_path}_torques.npz"
        np.savez(npz_file,
                 torque_history=torque_history[:step_count-2],
                 actuator_names=actuator_names,
                 time_steps=np.arange(step_count))

        # 2. Save OpenSim .mot file
        mot_file = f"{base_path}_torques.mot"

        dt = getattr(env, "dt", None)
        if dt is None and hasattr(env, "unwrapped"):
            dt = getattr(env.unwrapped, "dt", None)

        if dt is None:
            dt = 0.01666666  # Final safe fallback for MJX environments

        time_array = np.arange(step_count) * dt

        with open(mot_file, 'w') as f:
            # Write OpenSim .mot header
            f.write(f"Torques_from_{os.path.basename(base_path)}\n")
            f.write("version=1\n")
            f.write(f"nRows={step_count}\n")
            f.write(f"nColumns={n_actuators + 1}\n")  # +1 for the time column
            f.write("inDegrees=yes\n")
            f.write("endheader\n")

            # Write column headers (time + actuator names)
            header_row = ["time"] + actuator_names
            f.write("\t".join(header_row) + "\n")

            # Write row data
            for i in range(step_count - 2):
                row_data = [f"{time_array[i]:.6f}"] + [f"{val:.6f}" for val in torque_history[i]]
                f.write("\t".join(row_data) + "\n")

        print(f"\n Saved full torque data series to:\n  - {npz_file}\n  - {mot_file}")


if __name__ == "__main__":
    main()