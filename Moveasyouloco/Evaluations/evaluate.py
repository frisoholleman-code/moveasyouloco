import os
import argparse
import csv

# 1. SET ENV VARS BEFORE JAX IMPORTS
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from scipy.spatial.transform import Rotation

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.task_factories import CustomDatasetConf
from loco_mujoco.environments import SkeletonTorque
from omegaconf import OmegaConf


def get_env_data(env):
    data = getattr(env, "data", None)
    if data is None and hasattr(env, "unwrapped"):
        data = getattr(env.unwrapped, "data", None)
    return data


def pelvis_label_prefix(joint_name):
    lower_name = joint_name.lower()
    if "root" in lower_name or "pelvis" in lower_name:
        return "pelvis"
    return joint_name


def build_qpos_labels(model):
    labels = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        joint_type = int(model.jnt_type[joint_id])

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            prefix = pelvis_label_prefix(joint_name)
            labels.extend([
                f"{prefix}_tilt",
                f"{prefix}_list",
                f"{prefix}_rotation",
                f"{prefix}_tx",
                f"{prefix}_ty",
                f"{prefix}_tz",
            ])
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            labels.extend([
                f"{joint_name}_qw",
                f"{joint_name}_qx",
                f"{joint_name}_qy",
                f"{joint_name}_qz",
            ])
        else:
            labels.append(joint_name)

    return labels


def free_joint_quat_to_euler_deg(quat):
    w, x, y, z = quat
    return Rotation.from_quat([x, y, z, w]).as_euler('XYZ', degrees=False)


def project_qpos_to_output(qpos, model):
    out = []
    idx = 0
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            tx, ty, tz = qpos[idx:idx + 3]
            quat = qpos[idx + 3:idx + 7]
            tilt, list_, rotation = free_joint_quat_to_euler_deg(quat)
            out.extend([tilt, list_, rotation, tx, ty, tz])
            idx += 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            out.extend(qpos[idx:idx + 4].tolist())
            idx += 4
        else:
            out.append(qpos[idx])
            idx += 1
    return np.asarray(out, dtype=np.float32)


def build_actuator_to_joint_map(model, actuator_names):
    actuator_to_joint = {}
    for i, actuator_name in enumerate(actuator_names):
        trnid = int(model.actuator_trnid[i, 0])
        if trnid >= 0:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
            if joint_name is not None:
                actuator_to_joint[actuator_name] = joint_name
    return actuator_to_joint


def main():
    parser = argparse.ArgumentParser(description='Run evaluation with PPOJax and save both torques and kinematics.')
    parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
    parser.add_argument('--outputpath', type=str, required=True, help='Path to evaluation folder')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
    args = parser.parse_args()

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

            traj = Trajectory.load(npz_path)
            print(" Precomputing missing physics data for JAX reward calculation...")

            temp_env = SkeletonTorque()
            mj_model = temp_env.get_model()
            mj_data = mujoco.MjData(mj_model)

            n_frames = traj.data.qpos.shape[0]
            nsite, nbody = mj_model.nsite, mj_model.nbody

            site_xpos = np.zeros((n_frames, nsite, 3), dtype=np.float32)
            site_xmat = np.zeros((n_frames, nsite, 9), dtype=np.float32)
            subtree_com = np.zeros((n_frames, nbody, 3), dtype=np.float32)
            cvel = np.zeros((n_frames, nbody, 6), dtype=np.float32)
            xpos = np.zeros((n_frames, nbody, 3), dtype=np.float32)
            xquat = np.zeros((n_frames, nbody, 4), dtype=np.float32)

            qpos_np = np.array(traj.data.qpos)
            qvel_np = np.array(traj.data.qvel)

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

            factory_parameters.setdefault("th_params", {})
            factory_parameters["th_params"].update({
                "random_start": False,
                "fixed_start_conf": (0, 0)
            })

    # ==========================================
    # --- ENVIRONMENT SETUP ---
    # ==========================================
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"]=False
    config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
    env = factory.make(**config.experiment.env_params, **factory_parameters)

    # Handle standard vs gym environment data access
    mj_data = getattr(env, "data", None)
    if mj_data is None and hasattr(env, "unwrapped"):
        mj_data = getattr(env.unwrapped, "data", None)

    if mj_data is None:
        print(" Notice: Running MJX or stateless environment. Using theoretical torque calculation fallback.")

    # Setup for torques
    n_actuators = env.model.nu
    torque_history = np.zeros((args.n_steps, n_actuators))
    actuator_names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).replace("mot_", "")
                      for i in range(n_actuators)]

    actuator_to_joint = build_actuator_to_joint_map(env.model, actuator_names)

    # Setup for kinematics
    n_qpos = env.model.nq
    qpos_names = build_qpos_labels(env.model)
    qpos_output_size = len(qpos_names)

    if qpos_output_size <= 0:
        print("Warning: Failed to build qpos output labels. Using fallback numeric labels.")
        qpos_names = [f"qpos_{i}" for i in range(n_qpos)]
        qpos_output_size = len(qpos_names)
    
    # Create kinematics_history to store projected output
    kinematics_history = np.zeros((args.n_steps, qpos_output_size), dtype=np.float32)

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
        y, _ = agent_conf.network.apply(
            {'params': agent_state.train_state.params,
             'run_stats': agent_state.train_state.run_stats},
            obs,
            mutable=["run_stats"]
        )

        pi = y[0] if isinstance(y, tuple) else y
        action = jnp.atleast_2d(pi.mode())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Collect torque data
        raw_action = np.array(action).flatten()
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        gear_ratios = np.array(env.model.actuator_gear[:, 0])
        true_torques = clipped_action * gear_ratios
        torque_history[step_count] = true_torques

        # Collect kinematics data
        env_data = get_env_data(env)
        if env_data is not None:
            raw_qpos = np.array(env_data.qpos)
            kinematics_history[step_count] = project_qpos_to_output(raw_qpos, env.model)

        obs = next_obs
        step_count += 1
        env.render()

    print(f"COMPLETE: Collected data for {step_count} frames")

    # ==========================================
    # --- SAVE TORQUES ---
    # ==========================================
    base_path = os.path.splitext(args.outputpath)[0]
    torques_mot_file = f"{base_path}torques.mot"
    evaluation_stats_csv_file = f"{base_path}evaluation_stats.csv"

    dt = getattr(env, "dt", None)
    if dt is None and hasattr(env, "unwrapped"):
        dt = getattr(env.unwrapped, "dt", None)
    if dt is None:
        dt = 0.01666666

    time_array = np.arange(step_count) * dt

    # Save torque and kinematics statistics CSV
    csv_stats = []
    
    # Calculate torque statistics for each actuator
    for i, name in enumerate(actuator_names):
        torques = torque_history[:step_count, i]
        mean_torque = np.mean(np.abs(torques))
        max_torque = np.max(np.abs(torques))
        std_torque = np.std(torques)
        rms_torque = np.sqrt(np.mean(torques ** 2))

        csv_stats.append({
            "Actuator": name,
            "Mean_abs_Nm": f"{mean_torque:.4f}",
            "Max_abs_Nm": f"{max_torque:.4f}",
            "Std_Dev_Nm": f"{std_torque:.4f}",
            "RMS_Nm": f"{rms_torque:.4f}"
        })

    # Calculate overall torque statistics
    all_torques = torque_history[:step_count].flatten()
    overall_mean_torque = np.mean(np.abs(all_torques))
    overall_max_torque = np.max(np.abs(all_torques))
    overall_rms_torque = np.sqrt(np.mean(all_torques ** 2))

    csv_stats.append({
        "Actuator": "OVERALL",
        "Mean_abs_Nm": f"{overall_mean_torque:.4f}",
        "Max_abs_Nm": f"{overall_max_torque:.4f}",
        "Std_Dev_Nm": "N/A",
        "RMS_Nm": f"{overall_rms_torque:.4f}"
    })
    
    # Calculate kinematics statistics for each joint and add to CSV rows
    kinematics_stats = {}
    for j, joint_name in enumerate(qpos_names):
        angles = kinematics_history[:step_count, j]
        mean_angle = np.mean(np.abs(angles))
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        std_angle = np.std(angles)
        rms_angle = np.sqrt(np.mean(angles ** 2))
        
        kinematics_stats[joint_name] = {
            "Mean_abs_angle": f"{mean_angle:.4f}",
            "Min_angle": f"{min_angle:.4f}",
            "Max_angle": f"{max_angle:.4f}",
            "Std_Dev_angle": f"{std_angle:.4f}",
            "RMS_angle": f"{rms_angle:.4f}"
        }
    
    # Calculate overall kinematics statistics
    all_angles = kinematics_history[:step_count].flatten()
    overall_mean_angle = np.mean(np.abs(all_angles))
    overall_min_angle = np.min(all_angles)
    overall_max_angle = np.max(all_angles)
    overall_rms_angle = np.sqrt(np.mean(all_angles ** 2))
    
    kinematics_stats["OVERALL"] = {
        "Mean_abs_angle": f"{overall_mean_angle:.4f}",
        "Min_angle": f"{overall_min_angle:.4f}",
        "Max_angle": f"{overall_max_angle:.4f}",
        "Std_Dev_angle": "N/A",
        "RMS_angle": f"{overall_rms_angle:.4f}"
    }
    
    # Add kinematics stats to CSV rows
    for row in csv_stats:
        actuator_name = row["Actuator"]
        if actuator_name == "OVERALL":
            stats = kinematics_stats.get("OVERALL", {})
        else:
            stats = kinematics_stats.get(actuator_name, {})
            if not stats:
                mapped_joint = actuator_to_joint.get(actuator_name)
                stats = kinematics_stats.get(mapped_joint, {}) if mapped_joint else {}
            if not stats:
                stats = kinematics_stats.get("OVERALL", {})
        
        row.update(stats)

    with open(evaluation_stats_csv_file, mode='w', newline='') as f:
        fieldnames = ["Actuator", "Mean_abs_Nm", "Max_abs_Nm", "Std_Dev_Nm", "RMS_Nm", 
                  "Mean_abs_angle", "Min_angle", "Max_angle", "Std_Dev_angle", "RMS_angle"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_stats)

    print(f" Saved torque and kinematics statistics CSV to: {evaluation_stats_csv_file}")

    # Save torques MOT file
    with open(torques_mot_file, 'w') as f:
        f.write("Torque\n")
        f.write("version=1\n")
        f.write(f"nRows={step_count}\n")
        f.write(f"nColumns={n_actuators + 1}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")

        header_row = ["time"] + actuator_names
        f.write("\t".join(header_row) + "\n")

        for i in range(step_count):
            row_data = [f"{time_array[i]:.6f}"] + [f"{val:.6f}" for val in torque_history[i]]
            f.write("\t".join(row_data) + "\n")

    print(f" Saved torque data to: {torques_mot_file}")

    # ==========================================
    # --- SAVE KINEMATICS ---
    # ==========================================
    kinematics_mot_file = f"{base_path}kinematics.mot"

    with open(kinematics_mot_file, 'w') as f:
        f.write("Kinematics\n")
        f.write("version=1\n")
        f.write(f"nRows={step_count}\n")
        f.write(f"nColumns={qpos_output_size + 1}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")

        header_row = ["time"] + qpos_names
        f.write("\t".join(header_row) + "\n")

        for i in range(step_count):
            row_data = [f"{time_array[i]:.6f}"] + [f"{val:.6f}" for val in kinematics_history[i]]
            f.write("\t".join(row_data) + "\n")

    print(f" Saved kinematics data to: {kinematics_mot_file}")

    print(f"\n✓ Evaluation complete. Generated:\n  - {torques_mot_file}\n  - {kinematics_mot_file}\n  - {evaluation_stats_csv_file}")


if __name__ == "__main__":
    main()
