import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.task_factories import CustomDatasetConf

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax and analyze joint torques.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--n_steps', type=int, default=1000, help='Number of evaluation steps')
parser.add_argument('--save_torques', action='store_true', help='Save joint torques to file')
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

print(f"Dataset length: {traj.data.qpos.shape[0]} frames")

# create env - use factory_params dict instead of config
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimic"  # Use basic mimic without site visualization

env = factory.make(**config.experiment.env_params, **factory_params)

# ==========================================
# TORQUE EVALUATION
# ==========================================
print(f"\n🔧 Evaluating joint torques for {args.n_steps} steps...")

# Initialize arrays to store torque data
n_actuators = env.model.nu  # Number of actuators
torque_history = np.zeros((args.n_steps, n_actuators))
time_steps = np.arange(args.n_steps)

# Get actuator names for reference
actuator_names = []
for i in range(n_actuators):
    actuator_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    actuator_names.append(actuator_name)

print(f"📊 Tracking {n_actuators} actuators: {actuator_names}")

# Reset environment
reset_result = env.reset()
if len(reset_result) == 2:
    obs, info = reset_result
else:
    obs = reset_result[0]  # Assume obs is first
    info = {}  # Default empty info

done = False
step_count = 0

rng = jax.random.PRNGKey(0)  # Initialize random key

while not done and step_count < args.n_steps:
    # Get action from trained policy
    rng, _rng = jax.random.split(rng)
    y, updates = agent_conf.network.apply({'params': agent_state.train_state.params,
                                           'run_stats': agent_state.train_state.run_stats},
                                          obs, mutable=["run_stats"])

    # --- THE FIX ---
    agent_state = agent_state.replace(
        train_state=agent_state.train_state.replace(run_stats=updates['run_stats'])
    )
    # ---------------

    pi, _ = y
    action = pi.mode()
    action = jnp.atleast_2d(action)

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render if you want to see the visualizer
    env.render()

    # 1. Extract the raw neural network action
    raw_action = np.array(action).flatten()

    # --- THE FIX: Clip the raw action to [-1.0, 1.0] before calculating ---
    clipped_action = np.clip(raw_action, -1.0, 1.0)

    # 2. Extract the gear multipliers
    gear_ratios = env.model.actuator_gear[:, 0]

    # 3. Calculate the true physical torque applied to the simulation
    true_torques = clipped_action * gear_ratios

    # 4. Record the true torques
    torque_history[step_count] = true_torques

    obs = next_obs
    step_count += 1

print(f"✅ Collected torque data for {step_count} steps")

# ==========================================
# TORQUE ANALYSIS
# ==========================================
print("\n📈 JOINT TORQUE ANALYSIS")
print("=" * 50)

# Calculate statistics for each actuator
for i, name in enumerate(actuator_names):
    torques = torque_history[:step_count, i]

    # Basic statistics
    mean_torque = np.mean(np.abs(torques))  # Use absolute values for magnitude
    max_torque = np.max(np.abs(torques))
    std_torque = np.std(torques)
    rms_torque = np.sqrt(np.mean(torques ** 2))  # RMS torque

    # --- FIX 1: Corrected print statements with properly formatted f-strings ---
    print(f"\n{name}:")
    print(f"  Mean (abs): {mean_torque:.2f} Nm")
    print(f"  Max (abs) : {max_torque:.2f} Nm")
    print(f"  Std Dev   : {std_torque:.2f} Nm")
    print(f"  RMS       : {rms_torque:.2f} Nm")

# Overall statistics
all_torques = torque_history[:step_count].flatten()
overall_mean = np.mean(np.abs(all_torques))
overall_max = np.max(np.abs(all_torques))
overall_rms = np.sqrt(np.mean(all_torques ** 2))

print("\n🎯 OVERALL STATISTICS")
print("=" * 50)
# --- FIX 1 (cont.): Corrected overall print statements ---
print(f"  Overall Mean: {overall_mean:.2f} Nm")
print(f"  Overall Max : {overall_max:.2f} Nm")
print(f"  Overall RMS : {overall_rms:.2f} Nm")

# Save torque data if requested
if args.save_torques:
    output_file = f"{os.path.splitext(path)[0]}_torques.npz"
    np.savez(output_file,
             torque_history=torque_history[:step_count],
             actuator_names=actuator_names,
             time_steps=time_steps[:step_count])
    print(f"\n💾 Saved torque data to: {output_file}")

print("\n✅ Torque evaluation complete!")
