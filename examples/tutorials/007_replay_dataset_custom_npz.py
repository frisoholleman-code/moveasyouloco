import numpy as np
import gymnasium as gym
import loco_mujoco
import mujoco
import time

# ==========================================
# 1. Define Paths
# ==========================================
# UPDATE THIS to your specific .npz file
NPZ_PATH = "/home/frisokroes/loco-mujoco-linux/custom_data_prep/output_data/lopen.npz"

custom_data = np.load(NPZ_PATH)

if 'qpos' not in custom_data:
    raise ValueError("❌ 'qpos' key not found in dataset!")

qpos_traj = custom_data['qpos']

# Determine framerate for realistic playback speed
freq = float(custom_data['frequency']) if 'frequency' in custom_data else 100.0
dt = 1.0 / freq

# ==========================================
# 2. Initialize the Modern Gymnasium Env
# ==========================================
print(f"\nLoading Environment...")
# Initialize via Gymnasium.
# Note: Ensure the env_name matches your specific task (e.g., "SkeletonTorque.run")
env = gym.make("LocoMujoco", env_name="MjxSkeletonTorque", render_mode="human")
env.reset()

nq = env.unwrapped.model.nq

print(f"▶️ Replaying {len(qpos_traj)} frames from {NPZ_PATH}...")

# ==========================================
# 3. Custom Replay Loop
# ==========================================
try:
    for qpos in qpos_traj:
        # --- Handle DOF Mismatches ---
        if len(qpos) > nq:
            aligned_qpos = qpos[:nq]  # Prune extra DOFs
        elif len(qpos) < nq:
            aligned_qpos = np.pad(qpos, (0, nq - len(qpos)), 'constant')  # Pad missing DOFs
        else:
            aligned_qpos = qpos

        # A. Inject the positions directly into the environment's MuJoCo data
        env.unwrapped.data.qpos[:] = aligned_qpos

        # B. Update kinematics (moves the meshes without stepping physics)
        mujoco.mj_kinematics(env.unwrapped.model, env.unwrapped.data)

        # C. Render the frame to the Gymnasium window
        env.render()

        # D. Sleep to match the trajectory's framerate
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nPlayback interrupted by user.")
finally:
    env.close()
    print("✅ Replay finished.")