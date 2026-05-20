import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import time
import cv2
from tqdm import tqdm

from loco_mujoco.environments import SkeletonTorque
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from pathlib import Path

# ==========================================
# 1. Setup & Load Data
# ==========================================
# Extract base directory
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)

SHOW_INITIAL_STATE_ONLY = False
SAVE_VIDEO = False
VIDEO_FILENAME = "trajectory_render.mp4"
VIDEO_OUTPUT_PATH = BASE_DIR / "Data_Conversion" / "Output_Files" / VIDEO_FILENAME


NPZ_PATH = BASE_DIR / "Data_Conversion" / "Output_Files" / "squat5_smoothed_test_converted.npz"
custom_data = np.load(NPZ_PATH)

if 'qpos' not in custom_data:
    raise ValueError("WARNING: 'qpos' key not found in dataset!")

qpos_traj = custom_data['qpos']
N_steps = qpos_traj.shape[0]
print(f" Loaded trajectory with {N_steps} steps and {qpos_traj.shape[1]} DOFs.")

# Extract frequency or default to 100 Hz
if 'frequency' in custom_data:
    freq = float(custom_data['frequency'])
    print(f" Found 'frequency' in dataset: {freq} Hz")
else:
    freq = 60.0  # Default frequency
dt = 1.0 / freq

# DITISEENTESTLINE
# ==========================================
# 2. Initialize Environment
# ==========================================
# Note: You can swap UnitreeH1 out for SkeletonTorque here
env = SkeletonTorque(init_state_type="DefaultInitialStateHandler")

key = jax.random.PRNGKey(0)
env.reset(key)

model = env.get_model()
nq = model.nq
nv = model.nv

# ==========================================
# 3. Align DOFs (Prune or Pad)
# ==========================================
# Ensure the dataset qpos matches the model's nq
if qpos_traj.shape[1] > nq:
    print(f" Pruning {qpos_traj.shape[1] - nq} extra DOFs from qpos.")
    qpos_traj = qpos_traj[:, :nq]
elif qpos_traj.shape[1] < nq:
    print(f" Padding {nq - qpos_traj.shape[1]} missing DOFs with zeros.")
    qpos_traj = np.pad(qpos_traj, ((0, 0), (0, nq - qpos_traj.shape[1])), 'constant')

# ==========================================
# 4. Handle Velocity (qvel)
# ==========================================
qvel_traj = np.zeros((N_steps, nv))

if 'qvel' in custom_data:
    print(" Found 'qvel' in dataset, loading directly...")
    raw_qvel = custom_data['qvel']
    if raw_qvel.shape[1] > nv:
        qvel_traj = raw_qvel[:, :nv]
    elif raw_qvel.shape[1] < nv:
        qvel_traj = np.pad(raw_qvel, ((0, 0), (0, nv - raw_qvel.shape[1])), 'constant')
    else:
        qvel_traj = raw_qvel
else:
    print("WARNING: 'qvel' not found. Computing velocities using MuJoCo's native differentiator...")
    # We MUST use mj_differentiatePos because of quaternions (nq != nv for free joints)
    for i in range(N_steps - 1):
        mujoco.mj_differentiatePos(model, qvel_traj[i], dt, qpos_traj[i], qpos_traj[i + 1])

    # Duplicate the second-to-last frame's velocity for the final frame to avoid zero-dropping
    qvel_traj[-1] = qvel_traj[-2]

# ==========================================
# 5. Build the LocoMuJoCo Trajectory
# ==========================================
njnt = model.njnt
jnt_type = model.jnt_type
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]

# Create info: stores joint names, types, and framerate
traj_info = TrajectoryInfo(
    jnt_names,
    model=TrajectoryModel(njnt, jnp.array(jnt_type)),
    frequency=freq
)

# Create data: packs the JAX arrays. split_points dictate episode boundaries.
traj_data = TrajectoryData(
    jnp.array(qpos_traj),
    jnp.array(qvel_traj),
    split_points=jnp.array([0, N_steps])
)

# Combine into the final trajectory object
traj = Trajectory(traj_info, traj_data)

# ==========================================
# 6. Load and Replay / Record
# ==========================================
print("\n Replaying Trajectory...")
env.load_trajectory(traj)

if SAVE_VIDEO:
    import cv2
    from tqdm import tqdm

    print(f" Rendering and saving video to: {VIDEO_OUTPUT_PATH}...")

    # Initialize MuJoCo offscreen renderer
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(VIDEO_OUTPUT_PATH), fourcc, freq, (width, height))

    # Loop exactly once through the dataset
    for i in tqdm(range(N_steps), desc="Saving Video"):
        # Set state to current trajectory frame
        env.data.qpos[:] = qpos_traj[i]
        env.data.qvel[:] = qvel_traj[i]

        # Forward kinematics to update the geometry
        mujoco.mj_forward(model, env.data)

        # Render frame (using default free camera -1)
        renderer.update_scene(env.data, camera=-1)
        frame = renderer.render()

        # OpenCV expects BGR color format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write to the MP4 file
        video_writer.write(frame_bgr)

    # CRITICAL: Clean up properly (Notice we removed destroyAllWindows!)
    video_writer.release()

    print(f"\n✅ Video saved successfully to:\n{VIDEO_OUTPUT_PATH}")
elif SHOW_INITIAL_STATE_ONLY:
    print(" Showing initial state indefinitely (SHOW_INITIAL_STATE_ONLY=True)")

    # Set qpos and qvel to the initial frame from the trajectory
    env.data.qpos[:] = qpos_traj[0]
    env.data.qvel[:] = qvel_traj[0]

    # Update forward kinematics to apply the positions
    mujoco.mj_forward(model, env.data)

    # Launch the MuJoCo passive viewer
    with mujoco.viewer.launch_passive(model, env.data) as viewer:
        # viewer.is_running() keeps the loop alive until you close the window
        while viewer.is_running():
            viewer.sync()  # Synchronizes the viewer with the data state
            time.sleep(0.05)  # Pauses briefly so your CPU and the GUI event loop can breathe

else:
    # Call play_trajectory
    env.play_trajectory(n_steps_per_episode=N_steps, render=True)