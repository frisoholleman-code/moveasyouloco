import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import mujoco

# ==========================================
# 1. Configuration
# ==========================================
MOT_FILE = "/home/frisokroes/loco-mujoco-linux/custom_data_prep/input_data/squat1.mot"
OUTPUT_NPZ = "/home/frisokroes/loco-mujoco-linux/custom_data_prep/output_data/kroes_squat.npz"
XML_PATH = "/home/frisokroes/.virtualenvs/loco-mujoco-linux/lib/python3.12/site-packages/loco_mujoco_models/skeleton/skeleton_torque.xml"
GOLDEN_NPZ = "/home/frisokroes/loco-mujoco-linux/custom_data_prep/output_data/lopen.npz"

# ==========================================
# 2. Parse the .mot file
# ==========================================
with open(MOT_FILE, 'r') as f:
    lines = f.readlines()
header_idx = 0
for i, line in enumerate(lines):
    if line.strip() == "endheader":
        header_idx = i + 1
        break
df = pd.read_csv(MOT_FILE, skiprows=header_idx, sep=r'\s+')
n_frames = len(df)

# ==========================================
# 3. Load MuJoCo Model (for physics math)
# ==========================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
xml_qpos = np.zeros((n_frames, model.nq), dtype=np.float64)

# Map Root Translation and Rotation (Y-Up to Z-Up)
tx, ty, tz = df['pelvis_tx'].values, df['pelvis_ty'].values, df['pelvis_tz'].values
tilt, list_angle, rotation = df['pelvis_tilt'].values, df['pelvis_list'].values, df['pelvis_rotation'].values
R_align = R.from_euler('x', 90, degrees=True)

for i in range(n_frames):
    xml_qpos[i, 0] = tx[i]
    xml_qpos[i, 1] = -tz[i]
    xml_qpos[i, 2] = ty[i]

    rot_os = R.from_euler('zxy', [tilt[i], list_angle[i], rotation[i]], degrees=True)
    quat = (R_align * rot_os).as_quat()
    xml_qpos[i, 3], xml_qpos[i, 4], xml_qpos[i, 5], xml_qpos[i, 6] = quat[3], quat[0], quat[1], quat[2]

# Map Limbs dynamically
for mj_name in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]:
    if mj_name in df.columns and mj_name not in ['root', 'pelvis']:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)

        # Convert to radians
        angle = np.deg2rad(df[mj_name].values)

        # --- FIX 1: THE KNEE FLIP ---
        # If the joint is a knee, invert the angle!
        if "knee" in mj_name:
            angle = angle * -1.0

        xml_qpos[:, model.jnt_qposadr[joint_id]] = angle

# Calculate Velocities in the XML space
dt = df['time'].values[1] - df['time'].values[0]
xml_qvel = np.zeros((n_frames, model.nv), dtype=np.float64)
for i in range(n_frames - 1):
    mujoco.mj_differentiatePos(model, xml_qvel[i], dt, xml_qpos[i], xml_qpos[i + 1])
xml_qvel[-1] = xml_qvel[-2]

# ==========================================
# 4. The Stencil Extraction (Matching the Golden NPZ)
# ==========================================
golden = dict(np.load(GOLDEN_NPZ, allow_pickle=True))

golden_start_height = golden['qpos'][0, 2]
custom_start_height = xml_qpos[0, 2]
z_offset = golden_start_height - custom_start_height

print(f"🔧 Applying vertical offset of {z_offset:.4f} meters to drop skeleton to the floor.")
xml_qpos[:, 2] += z_offset


golden_names = [str(n).strip() for n in golden['joint_names']]

# Create the perfectly sized final arrays based on the Golden data
final_qpos = np.zeros((n_frames, golden['qpos'].shape[1]), dtype=np.float32)
final_qvel = np.zeros((n_frames, golden['qvel'].shape[1]), dtype=np.float32)

curr_qpos_idx = 0
curr_qvel_idx = 0

for name in golden_names:
    if name == 'root':
        # Root is a free joint: takes 7 spots in qpos, 6 spots in qvel
        final_qpos[:, 0:7] = xml_qpos[:, 0:7]
        final_qvel[:, 0:6] = xml_qvel[:, 0:6]
        curr_qpos_idx += 7
        curr_qvel_idx += 6
    else:
        # For standard hinge joints (1 spot each)
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            idx_qpos = model.jnt_qposadr[joint_id]
            idx_qvel = model.jnt_dofadr[joint_id]

            final_qpos[:, curr_qpos_idx] = xml_qpos[:, idx_qpos]
            final_qvel[:, curr_qvel_idx] = xml_qvel[:, idx_qvel]
        except Exception:
            print(f"⚠️ Warning: '{name}' not found in XML. Padding with zeros.")

        curr_qpos_idx += 1
        curr_qvel_idx += 1

# ==========================================
# 5. Build and Save the Final Dictionary
# ==========================================
# We copy the entire Golden dictionary to guarantee 100% metadata compliance,
# then just overwrite the parts we generated!
dataset_dict = golden.copy()
dataset_dict['qpos'] = final_qpos
dataset_dict['qvel'] = final_qvel
dataset_dict['split_points'] = np.array([0, n_frames], dtype=np.int32)
dataset_dict['frequency'] = np.float64(1.0 / dt)

np.savez(OUTPUT_NPZ, **dataset_dict)

print(f"✅ Conversion complete! Extracted {curr_qpos_idx} DOFs perfectly matching the Golden dataset.")
print(f"✅ Saved to: {OUTPUT_NPZ}")