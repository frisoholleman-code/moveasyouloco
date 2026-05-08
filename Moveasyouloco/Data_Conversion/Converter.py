"""
Converts .mot files to LocoMuJoCo-compatible .npz files.
Expects a specific .mot format with a header and data columns.
Must use the correct XML file from the desired environment.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot
import mujoco
import random

from wandb import controller

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Update these paths to point to your files.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input_Files"
OUTPUT_DIR = BASE_DIR / "Output_Files"
MODEL_DIR = BASE_DIR.parent / "Models" / "skeleton"

MOT_FILE_PATH = INPUT_DIR / "Friso9squat.mot"  # Input path, take from Input_Files.
XML_MODEL_PATH = MODEL_DIR / "skeleton_torque.xml"  # Path to xml model in Models/skeleton.
CONTROL_NPZ_PATH = BASE_DIR / "Control_Data" / "walk1_subject5.npz"  # Control data walk1subject5
OUTPUT_NPZ_PATH = OUTPUT_DIR / f"{MOT_FILE_PATH.stem}_test_converted.npz"  # Output path for the converted .npz file in Output_Files

# Set to True to print a visual mapping comparison for 5 random joints
ENABLE_VERIFICATION = True

# Joints that require angle inversion (e.g., due to coordinate system differences)
JOINTS_TO_INVERT = {"knee_angle_r", "knee_angle_l"}
INVERT_KNEE = False

def convert_mot_to_npz(mot_path: Path, xml_path: Path, control_npz_path: Path, output_path: Path, verify: bool = False):
    print(f"Conversion initiated, starting conversion for: {mot_path.name}")

    # ==========================================
    # 1. Parse the .mot file and Extract Data
    # ==========================================
    header_idx = -1
    with open(mot_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == "endheader":
                header_idx = i + 1
                break

    if header_idx == -1:
        raise ValueError(f" Malformed MOT file: 'endheader' not found in {mot_path.name}")

    print(f" Data starts at line {header_idx} (0-indexed).")
    dataframe = pd.read_csv(mot_path, skiprows=header_idx, sep=r'\s+')
    n_frames = len(dataframe)

    # ==========================================
    # 2. Load MuJoCo Model & Initialize States
    # ==========================================
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    xml_qpos = np.zeros((n_frames, model.nq), dtype=np.float64)

    # ==========================================
    # 3. Vectorized Root Translation and Rotation
    # ==========================================
    tx = dataframe['pelvis_tx'].values
    ty = dataframe['pelvis_ty'].values
    tz = dataframe['pelvis_tz'].values
    tilt = dataframe['pelvis_tilt'].values
    list_angle = dataframe['pelvis_list'].values
    rotation = dataframe['pelvis_rotation'].values

    xml_qpos[:, 0] = tx
    xml_qpos[:, 1] = -tz
    xml_qpos[:, 2] = ty

    R_ALIGN = rot.from_euler('x', 90, degrees=True)
    euler_angles = np.column_stack((tilt, list_angle, rotation))
    rot_os = rot.from_euler('ZXY', euler_angles, degrees=True)

    quats = (R_ALIGN * rot_os).as_quat()
    xml_qpos[:, 3:7] = quats[:, [3, 0, 1, 2]]

    # ==========================================
    # 4. Map Limbs
    # ==========================================
    for i in range(model.njnt):
        mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

        if mj_name in dataframe.columns and mj_name not in ['root', 'pelvis']:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            angle = np.deg2rad(dataframe[mj_name].values)

            if mj_name in JOINTS_TO_INVERT and INVERT_KNEE is True:
                angle *= -1.0

            xml_qpos[:, model.jnt_qposadr[joint_id]] = angle

    # ==========================================
    # 5. Calculate Velocities via MuJoCo Derivative
    # ==========================================
    time_values = dataframe['time'].values
    dt_values = np.diff(time_values)
    if not np.allclose(dt_values, dt_values[0], rtol=1e-3):
        print("️ WARNING: Timestep is not uniform. Using the first timestep for velocity calculation, but results may be inaccurate.")
    dt = dt_values[0]

    xml_qvel = np.zeros((n_frames, model.nv), dtype=np.float64)
    for i in range(n_frames - 1):
        mujoco.mj_differentiatePos(model, xml_qvel[i], dt, xml_qpos[i], xml_qpos[i + 1])
    xml_qvel[-1] = xml_qvel[-2]

    # ==========================================
    # 6. Stencil Extraction (Matching Golden NPZ)
    # ==========================================
    standard = dict(np.load(control_npz_path, allow_pickle=True))

    control_start_height = standard['qpos'][0, 2]
    custom_start_height = xml_qpos[0, 2]
    z_offset = control_start_height - custom_start_height

    print(f" Applying vertical offset of {z_offset:.4f} meters to match original .mot file.")
    xml_qpos[:, 2] += z_offset

    STANDARD_JOINT_NAMES = [str(n).strip() for n in standard['joint_names']]

    final_qpos = np.zeros((n_frames, standard['qpos'].shape[1]), dtype=np.float32)
    final_qvel = np.zeros((n_frames, standard['qvel'].shape[1]), dtype=np.float32)

    curr_qpos_idx = 0
    curr_qvel_idx = 0

    joint_to_final_idx = {}

    for name in STANDARD_JOINT_NAMES:
        if name == 'root':
            final_qpos[:, 0:7] = xml_qpos[:, 0:7]
            final_qvel[:, 0:6] = xml_qvel[:, 0:6]
            joint_to_final_idx['root'] = slice(0, 7)
            curr_qpos_idx += 7
            curr_qvel_idx += 6
        else:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

            if joint_id != -1:
                idx_qpos = model.jnt_qposadr[joint_id]
                idx_qvel = model.jnt_dofadr[joint_id]

                final_qpos[:, curr_qpos_idx] = xml_qpos[:, idx_qpos]
                final_qvel[:, curr_qvel_idx] = xml_qvel[:, idx_qvel]
                joint_to_final_idx[name] = curr_qpos_idx
            else:
                print(f" WARNING: '{name}' not found in XML. Padding with zeros.")

            curr_qpos_idx += 1
            curr_qvel_idx += 1

    # ==========================================
    # 7. Random Visual Verification Report
    # ==========================================
    if verify:
        print(f"\n RANDOM VERIFICATION REPORT: (First 10 frames)")
        valid_joints = [j for j in joint_to_final_idx.keys() if j in dataframe.columns]

        print(valid_joints)


        if not valid_joints:
            print("No valid joints available to verify.")
        else:
            num_to_pick = min(5, len(valid_joints))
            random_joints = random.sample(valid_joints, num_to_pick)

            for joint in random_joints:
                final_idx = joint_to_final_idx[joint]

                print(f"\n➤ Mapping Check:")
                print(f"  • .mot Column Name : '{joint}'")
                print(f"  • .npz Joint Name  : '{joint}' (Array Index: {final_idx})")

                mot_vals = dataframe[joint].values[:10]
                npz_vals = final_qpos[:10, final_idx]

                print(
                    f"{'Frame':<8} | {'MOT Value (deg)':<16} | {'NPZ Value (rad)':<16} | {'NPZ (re-converted to deg)':<25}")
                print("-" * 75)
                for f in range(min(10, n_frames)):
                    npz_deg = np.rad2deg(npz_vals[f])
                    print(f"{f:<8} | {mot_vals[f]:<16.4f} | {npz_vals[f]:<16.4f} | {npz_deg:<25.4f}")
        print("\n" + "=" * 75 + "\n")

    # ==========================================
    # 8. Build and Save Final Dictionary
    # ==========================================
    dataset_dict = standard.copy()
    dataset_dict['qpos'] = final_qpos
    dataset_dict['qvel'] = final_qvel
    dataset_dict['split_points'] = np.array([0, n_frames], dtype=np.int32)
    dataset_dict['frequency'] = np.float64(1.0 / dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dataset_dict)

    print(f"Conversion complete! Extracted {curr_qpos_idx} DOFs.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    convert_mot_to_npz(
        mot_path=MOT_FILE_PATH,
        xml_path=XML_MODEL_PATH,
        control_npz_path=CONTROL_NPZ_PATH,
        output_path=OUTPUT_NPZ_PATH,
        verify=ENABLE_VERIFICATION
    )
