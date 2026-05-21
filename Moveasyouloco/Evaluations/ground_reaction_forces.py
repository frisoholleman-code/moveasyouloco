import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import pandas as pd
from loco_mujoco.environments import SkeletonTorque
from pathlib import Path

# ==========================================
# 1. Setup & Configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
NPZ_PATH = BASE_DIR / "Data_Conversion" / "Output_Files" / "Friso9squat_test_converted.npz"

# Output Files
MOT_OUT_PATH = BASE_DIR / "Data_Conversion" / "Output_Files" / "Friso9squat_grf_est.mot"
XML_OUT_PATH = BASE_DIR / "Data_Conversion" / "Output_Files" / "Friso9squat_external_loads.xml"

# *** IMPORTANT: Define your MuJoCo foot geometry names here ***
# The script uses these to separate Left and Right GRF
# *** IMPORTANT: Define your MuJoCo foot geometry names here ***
LEFT_FOOT_GEOMS = ["l_foot", "l_bofoot", "foot_box_l"]
RIGHT_FOOT_GEOMS = ["r_foot", "r_bofoot", "foot_box_r"]

custom_data = np.load(NPZ_PATH)
qpos_traj = custom_data['qpos']
N_steps = qpos_traj.shape[0]
freq = float(custom_data.get('frequency', 60.0))
dt = 1.0 / freq

# ==========================================
# 2. Initialize Environment & Align Data
# ==========================================
env = SkeletonTorque(init_state_type="DefaultInitialStateHandler")
key = jax.random.PRNGKey(0)
env.reset(key)

model = env.get_model()

geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
print("\n--- ALL MUJOCO GEOMETRIES ---")
print(geom_names)
print("-----------------------------\n")

nq, nv = model.nq, model.nv

if qpos_traj.shape[1] > nq:
    qpos_traj = qpos_traj[:, :nq]
elif qpos_traj.shape[1] < nq:
    qpos_traj = np.pad(qpos_traj, ((0, 0), (0, nq - qpos_traj.shape[1])), 'constant')

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

# ==========================================
# 3. Extract Separated GRF and CoP
# ==========================================
mot_data = []
print("Simulating and extracting Left/Right GRF & CoP...")

for i in range(N_steps):
    env.data.qpos[:] = qpos_traj[i]
    env.data.qvel[:] = qvel_traj[i]
    mujoco.mj_forward(model, env.data)
    print(f"Step {i}: Number of contacts detected = {env.data.ncon}")

    forces = {"r": np.zeros(3), "l": np.zeros(3)}
    cop = {"r": np.zeros(3), "l": np.zeros(3)}
    torques = {"r": np.zeros(3), "l": np.zeros(3)}  # Simplified: assuming 0 torque for now

    # Track sum of normal forces for weighted CoP calculation
    sum_fz = {"r": 0.0, "l": 0.0}

    for c_idx in range(env.data.ncon):
        contact = env.data.contact[c_idx]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""

        is_geom1_floor = (model.geom_type[contact.geom1] == mujoco.mjtGeom.mjGEOM_PLANE) or any(x in geom1_name.lower() for x in ["floor", "ground"])
        is_geom2_floor = (model.geom_type[contact.geom2] == mujoco.mjtGeom.mjGEOM_PLANE) or any(
            x in geom2_name.lower() for x in ["floor", "ground"])

        if is_geom1_floor or is_geom2_floor:
            # Determine which foot is making contact
            body_geom = geom2_name if is_geom1_floor else geom1_name
            side = None
            if any(lf in body_geom for lf in LEFT_FOOT_GEOMS):
                side = "l"
            elif any(rf in body_geom for rf in RIGHT_FOOT_GEOMS):
                side = "r"

            if side:
                c_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, env.data, c_idx, c_force)
                frame = contact.frame.reshape(3, 3)
                force_world = np.dot(c_force[:3], frame)

                if not is_geom1_floor: force_world = -force_world

                forces[side] += force_world

                # Weighted Center of Pressure Calculation
                # P_cop = sum(P_contact * Fz_contact) / sum(Fz_contact)
                contact_pos = contact.pos
                fz = abs(force_world[2])
                cop[side] += contact_pos * fz
                sum_fz[side] += fz

    # Finalize CoP division
    for s in ["r", "l"]:
        if sum_fz[s] > 0.001:
            cop[s] = cop[s] / sum_fz[s]
        else:
            cop[s] = np.zeros(3)  # No contact, CoP is 0

    # Append to MOT row
    mot_data.append([
        i * dt,
        forces["r"][0], forces["r"][1], forces["r"][2],  # Right Force
        cop["r"][0], cop["r"][1], cop["r"][2],  # Right CoP
        torques["r"][0], torques["r"][1], torques["r"][2],  # Right Torque
        forces["l"][0], forces["l"][1], forces["l"][2],  # Left Force
        cop["l"][0], cop["l"][1], cop["l"][2],  # Left CoP
        torques["l"][0], torques["l"][1], torques["l"][2]  # Left Torque
    ])

# ==========================================
# 4. Write OpenSim .mot File
# ==========================================
columns = [
    "time",
    "ground_force_r_vx", "ground_force_r_vy", "ground_force_r_vz",
    "ground_force_r_px", "ground_force_r_py", "ground_force_r_pz",
    "ground_torque_r_x", "ground_torque_r_y", "ground_torque_r_z",
    "ground_force_l_vx", "ground_force_l_vy", "ground_force_l_vz",
    "ground_force_l_px", "ground_force_l_py", "ground_force_l_pz",
    "ground_torque_l_x", "ground_torque_l_y", "ground_torque_l_z"
]

MOT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MOT_OUT_PATH, "w") as f:
    f.write(f"Friso9squat_grf_est\n")
    f.write(f"version=1\n")
    f.write(f"nRows={N_steps}\n")
    f.write(f"nColumns={len(columns)}\n")
    f.write(f"inDegrees=yes\n")
    f.write(f"endheader\n")
    f.write("\t".join(columns) + "\n")

    for row in mot_data:
        f.write("\t".join([f"{val:.6f}" for val in row]) + "\n")

print(f"Saved OpenSim MOT file to: {MOT_OUT_PATH}")

# ==========================================
# 5. Write OpenSim .xml File
# ==========================================
xml_content = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ExternalLoads name="synthetic_external_loads">
        <objects>
            <ExternalForce name="RightGRF">
                <applied_to_body>calcn_r</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>ground_force_r_v</force_identifier>
                <point_identifier>ground_force_r_p</point_identifier>
                <torque_identifier>ground_torque_r_</torque_identifier>
            </ExternalForce>
            <ExternalForce name="LeftGRF">
                <applied_to_body>calcn_l</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>ground_force_l_v</force_identifier>
                <point_identifier>ground_force_l_p</point_identifier>
                <torque_identifier>ground_torque_l_</torque_identifier>
            </ExternalForce>
        </objects>
        <groups />
        <datafile>{MOT_OUT_PATH.name}</datafile>
        <external_loads_model_kinematics_file />
        <lowpass_cutoff_frequency_for_load_kinematics>-1</lowpass_cutoff_frequency_for_load_kinematics>
    </ExternalLoads>
</OpenSimDocument>
"""

with open(XML_OUT_PATH, "w") as f:
    f.write(xml_content)

print(f"Saved OpenSim XML file to: {XML_OUT_PATH}")