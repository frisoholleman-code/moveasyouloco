import os
import subprocess
DATE="2026-05-29"
TIME="14-10-18" 
EVALUATE= False  # set to False when run is already evaluated and you just want to generate the graph
PLOT_TORQUES= True   # set to True to evaluate and/or plot torque output
PLOT_KINEMATICS= True   # set to True to evaluate and/or plot kinematics output
GRAPH_NAMES = ["knee"] # list of graph name prefixes to be generated
JOINTS_TO_PLOT = [['knee_angle_r','knee_angle_l']]

    #'lumbar_ext','lumbar_bend','lumbar_rot',
    #'shoulder_flex_g br','shoulder_add_r','shoulder_rot_r','shoulder_flex_l','shoulder_add_l','shoulder_rot_l','elbow_flex_r','elbow_flex_l',
    #'pro_sup_r','pro_sup_l',
    #'wrist_flex_r','wrist_flex_l','wrist_dev_r','wrist_dev_l',
    #'hip_flexion_r','hip_adduction_r','hip_rotation_r','hip_flexion_l','hip_adduction_l','hip_rotation_l',
    #'knee_angle_r','knee_angle_l',
    #'ankle_angle_r','ankle_angle_l'

args_path = f"outputs/{DATE}/{TIME}/PPOJax_saved.pkl"
output_dir = f"outputs/{DATE}/{TIME}/evaluations"
os.makedirs(output_dir, exist_ok=True)
args_output = f"{output_dir}/"

if EVALUATE:
    # Run combined evaluation (produces both torques.mot and kinematics.mot in one pass)
    subprocess.run(["python", "Moveasyouloco/Evaluations/evaluate.py", 
                    "--path", args_path,
                    "--outputpath", args_output,
                    "--n_steps", "2000"])

if PLOT_TORQUES:
    for name in GRAPH_NAMES:
        subprocess.run(["python", "Moveasyouloco/Evaluations/graph_generator_joint.py",
                        "--path", f"{args_output}torques.mot",
                        "--outputpath", f"{args_output}torque_{name}.png",
                        "--joints"] + JOINTS_TO_PLOT[GRAPH_NAMES.index(name)])

if PLOT_KINEMATICS:
    for name in GRAPH_NAMES:
        subprocess.run(["python", "Moveasyouloco/Evaluations/graph_generator_joint.py",
                        "--path", f"{args_output}kinematics.mot",
                        "--outputpath", f"{args_output}kinematics_{name}.png",
                        "--joints"] + JOINTS_TO_PLOT[GRAPH_NAMES.index(name)])

