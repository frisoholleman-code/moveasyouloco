import os
import subprocess
DATE="2026-05-21"
TIME="12-51-46" 
EVALUATE=False   #set to False when run is already evaluated and you just want to generate the graph, set to True when you want to run the evaluation and generate the graph
PLOT_GRAPH=True #set to False to skip graph generation, set to True to generate the graph after evaluation, can be set to True even if EVALUATE is False, as long as the evaluation has already been run and the necessary .csv file is available in the output directory
GRAPH_NAME = "hip_torque_graph"
JOINTS_TO_PLOT = [
    #'lumbar_ext',
    #'lumbar_bend',
    #'lumbar_rot',
    #'shoulder_flex_r',
    #'shoulder_add_r',
    #'shoulder_rot_r',
    #'shoulder_flex_l',
    #'shoulder_add_l',
    #'shoulder_rot_l',    
    #'elbow_flex_r',
    #'elbow_flex_l',
    #'pro_sup_r',
    #'pro_sup_l',
    #'wrist_flex_r',
    #'wrist_flex_l',
    #'wrist_dev_r',
    #'wrist_dev_l',
    #'hip_flexion_r',
    #'hip_adduction_r',
    #'hip_rotation_r',
    #'hip_flexion_l',
    #'hip_adduction_l',
    #'hip_rotation_l',
    #'knee_angle_r',
    #'knee_angle_l',
    #'ankle_angle_r',
    #'ankle_angle_l'
]

args_path = f"outputs/{DATE}/{TIME}/PPOJax_saved.pkl"
output_dir = f"outputs/{DATE}/{TIME}/evaluations"
os.makedirs(output_dir, exist_ok=True)
args_output = f"{output_dir}/"

if EVALUATE:
    #Evaluate torques, run eval_torques
    subprocess.run(["python", "Moveasyouloco/Tests/eval_torques.py", 
                    "--path", args_path,
                    "--outputpath", args_output,
                    "--n_steps", "1000",
                    "--save_torques"])

    #Convert the .mot to .csv
    subprocess.run(["python", "Moveasyouloco/Tests/mot_to_csv_converter.py",
                    "--path", f"{args_output}torques.mot",
                    "--outputpath", f"{args_output}torques.csv"])

    #.mot no longer necessary, can be deleted if desired, uncomment line below to delete .mot
    os.remove(f"{args_output}torques.mot")

if PLOT_GRAPH:
    #Generate graphs
    subprocess.run(["python", "Moveasyouloco/Tests/graph_generator_joint.py",
                    "--path", f"{args_output}torques.csv",
                    "--outputpath", f"{args_output}{GRAPH_NAME}.png",
                    "--joints"] + JOINTS_TO_PLOT)

