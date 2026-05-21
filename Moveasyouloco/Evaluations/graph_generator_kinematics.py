import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_mot_data(mot_filepath):
    """Helper function to load a .mot file into a pandas DataFrame."""
    skip_rows = 0
    try:
        with open(mot_filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.strip() == 'endheader':
                    skip_rows = i + 1
                    break
        return pd.read_csv(mot_filepath, sep='\t', skiprows=skip_rows)
    except FileNotFoundError:
        print(f"Error: The file '{mot_filepath}' was not found.")
        return None


def get_qpos_index(joint_names, jnt_types, target_joint):
    """
    Calculates the exact column index in the qpos matrix for a given joint.
    """
    if target_joint not in joint_names:
        return None

    if jnt_types is None or len(jnt_types) != len(joint_names):
        return joint_names.index(target_joint)

    current_idx = 0
    for name, jtype in zip(joint_names, jnt_types):
        if name == target_joint:
            return current_idx

        if jtype == 0:  # Free joint
            current_idx += 7
        elif jtype == 1:  # Ball joint
            current_idx += 4
        else:  # Slide (2) or Hinge (3) joint
            current_idx += 1

    return None


def plot_kinematics(mot_filepath_1=None, mot_filepath_2=None, npz_filepath=None, joints_to_plot=None, compare=False):
    """
    Reads and plots joint angles over time from up to two .mot files and one .npz file.
    """
    if joints_to_plot is None:
        joints_to_plot = []

    plt.figure(figsize=(10, 6))
    plotted_any = False

    # --- 1. Handle First .mot data ---
    if mot_filepath_1:
        df_mot1 = load_mot_data(mot_filepath_1)
        if df_mot1 is not None:
            if 'time' not in df_mot1.columns:
                print(f"Error: Could not find a 'time' column in {mot_filepath_1}.")
            else:
                time_mot1 = df_mot1['time']
                for joint in joints_to_plot:
                    if joint in df_mot1.columns:
                        label = f"{joint} (MOT 1)" if compare else joint
                        plt.plot(time_mot1, df_mot1[joint], label=label, linestyle='-', linewidth=2)
                        plotted_any = True
                    else:
                        print(f"Warning: Joint '{joint}' not found in first MOT file.")

    # --- 2. Handle Second .mot data ---
    if mot_filepath_2:
        df_mot2 = load_mot_data(mot_filepath_2)
        if df_mot2 is not None:
            if 'time' not in df_mot2.columns:
                print(f"Error: Could not find a 'time' column in {mot_filepath_2}.")
            else:
                time_mot2 = df_mot2['time']
                for joint in joints_to_plot:
                    if joint in df_mot2.columns:
                        label = f"{joint} (MOT 2)" if compare else joint
                        plt.plot(time_mot2, df_mot2[joint], label=label, linestyle='--', linewidth=2)
                        plotted_any = True
                    else:
                        print(f"Warning: Joint '{joint}' not found in second MOT file.")

    # --- 3. Handle .npz data ---
    if npz_filepath:
        try:
            npz_data = np.load(npz_filepath, allow_pickle=True)

            if 'qpos' in npz_data and 'joint_names' in npz_data:
                qpos_matrix = npz_data['qpos']

                raw_names = npz_data['joint_names']
                joint_names = [j.decode('utf-8') if isinstance(j, bytes) else str(j) for j in raw_names]

                jnt_types = npz_data['jnt_type'] if 'jnt_type' in npz_data else None

                # --- FIX: Reconstruct proper time array using frequency ---
                if 'time' in npz_data:
                    time_npz_plot = npz_data['time']
                elif 'frequency' in npz_data:
                    # Extract frequency value safely and calculate time
                    freq = np.asarray(npz_data['frequency']).item()
                    time_npz_plot = np.arange(len(qpos_matrix)) / freq
                else:
                    time_npz_plot = np.arange(len(qpos_matrix))

                for joint in joints_to_plot:
                    idx = get_qpos_index(joint_names, jnt_types, joint)

                    if idx is not None:
                        # Extract the specific column for this joint
                        joint_data = qpos_matrix[:, idx]

                        # --- FIX: Convert radians to degrees ---
                        joint_data_deg = np.rad2deg(joint_data)

                        label = f"{joint} (NPZ)" if compare else joint
                        linestyle = '-.' if compare else '-'
                        plt.plot(time_npz_plot, joint_data_deg, label=label, linestyle=linestyle, linewidth=2)
                        plotted_any = True
                    else:
                        print(f"Warning: Joint '{joint}' not found in NPZ 'joint_names'.")
            else:
                print("Error: NPZ file does not contain 'qpos' or 'joint_names'.")

        except FileNotFoundError:
            print(f"Error: The file '{npz_filepath}' was not found.")

    # --- 4. Finalize Plot ---
    if plotted_any:
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (Degrees)')
        title = 'Joint Kinematics Comparison' if compare else 'Joint Kinematics Over Time'
        plt.title(title)

        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid joints were plotted.")


if __name__ == "__main__":
    # --- File Paths ---
    mot_file_1 = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Input_Files/squat5.mot"

    mot_file_2 = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/squat5_smoothed.mot"

    # Add the path to your converted .npz file here:
    npz_file = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/squat5_converted.npz"

    # --- Parameters ---
    joints = ['knee_angle_l', 'knee_angle_r']

    # Set to True to plot both files together.
    # Set to False to just plot the .mot file (or just the .npz file if mot_file=None)
    compare_datasets = True

    plot_kinematics(mot_filepath_1=mot_file_1,
                    mot_filepath_2=mot_file_2,
                    # npz_filepath=npz_file,
                    joints_to_plot=joints,
                    compare=compare_datasets)