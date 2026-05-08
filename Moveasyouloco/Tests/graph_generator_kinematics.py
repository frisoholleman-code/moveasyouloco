import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_mot_kinematics(mot_filepath, joints_to_plot):
    """
    Reads a .mot file and plots the specified joint angles over time.
    """
    # 1. Find the end of the header
    skip_rows = 0
    with open(mot_filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                skip_rows = i + 1
                break

    # 2. Read the data into a pandas DataFrame
    # .mot files are typically tab-separated
    try:
        df = pd.read_csv(mot_filepath, sep='\t', skiprows=skip_rows)
    except FileNotFoundError:
        print(f"Error: The file '{mot_filepath}' was not found.")
        return

    # 3. Verify 'time' column exists
    if 'time' not in df.columns:
        print("Error: Could not find a 'time' column in the data.")
        return

    time_data = df['time']

    # 4. Plot the data
    plt.figure(figsize=(10, 6))

    plotted_any = False
    for joint in joints_to_plot:
        if joint in df.columns:
            plt.plot(time_data, df[joint], label=joint)
            plotted_any = True
        else:
            print(f"Warning: Joint '{joint}' not found in the file's columns.")

    if plotted_any:
        plt.xlabel('Time (s)')
        plt.ylabel('Angle')
        plt.title('Joint Angles Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid joints were plotted. Check your column names.")


if __name__ == "__main__":
    # Example usage:
    # Replace 'kinematics.mot' with the name of your file
    mot_file = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Input_Files/squat1_29april.mot"

    # Specify the exact column names you want to visualize from your file
    joints = ['knee_angle_r', 'hip_flexion_r', 'ankle_angle_r']

    plot_mot_kinematics(mot_file, joints)