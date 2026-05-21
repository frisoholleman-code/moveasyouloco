import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ==========================================
# --- CONFIGURATION ---
# ==========================================

parser = argparse.ArgumentParser(description='generate graph from .csv')
parser.add_argument('--path', type=str, required=True, help='path to the .csv file')
parser.add_argument('--outputpath', type=str, required=True, help='path to the output image file')
parser.add_argument('--joints', nargs='+', required=True, help='list of joints to plot (space-separated)')
args = parser.parse_args()

# ==========================================
def generate_torque_plot(csv_file, output_file, joints):
    print(f" Loading data from {csv_file}...")

    # Load the CSV data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f" Error: Could not find the file '{csv_file}'. Make sure it is in the same folder.")
        return

    # Ensure the time column exists for the X-axis
    if 'time' not in df.columns:
        print("Error: The CSV file must contain a 'time' column.")
        return

    print(f" Plotting {len(joints)} joints...")

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each requested joint
    plotted_count = 0
    for joint in joints:
        if joint in df.columns:
            ax.plot(df['time'], df[joint], label=joint, linewidth=1.5)
            plotted_count += 1
        else:
            print(f" Warning: Joint '{joint}' not found in the CSV! Skipping...")

    if plotted_count == 0:
        print(" Error: None of the requested joints were found in the CSV. Plot cancelled.")
        return

    # Formatting the graph to look clean and professional
    ax.set_title('Joint Torques over Time', fontsize=14, pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Torque (Nm)', fontsize=12)

    # Move the legend outside the plot area if there are many joints
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    ax.grid(True, linestyle='--', alpha=0.7)

    # Prevent the legend from being cut off when saving
    plt.tight_layout()

    # Save the image
    plt.savefig(output_file, dpi=300)
    print(f" Success! Plot saved locally as: {output_file}")

    # Display the plot in a window
    plt.show()

generate_torque_plot(args.path, args.outputpath, args.joints)