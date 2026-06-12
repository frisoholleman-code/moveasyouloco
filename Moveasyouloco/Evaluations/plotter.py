import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_biomechanical_envelopes(csv_filepath, joints_to_plot, output_image_path=None):
    """
    Plots mean +/- SD envelopes for selected joints across 0-100% cycle duration.

    Parameters:
    - csv_filepath: Path to the '_normalized_envelope.csv' file.
    - joints_to_plot: List of string names of the coordinates to visualize.
    - output_image_path: Path to save the high-resolution figure (e.g., 'figure7.png').
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: The data file '{csv_filepath}' does not exist. Please run the processing script first.")
        return

    # 1. Load the normalized cycle data
    df = pd.read_csv(csv_filepath)
    x = df['percent_duration'].values  # This goes from 0 to 100%

    # 2. Configure a clean, academic layout
    num_plots = len(joints_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(7, 3.5 * num_plots), sharex=True)

    # If plotting only a single joint, make axes iterable
    if num_plots == 1:
        axes = [axes]

    # Standard academic color scheme (using distinct shades for different curves if needed)
    line_color = '#1f77b4'  # Deep blue for the mean curve
    shade_color = '#1f77b4'  # Matching lighter blue for the SD band

    print(f"Generating envelope plots for: {joints_to_plot}...")

    # 3. Loop through each joint and construct the subplot
    for i, joint in enumerate(joints_to_plot):
        ax = axes[i]

        mean_col = f"{joint}_mean"
        sd_col = f"{joint}_sd"

        # Verify columns exist in the file
        if mean_col not in df.columns or sd_col not in df.columns:
            print(f"Warning: Columns for '{joint}' not found in the CSV. Skipping.")
            continue

        mean_vals = df[mean_col].values
        sd_vals = df[sd_col].values

        # Calculate boundaries for the SD envelope
        lower_bound = mean_vals - sd_vals
        upper_bound = mean_vals + sd_vals

        # Plot the main Mean trajectory line
        ax.plot(x, mean_vals, color=line_color, linewidth=2, label='RL Mean')

        # Fill the space between Mean - 1SD and Mean + 1SD
        ax.fill_between(x, lower_bound, upper_bound, color=shade_color, alpha=0.2, label='± 1 SD')

        # Formatting individual subplots
        ax.set_ylabel('Joint Torque (N·m)', fontsize=11, fontweight='bold')
        ax.set_title(f'Smoothed Torque Profile: {joint.replace("_", " ").title()}', fontsize=12, pad=10)

        # Clean, subtle gridlines for data reading
        ax.grid(True, linestyle='--', alpha=0.5, which='both')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)  # Zero line

        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    # 4. Format shared X-axis (0 to 100% Squat Duration)
    axes[-1].set_xlabel('Squat Cycle Duration (%)', fontsize=11, fontweight='bold')
    axes[-1].set_xlim(0, 100)

    plt.tight_layout()

    # 5. Save or Display the Figure
    if output_image_path:
        # Save at a high 300 DPI layout required by medical/engineering journals
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"High-resolution figure successfully saved to: {output_image_path}")
    else:
        plt.show()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Target the CSV output produced by the previous filtering step
    envelope_csv = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/torques_smoothed_ultra_RL_v2_r_normalized_envelope_v2.csv"

    # Define where you want the final high-quality paper image saved
    saved_figure_path = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Evaluations/RL_3_squat_torques_v2_r.png"

    # Select which coordinates you want to map as separate subplots.
    # You can add as many as you want (e.g., ['knee_angle_l', 'hip_flexion_l', 'lumbar_ext'])
    JOINTS_OF_INTEREST = ['knee_angle_r', 'knee_angle_l']

    plot_biomechanical_envelopes(
        csv_filepath=envelope_csv,
        joints_to_plot=JOINTS_OF_INTEREST,
        output_image_path=saved_figure_path
    )