import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent.parent




def get_joint_indices(joint_names):
    """
    Recreates the qpos indexing logic from the MOT-to-NPZ conversion script.
    Returns a dictionary mapping joint names to their start index in qpos.
    """
    joint_to_idx = {}
    curr_idx = 0

    for name in joint_names:
        name_str = str(name).strip()
        if name_str == 'root':
            joint_to_idx['root'] = slice(0, 7)
            curr_idx += 7
        else:
            joint_to_idx[name_str] = curr_idx
            curr_idx += 1

    return joint_to_idx


def load_data(npz_path, pkl_path):
    """Loads reference trajectory and agent rollout data."""
    # Load Reference Data (NPZ)
    ref_data = dict(np.load(npz_path, allow_pickle=True))
    ref_qpos = ref_data['qpos']
    joint_names = ref_data['joint_names']

    # Map the names to their indices
    joint_mapping = get_joint_indices(joint_names)

    # Load Agent Data (PKL)
    with open(pkl_path, 'rb') as f:
        agent_data = pickle.load(f)

    agent_qpos = agent_data['qpos']

    return ref_qpos, agent_qpos, joint_mapping


def normalize_cycle(data, num_points=100):
    """Interpolates a single gait cycle to a 0-100% format."""
    original_time = np.linspace(0, 1, len(data))
    normalized_time = np.linspace(0, 1, num_points)
    interpolator = interp1d(original_time, data, axis=0, kind='cubic')
    return interpolator(normalized_time)


def segment_gait_cycles(qpos_column, cycle_length_approx=100):
    """
    Placeholder for segmenting continuous rollout data into individual strides.
    You will need to replace this logic with actual gait event detection
    (e.g., heel strikes or kinematic peaks).
    """
    # Mock segmentation: splitting array into chunks
    num_cycles = len(qpos_column) // cycle_length_approx
    cycles = []

    for i in range(num_cycles):
        start = i * cycle_length_approx
        end = start + cycle_length_approx
        cycle_data = qpos_column[start:end]
        cycles.append(normalize_cycle(cycle_data))

    return np.array(cycles)


def plot_joint_kinematics(ax, ref_data, agent_data, ylabel, show_xlabel=False):
    """Plots the mean and std for a single joint on a given axis."""
    time_percent = np.linspace(0, 100, 100)

    # Calculate Mean and Std over the extracted cycles
    ref_mean = np.mean(ref_data, axis=0)
    ref_std = np.std(ref_data, axis=0)

    agent_mean = np.mean(agent_data, axis=0)
    agent_std = np.std(agent_data, axis=0)

    # Plot Agent (Blue)
    ax.plot(time_percent, agent_mean, label='Latent', color='#1f77b4')
    ax.fill_between(time_percent, agent_mean - agent_std, agent_mean + agent_std, color='#1f77b4', alpha=0.3)

    # Plot Reference (Orange)
    ax.plot(time_percent, ref_mean, label='Reference', color='#ff7f0e')
    ax.fill_between(time_percent, ref_mean - ref_std, ref_mean + ref_std, color='#ff7f0e', alpha=0.3)

    # Formatting
    ax.set_ylabel(ylabel, fontsize=14)
    if show_xlabel:
        ax.set_xlabel('t [cycle %]', fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.set_xlim(0, 100)


def main():
    walk_ref_path = BASE_DIR / "Data_Conversion" / "Output_Files" / "squat5_converted.npz"
    walk_agent_path = BASE_DIR.parent / "outputs" / "butterfly-118" / "PPOJax_saved_butterfly_118.pkl"

    walk_ref_qpos, walk_agent_qpos, walk_map = load_data(walk_ref_path, walk_agent_path)

    # Define the exact OpenSim/MOT joint names you want to plot
    # Change these if you are targeting the left leg ('_l')
    TARGET_JOINTS = {
        'Hip': 'hip_flexion_r',
        'Knee': 'knee_angle_r',
        'Ankle': 'ankle_angle_r'
    }

    # Data Processing function
    def process_joint(qpos_data, joint_name, joint_mapping):
        idx = joint_mapping[joint_name]
        joint_col = qpos_data[:, idx]
        return segment_gait_cycles(joint_col)

    data_dict = {
        'Walking': {}
    }

    # Process Walking
    for j_key, j_name in TARGET_JOINTS.items():
        ref_cycles = process_joint(walk_ref_qpos, j_name, walk_map)
        agent_cycles = process_joint(walk_agent_qpos, j_name, walk_map)
        data_dict['Walking'][j_key] = (ref_cycles, agent_cycles)

    # --- Plotting ---
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    conditions = ['Walking', 'Running']
    joints = [('Hip', 'Hip Flexion [rad]'), ('Knee', 'Knee Angle [rad]'), ('Ankle', 'Ankle Angle [rad]')]

    for col, condition in enumerate(conditions):
        axs[0, col].set_title(condition, fontsize=20, pad=15)

        for row, (joint_key, ylabel) in enumerate(joints):
            ref_cycles, agent_cycles = data_dict[condition][joint_key]
            show_x = (row == 2)
            plot_joint_kinematics(axs[row, col], ref_cycles, agent_cycles, ylabel, show_xlabel=show_x)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


if __name__ == "__main__":
    main()