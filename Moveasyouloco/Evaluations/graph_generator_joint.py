import argparse
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import yaml
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='generate graph from .mot')
parser.add_argument('--path', type=str, required=True, help='path to the .mot file')
parser.add_argument('--outputpath', type=str, required=True, help='path to the output image file')
parser.add_argument('--joints', nargs='+', required=True, help='list of joints to plot (space-separated)')
args = parser.parse_args()

def get_mot_data_type(mot_filepath):
    try:
        with open(mot_filepath, 'r') as f:
            first_line = f.readline().strip().lower()
            if 'torque' in first_line:
                return 'torque'
            if 'kinematics' in first_line:
                return 'kinematics'
            print(f"Warning: Could not determine data type from header '{first_line}'. Defaulting to 'torque'.")
            return 'torque'
    except FileNotFoundError:
        print(f"Error: The file '{mot_filepath}' was not found.")
        return None


def load_mot_data(mot_filepath):
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


def read_joint_limits(xml_path):
    joint_limits = {}
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Warning: Could not parse XML file '{xml_path}': {e}")
        return joint_limits
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        range_text = joint.get('range')
        if name and range_text:
            parts = range_text.strip().split()
            if len(parts) == 2:
                try:
                    joint_limits[name] = (float(parts[0]), float(parts[1]))
                except ValueError:
                    pass
    return joint_limits


def find_xml_path_from_hydra(mot_filepath):
    hydra_config = Path(mot_filepath).resolve().parent.parent / '.hydra' / 'config.yaml'
    if not hydra_config.exists():
        return None
    try:
        config = yaml.safe_load(hydra_config.read_text())
        if isinstance(config, dict):
            if 'env_params' in config and 'model_path' in config['env_params']:
                return config['env_params']['model_path']
            experiment = config.get('experiment', {})
            if isinstance(experiment, dict):
                env_params = experiment.get('env_params', {})
                if isinstance(env_params, dict) and 'model_path' in env_params:
                    return env_params['model_path']
    except Exception as e:
        print(f"Warning: Error loading config from '{hydra_config}': {e}")
    return None


def find_traj_path_from_hydra(mot_filepath):
    hydra_config = Path(mot_filepath).resolve().parent.parent / '.hydra' / 'config.yaml'
    if not hydra_config.exists():
        return None
    def normalize(raw_path):
        if not isinstance(raw_path, str):
            return None
        raw_path = raw_path.strip()
        if raw_path.endswith('_converted.npz'):
            return raw_path[:-len('_converted.npz')] + '.mot'
        if raw_path.endswith('.npz'):
            return raw_path[:-4] + '.mot'
        return raw_path
    try:
        config = yaml.safe_load(hydra_config.read_text())
        if isinstance(config, dict):
            cdc = config.get('custom_dataset_conf', {})
            if isinstance(cdc, dict) and 'traj' in cdc:
                return normalize(cdc['traj'])
            experiment = config.get('experiment', {})
            if isinstance(experiment, dict):
                env_params = experiment.get('env_params', {})
                if isinstance(env_params, dict):
                    cdc = env_params.get('custom_dataset_conf', {})
                    if isinstance(cdc, dict) and 'traj' in cdc:
                        return normalize(cdc['traj'])
                task_factory = experiment.get('task_factory', {})
                if isinstance(task_factory, dict):
                    params = task_factory.get('params', {})
                    if isinstance(params, dict):
                        cdc = params.get('custom_dataset_conf', {})
                        if isinstance(cdc, dict) and 'traj' in cdc:
                            return normalize(cdc['traj'])
    except Exception as e:
        print(f"Warning: Error loading config from '{hydra_config}': {e}")
    return None


def load_traj_data(traj_path):
    traj_p = Path(traj_path)
    if not traj_p.exists():
        alt = Path(__file__).resolve().parents[1] / traj_path
        if alt.exists():
            traj_p = alt
        else:
            return None
    suffix = traj_p.suffix.lower()
    try:
        if suffix == '.mot':
            return load_mot_data(str(traj_p))
        if suffix in {'.csv', '.txt'}:
            return pd.read_csv(traj_p)
        if suffix == '.npz':
            data = np.load(traj_p, allow_pickle=True)
            cols = {k: data[k] for k in data.files if k.endswith('_opencap')}
            if cols:
                df = pd.DataFrame(cols)
                if 'time' in data.files:
                    df['time'] = data['time']
                return df
            if hasattr(data, 'dtype') and getattr(data, 'dtype').names:
                return pd.DataFrame({n: data[n] for n in data.dtype.names})
    except Exception:
        return None
    return None


def resolve_joint_name(joint, columns):
    mapping = {
        'lumbar_ext': 'lumbar_extension',
        'lumbar_bend': 'lumbar_bending',
        'lumbar_rot': 'lumbar_rotation',
    }
    if joint in columns:
        return joint
    return mapping.get(joint, joint)


def generate_plot(mot_file, output_file, joints):
    print(f"Loading data from {mot_file}...")
    data_type = get_mot_data_type(mot_file)
    if data_type is None:
        return
    try:
        df = load_mot_data(mot_file)
        if df is None:
            return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    if 'time' not in df.columns:
        print("Error: The MOT file must contain a 'time' column.")
        return
    print(f"Plotting {len(joints)} joints...")

    joint_limits = {}
    traj_df = None
    if data_type == 'kinematics':
        xml_path = find_xml_path_from_hydra(mot_file)
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[1] / 'Models' / 'skeleton' / 'skeleton_torque.xml'
        xml_path = Path(xml_path)
        if xml_path.exists():
            joint_limits = read_joint_limits(xml_path)
        else:
            print(f"Warning: XML file for joint limits not found at '{xml_path}'. Skipping limit lines.")

        traj_path = find_traj_path_from_hydra(mot_file)
        if traj_path:
            loaded = load_traj_data(traj_path)
            if loaded is not None:
                traj_df = loaded
            else:
                date_folder = Path(mot_file).resolve().parent.parent
                alt = date_folder / traj_path
                if alt.exists():
                    traj_df = load_traj_data(alt)
                else:
                    repo_alt = Path(__file__).resolve().parents[1] / traj_path
                    if repo_alt.exists():
                        traj_df = load_traj_data(repo_alt)

    fig, ax = plt.subplots(figsize=(12, 6))

    plotted_count = 0
    for joint in joints:
        actual_joint = resolve_joint_name(joint, df.columns)
        if actual_joint in df.columns:
            line, = ax.plot(df['time'], df[actual_joint], label=joint, linewidth=1.5)
            plotted_count += 1
            if actual_joint in joint_limits:
                min_val, max_val = joint_limits[actual_joint]
                ax.axhline(min_val, color='red', linestyle=':', linewidth=1.5, alpha=0.9,
                           label='_nolegend_')
                ax.axhline(max_val, color='red', linestyle=':', linewidth=1.5, alpha=0.9,
                           label='_nolegend_')
                ax.plot([], [], color='red', linestyle=':', linewidth=1.5,
                        label=f"{joint} min/max")
            if traj_df is not None:
                overlay_series = None
                overlay_label = None
                if actual_joint in traj_df.columns:
                    overlay_series = traj_df[actual_joint]
                    overlay_label = f"{joint}_opencap"
                else:
                    joint_lower = joint.lower()
                    candidates = [c for c in traj_df.columns if c.lower().endswith('_opencap') and joint_lower in c.lower()]
                    if not candidates:
                        candidates = [c for c in traj_df.columns if c.lower().endswith('_opencap') and any(part in c.lower() for part in joint_lower.split('_'))]
                    if candidates:
                        overlay_series = traj_df[candidates[0]]
                        overlay_label = candidates[0]

                if overlay_series is not None:
                    overlay_values = np.deg2rad(np.asarray(overlay_series))
                    if 'knee' in joint.lower():
                        overlay_values = -overlay_values
                    if 'time' in traj_df.columns:
                        t_ref = np.asarray(traj_df['time'])
                        t_target = np.asarray(df['time'])
                        try:
                            y_interp = np.interp(t_target, t_ref, overlay_values)
                            ax.plot(t_target, y_interp, label=overlay_label, linestyle=':', linewidth=1.0,
                                    color=line.get_color(), alpha=0.9)
                        except Exception:
                            if len(traj_df) == len(df):
                                ax.plot(df['time'], overlay_values, label=overlay_label, linestyle=':', linewidth=1.0,
                                        color=line.get_color(), alpha=0.9)
                    elif len(traj_df) == len(df):
                        ax.plot(df['time'], overlay_values, label=overlay_label, linestyle=':', linewidth=1.0,
                                color=line.get_color(), alpha=0.9)
        else:
            print(f"Warning: Joint '{joint}' not found in the file! Skipping...")

    if plotted_count == 0:
        print("Error: None of the requested joints were found in the file. Plot cancelled.")
        return

    title = 'Joint Kinematics over Time' if data_type == 'kinematics' else 'Joint Torques over Time'
    ylabel = 'Angle (radians)' if data_type == 'kinematics' else 'Torque (Nm)'
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Success! Plot saved locally as: {output_file}")
    plt.show()

generate_plot(args.path, args.outputpath, args.joints)