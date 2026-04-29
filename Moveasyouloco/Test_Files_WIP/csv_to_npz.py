import argparse
import csv
import os

import mujoco
import numpy as np

from loco_mujoco.trajectory.dataclasses import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel

# Default configuration variables
INPUT_CSV_PATH = "/home/fholleman/Documents/Codefiles/BEP/loco-mujoco/onesteplong.csv"
OUTPUT_NPZ_PATH = None  # If None, output path is derived from the input path.
OUTPUT_FREQUENCY = None  # If provided, overrides the frequency inferred from the time column.

FREE_QPOS_SUFFIXES = ["_tx", "_ty", "_tz", "_qw", "_qx", "_qy", "_qz"]
FREE_QVEL_SUFFIXES = ["_vx", "_vy", "_vz", "_wx", "_wy", "_wz"]


def is_free_qpos_name(name):
    return any(name.endswith(suffix) for suffix in FREE_QPOS_SUFFIXES)


def is_free_qvel_name(name):
    return any(name.endswith(suffix) for suffix in FREE_QVEL_SUFFIXES)


def extract_joint_groups(qpos_names, qvel_names):
    joint_names = []
    joint_types = []
    i = 0
    j = 0

    while i < len(qpos_names):
        if is_free_qpos_name(qpos_names[i]):
            prefix = qpos_names[i][:-3]
            expected_qpos = [prefix + suffix for suffix in FREE_QPOS_SUFFIXES]
            expected_qvel = [prefix + suffix for suffix in FREE_QVEL_SUFFIXES]
            if qpos_names[i:i + 7] != expected_qpos:
                raise ValueError(f"Malformed free joint position names around {qpos_names[i:i + 7]}.")
            if qvel_names[j:j + 6] != expected_qvel:
                raise ValueError(f"Malformed free joint velocity names around {qvel_names[j:j + 6]}.")
            joint_names.append(prefix)
            joint_types.append(mujoco.mjtJoint.mjJNT_FREE)
            i += 7
            j += 6
        else:
            if j >= len(qvel_names) or qpos_names[i] != qvel_names[j]:
                raise ValueError(
                    f"Non-free qpos/qvel name mismatch: qpos={qpos_names[i]} qvel={qvel_names[j] if j < len(qvel_names) else None}"
                )
            joint_names.append(qpos_names[i])
            joint_types.append(mujoco.mjtJoint.mjJNT_HINGE)
            i += 1
            j += 1

    if j != len(qvel_names):
        raise ValueError("Extra qvel columns were found after parsing all qpos joints.")

    return joint_names, joint_types


def split_csv_columns(column_names):
    for split in range(1, len(column_names)):
        qpos_names = column_names[:split]
        qvel_names = column_names[split:]
        try:
            extract_joint_groups(qpos_names, qvel_names)
            return qpos_names, qvel_names
        except ValueError:
            continue
    raise ValueError("Unable to infer qpos/qvel partition from CSV header.")


def read_csv_trajectory(csv_path):
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV file is empty.")

        if header[0].strip().lower() != "time":
            raise ValueError("CSV file must begin with a 'time' column.")

        column_names = [name.strip() for name in header[1:]]
        qpos_names, qvel_names = split_csv_columns(column_names)

        rows = [row for row in reader if row]
        if not rows:
            raise ValueError("CSV file does not contain any data rows.")

        data = np.array(rows, dtype=float)
        times = data[:, 0]
        values = data[:, 1:]

        qpos = values[:, : len(qpos_names)]
        qvel = values[:, len(qpos_names) :]

    return times, qpos, qvel, qpos_names, qvel_names


def infer_frequency(times):
    if len(times) < 2:
        raise ValueError("Need at least two time steps to infer frequency.")
    dt = np.diff(times)
    if not np.allclose(dt, dt[0], atol=1e-8, rtol=1e-5):
        print("Warning: time steps are not perfectly uniform. Frequency will use the average step.")
    return float(1.0 / np.mean(dt))


def build_trajectory(times, qpos, qvel, qpos_names, qvel_names, frequency=None):
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.asarray(qvel, dtype=float)

    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError("qpos and qvel must be 2D arrays.")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError("qpos and qvel must have the same number of rows.")

    joint_names, joint_types = extract_joint_groups(qpos_names, qvel_names)

    if frequency is None:
        frequency = infer_frequency(times)

    info = TrajectoryInfo(
        joint_names=joint_names,
        model=TrajectoryModel(njnt=len(joint_types), jnt_type=np.array(joint_types, dtype=int)),
        frequency=frequency,
    )

    data = TrajectoryData(
        qpos=qpos,
        qvel=qvel,
        split_points=np.array([0, qpos.shape[0]], dtype=int),
    )

    return Trajectory(info=info, data=data)


def main():
    parser = argparse.ArgumentParser(description="Convert a LocoMuJoCo trajectory CSV file into a trajectory NPZ file.")
    parser.add_argument(
        "--csv",
        type=str,
        default=INPUT_CSV_PATH,
        help=f"Path to the input CSV file. Defaults to {INPUT_CSV_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_NPZ_PATH,
        help="Optional output path. If omitted, uses the same base name with .npz.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=OUTPUT_FREQUENCY,
        help="Optional trajectory frequency. If omitted, it is inferred from the time column.",
    )
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    times, qpos, qvel, qpos_names, qvel_names = read_csv_trajectory(csv_path)
    trajectory = build_trajectory(times, qpos, qvel, qpos_names, qvel_names, frequency=args.frequency)

    if args.output is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        args.output = f"{base}.npz"

    output_path = os.path.expanduser(args.output)
    trajectory.save(output_path)
    print(f"Saved NPZ to {output_path}")


if __name__ == "__main__":
    main()
