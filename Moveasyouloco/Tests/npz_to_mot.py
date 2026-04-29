import argparse
import os
import csv

import numpy as np

from loco_mujoco.trajectory.dataclasses import Trajectory

# Default configuration variables
INPUT_NPZ_PATH = "/home/fholleman/Downloads/onesteplong.npz"
OUTPUT_FORMAT = "csv"  # choose between "mot" or "csv"
WRITE_DEGREES = True  # if True, qpos values are written in degrees
OUTPUT_PATH = None  # if None, output path is derived from input pathn


def expand_joint_names(joint_names, joint_types):
    qpos_names = []
    qvel_names = []
    for name, jtype in zip(joint_names, joint_types):
        if jtype == 0:  # mjJNT_FREE
            qpos_names.extend([
                f"{name}_tx",
                f"{name}_ty",
                f"{name}_tz",
                f"{name}_qw",
                f"{name}_qx",
                f"{name}_qy",
                f"{name}_qz",
            ])
            qvel_names.extend([
                f"{name}_vx",
                f"{name}_vy",
                f"{name}_vz",
                f"{name}_wx",
                f"{name}_wy",
                f"{name}_wz",
            ])
        else:
            qpos_names.append(name)
            qvel_names.append(name)
    return qpos_names, qvel_names


def save_as_mot(output_path, times, qpos, qvel, qpos_names, qvel_names, in_degrees=False):
    n_rows = len(times)
    n_cols = 1 + qpos.shape[1] + qvel.shape[1]
    header_lines = [
        "Coordinates",
        "version=1",
        f"nRows={n_rows}",
        f"nColumns={n_cols}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "",
        "Units are S.I. units (second, meters, Newtons, ...)",
        "",
        "endheader",
    ]
    column_names = ["time"] + qpos_names + qvel_names
    with open(output_path, "w", newline="") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("\t".join(column_names) + "\n")
        for i in range(n_rows):
            row = [times[i]] + qpos[i].tolist() + qvel[i].tolist()
            f.write("\t".join(str(x) for x in row) + "\n")


def save_as_csv(output_path, times, qpos, qvel, qpos_names, qvel_names):
    column_names = ["time"] + qpos_names + qvel_names
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for i in range(len(times)):
            writer.writerow([times[i]] + qpos[i].tolist() + qvel[i].tolist())


def main():
    parser = argparse.ArgumentParser(description="Convert a LocoMuJoCo trajectory NPZ into a .mot or CSV file.")
    parser.add_argument(
        "--npz",
        type=str,
        default=INPUT_NPZ_PATH,
        help=f"Path to the input trajectory NPZ file. Defaults to {INPUT_NPZ_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Optional output path. If omitted, uses the same base name with .mot or .csv.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mot", "csv"],
        default=OUTPUT_FORMAT,
        help=f"Output format: mot or csv. Defaults to {OUTPUT_FORMAT}.",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        default=WRITE_DEGREES,
        help=f"Write qpos values in degrees rather than radians. Defaults to {WRITE_DEGREES}.",
    )
    args = parser.parse_args()

    npz_path = os.path.expanduser(args.npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Input NPZ file not found: {npz_path}")

    trajectory = Trajectory.load(npz_path, backend=np)
    qpos = np.array(trajectory.data.qpos)
    qvel = np.array(trajectory.data.qvel)

    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError("Trajectory qpos/qvel must be 2D arrays.")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError("qpos and qvel must have the same number of rows.")

    if args.degrees:
        qpos = np.degrees(qpos)

    joint_names = trajectory.info.joint_names
    joint_types = np.array(trajectory.info.model.jnt_type)
    qpos_names, qvel_names = expand_joint_names(joint_names, joint_types)

    dt = 1.0 / trajectory.info.frequency
    times = np.arange(qpos.shape[0]) * dt

    if args.output is None:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        args.output = f"{base}.{args.format}"

    output_path = os.path.expanduser(args.output)
    if args.format == "mot":
        save_as_mot(output_path, times, qpos, qvel, qpos_names, qvel_names, in_degrees=args.degrees)
    else:
        save_as_csv(output_path, times, qpos, qvel, qpos_names, qvel_names)

    print(f"Saved {args.format.upper()} to {output_path}")


if __name__ == "__main__":
    main()
