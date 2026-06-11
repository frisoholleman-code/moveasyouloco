import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


# ==========================================
# 1. Filter Definitions
# ==========================================

def butterworth_filter(time, data, cutoff_freq=6.0, order=4):
    """Applies a zero-phase Butterworth low-pass filter."""
    fs = 1 / np.round(np.mean(np.diff(time)), 16)
    wn = cutoff_freq / (fs / 2)

    if wn >= 1.0:
        print(f"Warning: Cutoff frequency >= Nyquist. Returning unfiltered data.")
        return data

    sos = signal.butter(order / 2, wn, btype='low', output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)


def moving_average_filter(data, window_size=7):
    """Applies a centered moving average filter."""
    if window_size > data.shape[0]: window_size = data.shape[0]
    return uniform_filter1d(data, size=window_size, axis=0)


def savitzky_golay_filter(data, window_size=15, polyorder=3):
    """Applies a Savitzky-Golay filter to preserve peaks while smoothing noise."""
    if window_size % 2 == 0: window_size += 1
    if window_size > data.shape[0]:
        window_size = data.shape[0] if data.shape[0] % 2 != 0 else data.shape[0] - 1
    return savgol_filter(data, window_length=window_size, polyorder=polyorder, axis=0, mode='interp')


# ==========================================
# 2. Time-Window Segmentation
# ==========================================

def segment_by_smart_windows(time_array, data_matrix, column_names, time_windows, tracking_column='knee_angle_r'):
    """
    Finds the deepest point of the squat within the rough time window,
    and extracts exactly symmetrical data around that center point.
    """
    normalized_points = 101
    all_cycles = []

    # How much time to grab before and after the deepest point (e.g., 0.8 seconds = 1.6s total cycle)
    # Adjust this if the squat is faster or slower!
    padding_seconds = 0.8

    try:
        track_idx = column_names.index(tracking_column)
    except ValueError:
        raise ValueError(f"Could not find '{tracking_column}' for alignment.")

    print(f"Aligning cycles dynamically around the deepest point of '{tracking_column}'...")

    for idx, (t_start, t_end) in enumerate(time_windows):
        start_idx = np.searchsorted(time_array, t_start)
        end_idx = np.searchsorted(time_array, t_end)

        # 1. Isolate the rough window
        window_time = time_array[start_idx:end_idx]
        window_data = data_matrix[start_idx:end_idx, track_idx]

        # 2. Find the exact frame of the deepest squat (minimum value in this window)
        # Note: If your joint goes positive during a squat, change np.argmin to np.argmax
        deepest_local_idx = np.argmin(window_data)
        deepest_absolute_idx = start_idx + deepest_local_idx
        deepest_time = time_array[deepest_absolute_idx]

        # 3. Create a perfect, symmetrical cut around that deepest point
        perfect_start_time = deepest_time - padding_seconds
        perfect_end_time = deepest_time + padding_seconds

        p_start_idx = np.searchsorted(time_array, perfect_start_time)
        p_end_idx = np.searchsorted(time_array, perfect_end_time)

        cycle_raw = data_matrix[p_start_idx:p_end_idx, :]
        cycle_len = p_end_idx - p_start_idx

        # Linearly interpolate each channel to the 0-100% time grid
        xp = np.linspace(0, 1, cycle_len)
        x_new = np.linspace(0, 1, normalized_points)

        cycle_interp = np.zeros((normalized_points, data_matrix.shape[1]))
        for col in range(data_matrix.shape[1]):
            cycle_interp[:, col] = np.interp(x_new, xp, cycle_raw[:, col])

        all_cycles.append(cycle_interp)
        print(
            f"  -> Squat {idx + 1} Aligned! Peak at {deepest_time:.2f}s. Extracted {perfect_start_time:.2f}s to {perfect_end_time:.2f}s")

    all_cycles = np.array(all_cycles)

    mean_envelope = np.mean(all_cycles, axis=0)
    sd_envelope = np.std(all_cycles, axis=0)

    return mean_envelope, sd_envelope


# ==========================================
# 3. Main File Processing
# ==========================================

def process_mot_file(input_filepath, output_filepath, filter_type='butterworth', **kwargs):
    print(f"Processing: {input_filepath}...")

    # Read Header
    header_lines = []
    with open(input_filepath, 'r') as file:
        for line in file:
            header_lines.append(line)
            if 'endheader' in line.strip(): break
    skip_rows = len(header_lines)

    # Read Data
    df = pd.read_csv(input_filepath, sep=r'\s+', skiprows=skip_rows, header=0)

    if 'time' not in df.columns: raise ValueError("The .mot file must contain a 'time' column.")

    time = df['time'].values
    data_columns = [col for col in df.columns if col != 'time']
    data_to_filter = df[data_columns].values

    # Apply Filter
    if filter_type == 'butterworth':
        cutoff, order = kwargs.get('bw_cutoff', 6.0), kwargs.get('bw_order', 4)
        print(f"Applying Butterworth filter (Cutoff: {cutoff}Hz, Order: {order})...")
        filtered_data = butterworth_filter(time, data_to_filter, cutoff_freq=cutoff, order=order)
    elif filter_type == 'savgol':
        window, poly = kwargs.get('sg_window', 15), kwargs.get('sg_poly', 3)
        print(f"Applying Savitzky-Golay filter (Window: {window}, Poly Order: {poly})...")
        filtered_data = savitzky_golay_filter(data_to_filter, window_size=window, polyorder=poly)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Process Cycles via Manual Time Windows
    time_windows = kwargs.get('time_windows', [])
    if time_windows:
        mean_data, sd_data = segment_by_smart_windows(time, filtered_data, data_columns, time_windows)

        if mean_data is not None:
            # Create a specialized DataFrame for the 0-100% Envelope Plotting
            norm_time = np.linspace(0, 100, 101)
            envelope_df = pd.DataFrame({'percent_duration': norm_time})

            for i, col_name in enumerate(data_columns):
                envelope_df[f'{col_name}_mean'] = mean_data[:, i]
                envelope_df[f'{col_name}_sd'] = sd_data[:, i]

            env_output = output_filepath.replace('.mot', '_normalized_envelope_v2.csv')
            envelope_df.to_csv(env_output, index=False)
            print(f"Saved biomechanical envelope data to: {env_output}")

    # Save continuous filtered file
    df_filtered = df.copy()
    df_filtered[data_columns] = filtered_data

    print(f"Saving smoothed continuous trial data to: {output_filepath}")
    with open(output_filepath, 'w') as file:
        file.writelines(header_lines)
    df_filtered.to_csv(output_filepath, sep='\t', index=False, mode='a', float_format='%.6f')
    print("Done!\n")


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":

    input_mot = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/RL_squat3.mot"
    output_mot = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/torques_smoothed_ultra_RL_v2_r.mot"

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    ACTIVE_FILTER = 'butterworth'

    # Kept slightly higher (12Hz) so we don't accidentally iron out all of the RL agent's peak torques
    BW_CUTOFF_FREQ = 6
    BW_ORDER = 4

    # The exact start and end times (in seconds) of the 3 squats based on visual inspection
    MANUAL_SQUAT_WINDOWS = [
        (1.0, 3.0),  # Squat 1
        (5.5, 7.5),  # Squat 2
        (9.0, 11.0)  # Squat 3
    ]
    # ---------------------------------------------------------

    if os.path.exists(input_mot):
        process_mot_file(
            input_mot,
            output_mot,
            filter_type=ACTIVE_FILTER,
            bw_cutoff=BW_CUTOFF_FREQ,
            bw_order=BW_ORDER,
            time_windows=MANUAL_SQUAT_WINDOWS
        )
    else:
        print(f"Error: Could not find '{input_mot}'.")