import numpy as np
from scipy import signal
import pandas as pd
import os


def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    """
    Applies a zero-phase Butterworth low-pass filter to the data.
    """
    # Calculate sampling frequency based on the time array
    fs = 1 / np.round(np.mean(np.diff(time)), 16)

    # Calculate normalized cutoff frequency (Nyquist frequency is fs/2)
    wn = lowpass_cutoff_frequency / (fs / 2)

    # Safety check: if cutoff is higher than Nyquist, don't filter
    if wn >= 1.0:
        print(
            f"Warning: Cutoff frequency ({lowpass_cutoff_frequency} Hz) is >= Nyquist frequency ({fs / 2:.2f} Hz). Returning unfiltered data.")
        return data

    # Create the filter using order/2 because sosfiltfilt applies the filter twice (forward and backward)
    sos = signal.butter(order / 2, wn, btype='low', output='sos')

    # Apply zero-phase forward-backward filter
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt


def smooth_mot_file(input_filepath, output_filepath, cutoff_freq=6.0, order=4):
    """
    Reads a .mot file, applies a low-pass filter to the data, and saves it.
    """
    print(f"Processing: {input_filepath}...")

    # 1. Read the .mot file header
    # .mot files have metadata at the top, ending with 'endheader'
    header_lines = []
    with open(input_filepath, 'r') as file:
        for line in file:
            header_lines.append(line)
            if 'endheader' in line.strip():
                break

    skip_rows = len(header_lines)

    # 2. Read the tabular data using pandas
    # .mot files are usually tab or space separated
    df = pd.read_csv(input_filepath, sep='\s+', skiprows=skip_rows, header=0)

    # Ensure 'time' column exists
    if 'time' not in df.columns:
        raise ValueError("The .mot file must contain a 'time' column.")

    time = df['time'].values

    # Separate the data columns from the time column
    data_columns = [col for col in df.columns if col != 'time']
    data_to_filter = df[data_columns].values

    # 3. Apply the filter
    print(f"Applying {order}th order low-pass filter at {cutoff_freq} Hz...")
    filtered_data = lowPassFilter(time, data_to_filter, cutoff_freq, order)

    # Replace old data with filtered data in the dataframe
    df_filtered = df.copy()
    df_filtered[data_columns] = filtered_data

    # 4. Write the new .mot file
    print(f"Saving smoothed data to: {output_filepath}")

    # Write the original header first
    with open(output_filepath, 'w') as file:
        file.writelines(header_lines)

    # Append the filtered data
    # .mot files conventionally use tabs for separation
    df_filtered.to_csv(output_filepath, sep='\t', index=False, mode='a', float_format='%.6f')
    print("Done!\n")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Define your input and output files
    input_mot = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Input_Files/squat5.mot"  # Replace with your actual OpenCap file path
    output_mot = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/Moveasyouloco/Data_Conversion/Output_Files/squat5_smoothed.mot"

    # Common cutoff frequency for human motion (walking/running) is usually between 6Hz and 15Hz
    CUTOFF_FREQUENCY = 6.0
    FILTER_ORDER = 4

    # Ensure the input file exists before trying to run
    if os.path.exists(input_mot):
        smooth_mot_file(input_mot, output_mot, cutoff_freq=CUTOFF_FREQUENCY, order=FILTER_ORDER)
    else:
        print(f"Please place a valid OpenCap .mot file named '{input_mot}' in the same directory as this script.")