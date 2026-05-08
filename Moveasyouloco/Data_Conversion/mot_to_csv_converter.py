import pandas as pd
import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)
model_to_evaluate = "butterfly-118"
INPUT_DIR = BASE_DIR / "outputs" / f"{model_to_evaluate}" / "PPOJax_saved_butterfly_118_torques.mot"
OUTPUT_CSV = BASE_DIR / "outputs" / f"{model_to_evaluate}" / f"{model_to_evaluate}.csv"

def mot_to_csv_pandas(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    try:
        # Step 1: Find where the header ends
        skip_rows = 0
        with open(input_file, 'r') as f:
            for line in f:
                skip_rows += 1
                if line.strip().lower() == 'endheader':
                    break

        # Step 2: Read the tab-separated data into a DataFrame
        # If your .mot file doesn't have an 'endheader' and is just plain tab-delimited,
        # you can remove the `skiprows=skip_rows` parameter.
        df = pd.read_csv(input_file, sep='\t', skiprows=skip_rows)

        # Step 3: Save the DataFrame to a CSV
        df.to_csv(output_file, index=False)
        print(f"Success! Converted '{input_file}' to '{output_file}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")


mot_to_csv_pandas(INPUT_DIR, OUTPUT_CSV)