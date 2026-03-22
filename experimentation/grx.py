# batch script that performs RX on all .mat files in a folder
import os
import argparse
import numpy as np
import scipy.io as sio
from spectral.algorithms.detectors import rx


def run_rx_on_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".mat"):
            continue

        dataset_name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"RX_{dataset_name}.mat")

        print(f"Processing {filename}...")

        try:
            mat = sio.loadmat(input_path)

            if "data" not in mat:
                print(f"  Skipping {filename}: variable 'data' not found.")
                continue

            hsi = mat["data"].astype(np.float64)

            # Run RX anomaly detection
            rx_result = rx(hsi)

            # Save result
            sio.savemat(output_path, {"show": rx_result})

            print(f"  Saved → {output_path}")

        except Exception as e:
            print(f"  Failed on {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch RX hyperspectral anomaly detection."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing .mat hyperspectral datasets")
    parser.add_argument("--output_dir", required=True, help="Folder to save RX results")

    args = parser.parse_args()

    run_rx_on_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()