import os
import time

import h5py

from directory_paths import MODELS_DIR, PARENT_DIR, SAVE_DIR


def compress_hdf5(input_file, output_file, num_samples=30000, chunksize=1000):
    """
    Compress an HDF5 file by saving only the first num_samples, optimized for speed.
    """
    try:
        with h5py.File(input_file, "r") as infile, h5py.File(
            output_file, "w"
        ) as outfile:
            for dataset_name in infile.keys():
                dataset = infile[dataset_name]
                shape = (min(num_samples, dataset.shape[0]), *dataset.shape[1:])
                dset = outfile.create_dataset(
                    dataset_name, shape=shape, dtype=dataset.dtype, compression="gzip"
                )
                # Process in chunks
                for i in range(0, shape[0], chunksize):
                    dset[i : i + chunksize] = dataset[i : i + chunksize]
                print(f"Dataset '{dataset_name}' compressed and saved.")

            # Copy attributes in bulk
            outfile.attrs.update(infile.attrs)
            print(f"File '{input_file}' compressed and saved to '{output_file}'.")
    except Exception as e:
        print(f"Error processing file '{input_file}': {e}")


if __name__ == "__main__":
    # Example Usage
    input_dir = SAVE_DIR
    output_dir = SAVE_DIR
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it doesn't exist

    # seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_affs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # n_affs = []
    file_name_base = "optimized_linear_extended_{seed}_{n_aff}_{n_aff}_flag_pcr_training_240Hz_50k_cleaned_smoothed.hdf5"
    file_names = [
        f"{SAVE_DIR}/{file_name_base.format(seed=seed, n_aff=n_aff)}"
        for seed in seeds
        for n_aff in n_affs
    ]
    # Iterate through all HDF5 files in the input directory
    for file_name in file_names:
        if os.path.isfile(file_name):
            start_time = time.time()
            print(f"Processing file: {file_name}")
            input_path = os.path.join(input_dir, file_name)
            # output files is same as file_name base name replacing _50k_ with _30k_
            output_path = file_name.replace("_50k_", "_30k_")
            try:
                compress_hdf5(input_path, output_path)
                print(f"Processing time: {time.time() - start_time:.2f} seconds")
                # Only delete the input file if the output file exists
                if os.path.isfile(output_path):
                    os.remove(input_path)
                    print(f"Deleted input file: {input_path}")
                else:
                    print(f"Output file not found, input file retained: {input_path}")
            except Exception as e:
                print(f"Error compressing file {file_name}: {e}")
