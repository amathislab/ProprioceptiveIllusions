"""
    Process HDF5 data and save to an output file with spindle afferent inputs per trial as 'data', and wrist xwy, shoulder angles and elbow angles as 'labels'
    Specify INPUT_FILES, and CONFIG_PATH and pass as arguments the seeds and n_aff (the number of afferents of type Ia and II)

"""

import argparse
import os

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from directory_paths import SAVE_DIR
from utils.spindle_FR_helper import *

# Constants loaded from configuration
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "configs/train_test_data_spindles_extended.yaml",
)

INPUT_FILE = f"{SAVE_DIR}/elbow_flex/elbow_flex_visualize_flat_100_240Hz.hdf5"
DEFAULT_SEEDS = [0]
NUM_SAMPLES = 30000

# Saves the data to the following path
# f"{output_path}/{output_prefix}_{seed}_{num_ia}_{num_ii}_{os.path.basename(input_file)}"
OUTPUT_PATH = SAVE_DIR
OUTPUT_PREFIX = "optimized_linear_extended"


# Utility function to load YAML configuration
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load configuration globally
CONFIG = load_config(CONFIG_PATH)


def process_chunk(
    data, coefficients, num_coefficients, muscles, chunk_size, sampled_coefficients
):
    """
    Process a chunk of data for a given muscle and coefficient set.

    Parameters
    ----------
    data : dict
        Dictionary containing data for muscle lengths, velocities, and accelerations.
    coefficients : dict
        Dictionary containing muscle coefficients.
    num_coefficients : list
        List of number of coefficients for each type (Ia and II).
    muscles : list
        List of muscle names.
    chunk_size : int
        Size of the chunk to process.
    sampled_coefficients : dict
        Dictionary containing sampled coefficients for each muscle and coefficient type.

    Returns
    -------
    chunk_data : ndarray
        Processed data for the given chunk.
    """
    chunk_data = np.zeros(
        (chunk_size, sum(num_coefficients), len(muscles), data["lengths"].shape[2])
    )

    # Iterate over muscles and coefficient types
    for muscle_idx, muscle in enumerate(muscles):
        for i, coeff_type in enumerate(["i_a", "ii"]):
            # Iterate over coefficients for the given type
            for j in range(num_coefficients[i]):
                idx = sum(num_coefficients[:i]) + j

                # Use the sampled coefficients for this muscle and coefficient type
                sampled_index = sampled_coefficients[coeff_type][muscle][j]
                coeffs = {
                    "k_l": coefficients[coeff_type][muscle]["k_l"][sampled_index],
                    "k_v": coefficients[coeff_type][muscle]["k_v"][sampled_index],
                    "e_v": coefficients[coeff_type][muscle]["e_v"][sampled_index],
                    "k_a": coefficients[coeff_type][muscle]["k_a"][sampled_index],
                    "k_c": coefficients[coeff_type][muscle]["k_c"][sampled_index],
                    "max_rate": coefficients[coeff_type][muscle]["max_rate"][
                        sampled_index
                    ],
                }
                # Process the chunk using the spindle transfer function
                chunk_data[:, idx, muscle, :] = (
                    clipped_spindle_transfer_function_coeffs(
                        data["lengths"][:, muscle, :],
                        data["velocities"][:, muscle, :],
                        data["accelerations"][:, muscle, :],
                        coeffs,
                    )
                )
    return chunk_data


def process_data(
    input_path, output_path, config, num_samples=NUM_SAMPLES, add_vel=False
):
    """
    Process HDF5 data and save to an output file.

    Parameters
    ----------
    input_path : str
        Path to the input HDF5 file.
    output_path : str
        Path to the output HDF5 file.
    config : dict
        Configuration dictionary containing parameters for processing.
    num_samples : int, optional
        Number of samples to process. If not specified, all samples are processed.
    add_vel : bool, optional
        If True, add velocities to the labels. Default is False.
    """
    np.random.seed(config["seed"])

    with h5py.File(input_path, "r") as source_file:
        # Determine total samples to process
        total_samples = source_file["muscle_lengths"].shape[0]

        # Load coefficients and sampled indices
        if num_samples is not None:
            total_samples = min(total_samples, num_samples)
        muscles = config["muscles"]
        coefficients = {
            key: load_coefficients(config[key + "_coeff_path"]) for key in ["i_a", "ii"]
        }
        num_coefficients = [config["num_i_a"], config["num_ii"]]
        sampled_coefficients = get_sampled_coefficients(
            config, num_coefficients, muscles, coefficients
        )

        # check if target_file already exists
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping processing.")
            return

        with h5py.File(output_path, "w") as target_file:
            # Initialize datasets
            target_file.create_dataset(
                "data",
                shape=(
                    total_samples,
                    sum(num_coefficients),
                    len(muscles),
                    config["time_steps"],
                ),
                dtype="float32",
                chunks=(
                    config["chunk_size"],
                    sum(num_coefficients),
                    len(muscles),
                    config["time_steps"],
                ),
            )
            target_file.create_dataset(
                "labels",
                shape=(total_samples, config["time_steps"], config["label_dims"]),
                dtype="float32",
                chunks=(
                    config["chunk_size"],
                    config["time_steps"],
                    config["label_dims"],
                ),
            )

            # Process in chunks
            for i in tqdm(range(0, total_samples, config["chunk_size"])):
                end_idx = min(i + config["chunk_size"], total_samples)
                chunk_size = end_idx - i
                data = {
                    "lengths": source_file["muscle_lengths"][i:end_idx],
                    "velocities": source_file["muscle_velocities"][i:end_idx],
                    "accelerations": source_file["muscle_accelerations"][i:end_idx],
                }
                data = normalize(
                    data["lengths"],
                    data["velocities"],
                    data["accelerations"],
                    config["optimal_lengths"],
                )

                chunk_data = process_chunk(
                    data,
                    coefficients,
                    num_coefficients,
                    muscles,
                    chunk_size,
                    sampled_coefficients,
                )
                target_file["data"][i:end_idx] = chunk_data

                # Process labels
                coords = np.transpose(
                    source_file["endeffector_coords"][i:end_idx], (0, 2, 1)
                )
                joints = np.transpose(source_file["joint_coords"][i:end_idx], (0, 2, 1))
                if add_vel:
                    # Compute velocities
                    dt = 1 / config["sampling_rate"]  # Ensure this value is correct
                    endeffector_velocities = np.gradient(coords, dt, axis=1)
                    joint_velocities = np.gradient(joints, dt, axis=1)

                    # Concatenate positions and velocities
                    labels = np.concatenate(
                        (coords, joints, endeffector_velocities, joint_velocities),
                        axis=2,
                    )
                    target_file["labels"][i:end_idx] = labels
                else:
                    labels = np.concatenate((coords, joints), axis=2)
                    target_file["labels"][i:end_idx] = labels

    print("Data processing complete!")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate test configurations.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=CONFIG_PATH,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=INPUT_FILE,
        help="Path to the input HDF5 file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_PATH,
        help="Path to the directory to save the output files",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds for the configurations",
    )
    parser.add_argument(
        "--n_aff",
        type=int,
        default=None,
        help="Number of afferents for each type (overrides config if provided)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config_path)
    save_vel = config.get("save_vel", False)
    output_path = args.output_dir
    output_prefix = config.get("output_prefix", OUTPUT_PREFIX)

    # Determine the number of afferents
    if args.n_aff is not None:
        num_ia = num_ii = args.n_aff
    else:
        num_ia = config["num_i_a"]
        num_ii = config["num_ii"]

    print(f"Using {num_ia} Ia afferents and {num_ii} II afferents.")

    for seed in args.seeds:
        # Prepare a fresh config for each seed
        updated_config = config.copy()
        updated_config.update(
            {
                "num_i_a": num_ia,
                "num_ii": num_ii,
                "input_path": args.input_file,
                "seed": seed,
            }
        )

        # Build output filename
        basename = os.path.basename(args.input_file)
        suffix = f"_{seed}_{num_ia}_{num_ii}_{basename}"
        output_file = os.path.join(output_path, f"{output_prefix}{suffix}")

        # Adjust filename for specific cases
        if NUM_SAMPLES == 30000:
            output_file = output_file.replace("_50k_", "_30k_")

        # if output file already exists, skip processing
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping processing.")
            continue
        
        # Specific adjustment for certain inputs
        if "EF3D" in args.input_file or "ES3D" in args.input_file:
            updated_config["chunk_size"] = 10

        updated_config["output_path"] = output_file

        print(f"Processing '{args.input_file}' -> Saving to '{output_file}'")

        process_data(
            args.input_file,
            output_file,
            updated_config,
            num_samples=NUM_SAMPLES,
            add_vel=save_vel,
        )


if __name__ == "__main__":
    main()
