"""
This file is used to upsample dataset objects from a hdf5 file
"""

import os
import argparse
import h5py
import numpy as np
from scipy.interpolate import interp1d

def upsample_hdf5(original_file_path, new_file_path, original_rate, new_rate):
    """
    Takes a hdf5 file with multiple datasets objects and upsamples them.

    Arguments:
        original_file_path (str): path to the hdf5 file to upsample
        new_file_path (str): path to save the upsampled data
        original_rate (float): sampling rate of the original data
        new_rate (float): target sample rate

    Returns:
    """
    with h5py.File(original_file_path, "r") as original_file:
        with h5py.File(new_file_path, "w") as new_file:

            # Iterate over each dataset in the original file
            for dataset_name in original_file.keys():
                print(dataset_name)
                data = original_file[dataset_name]  # [:]
                # if data.dtype == "float64": data = data.astype("float32")

                # datasets with time on the 3rd axis
                if dataset_name in [
                    "elbow_coords",
                    "elbow",
                    "endeffector_coords",
                    "end_effector_coord",
                    "joint_coords",
                    "muscle_coords",
                    "muscle_lengths",
                    "muscle_velocities",
                    "muscle_accelerations",
                    "shoulder_coords",
                    "shoulder",
                    "spindle_firing",
                    "spindle_FR",
                    "spindle_info",
                ]:
                    upsample_data(new_file, dataset_name, data, original_rate, new_rate)

                # datasets that shouldn't be upsampled
                else:
                    original_file.copy(dataset_name, new_file, name=dataset_name)


def upsample_data(file, name, data, original_rate, new_rate, batch_size=10000):
    """
    Upsamples a dataset object with time along the last or second-to-last axis,
    depending on its shape, and processes the data in batches.

    Arguments:
        data: dataset object of shape (N, ..., time) or (N, ..., time, features)
        original_rate (float): sampling rate of the original data
        new_rate (float): target sample rate
        batch_size (int): number of samples to process per batch

    Returns:
        upsampled_data: upsampled dataset object
    """
    # Determine the shape and dimensionality of the input
    is_multifeature = data.ndim == 4  # True if shape is (N, ..., time, features)
    data_shape = data.shape
    num_timepoints_original = data_shape[-2] if is_multifeature else data_shape[-1]
    duration = np.round(num_timepoints_original / original_rate, 1)
    num_timepoints_new = int(duration * new_rate)

    # Time axes
    t_original = np.linspace(0, duration, num_timepoints_original)
    t_new = np.linspace(0, duration, num_timepoints_new)

    # Initialize upsampled data array
    upsampled_shape = (
        data_shape[:-2] + (num_timepoints_new, data_shape[-1])
        if is_multifeature
        else data_shape[:-1] + (num_timepoints_new,)
    )
    # upsampled_data = np.empty(upsampled_shape, dtype=data.dtype)
    upsampled_data = file.create_dataset(name, shape=upsampled_shape, dtype=data.dtype)

    # Process data in batches
    num_samples = data_shape[0]
    for start_idx in range(0, num_samples, batch_size):
        print(f"Processing batch {start_idx}-{start_idx + batch_size}...")
        end_idx = min(start_idx + batch_size, num_samples)
        data_batch = data[start_idx:end_idx, ...]  # Current batch
        # upsampled_batch = np.empty((end_idx - start_idx,)+upsampled_shape[1:], dtype=data.dtype)

        # Initialize batch array
        if is_multifeature:
            # For shape (batch_size, ###, time, features)
            data_batch_interp = np.empty(
                data_batch.shape[:-2] + (num_timepoints_new, data_batch.shape[-1]),
                dtype=data.dtype,
            )
            for i in range(data_batch.shape[0]):  # Iterate over samples
                for j in range(
                    data_batch.shape[1]
                ):  # Iterate over the second dimension (e.g., muscles)
                    for k in range(data_batch.shape[-1]):  # Iterate over features
                        f_interp = interp1d(
                            t_original,
                            data_batch[i, j, :, k],
                            kind="linear",
                            fill_value="extrapolate",
                        )
                        data_batch_interp[i, j, :, k] = f_interp(t_new)
        else:
            # For shape (batch_size, ###, time)
            data_batch_interp = np.empty(
                data_batch.shape[:-1] + (num_timepoints_new,), dtype=data.dtype
            )
            for i in range(data_batch.shape[0]):  # Iterate over samples
                for j in range(
                    data_batch.shape[1]
                ):  # Iterate over the second dimension (e.g., muscles)
                    f_interp = interp1d(
                        t_original,
                        data_batch[i, j, :],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    data_batch_interp[i, j, :] = f_interp(t_new)

        # Assign interpolated batch to the output
        upsampled_data[start_idx:end_idx, ...] = data_batch_interp


def hdf5_structure(file_path):
    """
    Displays the structure of an hdf5 file containing multiple datasets

    Arguments:
        file_path (str): path to the hdf5 file

    Returns:
    """

    with h5py.File(file_path, "r") as hdf_file:

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Data type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        # Use the visititems method to traverse the file structure
        hdf_file.visititems(print_structure)
    print()


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--original_rate", type=float, default=60)
    parser.add_argument("--new_rate", type=float, default=240)
    params = parser.parse_args()

    output_dir = os.path.dirname(params.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Upsampling...")

    upsample_hdf5(params.input_file, params.output_file, params.original_rate, params.new_rate)

    print("Upsampling completed")
    print()
    print("Before upsampling:", params.input_file)
    hdf5_structure(params.input_file)
    print("After upsampling:", params.output_file)
    hdf5_structure(params.output_file)
