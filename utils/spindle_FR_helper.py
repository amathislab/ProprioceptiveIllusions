import csv
import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.muscle_names import MUSCLE_NAMES
from utils.plot_defaults import DEFAULT_COLORS

sample_rate = 240
GT_COLOR = "lightblue"
ELBOW_ANGLE_INDEX = 6


def normalize(lengths, velocities, accelerations, optimal_lengths):
    normalized_muscle_lengths = np.zeros_like(lengths)
    normalized_muscle_velocities = np.zeros_like(velocities)
    normalized_muscle_accelerations = np.zeros_like(accelerations)

    for i in range(len(optimal_lengths)):
        normalized_muscle_lengths[:, i, :] = lengths[:, i, :] / optimal_lengths[i] - 1
        normalized_muscle_velocities[:, i, :] = velocities[:, i, :] / optimal_lengths[i]
        normalized_muscle_accelerations[:, i, :] = (
            accelerations[:, i, :] / optimal_lengths[i]
        )

    data = {
        "lengths": normalized_muscle_lengths,
        "velocities": normalized_muscle_velocities,
        "accelerations": normalized_muscle_accelerations,
    }
    return data


def _convert_to_float_list(string):
    """
    Convert a string representation of a list into a float list.
    """
    return [float(x) for x in string.strip("[]").split(",")]


def load_coefficients(filepath):
    """
    Load spindle coefficients from a CSV file.
    """
    coefficients = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            muscle = int(row[0])
            coefficients[muscle] = {
                key: _convert_to_float_list(value)
                for key, value in zip(
                    ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate", "frac_zero"],
                    row[1:],
                )
            }
    return coefficients



def get_sampled_coefficients(config, num_coefficients, muscles, coefficients):
    # compute sampled coefficients
    """
    Compute and load sampled coefficients for each muscle and coefficient type.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing seed, coefficient paths, and number of coefficients.
    num_coefficients : list
        List of number of coefficients for each type (Ia and II).
    muscles : list
        List of muscle names.
    coefficients : dict
        Dictionary containing muscle coefficients.

    Returns
    -------
    sampled_coefficients : dict
        Dictionary containing sampled coefficients for each muscle and coefficient type.
    """
    seed = config["seed"]
    sampled_coefficients = {}
    for i, coeff_type in enumerate(["i_a", "ii"]):
        if config[coeff_type + "_sampled_coeff_path"] is not None:
            sampled_coefficients[coeff_type] = load_sampled_coefficients(
                config[coeff_type + "_sampled_coeff_path"]
            )
        else:
            num = num_coefficients[i]
            # set the sampled coefficients path
            coeffs_dir = config[coeff_type + "_coeff_path"]
            coeffs_dir = os.path.dirname(coeffs_dir)
            sample_coeff_path = os.path.join(
                coeffs_dir,
                f"sampled_coefficients_{coeff_type}_{num}_{seed}.csv",
            )
            config[coeff_type + "_sampled_coeff_path"] = sample_coeff_path
            print("sampled coef path", sample_coeff_path)
            # check if sampled coefficients exist already for seed and numIa
            if os.path.exists(sample_coeff_path):
                print("Loading samp coeffs from ", sample_coeff_path)
                sampled_coefficients[coeff_type] = load_sampled_coefficients(
                    sample_coeff_path
                )  # load sampled coefficients
            else:
                print("new coef sample")
                sampled_coefficients[coeff_type] = {}
                for muscle in muscles:
                    sampled_coefficients[coeff_type][muscle] = np.random.choice(
                        len(
                            coefficients[coeff_type][muscle]["k_l"]
                        ),  # total number of coeffs to chose from
                        num_coefficients[i],
                        replace=False,
                    )
                # print("saving new samp coef path", sample_coeff_path)
                with open(sample_coeff_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["muscle", "coefficients"])
                    for muscle in muscles:
                        writer.writerow(
                            [muscle, sampled_coefficients[coeff_type][muscle]]
                        )
    return sampled_coefficients



def load_sampled_coefficients(filepath):
    """
    Load indeces for sampled coefficients from a CSV file.
    """
    coefficients = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            muscle = int(row[0])
            cleaned_row = re.sub(
                r"\s+", " ", row[1].strip("[]")
            ).strip()  # Normalize spaces and trim leading/trailing spaces
            coefficients[muscle] = [
                int(x)
                for x in cleaned_row.split(" ")
                if x.isdigit()  # Split and ensure each part is a valid integer
            ]
    return coefficients


def spindle_transfer_function_coeffs(length, velocity, acceleration, coeffs):
    """
    Calculate spindle firing rate using transfer function coefficients.
    """
    firing_rate = (
        coeffs["k_l"] * length
        + coeffs["k_v"] * np.sign(velocity) * np.abs(velocity) ** coeffs["e_v"]
        + coeffs["k_a"] * acceleration
        + coeffs["k_c"]
    )
    return np.clip(firing_rate, 0, None)


def clipped_spindle_transfer_function_coeffs(length, velocity, acceleration, coeffs):
    """
    Clip spindle firing rates to the maximum firing rate.
    """
    unclipped = spindle_transfer_function_coeffs(length, velocity, acceleration, coeffs)
    return np.clip(unclipped, 0, coeffs["max_rate"])


def spindle_transfer_function(length, velocity, acceleration, k_l, k_v, e_v, k_a, k_c):
    firing_rate = (
        k_l * length
        + k_v * np.sign(velocity) * np.abs(velocity) ** e_v
        + k_a * acceleration
        + k_c
    )
    firing_rate = np.clip(firing_rate, 0, None)
    return firing_rate


def clipped_spindle_transfer_function(
    length, velocity, acceleration, k_l, k_v, e_v, k_a, k_c, max_rate
):
    unclipped_rates = spindle_transfer_function(
        length, velocity, acceleration, k_l, k_v, e_v, k_a, k_c
    )
    return np.clip(unclipped_rates, 0, max_rate)


# vibrations to firing rates
def generate_vib_config(
    vib_f,
    input_shape,
    channel_indices,
    muscles_to_vib,
    time_range,
    rand_max=None,
    i_a_sampled_coeff_path=None,
    i_a_coeff_path=None,
    ii_sampled_coeff_path=None,
    ii_coeff_path=None,
    n_ia=None,
):
    """
    Generate a vibration configuration dictionary for the input to the model.

    This function configures the vibration parameters applied to specific channels
    and muscles over a given time range. The vibration can either have randomly sampled
    maximum frequencies or load predefined coefficients.
    Assume channel indices correspond to Ia afferents.

    Parameters:
        vib_f (float): The vibration frequency to be used.
        channel_indices (list): List of indices corresponding to channels to be vibrated.
        muscles_to_vib (list): List of muscle names to apply vibrations.
        time_range (slice or array-like): The range of time indices for the vibration.
        rand_max (dict, optional): Dictionary containing randomization settings. Should have:
            - "is" (bool): Whether to randomize maximum frequencies.
            - "f_max_min" (int): Minimum random maximum frequency.
            - "f_max_max" (int): Maximum random maximum frequency.
        i_a_sampled_coeff_path (str, optional): Path to sampled coefficients for i_a.
        i_a_coeff_path (str, optional): Path to the main coefficients file for i_a.

    Returns:
        dict: A dictionary containing vibration configuration:
            - "is": Flag indicating vibrations are applied.
            - "vib_freq": The vibration frequency.
            - "vib_time_range": The time range of vibrations.
            - "vib_muscles": List of muscles being vibrated.
            - "muscle_f_max": Array of maximum frequencies per muscle and channel.
            - "indices_filter": Tensor indicating vibrated inputs.
            - "f_max": Tensor of maximum frequencies applied per muscle/channel.

    Raises:
        AssertionError: If `rand_max` is not provided and coefficient paths are missing.
    """
    muscle_indices = [MUSCLE_NAMES.index(muscle) for muscle in muscles_to_vib]
    num_channels, num_muscles, time = input_shape

    if n_ia is None:
        n_ia = int(num_channels / 2)  # default to half of channels

    # Initialize a tensor to indicate which inputs are vibrated
    vibration_array = torch.zeros((num_channels, num_muscles, time))

    if len(channel_indices) >= 1:  # if not empty
        if (type(channel_indices[0]) == int) or (type(channel_indices[0]) == float):
            # make channel_indices_all by repeating channel_indices for length of muscle_indices
            channel_indices_all = [channel_indices for i in range(len(muscle_indices))]
        else:
            channel_indices_all = channel_indices
    else:
        channel_indices_all = channel_indices

    # breakpoint()
    # Set vibration array values to 1 for specified muscles and channels
    for i, muscle_idx in enumerate(muscle_indices):
        for channel_idx in channel_indices_all[i]:
            vibration_array[channel_idx, muscle_idx, time_range] = 1
    ##### Fix below for different channels by muscles #####
    if rand_max and rand_max.get("is", False):
        # Randomly sample maximum frequencies
        f_max_min, f_max_max = rand_max["f_max_min"], rand_max["f_max_max"]
        # if only one value, repeat for all muscles 
        if type(f_max_min) == int or type(f_max_min) == float:
            f_max_mins = [f_max_min for i in range(len(muscle_indices))]
        else: 
            # check length of f_max_min
            if len(f_max_min) != len(muscle_indices):
                raise ValueError(
                    "Length of f_max_min does not match the number of muscles."
                )
            f_max_mins = f_max_min
        if type(f_max_max) == int or type(f_max_max) == float:
            f_max_maxs = [f_max_max for i in range(len(muscle_indices))]
        else: 
            # check length of f_max_max
            if len(f_max_max) != len(muscle_indices):
                raise ValueError(
                    "Length of f_max_max does not match the number of muscles."
                )
            f_max_maxs = f_max_max
        # print("Randomly sampling max frequencies in range", f_max_min, f_max_max)
        print("Randomly sampling max frequencies")
        f_max_filter = torch.zeros((num_channels, num_muscles))
        max_frequencies = np.zeros((len(muscle_indices), len(channel_indices)))

        for m, muscle_idx in enumerate(muscle_indices):
            f_max_min = f_max_mins[m]
            f_max_max = f_max_maxs[m]
            for c, channel_idx in enumerate(channel_indices_all[m]):
                max_frequencies[m, c] = np.random.randint(f_max_min, f_max_max)
                f_max_filter[channel_idx, muscle_idx] = max_frequencies[m, c]
    else:
        # Load predefined coefficients for maximum frequencies
        assert i_a_sampled_coeff_path is not None, "i_a_sampled_coeff_path is required."
        assert i_a_coeff_path is not None, "i_a_coeff_path is required."

        print("Loading max frequencies from coefficients")
        sampled_coefficients = {
            "i_a": load_sampled_coefficients(i_a_sampled_coeff_path)
        }
        coefficients_ia = load_coefficients(i_a_coeff_path)

        # also vibrate some II afferents
        if max(channel_indices) >= n_ia:
            # load ii coefficients
            assert (
                ii_sampled_coeff_path is not None
            ), "ii_sampled_coeff_path is required."
            assert ii_coeff_path is not None, "ii_coeff_path is required."
            sampled_coefficients["ii"] = load_sampled_coefficients(
                ii_sampled_coeff_path
            )
            coefficients_ii = load_coefficients(ii_coeff_path)

        f_max_filter = torch.zeros((num_channels, num_muscles))
        max_frequencies = np.zeros((len(muscle_indices), len(channel_indices)))

        for m, muscle_idx in enumerate(muscle_indices):
            for c, channel_idx in enumerate(channel_indices_all[m]):
                if channel_idx >= n_ia:
                    sampled_index = sampled_coefficients["ii"][muscle_idx][
                        channel_idx - n_ia
                    ]
                    max_frequencies[m, c] = coefficients_ii[muscle_idx]["max_rate"][
                        sampled_index
                    ]
                    f_max_filter[channel_idx, muscle_idx] = max_frequencies[m, c]
                else:
                    sampled_index = sampled_coefficients["i_a"][muscle_idx][channel_idx]
                    max_frequencies[m, c] = coefficients_ia[muscle_idx]["max_rate"][
                        sampled_index
                    ]
                    f_max_filter[channel_idx, muscle_idx] = max_frequencies[m, c]

    # Create the vibration configuration dictionary
    vib_config = {
        "is": True,
        "vib_freq": vib_f,
        "vib_time_range": [time_range[0], time_range[1] - 1],
        "vib_muscles": muscles_to_vib,
        "muscle_f_max": max_frequencies,
        "indices_filter": vibration_array,
        "f_max": f_max_filter,
    }

    return vib_config


class Afferent:
    def __init__(self, muscle_idx, channel_idx, aff_type="i_a"):
        self.muscle_idx = muscle_idx
        self.muscle_name = MUSCLE_NAMES[muscle_idx]
        self.channel_idx = channel_idx
        self.aff_type = aff_type  # "i_a" or "ii"
        self.f_max = None

    def set_f_max(self, f_max):
        self.f_max = f_max

    def set_f_max_from_coeffs(self, sampled_coefficients, coefficients_ia):
        sampled_index = sampled_coefficients["i_a"][self.muscle_idx][self.channel_idx]
        self.f_max = coefficients_ia[self.muscle_idx]["max_rate"][sampled_index]


def add_vibration_to_array(vibration_array, vib_range, muscle_indices, channel_indices):
    for muscle_idx in muscle_indices:
        for channel_idx in channel_indices:
            vibration_array[channel_idx, muscle_idx, vib_range] = 1
    return vibration_array


def add_vibration_to_array(vibration_array, vib_range, afferents):
    for aff in afferents:
        muscle_idx = aff.muscle_idx
        channel_idx = aff.channel_idx
        vibration_array[channel_idx, muscle_idx, vib_range] = 1
    return vibration_array


def generate_vib_config_multiple(
    vib_f,
    input_shape,
    vib_time_ranges,
    muscles_to_vib_all,
    channel_indices,
    rand_max=None,
    i_a_sampled_coeff_path=None,
    i_a_coeff_path=None,
    ii_sampled_coeff_path=None,
    ii_coeff_path=None,
    n_ia=None,
):

    # muscle_indices = [MUSCLE_NAMES.index(muscle) for muscle in muscles_to_vib]
    assert len(vib_time_ranges) == len(muscles_to_vib_all)

    vib_configs = []
    # Iterate over the set of muscles to vibrate
    for i, vib_range in enumerate(vib_time_ranges):
        print(vib_range)
        # Initialize a tensor to indicate which inputs are vibrated
        # vibration_array = torch.zeros((num_channels, num_muscles, time))
        # if multiple vib freq are specificed for each vib_range
        if type(vib_f) == list:
            vib_f_i = vib_f[i]
        else:
            vib_f_i = vib_f
        muscles_to_vib = muscles_to_vib_all[i]
        vib_config = generate_vib_config(
            vib_f_i,
            input_shape,
            channel_indices,
            muscles_to_vib,
            range(vib_range[0], vib_range[1]),
            rand_max,
            i_a_sampled_coeff_path,
            i_a_coeff_path,
            ii_sampled_coeff_path,
            ii_coeff_path,
            n_ia,
        )
        vib_configs.append(vib_config)
    return vib_configs


# # vibrations to firing rates
# def generate_vib_config_multiple_affs(
#     vib_f,
#     input_shape,
#     vib_time_ranges,
#     vib_affs,
#     rand_max=None,
#     i_a_sampled_coeff_path=None,
#     i_a_coeff_path=None,
# ):
#     """
#     Generate a vibration configuration dictionary for the input to the model.

#     This function configures the vibration parameters applied to specific channels
#     and muscles over a given time range. The vibration can either have randomly sampled
#     maximum frequencies or load predefined coefficients.
#     Assume channel indices correspond to Ia afferents.

#     Parameters:
#         vib_f (float): The vibration frequency to be used at each time_range. Float if all the same otherwise list of list for each vib_range and each aff
#         vib_time_ranges(list): List of the range of time indices for the vibration.
#         vib_affs (list): list of same shape as vib_time_ranges with list of afferents to vibrate at each time range
#         rand_max (dict, optional): Dictionary containing randomization settings. Should have:
#             - "is" (bool): Whether to randomize maximum frequencies.
#             - "f_max_min" (int): Minimum random maximum frequency.
#             - "f_max_max" (int): Maximum random maximum frequency.
#         i_a_sampled_coeff_path (str, optional): Path to sampled coefficients for i_a.
#         i_a_coeff_path (str, optional): Path to the main coefficients file for i_a.

#     Returns:
#         list of dict: A dictionary containing vibration configuration:
#             - "is": Flag indicating vibrations are applied.
#             - "vib_freq": The vibration frequency.
#             - "vib_time_range": The time range of vibrations.
#             - "vib_muscles": List of muscles being vibrated.
#             - "muscle_f_max": Array of maximum frequencies per muscle and channel.
#             - "indices_filter": Tensor indicating vibrated inputs.
#             - "f_max": Tensor of maximum frequencies applied per muscle/channel.

#     Raises:
#         AssertionError: If `rand_max` is not provided and coefficient paths are missing.
#     """
#     # muscle_indices = [MUSCLE_NAMES.index(muscle) for muscle in muscles_to_vib]
#     num_channels, num_muscles, time = input_shape
#     assert len(vib_time_ranges) == len(vib_affs)

#     vib_configs = {}
#     # Iterate over the set of muscles to vibrate
#     for i, vib_range in enumerate(vib_time_ranges):
#         # Initialize a tensor to indicate which inputs are vibrated
#         # vibration_array = torch.zeros((num_channels, num_muscles, time))
#         # if multiple vib freq are specificed for each vib_range
#         if type(vib_f) == list:
#             vib_f_i = vib_f[i]
#         elif type(vib_f) == float:
#             vib_f_i = vib_f
#         else:
#             print("vib_f should be float or list", vib_f)
#         afferents = vib_affs[i]
#         muscles_to_vib = []  # list of muscles to vib
#         channel_indices = []  # list of list with channels for each muscle
#         for aff in afferents:
#             muscle_idx = aff.muscle_idx
#             channel_idx = aff.channel_idx
#             if muscle_idx in muscles_to_vib:
#                 i = muscles_to_vib.index(muscle_idx)
#                 if channel_idx not in channel_indices[i]:
#                     channel_indices[i].append(channel_idx)
#             else:  # new muscle
#                 muscles_to_vib.append(muscle_idx)
#                 channel_indices.append([channel_idx])

#         vib_config = generate_vib_config(
#             vib_f_i,
#             input_shape,
#             channel_indices,
#             muscles_to_vib,
#             vib_range,
#             rand_max,
#             i_a_sampled_coeff_path,
#             i_a_coeff_path,
#         )
#         vib_configs.append(vib_config)

#     return vib_configs


# def compute_vib_metrics_angles(theta_n, theta_v, theta_t, vib_range):
#     """
#     Compute vibration-related metrics for given angles.

#     Args:
#         theta_n (np.ndarray): Normal angles (no vibration).
#         theta_v (np.ndarray): Vibrated angles.
#         theta_t (np.ndarray): True angles (ground truth).
#         vib_range (slice): Range of indices where vibration is active.

#     Returns:
#         dict: A dictionary containing vibration-related metrics.
#     """
#     vib_metrics = {}
#     vib_range = list(range(vib_range[0], vib_range[1]))

#     # Compute offsets within vibration range
#     vib_metrics["offset_t_n_vib"] = np.mean(theta_t[vib_range] - theta_n[vib_range])
#     vib_metrics["offset_t_v_vib"] = np.mean(theta_t[vib_range] - theta_v[vib_range])
#     vib_metrics["offset_n_v_vib"] = np.mean(theta_n[vib_range] - theta_v[vib_range])

#     # Define the complement of the vibration range
#     full_range = range(len(theta_n))
#     non_vib_range = [i for i in full_range if i not in vib_range]

#     # Compute offsets outside vibration range
#     vib_metrics["offset_t_n_non_vib"] = np.mean(
#         theta_t[non_vib_range] - theta_n[non_vib_range]
#     )
#     vib_metrics["offset_t_v_non_vib"] = np.mean(
#         theta_t[non_vib_range] - theta_v[non_vib_range]
#     )
#     vib_metrics["offset_n_v_non_vib"] = np.mean(
#         theta_n[non_vib_range] - theta_v[non_vib_range]
#     )
#     # breakpoint()

#     # Compute difference between vibration angles 10 steps before and after start of vibration
#     # start_idx = vib_range.start
#     # end_idx = vib_range.start + 10
#     # Compute difference between vibration angles 10 steps before and after start of vibration
#     before_idx = vib_range[0] - 10
#     after_idx = vib_range[0] + 100
#     if before_idx >= 10 and after_idx + 10 <= len(theta_v):
#         before_vib = np.mean(theta_v[before_idx - 10 : before_idx])
#         # during_vib = np.mean(theta_v[end_idx : end_idx + 10])
#         during_vib = np.mean(theta_v[after_idx : after_idx + 10])
#         # breakpoint()
#         vib_metrics["vib_angle_diff"] = before_vib - during_vib
#     else:
#         vib_metrics["vib_angle_diff"] = None  # Not enough data to compute this metric
#     return vib_metrics


def compute_vib_metrics_angles(
    theta_n, theta_v, theta_t, vib_ranges, name_suffix="elbow"
):
    """
    Compute vibration-related metrics for given angles with a name suffix.

    Args:
        theta_n (np.ndarray): Normal angles (no vibration).
        theta_v (np.ndarray): Vibrated angles.
        theta_t (np.ndarray): True angles (ground truth).
        vib_ranges (slice or list of slices): Range(s) of indices where vibration is active.
        name_suffix (str): Suffix to append to each metric name.

    Returns:
        dict: A dictionary containing vibration-related metrics with modified names.
              If multiple ranges are provided, each metric is a list of values.
    """
    if not isinstance(vib_ranges, list):
        vib_ranges = [vib_ranges]

    # Initialize dictionary to store metrics
    metrics = {
        # f"offset_t_n_vib_{name_suffix}": [],
        # f"offset_t_v_vib_{name_suffix}": [],
        # f"offset_n_v_vib_{name_suffix}": [],
        # f"offset_t_n_non_vib_{name_suffix}": [],
        # f"offset_t_v_non_vib_{name_suffix}": [],
        # f"offset_n_v_non_vib_{name_suffix}": [],
        f"vib_angle_diff_{name_suffix}": [],
    }

    full_range = range(len(theta_n))  # Full range of indices
    before_idx = vib_ranges[0][0] - 30

    # breakpoint()
    for i, vib_range_tpl in enumerate(vib_ranges):
        vib_range = list(range(vib_range_tpl[0], vib_range_tpl[1]))

        # # Compute offsets within vibration range
        # metrics[f"offset_t_n_vib_{name_suffix}"].append(
        #     np.mean(theta_t[vib_range] - theta_n[vib_range])
        # )
        # metrics[f"offset_t_v_vib_{name_suffix}"].append(
        #     np.mean(theta_t[vib_range] - theta_v[vib_range])
        # )
        # metrics[f"offset_n_v_vib_{name_suffix}"].append(
        #     np.mean(theta_n[vib_range] - theta_v[vib_range])
        # )

        # # Define the complement of the vibration range
        # non_vib_range = [i for i in full_range if i not in vib_range]

        # # Compute offsets outside vibration range
        # metrics[f"offset_t_n_non_vib_{name_suffix}"].append(
        #     np.mean(theta_t[non_vib_range] - theta_n[non_vib_range])
        # )
        # metrics[f"offset_t_v_non_vib_{name_suffix}"].append(
        #     np.mean(theta_t[non_vib_range] - theta_v[non_vib_range])
        # )
        # metrics[f"offset_n_v_non_vib_{name_suffix}"].append(
        #     np.mean(theta_n[non_vib_range] - theta_v[non_vib_range])
        # )

        # Compute difference between vibration angles 10 steps before and after start of vibration
        after_idx = vib_range[0] + 100
        if before_idx >= 10 and after_idx + 10 <= len(theta_v):
            before_vib = np.mean(theta_v[before_idx - 10 : before_idx])
            during_vib = np.mean(theta_v[after_idx : after_idx + 10])
            # if name_suffix == "elbow":
            #     breakpoint()
            metrics[f"vib_angle_diff_{name_suffix}"].append(before_vib - during_vib)
        else:
            metrics[f"vib_angle_diff_{name_suffix}"].append(
                None
            )  # Not enough data to compute this metric

    # If there was only one range, return single values instead of lists
    if len(vib_ranges) == 1:
        metrics = {key: value[0] for key, value in metrics.items()}

    return metrics


def compute_vib_metrics_xyz(pos_v, vib_ranges, name_suffix="xyz"):

    if not isinstance(vib_ranges, list):
        vib_ranges = [vib_ranges]

    # Initialize dictionary to store metrics
    metrics = {
        # f"offset_t_n_vib_{name_suffix}": [],
        # f"offset_t_v_vib_{name_suffix}": [],
        # f"offset_n_v_vib_{name_suffix}": [],
        # f"offset_t_n_non_vib_{name_suffix}": [],
        # f"offset_t_v_non_vib_{name_suffix}": [],
        # f"offset_n_v_non_vib_{name_suffix}": [],
        f"pos_diff_{name_suffix}": [],
    }

    # full_range = range(len(theta_n))  # Full range of indices
    before_idx = vib_ranges[0][0] - 30

    for i, vib_range_tpl in enumerate(vib_ranges):
        vib_range = list(range(vib_range_tpl[0], vib_range_tpl[1]))

        # # Compute offsets within vibration range
        # metrics[f"offset_t_n_vib_{name_suffix}"].append(
        #     np.mean(theta_t[vib_range] - theta_n[vib_range])
        # )
        # metrics[f"offset_t_v_vib_{name_suffix}"].append(
        #     np.mean(theta_t[vib_range] - pos_v[vib_range])
        # )
        # metrics[f"offset_n_v_vib_{name_suffix}"].append(
        #     np.mean(theta_n[vib_range] - pos_v[vib_range])
        # )

        # # Define the complement of the vibration range
        # non_vib_range = [i for i in full_range if i not in vib_range]

        # # Compute offsets outside vibration range
        # metrics[f"offset_t_n_non_vib_{name_suffix}"].append(
        #     np.mean(theta_t[non_vib_range] - theta_n[non_vib_range])
        # )
        # metrics[f"offset_t_v_non_vib_{name_suffix}"].append(
        #     np.mean(theta_t[non_vib_range] - pos_v[non_vib_range])
        # )
        # metrics[f"offset_n_v_non_vib_{name_suffix}"].append(
        #     np.mean(theta_n[non_vib_range] - pos_v[non_vib_range])
        # )

        # Compute difference between mean position 10 steps before and after start of vibration
        after_idx = vib_range[0] + 100
        if before_idx >= 10 and after_idx + 10 <= len(pos_v):
            before_vib = np.mean(pos_v[before_idx - 10 : before_idx])
            during_vib = np.mean(pos_v[after_idx : after_idx + 10])
            metrics[f"pos_diff_{name_suffix}"].append(before_vib - during_vib)
        else:
            metrics[f"pos_diff_{name_suffix}"].append(
                None
            )  # Not enough data to compute this metric

    # If there was only one range, return single values instead of lists
    if len(vib_ranges) == 1:
        metrics = {key: value[0] for key, value in metrics.items()}

    return metrics


# def compute_vib_metrics_angles(
#     theta_n, theta_v, theta_t, vib_ranges, name_suffix="elbow"
# ):
#     """
#     Compute vibration-related metrics for given angles.

#     Args:
#         theta_n (np.ndarray): Normal angles (no vibration).
#         theta_v (np.ndarray): Vibrated angles.
#         theta_t (np.ndarray): True angles (ground truth).
#         vib_ranges (slice or list of slices): Range(s) of indices where vibration is active.

#     Returns:
#         dict: A dictionary containing vibration-related metrics.
#               If multiple ranges are provided, each metric is a list of values.
#     """
#     # if not isinstance(vib_ranges, list):
#     #     vib_ranges = [
#     #         list(range(vib_ranges[0], vib_ranges[1]))
#     #     ]  # Convert single range to a list
#     # else:
#     #     vib_ranges = [
#     #         list(range(vib_range[0], vib_range[1])) for vib_range in vib_ranges
#     #     ]
#     if not isinstance(vib_ranges, list):
#         vib_ranges = [vib_ranges]

#     # Initialize dictionary to store metrics
#     metrics = {
#         "offset_t_n_vib": [],
#         "offset_t_v_vib": [],
#         "offset_n_v_vib": [],
#         "offset_t_n_non_vib": [],
#         "offset_t_v_non_vib": [],
#         "offset_n_v_non_vib": [],
#         "vib_angle_diff": [],
#     }

#     full_range = range(len(theta_n))  # Full range of indices
#     before_idx = vib_ranges[0][0] - 10

#     for i, vib_range_tpl in enumerate(vib_ranges):
#         # print(vib_range_tpl, i)
#         vib_range = list(range(vib_range_tpl[0], vib_range_tpl[1]))
#         # Compute offsets within vibration range
#         metrics["offset_t_n_vib"].append(
#             np.mean(theta_t[vib_range] - theta_n[vib_range])
#         )
#         metrics["offset_t_v_vib"].append(
#             np.mean(theta_t[vib_range] - theta_v[vib_range])
#         )
#         metrics["offset_n_v_vib"].append(
#             np.mean(theta_n[vib_range] - theta_v[vib_range])
#         )

#         # breakpoint()
#         # Define the complement of the vibration range
#         non_vib_range = [i for i in full_range if i not in vib_range]

#         # Compute offsets outside vibration range
#         metrics["offset_t_n_non_vib"].append(
#             np.mean(theta_t[non_vib_range] - theta_n[non_vib_range])
#         )
#         metrics["offset_t_v_non_vib"].append(
#             np.mean(theta_t[non_vib_range] - theta_v[non_vib_range])
#         )
#         metrics["offset_n_v_non_vib"].append(
#             np.mean(theta_n[non_vib_range] - theta_v[non_vib_range])
#         )

#         # Compute difference between vibration angles 10 steps before and after start of vibration
#         # before_idx = vib_range[0] - 10
#         after_idx = vib_range[0] + 100
#         if before_idx >= 10 and after_idx + 10 <= len(theta_v):
#             before_vib = np.mean(theta_v[before_idx - 10 : before_idx])
#             # during_vib = np.mean(theta_v[end_idx : end_idx + 10])
#             during_vib = np.mean(theta_v[after_idx : after_idx + 10])
#             # breakpoint()
#             metrics["vib_angle_diff"].append(before_vib - during_vib)
#         else:
#             metrics["vib_angle_diff"].append(
#                 None
#             )  # Not enough data to compute this metric

#     # If there was only one range, return single values instead of lists
#     if len(vib_ranges) == 1:
#         metrics = {key: value[0] for key, value in metrics.items()}
#     # print("metrics", metrics)

#     return metrics


def compute_vib_metrics(
    scores, scores_vib, batch_y_s, vib_range, task="letter_reconstruction_joints"
):
    """
    Compute vibration metrics for a batch of samples and store them in a dictionary of lists.

    Arguments:
        scores: Tensor of predicted scores without vibrations (batch_size x time x features).
        scores_vib: Tensor of predicted scores with vibrations (batch_size x time x features).
        batch_y_s: Tensor of ground truth values (batch_size x time x features).
        vib_range: Slice or indices defining the vibration range.

    Returns:
        dict: A dictionary where each key is a metric name and the value is a list of the metric values
              for each sample in the batch.
    """
    elbow_angle_idx = 6  # Index for the elbow angle
    num_outputs = scores.shape[2]
    vib_metrics = {}

    for i in range(scores.shape[0]):
        # Extract data for the current sample
        theta_n = scores[i, :, elbow_angle_idx].detach().cpu().numpy()
        theta_v = scores_vib[i, :, elbow_angle_idx].detach().cpu().numpy()
        theta_t = batch_y_s[i, :, elbow_angle_idx].cpu().numpy()

        sample_metrics = {}
        # Compute metrics for the current sample
        sample_metrics_e = compute_vib_metrics_angles(
            theta_n, theta_v, theta_t, vib_range, "elbow"
        )
        sample_metrics.update(sample_metrics_e)

        # add metrics for the other angles
        for angle_idx in range(3, elbow_angle_idx):
            theta_n = scores[i, :, angle_idx].detach().cpu().numpy()
            theta_v = scores_vib[i, :, angle_idx].detach().cpu().numpy()
            theta_t = batch_y_s[i, :, angle_idx].cpu().numpy()

            # Compute metrics for the current sample
            sample_metrics_e = compute_vib_metrics_angles(
                theta_n, theta_v, theta_t, vib_range, f"shoulder_{angle_idx}"
            )
            sample_metrics.update(sample_metrics_e)

        # add metrics for xyz coord
        for idx in range(3):
            # pos_n = scores[i, :, idx].detach().cpu().numpy()
            pos_v = scores_vib[i, :, idx].detach().cpu().numpy()
            # pos_t = batch_y_s[i, :, idx].cpu().numpy()

            # Compute metrics for the current sample
            sample_metrics_e = compute_vib_metrics_xyz(pos_v, vib_range, f"xyz_{idx}")
            sample_metrics.update(sample_metrics_e)

        # add velocity metrics
        if (
            num_outputs > elbow_angle_idx + 1
            and task == "letter_reconstruction_joints_vel"
        ):
            # positions vel
            for idx in range(elbow_angle_idx + 1, elbow_angle_idx + 4):
                # pos_n = scores[i, :, idx].detach().cpu().numpy()
                pos_v = scores_vib[i, :, idx].detach().cpu().numpy()
                # pos_t = batch_y_s[i, :, idx].cpu().numpy()

                # Compute metrics for the current sample
                sample_metrics_e = compute_vib_metrics_xyz(
                    pos_v, vib_range, f"vel_xyz_{idx}"
                )
                sample_metrics.update(sample_metrics_e)
            # angles
            for idx in range(elbow_angle_idx + 4, num_outputs - 1):
                theta_n = scores[i, :, idx].detach().cpu().numpy()
                theta_v = scores_vib[i, :, idx].detach().cpu().numpy()
                theta_t = batch_y_s[i, :, idx].cpu().numpy()

                # Compute metrics for the current sample
                sample_metrics_e = compute_vib_metrics_angles(
                    theta_n, theta_v, theta_t, vib_range, f"vel_shoulder_{idx}"
                )
                sample_metrics.update(sample_metrics_e)
            # elbow
            idx = num_outputs - 1
            theta_n = scores[i, :, idx].detach().cpu().numpy()
            theta_v = scores_vib[i, :, idx].detach().cpu().numpy()
            theta_t = batch_y_s[i, :, idx].cpu().numpy()

            # Compute metrics for the current sample
            sample_metrics_e = compute_vib_metrics_angles(
                theta_n, theta_v, theta_t, vib_range, f"vel_elbow"
            )
            sample_metrics.update(sample_metrics_e)

        # Add metrics to vib_metrics dictionary
        for metric_name, value in sample_metrics.items():
            if metric_name not in vib_metrics:
                vib_metrics[metric_name] = []
            vib_metrics[metric_name].append(value)

    return vib_metrics


# def compute_vib_metrics(scores, scores_vib, batch_y_s, vib_ranges):
#     """
#     Compute vibration metrics for a batch of samples and organize them by vibration ranges.

#     Arguments:
#         scores: Tensor of predicted scores without vibrations (batch_size x time x features).
#         scores_vib: Tensor of predicted scores with vibrations (batch_size x time x features).
#         batch_y_s: Tensor of ground truth values (batch_size x time x features).
#         vib_ranges: Slice or list of slices defining the vibration range(s).

#     Returns:
#         dict: A dictionary where each key is a vibration range index, and the value is a dictionary
#               of metrics for all samples in that range.
#     """
#     elbow_angle_idx = 6  # Index for the elbow angle
#     vib_metrics_by_range = {}

#     # Ensure vib_ranges is a list for consistent handling
#     if isinstance(vib_ranges, slice):
#         vib_ranges = [vib_ranges]

#     for range_idx, vib_range in enumerate(vib_ranges):
#         vib_metrics_by_range[range_idx] = {}  # Initialize dictionary for this range

#         for i in range(scores.shape[0]):
#             # Extract data for the current sample
#             theta_n = scores[i, :, elbow_angle_idx].detach().cpu().numpy()
#             theta_v = scores_vib[i, :, elbow_angle_idx].detach().cpu().numpy()
#             theta_t = batch_y_s[i, :, elbow_angle_idx].cpu().numpy()

#             # Compute metrics for the current sample and vibration range
#             sample_metrics = compute_vib_metrics_angles(
#                 theta_n, theta_v, theta_t, vib_range
#             )

#             # Organize metrics by vibration range and sample
#             for metric_name, value in sample_metrics.items():
#                 if metric_name not in vib_metrics_by_range[range_idx]:
#                     vib_metrics_by_range[range_idx][metric_name] = []
#                 vib_metrics_by_range[range_idx][metric_name].append(value)

#     return vib_metrics_by_range


def plot_vibration_metrics(
    vib_metrics,
    elbow_angles,
    num_samples=None,
    save_path=None,
    plot_types=["bar", "scatter"],
    save_suffix="",
    swap_elbow_angle=False,
):
    """
    Plots vibration metrics for each metric across all batch samples.

    Args:
        vib_metrics (dict): Dictionary of metrics with keys as metric names
                            and values as lists for each batch sample.
    """
    metrics = list(vib_metrics.keys())  # All metric names
    num_metrics = len(metrics)

    if "bar" in plot_types:
        # Create subplots for each metric
        fig, axes = plt.subplots(
            1, num_metrics, figsize=(5 * num_metrics, 5), sharey=True
        )

        # Ensure axes is iterable (even if there's one metric)
        if num_metrics == 1:
            axes = [axes]

        if num_samples is None:
            num_samples = len(vib_metrics[metrics[0]])
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Plot metric values for each sample
            ax.bar(
                range(num_samples),
                vib_metrics[metric][:num_samples],
                color=DEFAULT_COLORS[0],
                edgecolor="black",
            )
            ax.set_title(metric.replace("_", " ").capitalize())
            ax.set_xlabel("Batch Sample Index")
            ax.set_ylabel("Angle difference (degrees)")
            ax.set_xticks(range(num_samples))

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(save_path + f"/vibration_metrics_bar{save_suffix}.pdf")
        # plt.show()
        plt.close()

    if "scatter" in plot_types:
        if swap_elbow_angle:
            elbow_angles = 180 - elbow_angles
        # Create subplots for each metric
        fig, axes = plt.subplots(
            1, num_metrics, figsize=(5 * num_metrics, 5), sharey=True
        )

        # Ensure axes is iterable (even if there's one metric)
        if num_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Plot metric values for each sample
            ax.scatter(
                elbow_angles,
                vib_metrics[metric],
                color=DEFAULT_COLORS[0],
                edgecolor="black",
                s=100,
            )
            ax.hl = ax.axhline(0, color="black", linestyle="-")
            ax.set_title(metric.replace("_", " ").capitalize())
            ax.set_xlabel("Elbow angle")
            ax.set_ylabel("Angle difference (deg)")
            # ax.set_xticks(range(batch_size))

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(save_path + f"/vibration_metrics_scatter{save_suffix}.pdf")
        # plt.show()
        plt.close()


def plot_vibration_analysis(df, save_path, swap_elbow_angle=False):
    """
    Plot 'offset_n_v_vib' and 'vib_angle_diff' vs. vibration frequency,
    colored by elbow angle ranges.

    Args:
        df (pd.DataFrame): DataFrame containing trial results.
        save_path (str): Directory to save the generated plot.
    """
    # Define elbow angle ranges
    if swap_elbow_angle:
        df["elbow_angle"] = 180 - df["elbow_angle"]
        elbow_angle_ranges = [30, 60, 90, 120, 150, 180]
        range_labels = ["30-60°", "60-90°", "90-120°", "120-150°", "150-180°"]
    else:
        elbow_angle_ranges = [0, 30, 60, 90, 120, 150]
        range_labels = ["0-30°", "30-60°", "60-90°", "90-120°", "120-150°"]
    df["elbow_angle_range"] = pd.cut(
        df["elbow_angle"], bins=elbow_angle_ranges, labels=range_labels
    )
    # breakpoint()

    if "vib_angle_diff" in df.columns:
        vib_angle_diff_name = "vib_angle_diff"
    elif "vib_angle_diff_elbow" in df.columns:
        vib_angle_diff_name = "vib_angle_diff_elbow"
    else:
        print("vib_angle_diff or vib_angle_diff_elbow not found in df")

    # Map ranges to colors
    range_colors = cm.cool(np.linspace(0, 1, len(range_labels)))
    range_color_map = dict(zip(range_labels, range_colors))
    vib_freq_name = "vib_freq_str" if type(df["vib_freq"][0]) == list else "vib_freq"

    # Create figure and subplots
    # check number of values of vib_angle_diff computed
    if type(df[vib_angle_diff_name][0]) == list:
        num_vib_angle_diff_values = len(df[vib_angle_diff_name][0])
    else:
        num_vib_angle_diff_values = 1

    print("Plotting ", num_vib_angle_diff_values, " vib_angle_diff values")
    MAX_COLS = 5
    num_cols = min(num_vib_angle_diff_values, MAX_COLS)
    num_rows = (num_vib_angle_diff_values + num_cols - 1) // num_cols
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * (num_vib_angle_diff_values), 5 * num_rows),
        sharey=True,
    )
    axes = np.array(axs).ravel()
    # ax1 = axes[0]

    # # First subplot: offset_n_v_vib vs. vibration frequency
    # for label, color in range_color_map.items():
    #     subset = df[df["elbow_angle_range"] == label]
    #     ax1.scatter(
    #         subset[vib_freq_name],
    #         subset["mean_offset_n_v_vib"],
    #         label=f"{label}",
    #         color=color,
    #         alpha=0.7,
    #     )
    # ax1.set_title("Offset between No Vib and Vib")
    # ax1.set_xlabel("Vibration Frequency (Hz)")
    # ax1.set_ylabel("Offset between no vib and vib (deg)")
    # # ax1.grid(True)

    # Second subplot: vib_angle_diff vs. vibration frequency
    # for each computed "vib_angle_diff", plot one figure
    if num_vib_angle_diff_values > 1:
        for i in range(num_vib_angle_diff_values):
            for label, color in range_color_map.items():
                subset = df[df["elbow_angle_range"] == label]
                ax2 = axes[i]
                # Extract the ith element from the list in the "vib_angle_diff" column for all rows
                vib_angle_diff = subset[vib_angle_diff_name].apply(
                    lambda x: x[i] if i < len(x) else None
                )

                for label, color in range_color_map.items():
                    label_subset = subset[subset["elbow_angle_range"] == label]
                    ax2.scatter(
                        label_subset[vib_freq_name],
                        vib_angle_diff[
                            label_subset.index
                        ],  # Ensure alignment with the subset
                        color=color,
                        alpha=0.7,
                    )
                ax2.set_title(f"Vib Angle Diff range {i}")
                ax2.set_xlabel("Vibration Frequency (Hz)")
                ax2.set_ylabel("Vib Angle Diff (deg)")
    else:
        ax2 = axes[0]
        for label, color in range_color_map.items():
            subset = df[df["elbow_angle_range"] == label]
            ax2.scatter(
                subset[vib_freq_name],
                subset[vib_angle_diff_name],
                # label=f"{label}",
                color=color,
                alpha=0.7,
            )
        ax2.set_title("Vib Angle Diff")
        ax2.set_xlabel("Vibration Frequency (Hz)")
        ax2.set_ylabel("Vib Angle Diff (deg)")
        # ax2.grid(True)

    # Add a single legend
    fig.legend(
        loc="upper center",
        ncol=len(range_labels),
        title="Elbow Angle Ranges",
        fontsize="large",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout for legend
    plt.savefig(f"{save_path}/combined_vibration_analysis.pdf")
    plt.close()


def plot_inputs_and_elbow_angles(
    df,
    muscle_idx_to_plot,
    channel_indices_to_plot,
    channels,
    save_path,
    swap_elbow_angle=False,
):
    """
    Generate combined plots for inputs and elbow angles with and without vibrations.

    Args:
        df (pd.DataFrame): DataFrame containing trial results.
        muscle_idx_to_plot (list): List of muscle indices to plot.
        channel_indices_to_plot (list): List of channel indices to plot.
        save_path (str): Directory to save the generated plots.
    """
    # Filter rows with non-NaN values for `true_outputs` and `predicted_outputs`
    filtered_df = df.dropna(subset=["true_outputs", "predicted_outputs"])

    if type(filtered_df.iloc[0]["vib_freq"]) == list:
        # Flatten vib_freq lists into unique scalar values
        vib_freqs = filtered_df["vib_freq_str"].unique()
    else:
        vib_freqs = filtered_df["vib_freq"].unique()

    vib_freqs.sort()
    colors = cm.inferno(np.linspace(0, 1, len(vib_freqs)))  # Red-Orange Colormap
    freq_color_map = dict(zip(vib_freqs, colors))

    for trial in filtered_df["trial"].unique():
        trial_data = filtered_df[filtered_df["trial"] == trial]
        true_outputs = trial_data.iloc[0]["true_outputs"]  # Same for all vib_freqs
        time_steps = np.arange(true_outputs.shape[0])  # Time axis
        time_steps = time_steps / sample_rate  # Convert to seconds

        # Determine subplot layout
        num_muscles = len(muscle_idx_to_plot)
        num_channels = len(channel_indices_to_plot)
        total_subplots = num_muscles * num_channels + 1  # +1 for elbow angles
        cols = 2
        rows = (total_subplots + cols - 1) // cols  # Compute rows for subplots

        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
        axes = axes.flatten()

        subplot_idx = 0
        handles = []  # To store line handles for the legend
        labels = []  # To store corresponding labels

        # Plot inputs with and without vibration for each muscle and channel
        for muscle in muscle_idx_to_plot:
            muscle_name = MUSCLE_NAMES[muscle]
            for c in channel_indices_to_plot:
                channel = channels[c]
                ax = axes[subplot_idx]
                inputs_no_vib = trial_data.iloc[0]["inputs_no_vib"][channel, muscle, :]
                for _, row in trial_data.iterrows():
                    if muscle_name in row["vib_muscles"]:
                        # Find index of muscle_name in vib_muscles
                        f_max_muscle_index = row["vib_muscles"].index(muscle_name)
                    else:
                        f_max_muscle_index = None
                    if (
                        f_max_muscle_index is not None
                        and row["muscle_f_max"].shape[1] > 0
                    ):
                        f_max = row["muscle_f_max"][f_max_muscle_index, c]
                    else:
                        f_max = "N/A"
                    # vib_f = row["vib_freq"]
                    vib_f = (
                        ", ".join(map(str, row["vib_freq"]))
                        if type(row["vib_freq"]) == list
                        else row["vib_freq"]
                    )
                    inputs_vib = row["inputs_vib"][channel, muscle, :]
                    (line,) = ax.plot(
                        time_steps,
                        inputs_vib,
                        label=f"{vib_f} Hz",
                        linestyle="-",
                        color=freq_color_map[vib_f],
                    )
                    # if vib_f not in labels:  # Avoid duplicate labels in legend
                    # handles.append(line)
                    # labels.append(f"{vib_f} Hz")
                ax.plot(
                    time_steps,
                    inputs_no_vib,
                    label="No Vib",
                    linestyle="--",
                    color=GT_COLOR,
                )
                if f_max != "N/A":
                    title = f"{muscle_name}, Channel: {channel}, Max FR: {f_max:.2f}"
                else:
                    title = f"{muscle_name}, Channel: {channel}"
                ax.set_title(title)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Firing Rate (Hz)")
                subplot_idx += 1

        # Plot predicted and true elbow angles
        ax = axes[subplot_idx]
        predicted_outputs = trial_data.iloc[0]["predicted_outputs"][
            :, ELBOW_ANGLE_INDEX
        ]
        if swap_elbow_angle:
            predicted_outputs = 180 - predicted_outputs
        for _, row in trial_data.iterrows():
            # vib_f = row["vib_freq"]
            vib_f = (
                ", ".join(map(str, row["vib_freq"]))
                if type(row["vib_freq"]) == list
                else row["vib_freq"]
            )
            predicted_vib_outputs = row["predicted_vib_outputs"][:, ELBOW_ANGLE_INDEX]
            if swap_elbow_angle:
                predicted_vib_outputs = 180 - predicted_vib_outputs
            (line,) = ax.plot(
                time_steps,
                predicted_vib_outputs,
                label=f"{vib_f} Hz",
                linestyle="-",
                color=freq_color_map[vib_f],
            )
            if vib_f not in labels:  # Avoid duplicate labels in legend
                handles.append(line)
                labels.append(f"{vib_f} Hz")
        (line,) = ax.plot(
            time_steps,
            predicted_outputs,
            label="No Vib",
            linestyle="--",
            color=GT_COLOR,
        )
        handles.append(line)
        labels.append(f"No Vib")
        true_elbow_angle = true_outputs[:, ELBOW_ANGLE_INDEX]
        if swap_elbow_angle:
            true_elbow_angle = 180 - true_elbow_angle
        (line,) = ax.plot(
            time_steps,
            true_elbow_angle,
            label="Ground Truth",
            color=GT_COLOR,
            # linewidth=2,
        )
        handles.append(line)
        labels.append(f"GT")
        ax.set_title(f"Elbow Angles")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Elbow Angle (deg)")
        subplot_idx += 1

        # Remove any unused subplots
        for i in range(subplot_idx, len(axes)):
            fig.delaxes(axes[i])

        # fig.suptitle(f"Trial: {trial}, Elbow Angle: {row['elbow_angle']:.2f} deg")
        # Add a single legend for the entire figure
        fig.legend(handles, labels, loc="upper center", ncol=6, fontsize="medium")
        # fig.legend(handles, labels, loc="best", ncol=2, fontsize="small")

        plt.tight_layout(
            rect=[0, 0, 1, 0.9]
        )  # Adjust layout to make room for the legend
        # plt.tight_layout(rect=[0, 0, 1.1, 1])  # Adjust layout to make room for the legend
        plt.savefig(f"{save_path}/combined_plot_trial_{trial}.pdf")
        plt.close()
