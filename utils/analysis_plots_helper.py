import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from directory_paths import SAVE_DIR
from train.train_model_utils import *
from utils.muscle_names import MUSCLE_NAMES
from utils.spindle_FR_helper import get_sampled_coefficients, load_coefficients

SAMPLE_RATE = "240Hz"
sample_rate = 240
ELBOW_ANGLE_INDEX = 6

BASE_FIG_SIZE_1COL = (3.5, 2.7)
# BASE_FIG_SIZE_1COL = (7.0, 5.0)
BASE_FIG_SIZE_2COL = (7.0, 5.0)
# BASE_FIG_SIZE_2COL = (14.0, 10.0)
BASE_FIG_SIZE_MULTCOL = (7.0, 8.0)

LW = 1
LW_SMALL = 0.5
LW_LARGE = 1.5
MS = 20
MS_line = 5
MS_BIG = 35

ALPHA = 0.8

MARKER_LIST = [
    "o",
    "X",
    "^",
    "p",
    "d",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]
seed_marker_map = {i: MARKER_LIST[i] for i in range(10)}

original_hex_biceps = "#FF3B30"
original_hex_tri = "#00C7BE"

coord_name_mapping = {
    "pos_diff_xyz_0": "wrist x",
    "pos_diff_xyz_1": "wrist y",
    "pos_diff_xyz_2": "wrist z",
    "vib_angle_diff_shoulder_3": "elevation",
    "vib_angle_diff_shoulder_4": "shoulder elevation",
    "vib_angle_diff_shoulder_5": "shoulder rotation",
    "vib_angle_diff_elbow": "elbow angle",
}
# coord_name_mapping[]
coord_name_mapping["pos_diff_vel_xyz_7"] = "wrist x vel"
coord_name_mapping["pos_diff_vel_xyz_8"] = "wrist y vel"
coord_name_mapping["pos_diff_vel_xyz_9"] = "wrist z vel"
coord_name_mapping["vib_angle_diff_vel_shoulder_10"] = "elevation vel"
coord_name_mapping["vib_angle_diff_vel_shoulder_11"] = "shoulder elevation vel"
coord_name_mapping["vib_angle_diff_vel_shoulder_12"] = "shoulder rotation vel"
coord_name_mapping["vib_angle_diff_vel_elbow"] = "elbow angle vel"

path_save = SAVE_DIR + "/plots"
if not os.path.exists(path_save):
    os.makedirs(path_save)
    print("created directory ", path_save)

## Plot style 
def set_publication_style():
    scale_factor = 1
    """Set consistent styling for publication figures"""
    # Font settings - use a journal-compatible font
    # plt.rcParams['font.family'] = 'Arial'  # Common journal font, or try 'Times New Roman'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 9 * scale_factor # Base font size in points
    
    # Line settings
    plt.rcParams['lines.linewidth'] = 1.0  # Line width in points
    plt.rcParams['axes.linewidth'] = 0.8  # Frame line width
    
    # Tick parameters
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    plt.rcParams['xtick.major.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 3.0
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['xtick.labelsize'] = 8* scale_factor
    plt.rcParams['ytick.labelsize'] = 8* scale_factor
    
    # Axes labels
    plt.rcParams['axes.labelsize'] = 9* scale_factor
    plt.rcParams['axes.titlesize'] = 10* scale_factor
    
    # Legend
    plt.rcParams['legend.fontsize'] = 8* scale_factor
    plt.rcParams['legend.frameon'] = False
    
    # Figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    plt.rcParams.update({
        "figure.dpi": 300,                 # High-resolution figures
        "savefig.dpi": 300,                # Save figures with high resolution
        "savefig.format": "svg",           # Default format for saving figures
        "savefig.bbox": "tight",           # Tight layout when saving
        # "savefig.fonttype": 42             # Use TrueType fonts
    })
    plt.rcParams['svg.fonttype'] = 'none'  # to load text to figma 
    
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Seaborn specific styling
    sns.set_context("paper")
    # sns.set_style("ticks")
    
    # Use a consistent color palette
    sns.set_palette("colorblind")  # Accessible color palette
    


## Load data
def load_vibration_data(
    model_path: str,
    test_exp_dir: str,
    vib_muscles=None,
    vib_range=None,
    columns_to_load=None,
    coef_seeds_to_load=None,
    train_seeds_to_load=None,
    input_data: str = "ELBOW",
    sample_rate: str = "240Hz",
    exact_vib_muscles: bool = False,
    addChannels: bool = False
) -> pd.DataFrame:
    """
    Load H5 files from a specific folder structure with multiple filtering options.

    Parameters:
    -----------
    model_path : str
        Base path to the model directories
    INPUT_DATA : str
        Input data subdirectory name
    SAMPLE_RATE : str
        Sample rate subdirectory name
    test_exp_dir : str
        Test experiment directory name
    vib_muscles : list, optional
        List of muscle names to filter
    vib_range : str, optional
        Specific vibration range to filter
    columns_to_load : list, optional
        Specific columns to load from H5 files
    coef_seeds_to_load : list, optional
        Specific coefficient seeds to load
    train_seeds_to_load : list, optional
        Specific training seeds to load

    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame with all matching H5 file contents
    """
    dataframes = []

    # Walk through the directory structure
    for seed_path in os.listdir(model_path):
        # Construct full path to vibration data
        path_to_vib = os.path.join(
            model_path, seed_path, "test", input_data, sample_rate, test_exp_dir
        )

        # Skip if path doesn't exist
        if not os.path.isdir(path_to_vib):
            # check different path structure
            path_to_vib = os.path.join(
                model_path, seed_path, "test", input_data, test_exp_dir
            )
            if not os.path.isdir(path_to_vib):
                continue

        # Process muscle subdirectories
        for muscle_subdir in os.listdir(path_to_vib):
            muscle_path = os.path.join(path_to_vib, muscle_subdir)

            # Filter muscles if specified
            if exact_vib_muscles:
                if vib_muscles is not None and not any(
                    vm == muscle_subdir for vm in vib_muscles
                ):
                    continue
            else:
                if vib_muscles is not None and not any(
                    vm in muscle_subdir for vm in vib_muscles
                ):
                    continue

            # check muscle_path is a directory
            if not os.path.isdir(muscle_path):
                continue
            # Process experiment subdirectories
            for exp_subdir in os.listdir(muscle_path):
                # Filter vibration range if specified
                if vib_range is not None and vib_range not in exp_subdir:
                    continue

                exp_path = os.path.join(muscle_path, exp_subdir)
                h5_file_path = os.path.join(exp_path, "vib_results.h5")

                # Check if H5 file exists
                if not os.path.exists(h5_file_path):
                    continue

                # Extract seeds
                coef_seed = seed_path.split("_")[-2]
                train_seed = seed_path.split("_")[-1]

                # Filter seeds if specified
                if (
                    coef_seeds_to_load is not None
                    and coef_seed not in coef_seeds_to_load
                ) or (
                    train_seeds_to_load is not None
                    and train_seed not in train_seeds_to_load
                ):
                    continue

                # Load DataFrame
                # df = pd.read_hdf(h5_file_path, columns=columns_to_load)
                print(f"Loading {h5_file_path}")
                if columns_to_load is not None:
                    df = pd.read_hdf(h5_file_path)[columns_to_load]
                else:
                    df = pd.read_hdf(h5_file_path)

                # Add metadata columns
                df["seed_path"] = seed_path
                df["coef_seed"] = coef_seed
                df["train_seed"] = train_seed
                
                if addChannels: 
                    # Add channels vibrated 
                    channel_str = exp_subdir.split("_")[-2]
                    print(channel_str)
                    if channel_str[0] == "c": # only one channel vibrated
                        channel = channel_str[-1]
                        df["channels"] = channel
                    else: # multiple channels vibrated
                        # num channels is the number before c in channel_str
                        num_channels = int(channel_str.split("c")[0]) 
                        channels = [x for x in range(num_channels)]
                        # channels = [x for x in range(5,5+num_channels)]
                        # make into a string of the form "0_1_2_3"
                        channels_str = "_".join([str(x) for x in channels])
                        df["channels"] = channels_str

                dataframes.append(df)

    # Concatenate all DataFrames
    if not dataframes:
        raise ValueError("No DataFrames were loaded. Check your filtering criteria.")

    df_all = pd.concat(dataframes, ignore_index=True)
    # vib_muscles_str combines vib_muscles with _ in between
    df_all["vib_muscles_str"] = df_all["vib_muscles"].apply(lambda x: "_".join(x))
    return df_all


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """
    Provide a comprehensive summary of the loaded DataFrame with specific details.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to summarize

    Returns:
    --------
    dict
        Dictionary containing detailed summary statistics
    """
    summary = {
        "unique_coef_seeds": sorted(df["coef_seed"].unique().tolist()),
        "unique_train_seeds": sorted(df["train_seed"].unique().tolist()),
        "train_seeds_per_coef_seed": df.groupby("coef_seed")["train_seed"]
        .nunique()
        .to_dict(),
        "trials_per_coef_seed": df.groupby("coef_seed").size().to_dict(),
        "trials_per_train_seed": df.groupby("train_seed").size().to_dict(),
        "vib_muscles_summary": {},
        "vib_freq_summary": {},
    }

    # Count trials per unique combination of coef_seed and train_seed
    summary["trials_per_coef_seed_train_seed"] = (
        df.groupby(["coef_seed", "train_seed"]).size().to_dict()
    )

    # Vib muscles per coef_seed and train_seed
    vib_muscles_by_seed = (
        df.groupby(["coef_seed", "train_seed"])["vib_muscles_str"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    summary["vib_muscles_summary"] = vib_muscles_by_seed

    # Vib frequencies per coef_seed, train_seed, and vib_muscles_str
    vib_freq_summary = (
        df.groupby(["coef_seed", "train_seed", "vib_muscles_str"])["vib_freq"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    summary["vib_freq_summary"] = vib_freq_summary

    return summary

def get_coefs(config):
    """
        Returns the coefficients for the given configuration as a dictionary.
    """
    coefs_names = ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate", "frac_zero"]
    # load coefficients
    coefficients = {
        key: load_coefficients(config[key + "_coeff_path"]) for key in ["i_a", "ii"]
    }
    muscles = config["muscles"]
    num_coefficients = [config["num_i_a"], config["num_ii"]]
    sampled_coefficients = get_sampled_coefficients(
        config, num_coefficients, muscles, coefficients
    )
    # print(sampled_coefficients)
    
    all_coefs = {}
    for i, coeff_type in enumerate(["i_a", "ii"]):
        all_coefs[coeff_type] = {}
        for muscle in muscles:
            all_coefs[coeff_type][muscle] = {}
            muscle_idx = MUSCLE_NAMES.index(muscle)
            print("muscle ", muscle, muscle_idx)
            # Iterate over coefficients for the given type
            sampled_coefficients_m = sampled_coefficients[coeff_type][muscle_idx]
            print(sampled_coefficients_m)
            for coef_name in coefs_names:
                all_coefs[coeff_type][muscle][coef_name] = []
            for j in range(num_coefficients[i]):
                # print('aff ', j)
                idx = sum(num_coefficients[:i]) + j
                # Use the sampled coefficients for this muscle and coefficient type
                sampled_index = sampled_coefficients_m[j]
                # print("sampled idx ", sampled_index)
                for coef_name in coefs_names:
                    all_coefs[coeff_type][muscle][coef_name].append(
                        coefficients[coeff_type][muscle_idx][coef_name][sampled_index]
                    )
    return all_coefs


def find_nearest_trial(df, target_angle=90, tolerance=5):
    """Find the trial with an elbow_angle near the target angle."""
    df = df.dropna(subset=["predicted_vib_outputs"])
    mask_angle = (df["elbow_angle"] >= target_angle - tolerance) & (df["elbow_angle"] <= target_angle + tolerance)
    # select the value of the first trial that matches
    filtered_df = df[mask_angle]
    trial = filtered_df.iloc[0]["trial"]
    # mask_vib_freq = df["vib_freq"] == vib_freq
    # filtered_df = df[mask_angle & mask_vib_freq]
    # return filtered_df
    return trial

import colorsys


def adjust_color(hex_color, hue_shift, saturation=1.0, value=0.78):
    # Convert hex to RGB (0-255 scale)
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Convert RGB to HSV (0-1 scale)
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    # Adjust hue, ensure it stays within [0, 1]
    h = (h + hue_shift/360) % 1
    # Convert back to RGB (0-1 scale)
    r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
    # Convert RGB to hex
    return f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}'

    
## Plots
def plot_vibration_heatmap(df, frequency, save_path, plot_xyz=False, suffix=""):
    # Filter data for the specific frequency
    df_freq = df[df["vib_freq"] == frequency].copy()

    df_freq = df_freq.sort_values(by=["vib_muscles_str"])

    # Convert the vib_muscles lists to strings for grouping
    # df_freq['vib_muscles_str'] = df_freq['vib_muscles'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    # Define the metrics to analyze

    if plot_xyz:
        metrics = ["pos_diff_xyz_0", "pos_diff_xyz_1", "pos_diff_xyz_2"]
    else:
        metrics = [
            "vib_angle_diff_elbow",
            "vib_angle_diff_shoulder_3",
            "vib_angle_diff_shoulder_4",
            "vib_angle_diff_shoulder_5",
        ]

    # check the metrics in df_freq
    metrics = [x for x in metrics if x in df_freq.columns]
    print(metrics)
    if len(metrics) == 0:
        print("No metrics found in the DataFrame.")
        return

    # Group the data by vib_muscles_str and compute mean, standard deviation, and sample size
    grouped = df_freq.groupby("vib_muscles_str", as_index=False)[metrics].agg(
        ["mean", "std", "count"]
    )

    # Extract mean, standard deviation, and sample size per group
    pivot_data = grouped.xs("mean", axis=1, level=1)  # Extract mean values
    pivot_data_std = grouped.xs("std", axis=1, level=1)  # Extract std values
    n_samples = grouped.xs(
        "count", axis=1, level=1
    )  # Extract sample counts (per group)

    # Ensure standard error is computed per group
    standard_error = pivot_data_std / np.sqrt(n_samples)

    # Format annotations as "mean ± SE"
    annot_text = (
        pivot_data.round(3).astype(str) + " ± " + standard_error.round(3).astype(str)
    )

    # Set up the plot
    plt.figure(figsize=BASE_FIG_SIZE_2COL)

    if plot_xyz:
        hm_label = "Perceived Vibration Effects (cm)"
    else:
        hm_label = "Perceived Vibration Effects (deg)"
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=annot_text,  # Show original values in annotations
        fmt="",
        cmap="RdYlBu_r",  # Red-Yellow-Blue colormap (reversed)
        center=0,
        cbar_kws={"label": hm_label},
    )

    # Customize the plot
    plt.title(f"Vibration Effects Heatmap (Frequency: {frequency}Hz)")
    plt.ylabel("Vibrated Muscles")
    plt.xlabel("Metrics")
    # change x labels according to coord_name_mapping
    plt.xticks(
        ticks=np.arange(len(metrics)) + 0.5,
        labels=[coord_name_mapping[metric] for metric in metrics],
        rotation=45,
        ha="right",
    )

    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45, ha='right')

    fig_path = f"{save_path}/heatmap_vib_angle_diff_vsfreq_{frequency}_{suffix}.svg"
    if plot_xyz:
        fig_path = (
            f"{save_path}/heatmap_xyz_vib_angle_diff_vsfreq_{frequency}_{suffix}.svg"
        )
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f"{fig_path}")
    plt.show()


def plot_vibration_violins(df, frequency, save_path, plot_xyz=False, suffix=""):
    # Filter data for the specific frequency
    df_freq = df[df["vib_freq"] == frequency].copy()
    # sort by vib_muscles_str
    df_freq = df_freq.sort_values(by=["vib_muscles_str"])

    # Convert the vib_muscles lists to strings for grouping
    # df_freq['vib_muscles_str'] = df_freq['vib_muscles'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    # Define the metrics to analyze
    if plot_xyz:
        metrics = ["pos_diff_xyz_0", "pos_diff_xyz_1", "pos_diff_xyz_2"]
    else:
        metrics = [
            "vib_angle_diff_elbow",
            "vib_angle_diff_shoulder_3",
            "vib_angle_diff_shoulder_4",
            "vib_angle_diff_shoulder_5",
        ]

    metrics = [x for x in metrics if x in df_freq.columns]
    print(metrics)
    if len(metrics) == 0:
        print("No metrics found in the DataFrame.")
        return

    # Group the data by vib_muscles_str and compute mean, standard deviation, and sample size
    # Create figure
    fig, axs = plt.subplots(1, len(metrics), figsize=(BASE_FIG_SIZE_1COL[0] * len(metrics), BASE_FIG_SIZE_2COL[1]), sharey=True)
    # fig.suptitle(f"Vibration Effects Distribution (Frequency: {frequency}Hz)")

    # Create a violin plot for each metric
    for idx, metric in enumerate(metrics):
        # Prepare data for violin plot
        sns.violinplot(data=df_freq, x="vib_muscles_str", y=metric, ax=axs[idx])

        # Customize each subplot
        axs[idx].set_title(coord_name_mapping[metric])
        axs[idx].set_xlabel("")  # Remove x label as it's redundant

        if plot_xyz:
            axs[idx].set_ylabel("Perceived Vibration Effects (cm)" if idx == 0 else "")
        else:
            axs[idx].set_ylabel("Perceived Vibration Effects (deg)" if idx == 0 else "")

        # Rotate x-axis labels for better readability
        axs[idx].tick_params(axis="x", rotation=45)

        # Add grid lines for better readability
        axs[idx].grid(True, axis="y", linestyle="--", alpha=0.7)

        # # Calculate and annotate mean values
        # means = df_freq.groupby('vib_muscles_str')[metric].mean()
        # for i, mean_val in enumerate(means):
        #     axs[idx].text(i, axs[idx].get_ylim()[1],
        #                  f'μ={mean_val:.3f}',
        #                  ha='center', va='bottom')

    # Generate save path
    fig_path = f"{save_path}/violin_vib_angle_diff_vsfreq_{frequency}_{suffix}.svg"
    if plot_xyz:
        fig_path = (
            f"{save_path}/violin_xyz_vib_angle_diff_vsfreq_{frequency}_{suffix}.svg"
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(fig_path)
    # plt.show()


def plot_predicted_outputs(
    df, save_path, cols_to_group=["trial", "vib_muscles_str", "vib_freq"]
):
    # Filter rows where predicted_vib_outputs is not NaN
    """
    Plot predicted vibration outputs for each group of trials, muscles, and frequencies.

    Args:
        df (pd.DataFrame): DataFrame containing the trial data with 'predicted_vib_outputs'.
        save_path (str): Directory to save the generated plots.
        cols_to_group (list, optional): Columns to group the data by before plotting. Defaults to ["trial", "vib_muscles_str", "vib_freq"].

    This function filters the DataFrame to include only rows where 'predicted_vib_outputs' is not NaN.
    It then groups the data by the specified columns and generates a plot for each group. Each plot
    includes subplots for each output across time, showing individual trials and the mean output.
    The plots are saved in the specified directory.
    """

    mask = df["predicted_vib_outputs"].notna()
    df_filtered = df[mask]

    # Group by the specified columns
    groups = df_filtered.groupby(cols_to_group)

    # Create plots for each group
    for _, group_df in groups:
        # Get the 7xtimesteps array
        predicted_outputs_list = list(group_df["predicted_vib_outputs"])
        n_timesteps = predicted_outputs_list[0].shape[0]
        n_outputs = predicted_outputs_list[0].shape[1]
        time_points = np.arange(n_timesteps) / sample_rate

        #  Create figure with two columns of subplots
        # fig = plt.figure(figsize=(15, 12))
        fig = plt.figure(figsize=(BASE_FIG_SIZE_1COL[0]*4, BASE_FIG_SIZE_1COL[1]*2))
        # Create GridSpec with 4 rows (maximum needed for either column)
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Create axes - first column has 3 plots, second column has 4 plots
        axs = []
        # First column (3 plots)
        for i in range(3):
            axs.append(fig.add_subplot(gs[i, 0]))
        # Second column (4 plots)
        for i in range(4):
            axs.append(fig.add_subplot(gs[i, 1]))

        # share x axis
        for i in range(7):
            axs[i].sharex(axs[0])

        # share y axis
        num_trials = len(group_df["trial"].unique())
        muscles = group_df["vib_muscles_str"].iloc[0]
        freq = group_df["vib_freq"].iloc[0]
        # fig.suptitle(f'Trial: {trial} Muscles: {muscles} Frequency: {freq} Hz',
        fig.suptitle(
            f"Trials: {num_trials} Muscles: {muscles} Frequency: {freq} Hz", fontsize=20
        )

        # Labels for each output
        output_labels = [coord_name_mapping[key] for key in coord_name_mapping.keys()]

        # Calculate mean across all rows
        all_outputs = np.array(predicted_outputs_list)
        mean_outputs = np.mean(all_outputs, axis=0)

        # Plot each output in its subplot
        # Plot each output in its subplot
        for i in range(n_outputs):
            # Plot individual trials with low opacity
            for pred_output in predicted_outputs_list:
                axs[i].plot(
                    time_points, pred_output[:, i], "b-", alpha=0.2, linewidth=1
                )

            # Plot mean with high opacity and thicker line
            axs[i].plot(
                time_points,
                mean_outputs[:, i],
                "b-",
                alpha=1.0,
                linewidth=3,
                label="Mean",
            )

            # axs[i].set_ylabel(output_labels[i])
            axs[i].set_title(output_labels[i])
            axs[i].grid(False)

            # Add legend to first subplot
            if i == 0:
                axs[i].legend()

            # Only add x-label to bottom subplot
            if i == 6:
                axs[i].set_xlabel("Time (s)")

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/predicted_outputs_trial_{num_trials}_muscles_{muscles}_freq_{freq}.svg"
        )
        plt.show()

def plot_angleOffset_vs_vibration(
    df, save_path, color, suffix="", col_name="vib_angle_diff_elbow", per_seed=None, ylim=None, group_by=["trial", "vib_freq"], seed_marker_map = seed_marker_map, ls_map=None, maxLine=False, figsize=BASE_FIG_SIZE_1COL
):
    """
    Plot 'offset_n_v_vib' and 'vib_angle_diff' vs. vibration frequency,
    with improved visuals:
    - Jittered scatter plot
    - Triangular markers
    - Thin lines connecting trials
    - A thick line representing the mean

    Args:
        df (pd.DataFrame): DataFrame containing trial results.
        save_path (str): Directory to save the generated plot.
        color (str): Color for the markers and lines.
        suffix (str): Suffix for the filename of the saved plot.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    jitter_strength = 3  # Adjust this value for more or less jitter
    
    if per_seed is not None:
        # Loop over each unique training_seed and assign a unique marker
        unique_seeds = df[per_seed].unique()
        unique_seeds.sort()
        grad = generate_color_gradient(color, num_shades=len(unique_seeds))
        for i, seed in enumerate(unique_seeds):
            seed_data = df[df[per_seed] == seed]

            # Compute the mean value of col_name for each (trial, vib_freq) combination
            aggregated_data = (
                seed_data.groupby(group_by)
                .agg(col_name_mean=(col_name, "mean"))
                .reset_index()
            )

            # Add jitter to vibration frequencies for better visualization
            if len(aggregated_data["vib_freq"]) == len(aggregated_data["vib_freq"].unique()):
                print(len(aggregated_data["vib_freq"]))
                jittered_freqs = aggregated_data["vib_freq"]
                s=MS_BIG
            else:
                jittered_freqs = aggregated_data["vib_freq"] + np.random.uniform(
                    -jitter_strength, jitter_strength, size=len(aggregated_data)
                )
                s=MS

            # if type(seed) == str:
            #     marker = seed_marker_map[int(seed)]
            # else:
            #     marker = seed_marker_map[seed]
            marker = seed_marker_map[seed]
            # Scatter plot
            ax.scatter(
                jittered_freqs,  # Jittered vibration frequencies
                aggregated_data[
                    "col_name_mean"
                ],  # Mean value of col_name for each trial and vib_freq
                color=grad[i],
                edgecolor="black",
                alpha=ALPHA,
                marker=marker,  
                s=s,
                label=f"{per_seed} {seed}",
            )
            if ls_map is None:
                ls = '-'
            else:
                ls = ls_map[seed]
            # plot the mean of the plotted data
            ax.plot(
                aggregated_data["vib_freq"].unique(),
                aggregated_data.groupby("vib_freq")["col_name_mean"].mean(),
                color=grad[i],
                ls=ls,
                lw=LW,
            )
        ax.legend()
    else:
        # Compute the mean value of col_name for each (trial, vib_freq) combination
        aggregated_data = (
            df.groupby(["trial", "vib_freq"])
            .agg(col_name_mean=(col_name, "mean"))
            .reset_index()
        )

        # Add jitter to vibration frequencies for better visualization
        jittered_freqs = aggregated_data["vib_freq"] + np.random.uniform(
            -jitter_strength, jitter_strength, size=len(aggregated_data)
        )

        # Scatter plot
        ax.scatter(
            jittered_freqs,  # Jittered vibration frequencies
            aggregated_data[
                "col_name_mean"
            ],  # Mean value of col_name for each trial and vib_freq
            color=color,
            edgecolor="black",
            alpha=ALPHA,
            marker="o",  # Triangle marker
            s=MS,
            label="Trials",
        )
        # plot the mean of the plotted data
        ax.plot(
            aggregated_data["vib_freq"].unique(),
            aggregated_data.groupby("vib_freq")["col_name_mean"].mean(),
            color=color,
            lw=LW,
        )
        # add vertical line at vib_freq of max "col_name_mean"
        if maxLine:
            if ylim is None:
                ylim = ax.get_ylim()[1]
            max_freq = aggregated_data.groupby("vib_freq")["col_name_mean"].mean().abs().idxmax()
            ax.vlines(max_freq, -ylim, ylim, linestyles="dashed", color="black", lw=LW_SMALL)
            # ax.text(max_freq, 200, f"{max_freq:.2f} Hz", ha="center", va="bottom", fontsize=12)

    ax.hlines(0, 0, 200, linestyles="dashed", color="black", lw=LW_SMALL)
    # Add labels, grid, and legend
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)
    # else:
        # ax.set_ylim(-200, 200)
        
    ax.set_xlabel("Vibration Frequency (Hz)")
    if col_name == "vib_angle_diff_elbow":
        ax.set_ylabel(f"Predicted elbow angle diff. (deg)")
    else:
        ax.set_ylabel(f"{col_name} (deg)")
    # ax.legend(fontsize=16)
    # plt.title(f"{col_name} vs. Vibration Frequency (mean)")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{save_path}/mean_vib_angle_diff_vsfreq_{suffix}_per_{per_seed}.svg",
                format='svg',
                bbox_inches='tight')
    plt.savefig(f"{save_path}/mean_vib_angle_diff_vsfreq_{suffix}_per_{per_seed}.png")
    print("saved as ", f"mean_vib_angle_diff_vsfreq_{suffix}_per_{per_seed}.svg")
    plt.show()
    plt.close()


def plot_angleOffset_vs_vibration_elbowRange(df, save_path, suffix="", base_color="black", col_name=None, 
                                            margin_percent=0.1, y_min=None, y_max=None, num_ranges=5):
    """
    Plot 'offset_n_v_vib' and 'vib_angle_diff' vs. vibration frequency,
    colored by elbow angle ranges with dynamically adjusted axis ranges and dynamic elbow ranges.
    
    Args:
        df (pd.DataFrame): DataFrame containing trial results.
        save_path (str): Directory to save the generated plot.
        suffix (str): Suffix to add to the filename.
        base_color (str): Color for the mean line.
        col_name (str, optional): Column name to plot. If None, will try to find appropriate column.
        margin_percent (float): Percentage margin to add to data range (default: 10%).
        y_min (float, optional): Force minimum y-axis value. If None, calculated from data.
        y_max (float, optional): Force maximum y-axis value. If None, calculated from data.
        num_ranges (int): Number of elbow angle ranges to create (default: 5).
    """
    
    # Dynamically determine elbow angle ranges based on data
    # min_angle = df["elbow_angle"].min()
    # max_angle = df["elbow_angle"].max()
    
    # # Round to nearest 5 or 10 for cleaner bin edges
    # min_angle = np.floor(min_angle / 5) * 5
    # max_angle = np.ceil(max_angle / 5) * 5
    
    # # Create evenly spaced bins between min and max
    # elbow_angle_ranges = np.linspace(min_angle, max_angle, num_ranges + 1)
    elbow_angle_ranges =  [30, 60, 90, 120, 150]
    
    # Create range labels
    range_labels = [f"{int(elbow_angle_ranges[i])}-{int(elbow_angle_ranges[i+1])}°" 
                   for i in range(len(elbow_angle_ranges)-1)]
    
    df["elbow_angle_range"] = pd.cut(
        df["elbow_angle"], bins=elbow_angle_ranges, labels=range_labels
    )
    
    # Determine column to plot
    if col_name is None:
        if "vib_angle_diff" in df.columns:
            col_name = "vib_angle_diff"
        elif "vib_angle_diff_elbow" in df.columns:
            col_name = "vib_angle_diff_elbow"
        else:
            print("vib_angle_diff or vib_angle_diff_elbow not found in df")
            return
    
    jitter_strength = 3
    
    # Map ranges to colors
    if "TRI" in suffix:
        range_colors = cm.GnBu(np.linspace(0, 1, len(range_labels)))
    else:
        range_colors = cm.OrRd(np.linspace(0, 1, len(range_labels)))
    range_color_map = dict(zip(range_labels, range_colors))
    
    # Jittering setup
    vib_freq_name = "vib_freq_str" if isinstance(df["vib_freq"].iloc[0], list) else "vib_freq"
    df["jittered_vib_freq"] = df[vib_freq_name] + np.random.uniform(
        -jitter_strength, jitter_strength, size=len(df)
    )
    
    # Create figure and subplots (assuming BASE_FIG_SIZE_1COL is defined elsewhere)
    # If BASE_FIG_SIZE_1COL isn't defined in your environment, use a default size
    try:
        fig_size = BASE_FIG_SIZE_1COL
    except NameError:
        fig_size = (6, 5)  # Default size if BASE_FIG_SIZE_1COL is not defined
        
    fig, ax = plt.subplots(1, 1, figsize=fig_size, sharey=True)
    
    # Plotting
    for label, color in range_color_map.items():
        subset = df[df["elbow_angle_range"] == label]
        if not subset.empty:
            ax.scatter(
                subset["jittered_vib_freq"],
                subset[col_name],
                color=color,
                edgecolor="black",
                alpha=0.9,
                label=label,
                s=MS if 'MS' in globals() else 50  # Use default if MS not defined
            )
    
    # Plot the mean line if there are multiple vibration frequencies
    if len(df["vib_freq"].unique()) > 1:
        means = df.groupby("vib_freq")[col_name].mean()
        ax.plot(
            means.index,
            means.values,
            color=base_color,
            # marker='o',
            label='Mean'
        )
    
    # Dynamically set axis limits based on data
    # X-axis limits
    x_data = df["jittered_vib_freq"]
    x_min, x_max = x_data.min(), x_data.max()
    x_range = x_max - x_min
    x_margin = x_range * margin_percent
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    # Y-axis limits
    y_data = df[col_name]
    # Use provided limits if specified, otherwise calculate from data
    if y_min is None:
        y_min = y_data.min()
    if y_max is None:
        y_max = y_data.max()
    y_range = y_max - y_min
    y_margin = y_range * margin_percent
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Improve grid and appearance
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels
    ax.set_xlabel("Vibration Frequency (Hz)")
    ax.set_ylabel(f"{col_name.replace('_', ' ').title()} (deg)")
    
    # Add legend with better placement
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    ax.legend()
    
    # Save the plot with higher DPI
    plt.tight_layout()
    plt.savefig(f"{save_path}/vib_angle_diff_vsfreq_ElbowAngleRange_{suffix}.svg", 
                bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}/vib_angle_diff_vsfreq_ElbowAngleRange_{suffix}.png", 
                bbox_inches='tight', dpi=300)
    
    plt.show()
    
    return fig, ax  # Return the figure and axis objects for further customization if needed

from matplotlib.colors import to_rgba


def generate_color_gradient(base_hex, num_shades=5, factor=0.1):
    """
    Generates a gradient of colors based on the base color (lighter to darker shades).
    base_hex: Base hex color for the gradient.
    num_shades: Number of color shades to generate.
    factor: How much to lighten or darken each subsequent shade.
    """
    base_rgb = np.array(to_rgba(base_hex)[:3])  # Convert hex to RGB
    shades = [base_rgb]

    # Generate shades from lighter to darker
    for i in range(1, num_shades):
        # Darken the color by a factor for each subsequent shade
        new_rgb = np.clip(base_rgb * (1 - factor * i), 0, 1)
        shades.append(new_rgb)

    # Convert RGB back to RGBA with alpha=1
    return [to_rgba(rgb, alpha=1) for rgb in shades]


# Function to plot data for a specific muscle
def plot_vibration_vs_input(
    data,
    muscle_names,
    time_point,
    channels=[0, 1, 2, 3, 4],
    per_seed=None,
    grad_color="black",
    suffix="",
    save_path=None,
    seed_marker_map = None, 
    ls_map=None,
    colors=None,
):
    muscle_idxs = [MUSCLE_NAMES.index(muscle_name) for muscle_name in muscle_names]

    # Extract vibration frequencies
    vib_freqs = data["vib_freq"].unique()
    vib_freqs.sort()  # Ensure frequencies are sorted

    if per_seed is not None:
        seeds = data[per_seed].unique()
        num_lines = len(seeds) * len(muscle_idxs) * len(channels)
    else:
        num_lines = len(muscle_idxs) * len(channels)
    if colors is None:
        grad = generate_color_gradient(grad_color, num_shades=num_lines)
    else:
        grad = colors

    fig, ax = plt.subplots(1, 1, figsize=BASE_FIG_SIZE_1COL)
    # Loop through channels
    c_idx = 0
    for j, muscle_idx in enumerate(muscle_idxs):
        for c, channel in enumerate(channels):
            if per_seed is not None:
                seeds = data[per_seed].unique()
                seeds.sort()
                for i, seed in enumerate(seeds):
                    inputs_at_time = []
                    m = seed_marker_map[seed]
                    d = data[data[per_seed] == seed]
                    for freq in vib_freqs:
                        # Filter data for the current frequency
                        freq_data = d[d["vib_freq"] == freq]
                        # Extract input values at the specific time point
                        values = [
                            np.array(row.inputs_vib)[channel, muscle_idx, time_point]
                            for row in freq_data.itertuples()
                        ]
                        # print(len(values))
                        inputs_at_time.append(
                            np.mean(values)
                        )  # Average across trials for this frequency
                    color = grad[c_idx]
                    c_idx += 1
                    # Plot the result for the current channel
                    ax.plot(
                        vib_freqs,
                        inputs_at_time,
                        label=f"Aff {channel}, Muscle {muscle_names[j]}, {per_seed} {seed}",
                        marker=m,
                        ms=MS_line,
                        linestyle="--",
                        color=color,
                    )
            else:
                inputs_at_time = []
                if seed_marker_map is not None:
                    m = seed_marker_map[str(channel)]
                else:
                    m='o'
                if ls_map is not None:
                    ls = ls_map[str(channel)]
                else:
                    ls = '--'
                for freq in vib_freqs:
                    # Filter data for the current frequency
                    freq_data = data[data["vib_freq"] == freq]
                    # Extract input values at the specific time point
                    values = [
                        np.array(row.inputs_vib)[channel, muscle_idx, time_point]
                        for row in freq_data.itertuples()
                    ]
                    inputs_at_time.append(
                        np.mean(values)
                    )  # Average across trials for this frequency
                # Plot the result for the current channel
                ax.plot(
                    vib_freqs,
                    inputs_at_time,
                    label=f"Aff {channel}, Muscle {muscle_names[j]}",
                    marker=m,
                    ms=MS_line,
                    linestyle=ls,
                    color=grad[c_idx],
                )
                c_idx += 1

    # concatenate muscle names into str
    muscle_names = "_".join(muscle_names)
    # ax.set_title(f"Firing rate (Hz)")
    ax.set_xlabel("Vibration Frequency (Hz)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            f"{save_path}/inputs_vs_vibfreq_{muscle_names}_{suffix}_per_{per_seed}.png"
        )
        plt.savefig(
            f"{save_path}/inputs_vs_vibfreq_{muscle_names}_{suffix}_per_{per_seed}.svg"
        )
    plt.show()
    plt.close()

def compute_correlation(data, labels, muscle_indices=None, FR_range=None):
    if muscle_indices is None:
        muscle_indices = range(data.shape[1])

    num_muscles = len(muscle_indices)  # 25 muscles
    num_trials = data.shape[0]  # Number of trials
    correlations = np.zeros(num_muscles)

    for i, m_idx in enumerate(muscle_indices):
        muscle_data = data[:, m_idx, :]  # Shape: (num_trials, num_timesteps)
        output_data = labels  # Shape: (num_trials, num_timesteps)

        trial_corrs = []
        for t in range(num_trials):
            muscle_values = muscle_data[t]
            output_values = output_data[t]

            # Apply filtering based on input range
            if FR_range is not None:
                min_val, max_val = FR_range
                mask = (muscle_values >= min_val) & (muscle_values <= max_val)
                muscle_values = muscle_values[mask]
                output_values = output_values[mask]

            # Compute correlation if we have valid values after filtering
            if len(muscle_values) > 1:
                trial_corrs.append(np.corrcoef(muscle_values, output_values)[0, 1])

        # Compute mean correlation across trials, ignoring NaNs
        correlations[i] = np.nanmean(trial_corrs) if trial_corrs else np.nan

    return correlations


def compute_saliency_optimized(model, inputs, output_idx, swap_elbow_angle=True):
    """
    Computes saliency maps efficiently using a single forward pass.

    Parameters:
    - model: Trained PyTorch model
    - inputs: Tensor [trials, channels, muscles, time]
    - output_idx: Index of the output variable to analyze
    - input_range: Dict of the form {"min": min_value, "max": max_value, "m_idx": muscle_idx, "channel": channel_idx}

    Returns:
    - saliency_maps: Saliency maps [trials, channels, muscles, time]
    """
    inputs = inputs.clone().detach().requires_grad_(True)  # Track gradients
    model.eval()

    # Forward pass once
    preds, _, _ = model(inputs)  # Shape: (trials, time, output_dim)

    print(preds.shape)
    if swap_elbow_angle:
        preds[..., ELBOW_ANGLE_INDEX] = 180 - preds[..., ELBOW_ANGLE_INDEX]

    # Select the relevant output for gradient computation
    target_output = preds[:, :, output_idx]  # Shape: (trials, time)

    # Create a tensor of ones for the gradient computation
    grad_outputs = torch.ones_like(target_output)

    # Compute gradients of target output w.r.t. inputs in a single step
    gradients = torch.autograd.grad(
        outputs=target_output,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
    )[
        0
    ]  # Shape: (trials, channels, muscles, times)

    # return mean over the trials
    # gradients = torch.mean(gradients, dim=0)
    return gradients.detach().cpu().numpy()  # Convert to NumPy at the end


def compute_integrated_gradients(
    model,
    inputs,
    baseline_inputs=None,
    output_idx=6,
    perturbed_muscle_idx=None,
    steps=50,
    muscle_names=MUSCLE_NAMES,
    plot=False,
    swap_elbow_angle=True,
):
    """
    Compute Integrated Gradients attribution for model predictions.
    If perturbed_muscle_idx is provided, will analyze attribution differences
    when that muscle is perturbed.

    Args:
        model: Trained PyTorch model
        inputs: Input tensor of shape (trials, channels, muscles, times)
        baseline_inputs: Baseline inputs (zeros by default)
        output_idx: Index of output to analyze
        perturbed_muscle_idx: Optional muscle to perturb for comparison
        steps: Number of steps for integration
        muscle_names: Optional list of muscle names for plotting

    Returns:
        attributions: Attribution scores per muscle
    """
    if baseline_inputs is None:
        baseline_inputs = torch.zeros_like(inputs)

    model.eval()

    # Prepare for interpolation
    input_diff = inputs - baseline_inputs
    scaled_inputs = [
        baseline_inputs + (float(i) / steps) * input_diff for i in range(steps + 1)
    ]

    # If analyzing a perturbed condition
    if perturbed_muscle_idx is not None:
        perturbed_inputs = inputs.clone()
        perturbation = torch.zeros_like(inputs)
        perturbation[:, :, perturbed_muscle_idx, :] = 5.0  # Apply 5.0 perturbation
        perturbed_inputs = perturbed_inputs + perturbation

        perturbed_diff = perturbed_inputs - baseline_inputs
        perturbed_scaled = [
            baseline_inputs + (float(i) / steps) * perturbed_diff
            for i in range(steps + 1)
        ]

    # Calculate gradients at each step
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_(True)
        output, *_ = model(scaled_input)
        if swap_elbow_angle:
            output[..., ELBOW_ANGLE_INDEX] = 180 - output[..., ELBOW_ANGLE_INDEX]
        target = output[:, :, output_idx]

        grad_outputs = torch.ones_like(target)
        grad = torch.autograd.grad(
            outputs=target,
            inputs=scaled_input,
            grad_outputs=grad_outputs,
            create_graph=False,
        )[0]
        grads.append(grad.detach())
        scaled_input.requires_grad_(False)

    # Average gradients
    avg_grads = torch.stack(grads).mean(0)
    attributions = avg_grads * input_diff

    # Compute attributions for perturbed input if requested
    if perturbed_muscle_idx is not None:
        perturbed_grads = []
        for p_input in perturbed_scaled:
            p_input.requires_grad_(True)
            output, *_ = model(p_input)
            if swap_elbow_angle:
                output[..., ELBOW_ANGLE_INDEX] = 180 - output[..., ELBOW_ANGLE_INDEX]
            target = output[:, :, output_idx]

            grad_outputs = torch.ones_like(target)
            grad = torch.autograd.grad(
                outputs=target,
                inputs=p_input,
                grad_outputs=grad_outputs,
                create_graph=False,
            )[0]
            perturbed_grads.append(grad.detach())
            p_input.requires_grad_(False)

        avg_perturbed_grads = torch.stack(perturbed_grads).mean(0)
        perturbed_attributions = avg_perturbed_grads * perturbed_diff

        # Compute attribution differences
        attribution_diff = perturbed_attributions - attributions

        # Aggregate over time and trials for visualization
        attribution_per_muscle = attribution_diff.sum(dim=(0, 1, 3)).cpu().numpy()

        if muscle_names is None:
            muscle_names = [f"Muscle {i+1}" for i in range(len(attribution_per_muscle))]

        if plot:
            # Plot attribution differences
            plt.figure(figsize=BASE_FIG_SIZE_1COL)
            plt.bar(muscle_names, attribution_per_muscle)
            plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            plt.title(
                f"Attribution Difference When Perturbing Muscle {perturbed_muscle_idx+1}"
            )
            plt.ylabel("Attribution Difference")
            plt.xlabel("Muscles")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        return attribution_diff.detach().cpu().numpy()

    # Aggregate attributions per muscle for standard analysis
    attribution_per_muscle = attributions.sum(dim=(0, 1, 3)).cpu().numpy()

    if plot:
        # Plot attributions
        plt.figure(figsize=BASE_FIG_SIZE_1COL)
        plt.bar(muscle_names, attribution_per_muscle)
        plt.title(f"Feature Attribution for Output {output_idx}")
        plt.ylabel("Attribution Score")
        plt.xlabel("Muscles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return attributions.detach().cpu().numpy()


def compute_global_saliency(saliency_maps, inputs=None, input_range=None):
    """
    Compute a single measure of influence for each input channel/muscle on elbow angle.

    Args:
        saliency_maps (numpy array): (num_trials, num_channels, num_muscles, time_steps)

    Returns:
        effect_strength (numpy array): (num_channels, num_muscles), showing absolute influence
        effect_direction (numpy array): (num_channels, num_muscles), showing signed influence
    """
    # Apply input filtering if a range is provided
    # print(f"Number of non-zero values in measure: {np.sum(saliency_maps != 0)}")
    if input_range is not None:
        saliency_maps = saliency_maps.copy()
        channels = input_range["channels"]  # Channel to check
        m_idx = input_range["m_idx"]  # Muscle index to check
        min_value = input_range["min"]  # Minimum threshold
        max_value = input_range["max"]  # Maximum threshold

        # Ensure inputs are provided
        if inputs is None:
            raise ValueError("inputs must be provided if input_range is not None")
        if inputs.shape[0] != saliency_maps.shape[0]:
            raise ValueError(
                "inputs and saliency_maps must have the same number of trials"
            )

        mask = np.zeros_like(saliency_maps, dtype=bool)

        # Apply mask for all specified channels
        for channel in channels:
            channel_mask = (inputs[:, :, channel, m_idx, :] >= min_value) & (
                inputs[:, :, channel, m_idx, :] <= max_value
            )

            # Expand dimensions to match (num_trials, num_channels, num_muscles, time_steps)
            channel_mask = np.expand_dims(channel_mask, axis=2)  # Add muscle dimension
            channel_mask = np.expand_dims(channel_mask, axis=2)  # Add channel dimension

            # Broadcast across channels & muscles
            channel_mask = np.broadcast_to(channel_mask, saliency_maps.shape)

            mask |= channel_mask  # Combine with mask

        # print(f"Number of non-zero values in mask: {np.sum(mask)}")

        # If mask is completely empty, return NaN
        if np.sum(mask) == 0:
            saliency_maps = np.full_like(saliency_maps, np.nan)
        else:
            saliency_org = saliency_maps
            # copy to avoid overwriting
            saliency_maps *= mask  # Apply mask

    # if all values are zero or nan
    if np.all(saliency_maps == 0) or np.all(np.isnan(saliency_maps)):
        print("all values are zero or nan")
    # average over trials
    mean_signed_gradient = np.mean(
        saliency_maps, axis=1
    )  # (models, num_channels, num_muscles, time)
    # average over time
    mean_signed_gradient = np.mean(
        mean_signed_gradient, axis=-1
    )  # (models, num_channels, num_muscles)

    return mean_signed_gradient  # Function to plot correlation distributions


def compute_attribution_variability(
    model,
    test_data,
    muscle_idx,
    output_idx=0,
    input_ranges=[(10, 20), (20, 30), (30, 40), (40, 50)],
):
    """Analyze how attribution changes across different input ranges for a specific muscle."""
    results = {}

    for range_min, range_max in input_ranges:
        # Filter test data where the target muscle has inputs in the specified range
        mask = (test_data[:, :, muscle_idx, :] >= range_min) & (
            test_data[:, :, muscle_idx, :] <= range_max
        )
        if not torch.any(mask):
            continue

        # Create masked input containing only examples where muscle input is in range
        filtered_indices = torch.any(mask, dim=(1, 3))
        filtered_data = test_data[filtered_indices]

        if len(filtered_data) == 0:
            continue

        # Compute attributions for this input range
        attributions = compute_integrated_gradients(
            model=model, inputs=filtered_data, output_idx=output_idx
        )

        # Extract statistics about attributions for the target muscle
        muscle_attributions = (
            attributions[:, :, muscle_idx, :].sum(axis=(0, 1, 3)).item()
        )

        results[f"{range_min}-{range_max}"] = {
            "mean_attribution": muscle_attributions,
            "sample_count": len(filtered_data),
        }

    return results


def compute_attribution_consistency(attribution_ranges, m_idx):
    """Calculate consistency score for attributions across input ranges."""

    attributions = [data[m_idx] for data in attribution_ranges.values()]
    # remove nans
    attributions = [a for a in attributions if not np.isnan(a)]
    print(attributions)
    # Signs consistency (1.0 = perfectly consistent, 0.0 = completely inconsistent)
    signs = np.sign(attributions)
    sign_consistency = np.abs(np.sum(signs) / len(signs))

    # Coefficient of variation (normalized standard deviation)
    # Lower values indicate more consistent attribution magnitudes
    attribution_cv = np.std(np.abs(attributions)) / np.mean(np.abs(attributions))

    return {
        "sign_consistency": sign_consistency,
        "magnitude_variability": attribution_cv,
        "raw_attributions": attributions,
    }


def plot_correlation_distributions(
    filtered_data, filtered_labels, muscle_names=MUSCLE_NAMES
):
    plt.figure(figsize=BASE_FIG_SIZE_2COL)
    colors = sns.color_palette(
        "tab10", len(filtered_data)
    )  # Different colors for each dataset

    for i, (data, labels) in enumerate(zip(filtered_data, filtered_labels)):
        correlations = compute_correlation(data, labels)

        # Sort by absolute correlation
        sorted_indices = np.argsort(-np.abs(correlations))
        sorted_correlations = correlations[sorted_indices]
        sorted_muscle_names = [muscle_names[i] for i in sorted_indices]

        # Plot as a bar plot
        sns.barplot(
            x=sorted_muscle_names,
            y=sorted_correlations,
            color=colors[i],
            alpha=0.6,
            label=f"Dataset {i+1}",
        )

    plt.xlabel("Muscle Name")
    plt.ylabel("Mean Correlation (Channel 0 vs. Output 6)")
    plt.title("Correlation Distributions Across Multiple Datasets")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_side_by_side_correlation(
    filtered_data,
    filtered_labels,
    muscles_to_plot=None,
    FR_range=None,
    muscle_names=MUSCLE_NAMES,
    path_save=path_save,
    sort=False,
    swap_elbow_angle=True,
    suffix="",
    seed_name="Coefficient",
):
    num_datasets = len(filtered_data)

    if muscles_to_plot is not None:
        muscle_idxes = [muscle_names.index(i) for i in muscles_to_plot]
        muscle_names = muscles_to_plot
    else:
        muscle_idxes = range(len(muscle_names))

    num_muscles = len(muscle_names)
    bar_width = 0.15  # Adjusted for better spacing
    spacing_factor = 0.3  # Extra space between muscle groups
    x = np.arange(num_muscles) * (1 + spacing_factor)  # Add space between muscle groups

    # Get distinct colors for each muscle using the tab10 palette
    muscle_colors = sns.color_palette("tab10", num_muscles)

    handles = []  # Store bar handles for legend

    # plt.figure(figsize=(20, 6))
    plt.figure(figsize=(BASE_FIG_SIZE_2COL[0], BASE_FIG_SIZE_1COL[1]))

    for i, (data, labels) in enumerate(zip(filtered_data, filtered_labels)):
        correlations = compute_correlation(
            data, labels, muscle_indices=muscle_idxes, FR_range=FR_range
        )

        if i == 0:
            if sort:
                sorted_indices = np.argsort(-np.abs(correlations))
                muscle_names_sorted = [muscle_names[j] for j in sorted_indices]
            else:
                muscle_names_sorted = muscle_names
                sorted_indices = np.arange(len(muscle_names))

        sorted_correlations = correlations[sorted_indices]

        # Generate shades of the base muscle color for each dataset (seed)
        color_shades = [
            sns.light_palette(muscle_colors[j], n_colors=num_datasets)[i]
            for j in range(num_muscles)
        ]

        # Shift x positions for side-by-side bars
        x_shifted = x + (i - num_datasets / 2) * bar_width

        # Plot each bar with the correct shade and spacing
        for j in range(num_muscles):
            bar = plt.bar(
                x_shifted[j],
                sorted_correlations[j],
                width=bar_width,
                color=color_shades[j],
            )

        handles.append(bar[0])

    # X-axis labels
    plt.xticks(x, muscle_names_sorted, rotation=45, ha="right")
    plt.xlabel("Muscle Name")
    plt.ylabel("Mean Correlation")
    # Manually add legend with correct colors
    plt.legend(handles, [f"Seed {i}" for i in range(num_datasets)], title=seed_name)

    plt.grid(True, linestyle="--", alpha=0.6)
    if FR_range is not None:
        plt.title(f"Correlation per muscle - FR Range: {FR_range[0]}-{FR_range[1]} Hz")
        plt.savefig(
            f"{path_save}/correlations_inputs_elbow_angle_perSeed_{num_muscles}_swapElbow-{swap_elbow_angle}_rangeFR-{FR_range[0]}-{FR_range[1]}_{suffix}.jpeg"
        )
    else:
        plt.title(f"Correlation per muscle")
        plt.savefig(
            f"{path_save}/correlations_inputs_elbow_angle_perSeed_{num_muscles}_swapElbow-{swap_elbow_angle}_{suffix}.jpeg"
        )
    plt.show()


def plot_side_by_side_saliency(
    global_saliency,
    muscles_to_plot=None,
    path_save=path_save,
    sort=False,
    suffix="",
    seed_name="Coefficient",
    measure_name="Sensitivity",
):
    """
    Plot the global saliency influence of each muscle on elbow angle across multiple models.

    Args:
        effect_strengths (list of np.array): List of global saliency strengths from different models.
        effect_directions (list of np.array): List of global saliency directions from different models.
        muscles_to_plot (list of str): Subset of muscles to plot.
        muscle_names (list of str): Names of all muscles.
        path_save (str): Path to save the plot.
        sort (bool): Whether to sort muscles by absolute saliency strength.
        suffix (str): Suffix for saving the plot.
        seed_name (str): Label for differentiating models.
    """
    num_datasets = len(global_saliency)

    if muscles_to_plot is not None:
        muscle_idxes = [MUSCLE_NAMES.index(i) for i in muscles_to_plot]
        muscle_names = muscles_to_plot
    else:
        muscle_idxes = range(len(MUSCLE_NAMES))
        muscle_names = MUSCLE_NAMES

    num_muscles = len(muscle_names)
    bar_width = 0.15  # Adjusted for better spacing
    spacing_factor = 0.3  # Extra space between muscle groups
    x = np.arange(num_muscles) * (1 + spacing_factor)  # Add space between muscle groups

    # Get distinct colors for each muscle using the tab10 palette
    muscle_colors = sns.color_palette("tab10", num_muscles)

    handles = []  # Store bar handles for legend
    plt.figure(figsize=(BASE_FIG_SIZE_2COL[0], BASE_FIG_SIZE_1COL[1]))

    for i, saliency in enumerate(global_saliency):
        # Extract values for selected muscles
        saliency_values = saliency[muscle_idxes]

        # Combine for sorting if needed
        if i == 0:
            if sort:
                sorted_indices = np.argsort(-np.abs(saliency_values))
                muscle_names_sorted = [muscle_names[j] for j in sorted_indices]
            else:
                muscle_names_sorted = muscle_names
                sorted_indices = np.arange(len(muscle_names))

        sorted_strengths = saliency_values[sorted_indices]

        # Generate shades of the base muscle color for each dataset (seed)
        color_shades = [
            sns.light_palette(muscle_colors[j], n_colors=num_datasets)[i]
            for j in range(num_muscles)
        ]

        # Shift x positions for side-by-side bars
        x_shifted = x + (i - num_datasets / 2) * bar_width

        # Plot bars
        for j in range(num_muscles):
            bar = plt.bar(
                x_shifted[j],
                sorted_strengths[j],  # Preserve direction
                width=bar_width,
                color=color_shades[j],
            )
        handles.append(bar[0])

    # X-axis labels
    plt.xticks(x, muscle_names_sorted, rotation=45, ha="right")
    plt.xlabel("Muscle Name")
    plt.ylabel(f"{measure_name}")

    # Add legend
    plt.legend(handles, [f"Seed {i}" for i in range(num_datasets)], title=seed_name)

    plt.grid(True, linestyle="--", alpha=0.6)
    # plt.title(f"Global Saliency Influence per Muscle")
    plt.savefig(f"{path_save}/{measure_name}_perSeed_bar_{num_muscles}_{suffix}.jpeg")
    plt.show()


def plot_scatter_vib_angle_vs_correlation(
    max_data,
    color,
    muscle_name="",
    ax=None,
    path_save=path_save,
    col_name="vib_angle_diff_elbow",
    seed_col="coef_seed",
):
    """
    Plots a scatter plot of vib_angle_diff_elbow vs. correlation.
    - Uses different markers per train_seed.
    - Uses the given color for all points.

    Args:
        max_data (pd.DataFrame): DataFrame containing 'vib_angle_diff_elbow', 'correlation', and 'train_seed'.
        color (str): Color for the scatter points.
    """

    # Define unique markers for train_seed values
    # markers = MARKER_LIST

    unique_seeds = max_data[seed_col].unique()

    if ax is None:
        fig, ax = plt.subplots(
            figsize=BASE_FIG_SIZE_1COL
        )  # Create a new figure if no axis is provided

    # unique_seeds = max_data["coef_seed"].unique()
    # unique_seeds.sort()
    # seed_marker_map = {seed: MARKER_LIST[i] for i, seed in enumerate(unique_seeds)}
    if seed_col == "train_seed":
        colors = generate_color_gradient(color, num_shades=len(unique_seeds))
    else:
        colors = [color for _ in unique_seeds]

    # Plot each train_seed with a different marker
    for i, seed in enumerate(unique_seeds):
        subset = max_data[max_data[seed_col] == seed]
        coef_seed = subset["coef_seed"].iloc[0]
        ax.scatter(
            subset["correlation"],
            subset[col_name],
            color=colors[i],
            marker=seed_marker_map[coef_seed],  # Cycle through markers
            label=f"{seed_col} {seed}",
            edgecolors="black",
            alpha=0.8,
            s=200,
        )  # Adjust marker size

    # Labels and title
    # Labels and formatting
    ax.set_ylabel("Max Vib Angle Diff Elbow")
    ax.set_xlabel("Correlation")
    ax.set_title(f"Correlation vs. Vib Angle Diff ({muscle_name})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    # Add a horizontal dashed line at 0
    ax.axhline(0, linestyle="dashed", color="black", linewidth=1)
    ax.axvline(0, linestyle="dashed", color="black", linewidth=1)

    if ax is None:
        fig.savefig(f"{path_save}/scatter_vib_angle_vs_correlation_{muscle_name}.jpeg")


def plot_scatter_vib_angle_vs_measure(
    max_data,
    color,
    muscle_name="",
    ax=None,
    path_save=path_save,
    x_col="correlation",
    y_col="vib_angle_diff_elbow",
    seed_col="coef_seed",
):
    """
    Plots a scatter plot of vib_angle_diff_elbow vs. correlation.
    - Uses different markers per train_seed.
    - Uses the given color for all points.

    Args:
        max_data (pd.DataFrame): DataFrame containing 'vib_angle_diff_elbow', 'correlation', and 'train_seed'.
        color (str): Color for the scatter points.
    """

    # Define unique markers for train_seed values
    # markers = MARKER_LIST

    unique_seeds = max_data[seed_col].unique()

    if ax is None:
        fig, ax = plt.subplots(
            figsize=BASE_FIG_SIZE_1COL
        )  # Create a new figure if no axis is provided

    # unique_seeds = max_data["coef_seed"].unique()
    # unique_seeds.sort()
    # seed_marker_map = {seed: MARKER_LIST[i] for i, seed in enumerate(unique_seeds)}
    if seed_col == "train_seed":
        colors = generate_color_gradient(color, num_shades=len(unique_seeds))
    else:
        colors = [color for _ in unique_seeds]

    # Plot each train_seed with a different marker
    for i, seed in enumerate(unique_seeds):
        subset = max_data[max_data[seed_col] == seed]
        coef_seed = str(subset["coef_seed"].iloc[0])
        ax.scatter(
            subset[x_col],
            subset[y_col],
            color=colors[i],
            marker=seed_marker_map[coef_seed],  # Cycle through markers
            label=f"{seed_col} {seed}",
            edgecolors="black",
            alpha=0.8,
            s=200,
        )  # Adjust marker size

    # Labels and title
    # Labels and formatting
    ax.set_ylabel("Max Vib Angle Diff Elbow")
    ax.set_xlabel(f"{x_col}")
    ax.set_title(f"{muscle_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    # Add a horizontal dashed line at 0
    ax.axhline(0, linestyle="dashed", color="black", linewidth=1)
    ax.axvline(0, linestyle="dashed", color="black", linewidth=1)

    if ax is None:
        fig.savefig(f"{path_save}/scatter_vib_angle_vs_{x_col}_{muscle_name}.jpeg")


def plot_max_illusion_vs_measure(
    max_data,
    muscles_to_plot,
    path_save,
    suffix,
    seed_col,
    y_col_name,
    x_col_name,
    FR_range=None,
):
    num_muscles = len(muscles_to_plot)
    fig, axes = plt.subplots(1, num_muscles, figsize=(BASE_FIG_SIZE_1COL[0] * num_muscles, BASE_FIG_SIZE_1COL[1]), sharey=True)

    if num_muscles == 1:  # If only one muscle, make axes iterable
        axes = [axes]

    for ax, muscle_vibrated in zip(axes, muscles_to_plot):
        # Filter data for the muscle
        max_data_v = max_data[max_data["vib_muscles_str"] == muscle_vibrated]
        muscle_idx = MUSCLE_NAMES.index(muscle_vibrated)
        # select in x_col_name the element in muscle_idx
        max_data_v[x_col_name] = max_data_v[x_col_name].apply(lambda x: x[muscle_idx])

        if "TRI" in muscle_vibrated:
            color = original_hex_tri
        elif "BIC" in muscle_vibrated:
            color = original_hex_biceps
        else:
            color = "black"
        # Scatter plot for this muscle
        plot_scatter_vib_angle_vs_measure(
            max_data_v,
            color,
            muscle_vibrated,
            ax=ax,
            y_col=y_col_name,
            x_col=x_col_name,
            seed_col=seed_col,
        )

        ax.set_title(f"{muscle_vibrated}")

    plt.tight_layout()
    if FR_range is not None:
        plt.savefig(
            f"{path_save}/{y_col_name}_vs_{x_col_name}_{FR_range[0]}_{FR_range[1]}_{suffix}.jpeg"
        )
    else:
        plt.savefig(f"{path_save}/{y_col_name}_vs_{x_col_name}_{suffix}.jpeg")
    plt.show()


def compute_max_illusion(df_f, col_name):
    # Aggregate data
    aggregated_data = (
        df_f.groupby(["coef_seed", "train_seed", "vib_freq", "vib_muscles_str"])
        .agg(vib_angle_diff_elbow=(col_name, "mean"))
        .reset_index()
    )

    # Compute absolute value and sign
    aggregated_data[f"{col_name}_abs"] = np.abs(aggregated_data[col_name])
    aggregated_data[f"{col_name}_sign"] = np.sign(aggregated_data[col_name])

    # Get max absolute vib_angle_diff_elbow per (coef_seed, train_seed)
    idx_max = aggregated_data.groupby(["coef_seed", "train_seed", "vib_muscles_str"])[
        f"{col_name}_abs"
    ].idxmax()
    max_data = aggregated_data.loc[
        idx_max, ["coef_seed", "train_seed", "vib_freq", col_name, "vib_muscles_str"]
    ]
    return max_data


def analyze_multiple_muscles(
    muscles_vibrated, df_all, filtered_data, filtered_labels, path_save, FR_range=None
):
    num_muscles = len(muscles_vibrated)
    fig, axes = plt.subplots(1, num_muscles, figsize=(BASE_FIG_SIZE_1COL[0] * num_muscles, BASE_FIG_SIZE_1COL[1]), sharey=True)

    if num_muscles == 1:  # If only one muscle, make axes iterable
        axes = [axes]

    for ax, muscle_vibrated in zip(axes, muscles_vibrated):
        # Filter data for the muscle
        df_f = df_all[df_all["vib_muscles_str"] == muscle_vibrated]

        max_data = compute_max_illusion(df_f, "vib_angle_diff_elbow")
        # Compute correlation for the current muscle
        correlations = [
            compute_correlation(
                data, labels, [MUSCLE_NAMES.index(muscle_vibrated)], FR_range=FR_range
            )
            for data, labels in zip(filtered_data, filtered_labels)
        ]

        max_data["coef_seed"] = max_data["coef_seed"].astype(int)

        # Map correlations to max_data
        correlation_mapping = {i: correlations[i][0] for i in range(len(correlations))}
        max_data["correlation"] = max_data["coef_seed"].map(correlation_mapping)

        if "TRI" in muscle_vibrated:
            color = original_hex_tri
        elif "BIC" in muscle_vibrated:
            color = original_hex_biceps
        else:
            color = "black"
        # Scatter plot for this muscle
        plot_scatter_vib_angle_vs_correlation(max_data, color, muscle_vibrated, ax=ax)

        ax.set_title(f"{muscle_vibrated}")

    plt.tight_layout()
    if FR_range is not None:
        plt.savefig(
            f"{path_save}/correlation_vs_vib_angle_all_muscles_{FR_range[0]}_{FR_range[1]}.jpeg"
        )
    else:
        plt.savefig(f"{path_save}/correlation_vs_vib_angle_all_muscles.jpeg")
    plt.show()


# Function 1: Plot distribution of data for specific muscles
def plot_data_distribution(
    filtered_data, muscles_to_plot=None, muscle_names=MUSCLE_NAMES
):
    if muscles_to_plot is None:
        muscle_indices = range(len(muscle_names))
    else:
        muscle_indices = [MUSCLE_NAMES.index(i) for i in muscles_to_plot]
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=BASE_FIG_SIZE_MULTCOL)

    for i, data in enumerate(filtered_data):
        for m_idx in muscle_indices:
            # sns.kdeplot(data[:, m_idx, :].flatten(), label=f"seed {i} - Muscle {MUSCLE_NAMES[m_idx]}", fill=True, alpha=0.5)
            plt.hist(
                data[:, m_idx, :].flatten(),
                bins=100,
                label=f"seed {i} - Muscle {MUSCLE_NAMES[m_idx]}",
                alpha=0.5,
            )

    plt.xlabel("FR (Hz)")
    plt.ylabel("Density")
    plt.title("Distribution FR per muscle")
    plt.legend()
    plt.show()


# Function 2: Plot distribution of label values for all trials and times
def plot_labels_distribution(filtered_labels):
    fig, axs = plt.subplots(
        1, len(filtered_labels), figsize=(BASE_FIG_SIZE_1COL[0] * len(filtered_labels), BASE_FIG_SIZE_1COL[1]), sharey=True
    )

    for i, labels in enumerate(filtered_labels):
        if len(filtered_labels) == 1:
            ax = axs
        else:
            ax = axs[i]
        # sns.kdeplot(labels.flatten(), label=f"seed {i}", fill=True, alpha=0.5, ax=ax)
        ax.hist(labels.flatten(), bins=100, label=f"seed {i}", alpha=0.5)
        ax.set_xlabel("Elbow angle (deg)")
    fig.set_ylabel("Density")
    fig.suptitle("Distribution of elbow angle")
    fig.legend()
    fig.show()


# Function 3: Plot distribution of labels filtered by input data range
def plot_filtered_labels_distribution(
    filtered_data,
    filtered_labels,
    value_range,
    muscles_to_plot=None,
    muscle_names=MUSCLE_NAMES,
):
    if muscles_to_plot is None:
        muscle_indices = range(len(muscle_names))
    else:
        muscle_indices = [MUSCLE_NAMES.index(i) for i in muscles_to_plot]
    plt.figure(figsize=BASE_FIG_SIZE_MULTCOL)

    for i, (data, labels) in enumerate(zip(filtered_data, filtered_labels)):
        for m_idx in muscle_indices:
            mask = (data[:, m_idx, :].flatten() >= value_range[0]) & (
                data[:, m_idx, :].flatten() <= value_range[1]
            )
            filtered_labels_values = labels.flatten()[mask]

            if len(filtered_labels_values) > 0:
                sns.kdeplot(
                    filtered_labels_values,
                    label=f"Seed {i} - Muscle {MUSCLE_NAMES[m_idx]}",
                    fill=True,
                    alpha=0.5,
                )

    plt.xlabel("Elbow angle")
    plt.ylabel("Density")
    plt.title(f"{value_range}")
    plt.legend()
    plt.show()
    plt.show()
