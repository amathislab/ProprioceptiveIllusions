import argparse
import os

import torch
import yaml

from directory_paths import MODELS_DIR, SAVE_DIR
from inference.test_model_utils_new import (
    evaluate_model_with_vibrations,
    load_model,
    parse_config_value,
)
from train.new_spindle_dataset import SpindleDataset
from utils.muscle_names import MUSCLE_NAMES
from utils.spindle_FR_helper import (
    plot_inputs_and_elbow_angles,
    plot_vibration_analysis,
)

ELBOW_ANGLE_INDEX = 6
KEY = "spindle_info"

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "trained_models/experiment_causal_flag-pcr_optimizedLinearFR_30k_noLayerNorm_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_0",
)
INPUT_DATA = "ELBOW"
SAMPLE_RATE = "240Hz"
DEFAULT_SAVE_PATH = (
    MODEL_PATH
    + "/"
    + "test"
    + "/"
    + INPUT_DATA
    + "/"
    + SAMPLE_RATE
    + "/"
    + "vib_multiple_freqs"
)
DEFAULT_PATH_TO_DATA = os.path.join(
    SAVE_DIR, "optimized_linear_elbow_flex_visualize_flat_100_240Hz.hdf5"
)
# current directory
DEFAULT_IA_SAMPLED_COEFF_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../extract_data/sampled_coefficients_i_a.csv",
)
DEFAULT_IA_COEFF_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../spindle_coefficients/i_a/linear/coefficients.csv",
)


F_MAX_MAX = 150
F_MAX_MIN = 100

## Max near actual max_rate ## for vib_vary_multipleFs_fixFmax-splitPerMuscle_2
f_max_max_dict = {
    "BIClong": 151,
    "BICshort": 101,
    "TRIlong": 101,
    "TRIlat": 101,
    "TRImed": 51,
}
f_max_min_dict = {
    "BIClong": 150,
    "BICshort": 100,
    "TRIlong": 100,
    "TRIlat": 100,
    "TRImed": 50,
}
NUM_SAVE_SAMPLES = 10

DEFAULT_MUSCLES_TO_VIB = ["BIClong", "BICshort"]
# DEFAULT_MUSCLES_TO_VIB = ["BICshort"]
# DEFAULT_MUSCLES_TO_VIB = ["BIClong"]
# DEFAULT_MUSCLES_TO_VIB = ["TRIlat", "TRIlong", "TRImed"]
# DEFAULT_MUSCLES_TO_VIB = ["TRIlat"]
# DEFAULT_MUSCLES_TO_VIB = ["TRIlat", "TRIlong"]

CAUSAL = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test model with vibrations.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_PATH_TO_DATA,
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULT_SAVE_PATH,
        help="Directory to save results.",
    )
    parser.add_argument(
        "--vib_freqs",
        type=int,
        nargs="+",
        default=[0, 20, 40, 60, 80, 100, 110, 130, 150, 170, 190],
        help="List of vibration frequencies.",
    )
    parser.add_argument(
        "--channel_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of input channels to vibrate (indices).",
    )
    parser.add_argument(
        "--vib_start", type=int, default=200, help="Start time of vibration."
    )
    parser.add_argument(
        "--vib_end", type=int, default=900, help="End time of vibration."
    )
    parser.add_argument(
        "--muscles_to_vib",
        type=str,
        nargs="+",
        default=DEFAULT_MUSCLES_TO_VIB,
        help="Muscles to vibrate.",
    )
    parser.add_argument(
        "--rand_max",
        action="store_true",
        default=False,
        help="Randomize max frequencies.",
    )
    parser.add_argument(
        "--f_max_max",
        type=int,
        default=None,
        help="Max of range for max harmonic freq.",
    )
    parser.add_argument(
        "--f_max_min",
        type=int,
        default=None,
        help="Min of range for max harmonic freq.",
    )
    parser.add_argument(
        "--i_a_sampled_coeff_path",
        type=str,
        default=DEFAULT_IA_SAMPLED_COEFF_PATH,
        help="Path to the i_a_sampled_coeff_path dataset.",
    )
    parser.add_argument(
        "--i_a_coeff_path",
        type=str,
        default=DEFAULT_IA_COEFF_PATH,
        help="Path to the i_a_coeff_path dataset.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=False,
        help="Save plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.muscles_to_vib) == 1:
        vib_type = args.muscles_to_vib[0]
    elif ("BIClong" in args.muscles_to_vib) or ("BICshort" in args.muscles_to_vib):
        vib_type = "biceps"
    elif ("TRIlong" in args.muscles_to_vib) or ("TRIlat" in args.muscles_to_vib):
        vib_type = "triceps"
    else:
        vib_type = "mixed"

    if args.rand_max:
        # single value for all muscles
        # breakpoint()
        if args.f_max_max is not None and args.f_max_min is not None:
            print("Using same range for all muscles", args.f_max_max, args.f_max_min)
            rand_max_dict = {
                "is": True,
                "f_max_max": args.f_max_max,
                "f_max_min": args.f_max_min,
            }
        else:
            # different range per muscle
            print("Using for each muscle" + str(f_max_max_dict))
            rand_max_dict = {
                "is": True,
                "f_max_max": [f_max_max_dict[muscle] for muscle in args.muscles_to_vib],
                "f_max_min": [f_max_min_dict[muscle] for muscle in args.muscles_to_vib],
            }
            print(rand_max_dict)
            # breakpoint()
    else:
        rand_max_dict = None

    # check that args.model_path exists
    if not os.path.exists(os.path.join(args.model_path, "config.yaml")):
        # raise ValueError(f"model path {args.model_path} does not exist")
        print(f"model path {args.model_path} does not exist")
        return
    # check data path exists
    if not os.path.exists(args.data_path):
        # raise ValueError(f"data path {args.data_path} does not exist")
        print(f"data path {args.data_path} does not exist")
        return

    # add to save path range of vib freq and num channels vibed
    if args.channel_indices is None:
        channel_indices = []
    else:
        channel_indices = args.channel_indices
    num_channels = len(channel_indices)
    if num_channels == 1:
        channel_str = f"chan{channel_indices[0]}"
    else:
        channel_str = f"{num_channels}chan-{channel_indices[0]}-{channel_indices[-1]}"
    save_path = os.path.join(
        args.save_path,
        f"vib_{args.vib_freqs[0]}-{args.vib_freqs[-1]}Hz_{channel_str}_{len(args.muscles_to_vib)}{vib_type}",
    )
    os.makedirs(save_path, exist_ok=True)

    vib_results_path = os.path.join(save_path, f"vib_results.h5")
    # do not run if vib_results.h5 already exists
    if os.path.exists(vib_results_path):
        print(f"vib_results.h5 already exists at {vib_results_path}")
        return

    print("Model path: ", args.model_path)
    train_seed = int(args.model_path[-1])
    # Load model configuration
    with open(os.path.join(args.model_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = {key: parse_config_value(value) for key, value in config.items()}

    # if config has training seed, use it
    if "training_seed" in config.keys():
        if config["training_seed"] != train_seed:
            config["training_seed"] = train_seed  # issues with some experiments
        print(
            "loading model with seed ",
            config["seed"],
            "training seed ",
            config["training_seed"],
        )
    else:
        print("loading model with seed ", config["seed"])
    # Load test data
    test_data = SpindleDataset(
        args.data_path,
        dataset_type="test",
        key=KEY,
        task=config["task"],
        need_muscles=False,
    )

    # Load model
    # base_path is the part of args.model_path before /trained_models
    base_path = args.model_path.split("/trained_models")[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tester = load_model(
        config,
        args.model_path,
        config["task"],
        device,
        test_data,
        CAUSAL,
        base_path,
        MODELS_DIR,
    )

    # Evaluate model with vibrations
    df = evaluate_model_with_vibrations(
        tester,
        args.vib_freqs,
        channel_indices,
        args.vib_start,
        args.vib_end,
        args.muscles_to_vib,
        save_path,
        config,
        rand_max=rand_max_dict,
        i_a_sampled_coeff_path=args.i_a_sampled_coeff_path,
        i_a_coeff_path=args.i_a_coeff_path,
        num_plot_trials=NUM_SAVE_SAMPLES,
        plot_individual=args.save_plots,
    )

    muscle_indices_vibrated = [
        MUSCLE_NAMES.index(muscle) for muscle in df.iloc[0]["vib_muscles"]
    ]
    muscle_idx_to_plot = muscle_indices_vibrated
    # channel_indices_to_plot = [0]
    channel_indices_to_plot = [len(channel_indices) - 1]
    if args.save_plots:
        plot_inputs_and_elbow_angles(
            df,
            muscle_idx_to_plot,
            channel_indices_to_plot,
            channel_indices,
            save_path=save_path,
        )
    plot_vibration_analysis(df, args.save_path)

    # plot_samples_from_results(df, args.save_path)


if __name__ == "__main__":
    main()
