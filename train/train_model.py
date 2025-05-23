"""
This file is used to train and save the models
"""

import argparse

from directory_paths import MODELS_DIR, SAVE_DIR
from model.model_definitions import SpatiotemporalNetwork, SpatiotemporalNetworkCausal
from train.new_spindle_dataset import SpindleDataset
from train.train_model_utils import *

# ------------------------------------------------------------------------------------------------------
# Defaults here, set the values desired for this training in the main function

USE_GPU = True
TIME = 1152  # 320
# TIME = 320
NUM_MUSCLES = 25

# N_SKERNELS=[3, 16, 32, 64, 128]
DEFAULT_N_LAYERS = 4
DEFAULT_N_SKERNELS = [8, 8, 32, 64]
DEFAULT_N_TKERNELS = [8, 8, 32, 64]
DEFAULT_S_KERNELSIZE = 7
DEFAULT_T_KERNELSIZE = 7
DEFAULT_S_STRIDE = 1  # 2
DEFAULT_T_STRIDE = 1  # 2
DEFAULT_PADDING = 3  # for conv layer
DEFAULT_INPUT_SHAPE = [10, NUM_MUSCLES, TIME]  # num_input_channels x muscles x time
DEFAULT_P_DROP = 0.7  # for letter recognition
DEFAULT_SEED = 0

DEFAULT_NUM_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 128  # 256
DEFAULT_EARLY_STOP_MIN_EPOCH = 40
DEFAULT_EARLY_STOPPING_EPOCHS = 5
DEFAULT_END_TRAINING_NUM = 2
DEFAULT_LEARNING_RATE = 0.0005

FRACTION = None  # for Dataset if only train with subset
CAUSAL = True  # use causal output layer
DEFAULT_SEEDS = [0]
DEFAULT_N_AFF = 5

# ------------------------------------------------------------------------------------------------------


def train_with_config(config):
    input_shape = config.get("input_shape", DEFAULT_INPUT_SHAPE)
    if config["KEY"] == "spindle_info":
        input_shape = [2, NUM_MUSCLES, TIME]

    # build appropriate model
    if config["TASK"] == "letter_recognition":
        nclasses = 20
        normalize = False
    elif config["TASK"] == "letter_reconstruction":
        nclasses = 3
        normalize = True
    elif config["TASK"] == "letter_reconstruction_joints":  # 3end effector + joints
        nclasses = 7
        normalize = True
        # normalize = False
    elif config["TASK"] == "letter_reconstruction_joints_vel":  # 3end effector + joints
        nclasses = 14
        normalize = True

    net = (
        SpatiotemporalNetworkCausal
        if config.get("CAUSAL", True)
        else SpatiotemporalNetwork
    )

    param_seed = config.get("seed", DEFAULT_SEED)
    training_seed = config.get("training_seed", config.get("seed", DEFAULT_SEED))

    print(
        "creating model with seed",
        param_seed,
        "training seed ",
        training_seed,
    )
    model = net(
        experiment_id=config["EXPERIMENT_ID"] + config["TASK"],
        nclasses=nclasses,
        arch_type="spatiotemporal",
        nlayers=config.get("n_skernels", DEFAULT_N_LAYERS),
        n_skernels=config.get("n_skernels", DEFAULT_N_SKERNELS),
        n_tkernels=config.get("n_tkernels", DEFAULT_N_TKERNELS),
        s_kernelsize=config.get("s_kernelsize", DEFAULT_S_KERNELSIZE),
        t_kernelsize=config.get("t_kernelsize", DEFAULT_T_KERNELSIZE),
        s_stride=config.get("s_stride", DEFAULT_S_STRIDE),
        t_stride=config.get("t_stride", DEFAULT_T_STRIDE),
        padding=config.get("padding", DEFAULT_PADDING),
        input_shape=input_shape,
        p_drop=config.get("p_drop", DEFAULT_P_DROP),
        seed=param_seed,
        train=True,
        task=config["TASK"],
        outtime=TIME,
        my_dir=os.path.join(config.get("BASE_DIR", SAVE_DIR), MODELS_DIR),
        build_fc=config.get("build_fc", False),
        layer_norm=config.get("layer_norm", False),
        training_seed=training_seed,
    )

    print("main -> models created")

    # load the training data - check wether to use dataset or Spindledataset
    if config.get("spindle_dataset", True):  # spindle dataset
        train_data = SpindleDataset(
            config["PATH_TO_DATA"],
            dataset_type="train",
            key=config["KEY"],
            task=config["TASK"],
            aclass=None,
            new_size=TIME,
            start_end_idx=config["START_END_IDX"],
        )
    else:  # for non spindle inputs
        train_data = Dataset(
            config["PATH_TO_DATA"],
            dataset_type="train",
            key=config["KEY"],
            task=config["TASK"],
            aclass=None,
            new_size=TIME,
            fraction=FRACTION,
        )
    print("Data loaded.")

    # Create trainer
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and config.get("USE_GPU", True) else "cpu"
    )
    mytrainer = Trainer(model, train_data, device=device)
    print("Trainer created.")

    # Train model
    mytrainer.train(
        num_epochs=config.get("NUM_EPOCHS", DEFAULT_NUM_EPOCHS),
        batch_size=config.get("BATCH_SIZE", DEFAULT_BATCH_SIZE),
        early_stop_min_epoch=config.get(
            "EARLY_STOP_MIN_EPOCH", DEFAULT_EARLY_STOP_MIN_EPOCH
        ),
        early_stopping_epochs=config.get(
            "EARLY_STOPPING_EPOCHS", DEFAULT_EARLY_STOPPING_EPOCHS
        ),
        end_training_num=config.get("END_TRAINING_NUM", DEFAULT_END_TRAINING_NUM),
        learning_rate=config.get("LEARNING_RATE", DEFAULT_LEARNING_RATE),
        normalize=normalize,
        retrain=config.get("RETRAIN", False),
    )
    print(f"Training complete for configuration: {config['EXPERIMENT_ID']}")


def load_base_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with dynamic configs.")
    parser.add_argument(
        "--base_config", type=str, required=True, help="Path to base YAML config"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to dir with data for training"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="List of seeds"
    )
    parser.add_argument(
        "--training_seeds", type=int, nargs="+", default=[0], help="Training seeds"
    )
    parser.add_argument("--n_aff", type=int, default=5, help="Number of afferents")
    args = parser.parse_args()

    base_config = load_base_config(args.base_config)

    n_aff = args.n_aff
    seeds = args.seeds
    training_seeds = args.training_seeds
    
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = SAVE_DIR
    # base_dir is data_dir minus /data if /data is in the path
    if "/data" in data_dir:
        # find the last /data in the path
        last_data_idx = data_dir.rfind("/data")
        # split the path at that index
        base_dir = data_dir[:last_data_idx]
    else:
        base_dir = data_dir

    print(f"Running trainings for {len(seeds)} seeds and {n_aff} afferents.")

    for seed in seeds:
        for training_seed in training_seeds:
            config = copy.deepcopy(
                base_config
            )  # important to copy so you don't modify the original

            # Modify fields that depend on seed, training_seed, n_aff
            config["seed"] = seed
            config["training_seed"] = training_seed
            if config.get("PATH_TO_DATA") is None:
                data_path_prefix = config.get(
                    "DATA_PATH_PREFIX", "optimized_linear_extended"
                )
                config["PATH_TO_DATA"] = (
                    f"{data_dir}/{data_path_prefix}_{seed}_{n_aff}_{n_aff}_flag_pcr_training.hdf5"
                )
                config["BASE_DIR"] = base_dir
                if config.get("EXPERIMENT_ID") is None:
                    config["EXPERIMENT_ID"] = (
                        f"causal_flag-pcr_{data_path_prefix}_{n_aff}_{n_aff}_"
                    )
            config["input_shape"] = [
                n_aff + n_aff,
                NUM_MUSCLES,
                TIME,
            ]  # adjust if needed

            # Call training function
            train_with_config(config)
