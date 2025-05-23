"""
This file is used to test the model after training. It saves the performance on test datasets in txt file and plots some examples
"""

import argparse

from directory_paths import MODELS_DIR, SAVE_DIR
from inference.test_model_utils_new import Tester, parse_config_value
from model.model_definitions import SpatiotemporalNetwork, SpatiotemporalNetworkCausal
from train.new_spindle_dataset import SpindleDataset
from train.train_model_utils import *

# ------------------------------------------------------------------------------------------------------
# Change the constants here

# TIME = 320
TIME = 1152  # 320
SAMPLE_RATE = "240Hz"
# SAMPLE_RATE = "66_7Hz"
# for repeatable results
RAND = False
FLAG_SPLIT = 1342  # index of flag split in the flag-pcr test dataset

CAUSAL = True  # use causal output layer
# change device
USE_GPU = True

# change number of samples generated
VISUALIZE_SAMPLES = 5
# if INPUT_DATA == "FLAG":
#     VISUALIZE_SAMPLES = 2

# if animation
ANIMATE = False

DEFAULT_TRAINING_SEED = 9  # default seed is 9 for models trained wihtout seed


def load_base_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_model_config(test_config):
    task = test_config["TASK"]
    path_to_data = test_config["PATH_TO_DATA"]
    key = test_config["KEY"]
    input_data = test_config["input_data"]
    model_path = test_config["model_path"]
    sample_rate = test_config.get("sample_rate", SAMPLE_RATE)
    causal = test_config.get("CAUSAL", CAUSAL)
    # path_to_save = model_path + "/" + "test" + "/" + input_data + "/" + sample_rate
    path_to_save = model_path + "/" + "test" + "/" + input_data 

    flag_split = test_config.get("flag_split", FLAG_SPLIT)

    # if save path doesn't exist create it
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    print("main -> created folder: ", path_to_save)

    # check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"model path {model_path} does not exist")

    # load net parameters from config.yaml file of experiment
    with open(os.path.join(model_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = {key: parse_config_value(value) for key, value in config.items()}

    layer_norm = config.get("layer_norm", test_config.get("layer_norm", False))
    # make into boolean
    if isinstance(layer_norm, str):
        layer_norm = layer_norm.lower() == "true"

    if causal:
        net = SpatiotemporalNetworkCausal
    else:
        net = SpatiotemporalNetwork
    model = net(
        experiment_id=config["experiment_id"],
        nclasses=config["nclasses"],
        arch_type="spatiotemporal",
        nlayers=config["nlayers"],
        n_skernels=config["n_skernels"],
        n_tkernels=config["n_tkernels"],
        s_kernelsize=config["s_kernelsize"],
        t_kernelsize=config["t_kernelsize"],
        s_stride=config["s_stride"],
        t_stride=config["t_stride"],
        padding=config["padding"],
        input_shape=config["input_shape"],
        p_drop=config["p_drop"],
        seed=config["seed"],
        train=True,
        task=task,
        outtime=config["outtime"],
        my_dir=os.path.join(test_config.get("BASE_DIR", SAVE_DIR), MODELS_DIR),
        layer_norm=layer_norm,
        training_seed=config.get("training_seed", DEFAULT_TRAINING_SEED),
    )

    print("main -> model created")

    # load the testing data
    if test_config.get("spindle_dataset", True):  # spindle dataset
        test_data = SpindleDataset(
            path_to_data,
            dataset_type="test",
            key=key,
            task=task,
            aclass=None,
            need_muscles=False,
            new_size=config["input_shape"][-1],
        )
    else:
        test_data = Dataset(
            path_to_data,
            dataset_type="test",
            key=key,
            task=task,
            aclass=None,
            # need_muscles=KEY == "spindle_FR",
            need_muscles=False,
            new_size=config["input_shape"][-1],
        )
    print("main -> data loaded")

    # create tester
    mytester = Tester(
        model,
        test_data,
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu"
        ),
    )
    print("main -> tester created")
    # breakpoint()
    # load the parameters
    mytester.load()
    print("main -> model loaded")

    # find the accuracy
    evaluation_results = mytester.evaluate_model(n_split=flag_split)
    print(f"model accuracy is {evaluation_results['overall_accuracy']}")
    # save accuracy
    with open(os.path.join(path_to_save, "accuracy.txt"), "w") as f:
        f.write(path_to_data + "\n")
        # store evaluation_results dictionary
        for key, value in evaluation_results.items():
            f.write(f"{key}: {value}\n")

    # see some examples
    mytester.visualize_model(
        VISUALIZE_SAMPLES,
        path_to_save,
        animater=ANIMATE,
        pltshow=True,
        key=key,
        rand=RAND,
        input_data=input_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test configurations.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        default=[0],
        help="Seed for the configurations",
    )
    parser.add_argument(
        "--training_seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seed for the training",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to dir with data for training"
    )
    parser.add_argument("--n_aff", type=int, default=5, help="Number of afferents")
    parser.add_argument(
        "--base_config", type=str, required=True, help="Path to base YAML config"
    )
    args = parser.parse_args()

    base_config = load_base_config(args.base_config)

    # Extract command-line arguments
    seeds = args.seeds
    n_aff = args.n_aff
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

    datasets_to_test = base_config.get("datasets_to_test", ["FLAG_PCR", "EF3D", "ES3D"])
    model_path_prefix = base_config.get(
        "MODEL_PATH_PREFIX", "optimized_linear_extended"
    )
    task = base_config.get("task", "letter_reconstruction_joints")

    test_configs = []
    ##### Test spindle models with variable inputs ###########
    for input_data in datasets_to_test:
        for seed in seeds:
            for train_seed in training_seeds:
                config = copy.deepcopy(base_config)
                if input_data == "FLAG_PCR":
                    path_to_data = f"{data_dir}/{model_path_prefix}_{seed}_{n_aff}_{n_aff}_flag_pcr_test.hdf5"
                elif input_data == "EF3D":
                    path_to_data = f"{data_dir}/{model_path_prefix}_{seed}_{n_aff}_{n_aff}_EF3D.hdf5"
                elif input_data == "ES3D":
                    path_to_data = f"{data_dir}/{model_path_prefix}_{seed}_{n_aff}_{n_aff}_ES3D.hdf5"
                model_path = os.path.join(
                    base_dir,
                    f"trained_models/experiment_causal_flag-pcr_{model_path_prefix}_{n_aff}_{n_aff}_{task}",
                    f"spatiotemporal_4_{seed}_{train_seed}",
                )
                config["PATH_TO_DATA"] = path_to_data
                config["BASE_DIR"] = base_dir
                # config["EXPERIMENT_ID"] = (
                #     f"causal_flag-pcr_{model_path_prefix}_{n_aff}_{n_aff}_{task}"
                # )
                config["input_data"] = input_data
                config["model_path"] = model_path
                test_configs.append(config)
    print("Testing ", len(test_configs), "models-datasets combinations")
    for test_config in test_configs:
        test_model_config(test_config)
