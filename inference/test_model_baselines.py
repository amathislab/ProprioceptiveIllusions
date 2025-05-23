"""
This file is used to test the model after training
"""

import argparse
import json
import pickle as pkl

from directory_paths import MODELS_DIR, PARENT_DIR, SAVE_DIR
from inference.test_model_utils_new import Tester, parse_config_value

# from inference.test_model_utils_new import Tester, parse_config_value
from model.model_definitions import SpatiotemporalNetwork, SpatiotemporalNetworkCausal
from train.new_spindle_dataset import SpindleDataset
from train.train_model_utils import *

# ------------------------------------------------------------------------------------------------------
# Change the constants here

# flag - since data is saved slightly differently, must specify whether the PCR data from the paper (True)
# or the generated data from this project (False) is being used
# PCR_DATA = True

# TIME = 320
TIME = 1152  # 320
SAMPLE_RATE = "240Hz"
# SAMPLE_RATE = "66_7Hz"
# for repeatable results
RAND = False
FLAG_SPLIT = 1342

CAUSAL = True  # use causal output layer
# change device
USE_GPU = True

# change number of samples generated
VISUALIZE_SAMPLES = 5
# if INPUT_DATA == "FLAG":
#     VISUALIZE_SAMPLES = 2

# if animation
ANIMATE = False


def test_model_config(test_config):
    task = test_config["TASK"]
    path_to_data = test_config["PATH_TO_DATA"]
    key = test_config["KEY"]
    input_data = test_config["input_data"]
    model_path = test_config["model_path"]
    sample_rate = test_config.get("sample_rate", SAMPLE_RATE)
    causal = test_config.get("CAUSAL", CAUSAL)
    PATH_TO_SAVE = model_path + "/" + "test" + "/" + input_data + "/" + sample_rate
    # if save path doesn't exist create it
    if not os.path.exists(PATH_TO_SAVE):
        os.makedirs(PATH_TO_SAVE)
    print("main -> created folder: ", PATH_TO_SAVE)

    # load net parameters from config.yaml file of experiment
    with open(os.path.join(model_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = {key: parse_config_value(value) for key, value in config.items()}

    layer_norm = config.get("layer_norm", test_config.get("layer_norm", False))
    if layer_norm == "False":
        layer_norm = False
    elif layer_norm == "True":
        layer_norm = True
    print(layer_norm)

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
        my_dir=os.path.join(SAVE_DIR, MODELS_DIR),
        layer_norm=layer_norm,
        training_seed=config.get("training_seed", None),
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
            # need_muscles=KEY == "spindle_FR",
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
    evaluation_results = mytester.evaluate_model(n_split=FLAG_SPLIT)
    print(f"model accuracy is {evaluation_results['overall_accuracy']}")
    print("main -> model evaluated")
    # save accuracy
    with open(os.path.join(PATH_TO_SAVE, "accuracy.txt"), "w") as f:
        f.write(path_to_data + "\n")
        # store evaluation_results dictionary
        for key, value in evaluation_results.items():
            f.write(f"{key}: {value}\n")

    # see some examples
    mytester.visualize_model(
        VISUALIZE_SAMPLES,
        PATH_TO_SAVE,
        animater=ANIMATE,
        pltshow=True,
        key=key,
        rand=RAND,
        input_data=input_data,
    )
    print("main -> model visualised")

    print("done!")


if __name__ == "__main__":

    datasets_to_test = ["FLAG_PCR", "ELBOW-FLEX", "ELBOW"]
    # datasets_to_test = ["ELBOW-FLEX", "ELBOW"]

    test_configs = []
    ######### TEST baselines ##########
    for input_data in datasets_to_test:
        ## Elife rep ##
        model_path = os.path.join(
            SAVE_DIR,
            # f"trained_models/experiment_causal_flag-pcr_optimized_linearFR_extended_5_5_30k_noLayerNorm_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_{seed}",
            f"trained_models/experiment_eLife_repletter_reconstruction/spatiotemporal_4_8-8-32-64_7272_0",
        )
        if input_data == "FLAG_PCR":
            # path_to_data = os.path.join(
            #     PARENT_DIR,
            #     "flag_pcr_test_dec_16_240Hz_cleaned_smoothed.hdf5",
            # )
            continue
        elif input_data == "ELBOW-FLEX":
            path_to_data = os.path.join(
                PARENT_DIR,
                "elbow_flexion/elbow_flex_20.hdf5",
            )
        elif input_data == "ELBOW":
            path_to_data = os.path.join(
                SAVE_DIR,
                "elbow_flex/elbow_flex_visualize_flat_100.hdf5",
            )
        test_configs.append(
            {
                "PATH_TO_DATA": path_to_data,
                "EXPERIMENT_ID": "eLife_repletter_reconstruction",
                "TASK": "letter_reconstruction",
                "KEY": "spindle_info",
                "RETRAIN": False,
                "input_data": input_data,
                "flag_split": None,
                "model_path": model_path,
                "CAUSAL": False,
                "sample_rate": "66_7Hz",
                "layer_norm": True,
                "spindle_dataset": False,
                "training_seed": None,
            }
        )
        # experiment_eLife_rep_pcr_elbowAngles_stride1_240Hz_letter_reconstruction_joints
        model_path = os.path.join(
            SAVE_DIR,
            # f"trained_models/experiment_causal_flag-pcr_optimized_linearFR_extended_5_5_30k_noLayerNorm_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_{seed}",
            f"trained_models/experiment_eLife_rep_pcr_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_0",
        )
        if input_data == "FLAG_PCR":
            path_to_data = os.path.join(
                PARENT_DIR,
                "flag_pcr_test_dec_16_240Hz_cleaned_smoothed.hdf5",
            )
        elif input_data == "ELBOW-FLEX":
            path_to_data = os.path.join(
                PARENT_DIR,
                "elbow_flexion/elbow_flex_20_smoothed_cleaned.hdf5",
            )
        elif input_data == "ELBOW":
            path_to_data = os.path.join(
                SAVE_DIR,
                "elbow_flex/elbow_flex_visualize_flat_100_240Hz.hdf5",
            )
        test_configs.append(
            {
                "PATH_TO_DATA": path_to_data,
                "EXPERIMENT_ID": "eLife_rep_pcr_elbowAngles_stride1_240Hz_letter_reconstruction_joints",
                "TASK": "letter_reconstruction_joints",
                "KEY": "spindle_info",
                "RETRAIN": False,
                "input_data": input_data,
                "flag_split": 1342,
                "model_path": model_path,
                "CAUSAL": False,
                "layer_norm": True,
                "spindle_dataset": False,
                "training_seed": None,
            }
        )
        # experiment_eLife_rep_flag-pcr_l-v_elbowAngles_stride1_240Hz_letter_reconstruction_joints
        model_path = os.path.join(
            SAVE_DIR,
            f"trained_models/experiment_eLife_rep_flag-pcr_l-v_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_0",
        )
        if input_data == "FLAG_PCR":
            path_to_data = os.path.join(
                PARENT_DIR,
                "flag_pcr_test_dec_16_240Hz_cleaned_smoothed.hdf5",
            )
        elif input_data == "ELBOW-FLEX":
            path_to_data = os.path.join(
                PARENT_DIR,
                "elbow_flexion/elbow_flex_20_smoothed_cleaned.hdf5",
            )
        elif input_data == "ELBOW":
            path_to_data = os.path.join(
                SAVE_DIR,
                "elbow_flex/elbow_flex_visualize_flat_100_240Hz.hdf5",
            )
        test_configs.append(
            {
                "PATH_TO_DATA": path_to_data,
                "EXPERIMENT_ID": "eLife_rep_pcr_elbowAngles_stride1_240Hz_letter_reconstruction_joints",
                "TASK": "letter_reconstruction_joints",
                "KEY": "spindle_info",
                "RETRAIN": False,
                "input_data": input_data,
                "flag_split": 1342,
                "model_path": model_path,
                "CAUSAL": False,
                "layer_norm": True,
                "spindle_dataset": False,
                "training_seed": None,
            }
        )
        ### Spindle inputs #####
        model_path = os.path.join(
            SAVE_DIR,
            f"trained_models/experiment_causal_flag-pcr_optimizedLinearFR_30k_noLayerNorm_elbowAngles_stride1_240Hz_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_0",
        )
        if input_data == "FLAG_PCR":
            path_to_data = os.path.join(
                SAVE_DIR,
                "optimized_linear_flag_pcr_test_dec_16_240Hz_cleaned_smoothed.hdf5",
            )
        elif input_data == "ELBOW-FLEX":
            path_to_data = os.path.join(
                SAVE_DIR,
                "optimized_linear_elbow_flex_20_smoothed_cleaned.hdf5",
            )
        elif input_data == "ELBOW":
            path_to_data = os.path.join(
                SAVE_DIR,
                "optimized_linear_elbow_flex_visualize_flat_100_240Hz.hdf5",
            )
        test_configs.append(
            {
                # "PATH_TO_DATA": "/media/data4/adriana/ProprioPerception/optimized_linear_flag_pcr_test_dec_16_240Hz_cleaned_smoothed.hdf5",
                "PATH_TO_DATA": path_to_data,
                "EXPERIMENT_ID": "causal_flag-pcr_optimizedLinearFR_30k_noLayerNorm_elbowAngles_stride1_240Hz_",
                "TASK": "letter_reconstruction_joints",
                "KEY": "spindle_info",
                "RETRAIN": False,
                "input_data": input_data,
                "flag_split": FLAG_SPLIT,
                "model_path": model_path,
                "CAUSAL": True,
                "input_data": input_data,
                "layer_norm": False,
                "training_seed": None,
            }
        )
    print(len(test_configs))
    for test_config in test_configs:
        test_model_config(test_config)
