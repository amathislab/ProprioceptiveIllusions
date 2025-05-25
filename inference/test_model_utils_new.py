"""
This file contains all the functions needed to test the model
"""

import os

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import savgol_filter as sf

from directory_paths import MODELS_DIR, SAVE_DIR
from model.model_definitions import (
    LinearModel,
    SpatiotemporalNetwork,
    SpatiotemporalNetworkCausal,
)
from train.new_spindle_dataset import SpindleDataset
from utils.muscle_names import MUSCLE_NAMES
from utils.spindle_FR_helper import (
    compute_vib_metrics,
    generate_vib_config,
    generate_vib_config_multiple,
    plot_vibration_metrics,
)
from utils.visualize_sample import plot_outputs, plot_wristpred, plotter

ELBOW_ANGLE_INDEX = 6


LETTER_DICT = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "g",
    6: "h",
    7: "l",
    8: "m",
    9: "n",
    10: "o",
    11: "p",
    12: "q",
    13: "r",
    14: "s",
    15: "u",
    16: "v",
    17: "w",
    18: "y",
    19: "z",
}


class Tester:
    """Tests the model"""

    def __init__(self, model, dataset, device, batch_size=128):
        """
        Creates the model

        Arguments
            model : the `Conv`, `Affine` or `Recurrent` model to be evaluated. The test data is
            dataset : the `Dataset` object on which the model is to be evaluated.
            device : {'cpu', 'cuda:0'}, device on which the model is stored
            batch_size : int, batch size to test the model
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        #####################################################################################
        # delete this later
        # these are the values for the model as to not have to open the yaml file everytime
        # self.train_data_mean = 54.39655685424805
        # self.train_data_std = 1380.6353759765625
        # self.label_mean = 2.1344239711761475 # 25.170886993408203
        # self.label_max = 55.039005279541016 # 142.98316955566406
        # return
        #####################################################################################

        # Retrieve training mean, if data was normalized
        path_to_config_file = os.path.join(self.model.model_path, "config.yaml")
        with open(path_to_config_file, "r") as myfile:
            # model_config = yaml.load(myfile)
            config = yaml.full_load(myfile)

        self.train_data_mean = torch.tensor(config.get("train_mean", 0), device=device)
        self.train_data_std = torch.tensor(config.get("train_std", 1), device=device)
        self.label_mean = torch.tensor(config.get("label_mean", 0), device=device)
        self.label_max = torch.tensor(config.get("label_std", 1), device=device)

    def load(self):
        """
        loads the model parameters into self.model for inference.
        make sure to have initialized self.model
        """
        print("loading model from ", os.path.join(self.model.model_path, "model.ckpt"))
        # breakpoint()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model.model_path, "model.ckpt"), map_location=device
            )
        )
        self.model.eval()

    # def vibration_transfer_function(self, vib_freq, f_max):
    #     """
    #     Computes the effective vibration frequency based on the vibration frequency and max allowed frequency.

    #     Arguments:
    #         vib_freq: float, input vibration frequency.
    #         f_max: float, maximum allowable frequency.

    #     Returns:
    #         float: Spindle response to vibration freq
    #     """
    #     return max(0, min(vib_freq, 2 * f_max - vib_freq))

    def vibration_transfer_function(self, vib_freq, f_max_expanded):
        """
        Computes the effective frequency based on input vibration frequency and maximum frequency.

        Arguments:
            vib_freq: torch.Tensor, vibration frequency (can be scalar or tensor).
            f_max: torch.Tensor, maximum frequency (can be scalar or tensor).

        Returns:
            torch.Tensor: Effective frequency.
        """
        # Ensure f_max_expanded is a tensor
        if not isinstance(f_max_expanded, torch.Tensor):
            f_max_expanded = torch.tensor(
                f_max_expanded, dtype=vib_freq.dtype, device=vib_freq.device
            )

        # Ensure compatibility for min and max
        min_tensor = torch.zeros_like(vib_freq)  # Create tensor for lower bound
        # max_tensor = 2 * f_max_expanded - vib_freq  # Compute tensor for upper bound
        max_tensor = torch.clamp(2 * f_max_expanded - vib_freq, min=0)

        # Use torch.clamp with tensors
        eff_freq = torch.clamp(vib_freq, min=min_tensor, max=max_tensor)
        # check that no values are negative
        if torch.any(eff_freq < 0):
            raise ValueError("Effective frequency is negative")
        return eff_freq

    def add_vibrations(self, inputs, frequency=100.0, aff_info=None):
        """
        Perturbs inputs to simulate vibrations directly using effective frequency.

        Arguments:
            inputs: torch.Tensor, shape (batch_size, num_channels, num_muscles, time).
            frequency: float, base vibration frequency.
            aff_info: dict, contains:
                - "indices_filter": torch.Tensor, binary mask (num_channels, num_muscles, time).
                - "f_max": torch.Tensor, maximum frequency per channel and muscle (num_channels, num_muscles).

        Returns:
            torch.Tensor: Perturbed inputs with vibrations added.
        """
        # Extract the vibration filter and max frequencies
        vib_filter = aff_info[
            "indices_filter"
        ]  # Shape: (num_channels, num_muscles, time)
        f_max = aff_info["f_max"]  # Shape: (num_channels, num_muscles)

        # Ensure compatibility with inputs
        batch_size, num_channels, num_muscles, time_steps = inputs.shape
        assert vib_filter.shape == (num_channels, num_muscles, time_steps)
        assert f_max.shape == (num_channels, num_muscles)

        # Clone inputs to avoid in-place modification
        perturbed_inputs = inputs.clone()

        # Add vibrations
        # for channel in range(num_channels):
        #     for muscle in range(num_muscles):
        #         if torch.sum(vib_filter[channel, muscle]) > 0:  # Check if affected
        #             # Get maximum frequency for this channel and muscle
        #             max_freq = f_max[channel, muscle].item()

        #             # # Compute effective vibration frequency
        #             # eff_freq = self.vibration_transfer_function(frequency, max_freq)
        #             # # Update perturbed inputs: eff_freq * vib_filter when vib_filter > 0, else keep original inputs
        #             # ### set to only vib freq for each trial ###
        #             # if frequency > 0:
        #             #     perturbed_inputs[:, channel, muscle, :] = eff_freq * vib_filter[
        #             #         channel, muscle
        #             #     ] + inputs[:, channel, muscle, :] * (
        #             #         1 - vib_filter[channel, muscle]
        #             #     )
        #             #     ### sum response from vib freq and non vib ####
        #             #     # perturbed_inputs[trial, channel, muscle, :] += (
        #             #     #     eff_freq * vib_filter[channel, muscle]
        #             #     # )
        #             ### pass the freq of vib + FR with no vib for each trial ###
        #             for trial in range(batch_size):
        #                 input_at_freq = inputs[trial, channel, muscle, 0]
        #                 eff_freq = self.vibration_transfer_function(
        #                     frequency + input_at_freq, max_freq
        #                 )
        #                 if frequency > 0:
        #                     perturbed_inputs[
        #                         trial, channel, muscle, :
        #                     ] = eff_freq * vib_filter[channel, muscle] + inputs[
        #                         trial, channel, muscle, :
        #                     ] * (
        #                         1 - vib_filter[channel, muscle]
        #                     )
        ### use tensors for efficiency ###
        # Create a base frequency tensor (shape: num_channels, num_muscles, time_steps, batch_size)
        base_freq = torch.full(
            (num_channels, num_muscles, time_steps, batch_size),
            frequency,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        # Expand f_max to match the desired shape
        f_max_expanded = f_max.unsqueeze(-1).unsqueeze(
            -1
        )  # Shape: (num_channels, num_muscles, 1, 1)
        f_max_expanded = f_max_expanded.expand(
            num_channels, num_muscles, time_steps, batch_size
        )  # Shape: (num_channels, num_muscles, time_steps, batch_size)

        # Broadcast inputs for vibration computation
        vib_inputs = inputs.permute(
            1, 2, 3, 0
        )  # Shape: (num_channels, num_muscles, time_steps, batch_size)
        if vib_inputs.shape != base_freq.shape:
            raise ValueError(
                f"Shape mismatch: vib_inputs {vib_inputs.shape} vs base_freq {base_freq.shape}"
            )

        # Compute the effective frequency for all timesteps
        eff_freq = self.vibration_transfer_function(
            base_freq + vib_inputs, f_max_expanded
        )

        # Expand vib_filter to match the batch size and num_muscles
        vib_filter = vib_filter.unsqueeze(
            0
        )  # Shape: [1, num_channels, num_muscles, time_steps]
        vib_filter = vib_filter.expand(
            batch_size, num_channels, num_muscles, time_steps
        )  # Shape: [batch_size, num_channels, num_muscles, time_steps]

        # Ensure the eff_freq and vib_filter shapes are compatible with perturbed_inputs
        if eff_freq.shape != perturbed_inputs.shape:
            # eff_freq has the shape (num_channels, num_muscles, time_steps, batch_size)
            # perturbed_inputs has the shape (batch_size, num_channels, num_muscles, time_steps)
            # We need to permute eff_freq so that it matches the shape of perturbed_inputs
            eff_freq = eff_freq.permute(
                3, 0, 1, 2
            )  # Shape: (batch_size, num_channels, num_muscles, time_steps)

        # Now, apply the vibration only where vib_filter is non-zero
        perturbed_inputs = torch.where(
            vib_filter > 0, eff_freq * vib_filter, perturbed_inputs
        )
        # check that no values are negative
        if torch.any(perturbed_inputs < 0):
            raise ValueError("Effective frequency is negative")

        return perturbed_inputs

    def _add_vib(self, batch_X, vib_config):
        """Add vibrations specified in vib_config to batch_X

        Args:
            batch_X (torch.Tensor): unperturbed inputs
            vib_config (list of dict or dict): info on the vibrations to apply.
                assumes the vibrations are not overlapping in both time and muscles

        Returns:
            torch.Tensor: perturbed inputs
        """
        if type(vib_config) == list:
            for i in range(len(vib_config)):
                freq = vib_config[i]["vib_freq"]
                batch_X = self.add_vibrations(
                    batch_X, frequency=freq, aff_info=vib_config[i]
                )
            return batch_X
        else:
            freq = vib_config["vib_freq"]
            return self.add_vibrations(batch_X, frequency=freq, aff_info=vib_config)

    def _fetch_batch(self, step):
        """Fetch a batch of data."""
        # if key == "spindle_FR":
        #     muscles, muscles_normed, batch_X, batch_y = self.dataset.next_valbatch(
        #         self.batch_size, "test", step=step, flag=True
        #     )
        #     return batch_X, batch_y
        return self.dataset.next_valbatch(self.batch_size, "test", step=step)

    def _normalize_batch(self, batch_X, batch_y, epsilon=1e-8):
        """Normalize input and output data."""
        if type(self.train_data_mean) == torch.Tensor:
            batch_X_s = (
                batch_X.to(self.device) - self.train_data_mean.to(self.device)
            ) / (self.train_data_std.to(self.device) + epsilon)
        else:
            batch_X_s = (batch_X.to(self.device) - self.train_data_mean) / (
                self.train_data_std + epsilon
            )
        return batch_X_s, batch_y.to(self.device)

    def _compute_scores(self, batch_X_s):
        """Perform a forward pass through the model and unnormalize scores."""
        scores, prob, _ = self.model(batch_X_s)
        if type(self.label_max) == torch.Tensor:
            scores = scores.to(self.device) * self.label_max.to(
                self.device
            ) + self.label_mean.to(self.device)
        else:
            scores = scores.to(self.device) * self.label_max + self.label_mean
        return scores, prob

    def _evaluate_task(self, scores, prob, batch_y, elbow_scale):
        """Evaluate the model's performance based on the task."""
        wrist_xyz_loss, elbow_angle_loss = None, None

        if self.model.task == "letter_recognition":
            # accuracy = torch.mean(
            #     (batch_y == torch.argmax(prob, dim=1).squeeze(0)).type(
            #         torch.FloatTensor
            #     )
            # ).item()
            accuracy = (batch_y == torch.argmax(prob, dim=1)).float().mean().item()
            return accuracy, None

        if self.model.task in [
            "letter_reconstruction",
            "letter_reconstruction_joints",
            "letter_reconstruction_joints_vel",
        ]:
            wrist_xyz_loss = (
                (scores[:, :, :3] - batch_y[:, :, :3])
                .reshape(-1, 3)
                .norm(dim=1)
                .mean()
                .item()
            )
            if (
                self.model.task == "letter_reconstruction_joints"
                or self.model.task == "letter_reconstruction_joints_vel"
            ):
                elbow_angle_loss = (
                    (scores[:, :, 6] - batch_y[:, :, 6]).pow(2).mean().sqrt().item()
                )

        elif self.model.task in ["elbow_flex", "elbow_flex_joints"]:
            wrist_xyz_loss = self._evaluate_elbow_task(scores, batch_y, elbow_scale)

        else:
            raise NotImplementedError(f"Task {self.model.task} not supported.")

        return wrist_xyz_loss, elbow_angle_loss

    def _evaluate_elbow_task(self, scores, batch_y, elbow_scale):
        """Evaluate elbow flexion task."""
        scores = scores.cpu().detach().numpy()
        batch_y = self._adjust_ground_truth(batch_y, scores, elbow_scale)
        if self.model.task == "elbow_flex_joints":
            scores = scores[:, :, :3]
        wrist_xyz_loss = torch.norm(
            torch.subtract(
                torch.from_numpy(scores).reshape(-1, 3),
                torch.from_numpy(batch_y).reshape(-1, 3),
            ),
            dim=1,
        ).mean()
        return wrist_xyz_loss.item()

    def _adjust_ground_truth(self, batch_y, scores, elbow_scale):
        """Adjust ground truth for elbow flexion tasks."""
        # Move to CPU, detach from computation graph, and convert to NumPy
        batch_y = batch_y.cpu().detach().numpy() * elbow_scale

        # Reorder channels to [Z, X, Y]
        batch_y = batch_y[:, [2, 0, 1], :]

        # Compute the translation offset and align the ground truth with predictions
        translate = batch_y[0] - scores[:, :3, 0]
        batch_y -= translate[None, :, None]  # Adjust all time steps and batch samples
        # batch_y = batch_y.permute(1, 2, 0).cpu().detach().numpy() * elbow_scale
        # batch_y = np.transpose(
        #     np.concatenate(
        #         ([batch_y[:, 2, :]], [batch_y[:, 0, :]], [batch_y[:, 1, :]])
        #     ),
        #     (1, 0, 2),
        # )
        # translate = batch_y[0, :, :] - scores.transpose((1, 2, 0))[0, :3, :]
        # batch_y[:, :, :] -= translate
        # batch_y.transpose((2, 0, 1))
        return batch_y

    def _update_metrics(
        self,
        wrist_xyz_loss,
        elbow_angle_loss,
        overall_test_acc,
        wrist_xyz_losses,
        elbow_angle_losses,
        split_accuracies,
        split_angle_losses,
        step,
        n_split,
    ):
        """Update metrics based on the current batch."""
        overall_test_acc.append(wrist_xyz_loss)
        wrist_xyz_losses.append(wrist_xyz_loss)

        if elbow_angle_loss is not None:
            elbow_angle_losses.append(elbow_angle_loss)

        if step * self.batch_size < n_split:
            split_accuracies["first_n"].append(wrist_xyz_loss)
            if elbow_angle_loss is not None:
                split_angle_losses["first_n"].append(elbow_angle_loss)
        else:
            split_accuracies["rest"].append(wrist_xyz_loss)
            if elbow_angle_loss is not None:
                split_angle_losses["rest"].append(elbow_angle_loss)

    def _aggregate_results(
        self,
        overall_test_acc,
        wrist_xyz_losses,
        elbow_angle_losses,
        split_accuracies,
        split_angle_losses,
    ):
        """Aggregate metrics into a results dictionary."""
        return {
            "overall_accuracy": np.mean(overall_test_acc),
            "split_accuracy_first_n": (
                np.mean(split_accuracies["first_n"])
                if split_accuracies["first_n"]
                else None
            ),
            "split_accuracy_rest": (
                np.mean(split_accuracies["rest"]) if split_accuracies["rest"] else None
            ),
            "split_elbow_first_n": (
                np.mean(split_angle_losses["first_n"])
                if split_angle_losses["first_n"]
                else None
            ),
            "split_elbow_rest": (
                np.mean(split_angle_losses["rest"])
                if split_angle_losses["rest"]
                else None
            ),
            "elbow_angle_loss": (
                np.mean(elbow_angle_losses) if elbow_angle_losses else None
            ),
            "wrist_xyz_loss": np.mean(wrist_xyz_losses) if wrist_xyz_losses else None,
        }

    def _print_results(self, results, n_split):
        """Print evaluation results."""
        print(f"Overall Test Accuracy: {results['overall_accuracy']:.4f}")
        if results["split_accuracy_first_n"] is not None:
            print(
                f"First {n_split} Samples Accuracy: {results['split_accuracy_first_n']:.4f}"
            )
        if results["split_accuracy_rest"] is not None:
            print(f"Rest Samples Accuracy: {results['split_accuracy_rest']:.4f}")
        if results["split_elbow_first_n"] is not None:
            print(
                f"First {n_split} Samples Elbow Angle RMSE: {results['split_elbow_first_n']:.4f}"
            )
        if results["split_elbow_rest"] is not None:
            print(f"Rest Samples Elbow Angle RMSE: {results['split_elbow_rest']:.4f}")
        if results["elbow_angle_loss"] is not None:
            print(f"Elbow Angle RMSE: {results['elbow_angle_loss']:.4f}")
        if results["wrist_xyz_loss"] is not None:
            print(f"Wrist XYZ RMSE: {results['wrist_xyz_loss']:.4f}")

    def get_scores_probabilities(
        self, elbow_scale=80, vib_config=None, key="spindle_info", num_iter=None
    ):
        """
        Get the scores and probabilities of the model.

        Args:
            elbow_scale (float): Scale factor for ground truth in elbow flexion experiments.
            vib_config (dict): Configuration for vibrations, defaults to None.
            key (str): Specifies the type of data being evaluated (e.g., "spindle_info").

        Returns:
            tuple: Scores and probabilities of the model.
        """
        if num_iter is None:
            num_iter = max(self.dataset.test_data.shape[0] // self.batch_size, 1)

        scores = []
        batch_X_s = []
        batch_y_s = []

        for i in range(num_iter):
            # Fetch a batch of data
            batch_X, batch_y = self._fetch_batch(i)

            if vib_config is not None:
                batch_X = self._add_vib(batch_X, vib_config)

            # Normalize inputs
            batch_X_n, batch_y = self._normalize_batch(batch_X, batch_y)

            # Forward pass and compute scores
            s, p = self._compute_scores(batch_X_n)
            scores.append(s)
            batch_X_s.append(batch_X)
            batch_y_s.append(batch_y)
        return torch.cat(scores), torch.cat(batch_X_s), torch.cat(batch_y_s)

    def evaluate_model(
        self, elbow_scale=80, vib_config=None, key="spindle_info", n_split=1000
    ):
        """
        Evaluate the trained model.

        Args:
            elbow_scale (float): Scale factor for ground truth in elbow flexion experiments.
            vib_config (dict): Configuration for vibrations, defaults to None.
            key (str): Specifies the type of data being evaluated (e.g., "spindle_info").
            n_split (int): Number of initial samples for calculating split accuracy.

        Returns:
            dict: Evaluation metrics including accuracy and losses.
        """
        # if vib_config is None:
        #     vib_config = {"is": False}

        num_iter = max(self.dataset.test_data.shape[0] // self.batch_size, 1)

        overall_test_acc = []
        split_accuracies = {"first_n": [], "rest": []}
        elbow_angle_losses = []
        split_angle_losses = {"first_n": [], "rest": []}
        wrist_xyz_losses = []

        for i in range(num_iter):
            # Fetch a batch of data
            batch_X, batch_y = self._fetch_batch(i)

            if vib_config is not None:
                batch_X = self._add_vib(batch_X, vib_config)

            # Normalize inputs
            batch_X_s, batch_y = self._normalize_batch(batch_X, batch_y)

            # Forward pass and compute scores
            scores, prob = self._compute_scores(batch_X_s)

            # Task-specific evaluation
            wrist_xyz_loss, elbow_angle_loss = self._evaluate_task(
                scores, prob, batch_y, elbow_scale
            )

            # Update metrics
            self._update_metrics(
                wrist_xyz_loss,
                elbow_angle_loss,
                overall_test_acc,
                wrist_xyz_losses,
                elbow_angle_losses,
                split_accuracies,
                split_angle_losses,
                i,
                n_split,
            )

        # Aggregate and print results
        results = self._aggregate_results(
            overall_test_acc,
            wrist_xyz_losses,
            elbow_angle_losses,
            split_accuracies,
            split_angle_losses,
        )
        # self._print_results(results, n_split)

        return results

    def get_predictions(self):
        """
        Get the predictions of the model, along with the the X values.

        Returns:
            tuple: Input data (X) and the predictions.

        """
        num_iter = max(self.dataset.test_data.shape[0] // self.batch_size, 1)

        predictions = []
        batch_X_s = []
        batch_y_s = []

        for i in range(num_iter):
            print(f"Batch {i + 1}/{num_iter}")
            # Fetch a batch of data
            batch_X, batch_y = self._fetch_batch(i)

            # Normalize inputs
            batch_X_n, batch_y = self._normalize_batch(batch_X, batch_y)

            # Forward pass and compute
            s, p = self._compute_scores(batch_X_n)

            predictions.append(s)
            batch_X_s.append(batch_X)

        return torch.cat(predictions), torch.cat(batch_X_s)

    def evaluate_model_vibrations(
        self, elbow_scale=80, vib_config=None, key="spindle_info", n_split=1000
    ):
        """
        Evaluate the trained model with vibrations. Different quantities are saved

        Args:
            elbow_scale (float): Scale factor for ground truth in elbow flexion experiments.
            vib_config (dict): Configuration for vibrations, defaults to None.
            key (str): Specifies the type of data being evaluated (e.g., "spindle_info").
            n_split (int): Number of initial samples for calculating split accuracy.

        Returns:
            dict: Evaluation metrics including accuracy and losses.
        """
        if vib_config is None:
            print("No vibration configuration provided")
            return

        num_iter = max(self.dataset.test_data.shape[0] // self.batch_size, 1)

        overall_test_acc = []
        split_accuracies = {"first_n": [], "rest": []}
        elbow_angle_losses = []
        split_angle_losses = {"first_n": [], "rest": []}
        wrist_xyz_losses = []
        overall_test_acc = []
        split_accuracies = {"first_n": [], "rest": []}
        elbow_angle_losses = []
        split_angle_losses = {"first_n": [], "rest": []}
        wrist_xyz_losses = []

        for i in range(num_iter):
            # Fetch a batch of data
            batch_X, batch_y = self._fetch_batch(i)

            batch_X_vib = self._add_vib(batch_X, vib_config)

            # Normalize inputs
            batch_X_s, _ = self._normalize_batch(batch_X, batch_y)
            batch_X_s_vib, batch_y = self._normalize_batch(batch_X_vib, batch_y)

            # Forward pass and compute scores
            scores, prob = self._compute_scores(batch_X_s)
            scores_vib, prob = self._compute_scores(batch_X_s_vib)

            # Task-specific evaluation
            wrist_xyz_loss, elbow_angle_loss = self._evaluate_task(
                scores, prob, batch_y, elbow_scale
            )
            wrist_xyz_loss_vib, elbow_angle_loss_vib = self._evaluate_task(
                scores_vib, prob, batch_y, elbow_scale
            )

            # Update metrics
            self._update_metrics(
                wrist_xyz_loss,
                elbow_angle_loss,
                overall_test_acc,
                wrist_xyz_losses,
                elbow_angle_losses,
                split_accuracies,
                split_angle_losses,
                i,
                n_split,
            )

        # Aggregate and print results
        results = self._aggregate_results(
            overall_test_acc,
            wrist_xyz_losses,
            elbow_angle_losses,
            split_accuracies,
            split_angle_losses,
        )
        self._print_results(results, n_split)

        return results

    def visualize_model(
        self,
        num,
        path_to_save=None,
        vib_config=None,
        rand=True,
        pltshow=True,
        animater=False,
        key="spindle_info",
        input_data="ELBOW",
        indices_to_plot=None,
    ):
        """
        Generates examples to understand model output

        Arguments
            num: int, how many samples to generate
            path_to_save: string, where to save the visualizations
            elbow_scale: float to adjust scale of the ground truth in the elbow flexion experiment
            vib_config: {"is": {True, False},                                                                   whether to add vibrations
                        "length": {"only movement", "all"}                                                      where to add the vibrations
                        "VIB_FREQ": float,                                                                      frequency of vibrations
                        "VIB_AMP": float,                                                                       amplitude of vibrations
                        "MUSCLES_TO_VIB": list of strings of muscles or (["all"] for all muscles available),    which muscles to vibrate
                        "NUM_TIMEPOINTS": int}                                                                  time scale (320)
            rand: {True, False}, if to pick random samples or pick in order
            pltshow: {True, False}, wether to show the plots or not
        """
        ### not working ### to fix
        if pltshow:
            print("-----------------------------------")
            print("-----------------------------------")
            print("Some examples are being generated")
            print("-----------------------------------")

        # max number to pick from
        if self.batch_size > self.dataset.test_data.shape[0]:
            self.batch_size = self.dataset.test_data.shape[0]
        num_iter = self.dataset.test_data.shape[0] // self.batch_size

        # check num is smaller than num_iter - 1
        if num > num_iter:
            num = num_iter

        if indices_to_plot is None:
            if rand:
                # num random indeces to plot between 0 and num_iter
                indices_to_plot = np.random.randint(0, num_iter, num)
            else:
                # num sequential indeces to plot between 0 and num_iter
                indices_to_plot = np.arange(0, num)
        else:
            indices_to_plot = indices_to_plot[0:num]

        ## Florian ##
        # test_data_mean = torch.mean(self.dataset.test_data, dim=(0, 3)).reshape(1, 2, 25, 1)
        # test_data_std = torch.std(self.dataset.test_data, dim=(0, 3)).reshape(1, 2, 25, 1)

        # go through each example one at a time
        for i in range(num):
            index_to_plot = indices_to_plot[i]
            # Fetch a batch of data
            batch_X, batch_y = self._fetch_batch(index_to_plot)

            batch_X_s, batch_y = self._normalize_batch(batch_X, batch_y)

            scores, prob = self._compute_scores(batch_X_s)
            if vib_config is not None:
                batch_X_vib = self._add_vib(batch_X, vib_config)
                batch_X_vib_s, batch_y = self._normalize_batch(batch_X_vib, batch_y)
                # Forward pass and compute scores
                scores_vib, _ = self._compute_scores(batch_X_vib_s)

            # pick a random example from the batch
            idx = index_to_plot
            if path_to_save is not None:
                path_pref = f"{path_to_save}/{idx}_{input_data}"
            else:
                path_pref = None

            # visualize according to the task
            if self.model.task == "letter_recognition":

                # visualize meaning of outputs as letters printed to console
                true_letter = int(batch_y[idx].cpu().detach().numpy())
                prediction = torch.argmax(prob[idx]).squeeze(0)
                print(f"Given letter: \t\t{LETTER_DICT[int(true_letter)]}")
                print(f"Predicted letter: \t{LETTER_DICT[int(prediction)]}")
                print("-----------------------------------")

            elif self.model.task == "letter_reconstruction":

                # plot the letters in 3D
                true_letter = batch_y[idx].cpu().detach().numpy()
                prediction = scores[idx].cpu().detach().numpy()
                print("Not implemented yet")
                return
            elif (
                self.model.task == "letter_reconstruction_joints"
                or self.model.task == "letter_reconstruction_joints_vel"
            ):
                # plot in 3D
                true_letter = batch_y[idx].cpu().detach().numpy()
                prediction = scores[idx].cpu().detach().numpy()
                if vib_config is not None:
                    prediction_vib = scores_vib[idx].cpu().detach().numpy()
                    # plot_3D(
                    #     true_letter, prediction, prediction_vib, task=self.model.task
                    # )
                    plot_outputs(
                        true_letter, prediction, prediction_vib, num, path_pref
                    )
                else:
                    # plot_3D(true_letter, prediction, task=self.model.task)
                    plot_outputs(
                        true_letter, prediction, num=num, path_prefix=path_pref
                    )
                plot_wristpred(true_letter, prediction, num=num, path_prefix=path_pref)

            elif self.model.task == "elbow_flex":

                # # add vibrations if needed
                # scores_vib, batch_X_vib, _, _ = self.make_vib(
                #     vib_config,
                #     batch_X,
                #     vel=(
                #         None
                #         if key != "spindle_FR"
                #         else muscles.cpu().detach().numpy()[:, 1]
                #     ),
                # )
                print("NOT IMPLEMENTED YET")
                return


from collections import OrderedDict


def parse_config_value(value):
    # If the value is already an int or float, return it as-is
    if isinstance(value, (int, float)):
        return value

    # If the value is an OrderedDict, return it as-is (or handle as needed)
    if isinstance(value, OrderedDict):
        return value

    # If the value is already a list, return it as-is
    if isinstance(value, list):
        return value

    # Try to convert to int if it's a string that represents an integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try to convert to float if it's a string that represents a float
    try:
        return float(value)
    except ValueError:
        pass

    # Try to convert to a list of integers if it's a comma-separated string
    if isinstance(value, str):
        # Strip out brackets and split by commas
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]  # Remove the square brackets
        # Now try to split and convert to a list of integers
        if "," in value:
            try:
                return [int(x.strip()) for x in value.split(",")]
            except ValueError:
                pass  # Continue processing if split doesn't work

    # Return the value as-is if no conversion was possible
    return value


def load_model(
    config,
    model_path,
    task,
    device,
    test_data,
    causal=True,
    save_dir=SAVE_DIR,
    models_dir=MODELS_DIR,
):
    """Load the model from the specified path."""
    # model dir is model_path minus the last two directories
    model_dir = os.path.join(*model_path.split("/")[:-2])
    print(model_dir)
    if "LINEAR" in model_path:
        net = LinearModel
        experiment_id = config["experiment_id"]
        # remove from experiment_id "spindle_seed_{}_"
        experiment_id = experiment_id.replace(
            "spindle_seed_{}_".format(config["seed"]), ""
        )
        model = net(
            experiment_id=experiment_id,
            nclasses=config["nclasses"],
            input_shape=config["input_shape"],
            seed=config["seed"],
            train=True,
            task=task,
            outtime=config["outtime"],
            my_dir=os.path.join(save_dir, models_dir),
            training_seed=config.get("training_seed", None),
        )
    else:
        net = SpatiotemporalNetworkCausal if causal else SpatiotemporalNetwork
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
            p_drop=0.7,
            seed=config["seed"],
            train=True,
            task=task,
            outtime=config["outtime"],
            my_dir=os.path.join(save_dir, models_dir),
            # training_seed=config.get("training_seed", config.get("seed", 0)),
            training_seed=config.get(
                "training_seed", 9
            ),  # default train seed is 9 for mdoel path
        )
    tester = Tester(model, test_data, device=device)
    tester.load()
    return tester


def evaluate_model_with_vibrations_ranges(
    tester,
    vib_freqs,
    channel_indices,
    vib_ranges,
    muscles_to_vib_all,
    save_path,
    config,
    rand_max=None,
    i_a_sampled_coeff_path="/home/adriana/code/PropioPerception/extract_data/sampled_coefficients_i_a.csv",  # Replace with actual path if needed
    i_a_coeff_path="/home/adriana/code/PropioPerception/spindle_coefficients/i_a/linear/coefficients.csv",
    ii_sampled_coeff_path=None,
    ii_coeff_path=None,
    swap_elbow_angle=True,
    plot_individual=False,
    vib_type="alternating",
):
    """Evaluate the model for a range of vibration frequencies and save results."""
    if ii_sampled_coeff_path is None:
        # get from i_a_sampled_coeff_path replacing i_a with ii
        ii_sampled_coeff_path = i_a_sampled_coeff_path.replace("i_a", "ii")
        # check if ii_sampled_coeff_path exists
        if not os.path.exists(ii_sampled_coeff_path):
            ii_sampled_coeff_path = None

    if ii_coeff_path is None:
        # get from i_a_coeff_path replacing i_a with ii
        ii_coeff_path = i_a_coeff_path.replace("i_a", "ii")
        # check if ii_coeff_path exists
        if not os.path.exists(ii_coeff_path):
            ii_coeff_path = None

    results = []
    for vib_f in vib_freqs:
        vib_configs = generate_vib_config_multiple(
            vib_f,
            config["input_shape"],
            vib_ranges,
            muscles_to_vib_all,
            channel_indices,
            rand_max=rand_max,
            i_a_sampled_coeff_path=i_a_sampled_coeff_path,  # Replace with actual path if needed
            i_a_coeff_path=i_a_coeff_path,  # Replace with actual path if needed
            ii_sampled_coeff_path=ii_sampled_coeff_path,
            ii_coeff_path=ii_coeff_path,
        )
        print(f"Evaluating with vibration frequency: {vib_f} Hz")
        evaluation_results_vib = tester.evaluate_model(
            vib_config=vib_configs, n_split=1000
        )
        with open(os.path.join(save_path, "accuracy_vib.txt"), "a") as f:
            # store evaluation_results dictionary
            for key, value in evaluation_results_vib.items():
                f.write(f"{key}: {value}\n")

        scores, batch_x_s, _ = tester.get_scores_probabilities()
        scores_vib, batch_x_s_vib, batch_y_s = tester.get_scores_probabilities(
            vib_config=vib_configs
        )
        if swap_elbow_angle:
            batch_y_s[:, :, ELBOW_ANGLE_INDEX] = (
                180 - batch_y_s[:, :, ELBOW_ANGLE_INDEX]
            )
            scores[:, :, ELBOW_ANGLE_INDEX] = 180 - scores[:, :, ELBOW_ANGLE_INDEX]
            scores_vib[:, :, ELBOW_ANGLE_INDEX] = (
                180 - scores_vib[:, :, ELBOW_ANGLE_INDEX]
            )

        # vib_range = list(range(vib_start, vib_end))
        vib_metrics = compute_vib_metrics(scores, scores_vib, batch_y_s, vib_ranges)

        # Elbow angles
        elbow_angles = np.mean(
            batch_y_s[:, :, ELBOW_ANGLE_INDEX].cpu().detach().numpy(), axis=1
        )

        # Sort elbow angles and select indices
        sorted_indices = np.argsort(elbow_angles)
        selected_indices = sorted_indices[
            np.linspace(0, len(sorted_indices) - 1, num=10, dtype=int)
        ]

        # Collect results
        print(vib_metrics.keys())
        if "vib_angle_diff" in vib_metrics.keys():
            num_rows = len(vib_metrics["vib_angle_diff"])
        elif "vib_angle_diff_elbow" in vib_metrics.keys():
            num_rows = len(vib_metrics["vib_angle_diff_elbow"])
        for i in range(num_rows):
            row = {
                "vib_freq": vib_f,
                "vib_time_range": [[v_r[0], v_r[1]] for v_r in vib_ranges],
                "vib_muscles": muscles_to_vib_all,
                # "muscle_f_max": vib_config["muscle_f_max"],
                "trial": i,
                "elbow_angle": elbow_angles[i],
                "channel_indices": channel_indices,
            }
            row.update({key: vib_metrics[key][i] for key in vib_metrics})
            if i in selected_indices:
                row.update(
                    {
                        "true_outputs": batch_y_s[i].cpu().detach().numpy(),
                        "predicted_outputs": scores[i].cpu().detach().numpy(),
                        "predicted_vib_outputs": scores_vib[i].cpu().detach().numpy(),
                        "inputs_no_vib": batch_x_s[i].cpu().detach().numpy(),
                        "inputs_vib": batch_x_s_vib[i].cpu().detach().numpy(),
                    }
                )
            results.append(row)

        # ### not implemented for alternating yet ###
        # if plot_individual:
        #     plot_vibration_metrics(
        #         vib_metrics,
        #         elbow_angles,
        #         num_samples=NUM_SAVE_SAMPLES,
        #         save_path=save_path,
        #         plot_types=["scatter"],
        #         save_suffix=f"_vib_{vib_f}Hz_{vib_type}",
        #     )

    # Save results to HDF5
    df = pd.DataFrame(results)

    # Handle vibration frequencies
    if type(df.iloc[0]["vib_freq"]) == list:
        # Flatten vib_freq lists into unique scalar values
        df["vib_freq_str"] = df["vib_freq"].apply(lambda x: ", ".join(map(str, x)))
    df.to_hdf(os.path.join(save_path, f"vib_results.h5"), key="results", mode="w")
    return df


def evaluate_model_with_vibrations(
    tester,
    vib_freqs,
    channel_indices,
    vib_start,
    vib_end,
    muscles_to_vib,
    save_path,
    config,
    rand_max=None,
    i_a_sampled_coeff_path="/home/adriana/code/PropioPerception/extract_data/sampled_coefficients_i_a.csv",  # Replace with actual path if needed
    i_a_coeff_path="/home/adriana/code/PropioPerception/spindle_coefficients/i_a/linear/coefficients.csv",
    ii_sampled_coeff_path=None,
    ii_coeff_path=None,
    swap_elbow_angle=True,
    plot_individual=False,
    num_plot_trials=10,
):
    """Evaluate the model for a range of vibration frequencies and save results."""
    device = tester.device
    time_range = range(vib_start, vib_end)
    if len(muscles_to_vib) == 1:
        vib_type = muscles_to_vib[0]
    elif ("BIClong" in muscles_to_vib) or ("BICshort" in muscles_to_vib):
        vib_type = "biceps"
    elif ("TRIlong" in muscles_to_vib) or ("TRIlat" in muscles_to_vib):
        vib_type = "triceps"
    else:
        vib_type = "mixed"

    if ii_sampled_coeff_path is None:
        # get from i_a_sampled_coeff_path replacing i_a with ii
        ii_sampled_coeff_path = i_a_sampled_coeff_path.replace("i_a", "ii")
        # check if ii_sampled_coeff_path exists
        if not os.path.exists(ii_sampled_coeff_path):
            ii_sampled_coeff_path = None

    if ii_coeff_path is None:
        # get from i_a_coeff_path replacing i_a with ii
        ii_coeff_path = i_a_coeff_path.replace("i_a", "ii")
        # check if ii_coeff_path exists
        if not os.path.exists(ii_coeff_path):
            ii_coeff_path = None

    results = []
    for vib_f in vib_freqs:
        vib_config = generate_vib_config(
            vib_f,
            config["input_shape"],
            channel_indices,
            muscles_to_vib,
            time_range,
            rand_max=rand_max,
            i_a_sampled_coeff_path=i_a_sampled_coeff_path,  # Replace with actual path if needed
            i_a_coeff_path=i_a_coeff_path,  # Replace with actual path if needed
            ii_sampled_coeff_path=ii_sampled_coeff_path,
            ii_coeff_path=ii_coeff_path,
        )
        print(f"Evaluating with vibration frequency: {vib_f} Hz")
        evaluation_results_vib = tester.evaluate_model(
            vib_config=vib_config, n_split=1000
        )
        with open(os.path.join(save_path, "accuracy_vib.txt"), "a") as f:
            # f.write(PATH_TO_DATA + "\n")
            # store vib_config
            for key, value in vib_config.items():
                if key in ["vib_freq", "vib_time_range", "vib_muscles", "muscle_f_max"]:
                    f.write(f"{key}: {value}\n")
                # f.write(f"{key}: {value}\n")
            # store evaluation_results dictionary
            for key, value in evaluation_results_vib.items():
                f.write(f"{key}: {value}\n")

        scores, batch_x_s, _ = tester.get_scores_probabilities()
        scores_vib, batch_x_s_vib, batch_y_s = tester.get_scores_probabilities(
            vib_config=vib_config
        )
        if swap_elbow_angle:
            batch_y_s[:, :, ELBOW_ANGLE_INDEX] = (
                180 - batch_y_s[:, :, ELBOW_ANGLE_INDEX]
            )
            scores[:, :, ELBOW_ANGLE_INDEX] = 180 - scores[:, :, ELBOW_ANGLE_INDEX]
            scores_vib[:, :, ELBOW_ANGLE_INDEX] = (
                180 - scores_vib[:, :, ELBOW_ANGLE_INDEX]
            )

        vib_range = [(vib_start, vib_end)]
        # vib_range = list(range(vib_start, vib_end))
        # breakpoint()
        vib_metrics = compute_vib_metrics(
            scores, scores_vib, batch_y_s, vib_range, task=config["task"]
        )

        # print(vib_metrics)
        # Elbow angles
        elbow_angles = np.mean(
            batch_y_s[:, :, ELBOW_ANGLE_INDEX].cpu().detach().numpy(), axis=1
        )

        # Sort elbow angles and select indices
        sorted_indices = np.argsort(elbow_angles)
        selected_indices = sorted_indices[
            np.linspace(0, len(sorted_indices) - 1, num=10, dtype=int)
        ]
        # breakpoint()

        # Collect results one row one
        if "vib_angle_diff" in vib_metrics.keys():
            num_rows = len(vib_metrics["vib_angle_diff"])
        elif "vib_angle_diff_elbow" in vib_metrics.keys():
            num_rows = len(vib_metrics["vib_angle_diff_elbow"])
        for i in range(num_rows):
            row = {
                "vib_freq": vib_f,
                "vib_time_range": [vib_start, vib_end],
                "vib_muscles": muscles_to_vib,
                "muscle_f_max": vib_config["muscle_f_max"],
                "trial": i,
                "elbow_angle": elbow_angles[i],
                "channel_indices": channel_indices,
            }
            row.update({key: vib_metrics[key][i] for key in vib_metrics})
            if i in selected_indices:
                row.update(
                    {
                        "true_outputs": batch_y_s[i].cpu().detach().numpy(),
                        "predicted_outputs": scores[i].cpu().detach().numpy(),
                        "predicted_vib_outputs": scores_vib[i].cpu().detach().numpy(),
                        "inputs_no_vib": batch_x_s[i].cpu().detach().numpy(),
                        "inputs_vib": batch_x_s_vib[i].cpu().detach().numpy(),
                    }
                )
            results.append(row)

        if plot_individual:
            plot_vibration_metrics(
                vib_metrics,
                elbow_angles,
                num_samples=num_plot_trials,
                save_path=save_path,
                plot_types=["scatter"],
                save_suffix=f"_vib_{vib_f}Hz_{vib_type}",
            )

    # Save results to HDF5
    df = pd.DataFrame(results)
    df.to_hdf(os.path.join(save_path, f"vib_results.h5"), key="results", mode="w")
    return df
