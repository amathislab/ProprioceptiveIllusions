"""
This file contains the definitions for the spatiotemporal models that can predict
the letter and path based on proprioceptive inputs
Adapted from https://github.com/amathislab/DeepDraw/blob/master/code/nn_models.py
and https://github.com/amathislab/DeepDraw/blob/master/single_cell/nn_rmodels.py,
as well as based on descriptions from https://elifesciences.org/articles/81499#s4
"""

import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


class SpatiotemporalNetwork(nn.Module):
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(
        self,
        experiment_id,
        nclasses,
        arch_type,
        nlayers,
        n_skernels,
        n_tkernels,
        s_kernelsize,
        t_kernelsize,
        s_stride,
        t_stride,
        padding,
        input_shape,
        p_drop,
        seed=None,
        train=True,
        task="letter_recognition",
        outtime=320,
        my_dir="/media/data4/sebastian/part6_trained_models/",
        build_fc=True,  # whether to build the fully connected layer
        layer_norm=False,
        training_seed=None,
    ):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : string, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        padding : int, padding of convolutional layer.
        input_shape : list of three ints, size of input [C, S, T].
        p_drop : float, dropout probability of fully connected layer.
        seed : int, seed of the spindle parameters used.
        train : bool, is the network meant to be trained or not.
        task : {"letter_recognition", "letter_reconstruction"} that determine the output of the fully
            connected layer
        outtime : int, determines the resolution of output image when task="letter_reconstruction"
            ignore if task is "letter_recognition"
        my_dir : str, path to directory where models are saved
        build_fc : bool, whether to build the fully connected layer
        layer_norm : bool, whether to use layer normalization
        training_seed : int, seed for training. set at model definition
        """

        super(SpatiotemporalNetwork, self).__init__()

        # Set the random seed for reproducibility
        self.training_seed = training_seed
        if training_seed is not None:
            self._set_seed(training_seed)

        assert (
            len(n_skernels) == len(n_tkernels) == nlayers
        ), "Number of spatial and temporal processing layers must be equal!"

        if arch_type == "spatiotemporal":
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            t_stride = s_stride

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.padding = padding
        self.input_shape = input_shape
        self.p_drop = p_drop
        self.seed = seed
        self.task = task
        self.outtime = outtime
        self.layer_norm = layer_norm

        # Make model name
        # kernels = "-".join(str(i) for i in n_skernels)

        # parts_name = [
        #     arch_type,
        #     str(nlayers),
        #     kernels,
        #     "".join(str(i) for i in [s_kernelsize, s_stride, t_kernelsize, t_stride]),
        # ]
        parts_name = [
            arch_type,
            str(nlayers)
        ]

        # Create model directory
        self.name = "_".join(parts_name)

        if seed is not None:
            self.name += "_" + str(self.seed)
        if training_seed is not None:
            self.name += "_" + str(self.training_seed)
        if not train:
            self.name += "r"

        MODELS_DIR = os.path.join(os.path.dirname(CUR_DIR), my_dir)
        exp_dir = os.path.join(MODELS_DIR, f"experiment_{self.experiment_id}")
        self.model_path = os.path.join(exp_dir, self.name)
        print("model path model def", self.model_path)
        # breakpoint()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

        # build model
        self.model = []
        self.shapes = [input_shape]
        current_shape = input_shape  # (Channels, Height, Width)

        for i in range(len(n_skernels)):
            in_channels = current_shape[0] if i == 0 else n_skernels[i - 1]
            out_channels = n_skernels[i]

            # Compute the output spatial dimensions
            H_in, W_in = current_shape[1], current_shape[2]
            H_out = (
                H_in + 2 * ((s_kernelsize - 1) // 2) - s_kernelsize
            ) // s_stride + 1
            W_out = (
                W_in + 2 * ((t_kernelsize - 1) // 2) - t_kernelsize
            ) // t_stride + 1

            # Add the layer to the model
            if self.layer_norm:
                self.model.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(s_kernelsize, t_kernelsize),
                            stride=(s_stride, t_stride),
                            padding=(
                                (s_kernelsize - 1) // 2,
                                (t_kernelsize - 1) // 2,
                            ),  # SAME padding
                        ),
                        nn.LayerNorm(
                            [out_channels, H_out, W_out]
                            # normalized_shape=[out_channels, H_out]
                        ),  # Fixed normalized shape
                        nn.ReLU(),
                    )
                )
            else:
                self.model.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(s_kernelsize, t_kernelsize),
                            stride=(s_stride, t_stride),
                            padding=(
                                (s_kernelsize - 1) // 2,
                                (t_kernelsize - 1) // 2,
                            ),  # SAME padding
                        ),
                        nn.ReLU(),
                    )
                )

            # Update the current shape for the next layer
            current_shape = (out_channels, H_out, W_out)
            self.shapes.append(current_shape)

        self.model = nn.ModuleList(self.model)

        # linear classifier
        if build_fc:
            self.shapes.append(
                self.shapes[-1][0] * self.shapes[-1][1] * self.shapes[-1][2]
            )
            if self.task == "letter_recognition":
                self.fc = nn.Linear(self.shapes[-1], self.nclasses)
                # self.fc = nn.Linear(self.n_tkernels[-1], self.nclasses)
                self.dropout = nn.Dropout(p=self.p_drop)
            elif (
                self.task == "letter_reconstruction"
                or self.task == "elbow_flex"
                or self.task == "letter_reconstruction_joints"
                or self.task == "letter_reconstruction_joints_vel"
                or self.task == "elbow_flex_joints"
            ):
                # part of the predictions information will be encoded in the time steps
                # so only keep the required spatial information
                self.fc = nn.Linear(self.shapes[-1] // outtime, self.nclasses)
                self.dropout = None
        else:
            self.fc = None
            self.dropout = None
        print("shapes", self.shapes)

    def _set_seed(self, seed):
        """Set the random seed for reproducibility."""
        print("setting seed to ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def forward(self, X):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : torch.tensor [batch_size, 2, num_inputs, num_timesteps], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : torch.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : torch.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """

        # keep track of intermediate layers
        net = OrderedDict([])

        # useful values
        score = X
        batch_size = X.shape[0]

        # go through CNN
        for layer in range(self.nlayers):
            score = self.model[layer](score)
            net[f"spatiotemporal{layer}"] = score

        # flatten
        if self.task == "letter_recognition":
            score = score.view(batch_size, -1)
        elif (
            self.task == "letter_reconstruction"
            or self.task == "elbow_flex"
            or self.task == "letter_reconstruction_joints"
            or self.task == "letter_reconstruction_joints_vel"
            or self.task == "elbow_flex_joints"
        ):
            score = score.permute(0, 3, 1, 2)
            score = score.reshape(batch_size, self.outtime, -1)

        # go through linear fc layer
        if self.dropout is not None:
            score = self.dropout(score)

        score = self.fc(score)
        net["score"] = score

        # normalize
        probabilities = nn.functional.softmax(score, dim=1)

        return score, probabilities, net

    def find_layer_shape(self, input_size, kernel_size, stride, padding):
        """finds the size of the convolutional output for N-D convolutions

        Arguments
        ---------
        input_size : list of ints
        kernel_size : list of ints
        stride : list of ints
        padding : list of ints
        """

        out_size = [0] * len(input_size)
        for i in range(len(input_size)):
            out_size[i] = (input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[
                i
            ] + 1
        return out_size


class SpatiotemporalNetworkCausal(SpatiotemporalNetwork):
    """Extends SpatiotemporalNetwork with a causal fully connected layer per time step."""

    def __init__(self, *args, **kwargs):
        """Initialize the causal SpatiotemporalNetwork by extending the parent class."""
        kwargs["build_fc"] = False
        super(SpatiotemporalNetworkCausal, self).__init__(*args, **kwargs)

        if self.task != "letter_recognition":
            # Compute the flattened spatial dimensions from the final convolutional layer
            # spatial_size = self.shapes[-2][1] * self.shapes[-2][2]  # H_out * W_out
            spatial_size = self.shapes[-1][0] * self.shapes[-1][1]  # H_out * W_out
            self.shapes.append(spatial_size)

            # Replace the fully connected layer with one that processes each time step independently
            self.fc_per_time_step = nn.Linear(spatial_size, self.nclasses)
        else:
            raise NotImplementedError(
                "Causal fully connected layer is not implemented for letter recognition."
            )
        print("shapes", self.shapes)

    def forward(self, X):
        """Override forward pass to use causal fully connected layer."""
        net = OrderedDict([])
        score = X
        batch_size = X.shape[0]

        # Pass input through the convolutional layers
        for layer in range(self.nlayers):
            score = self.model[layer](score)
            net[f"spatiotemporal{layer}"] = score

        # Reshape for temporal processing
        if (
            self.task == "letter_reconstruction"
            or self.task == "elbow_flex"
            or self.task == "letter_reconstruction_joints"
            or self.task == "elbow_flex_joints"
            or self.task == "letter_reconstruction_joints_vel"
        ):
            score = score.permute(0, 3, 1, 2)  # [B, Time, Channels, Muscles]
            score = score.reshape(batch_size, self.outtime, -1)  # Flatten spatial dims

        # Apply dropout if defined
        if self.dropout is not None:
            score = self.dropout(score)

        # Apply the causal fully connected layer to each time step independently
        # print(score.shape)
        score = self.fc_per_time_step(score)  # [B, T, nclasses]
        net["score"] = score

        # Compute softmax probabilities along the class dimension
        probabilities = nn.functional.softmax(score, dim=-1)

        return score, probabilities, net


class PowerNet(nn.Module):
    """from https://github.com/amathislab/SpindleSim/blob/main/src/spindle_models/spindle_models.py"""

    def __init__(self, num_features=2, scaler_list=None, model_path=None):
        """
        PowerNet is a neural network that uses power transformation
        then uses linear transformation on the input to generate the output.
        It is based on the Prochazka-Gorassini model.

        Args:
            num_features (int, optional): Number of features. Defaults to 2.
        """
        super().__init__()
        if model_path is not None:
            model_ckpt_path = Path(os.path.join(model_path, f"fold_0.pth"))
            config_path = Path(os.path.join(model_path, "config.yml"))
            scaler_path = Path(os.path.join(model_path, "scaler.pkl"))

            ## Load the scaler
            with open(scaler_path, "rb") as f:
                self.scaler_list = pickle.load(f)

            config = self.load_config(config_path)
            self.num_features = config["num_feats"]
        else:
            self.num_features = num_features
            self.scaler_list = scaler_list
        self.power = nn.Parameter(
            torch.ones(self.num_features) * 0.5, requires_grad=True
        )
        self.linear = nn.Linear(self.num_features, 1)

        ## Load the model
        if model_path is not None:
            self.load_state_dict(torch.load(model_ckpt_path))

    def forward(self, x, inv_scaler=False, idx_scaler=0):
        """
        Forward pass of the PowerNet model.

        Args:
            x (torch.Tensor): Input data tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Power Transformation
        x = torch.sign(x) * torch.pow(
            torch.abs(x), torch.clamp(self.power, min=0.0, max=4.0)
        )
        # Linear layer
        x = self.linear(x)
        x = x.squeeze(dim=-1)  # Remove the last dimension
        if inv_scaler:
            x = self.scaler_list[idx_scaler].inverse_transform(x.detach().cpu().numpy())
        return x

    def load_config(self, config_file_path):
        with config_file_path.open("r") as file:
            return yaml.safe_load(file)


class LinearModel(nn.Module):
    """
    Defines a linear model as a baseline for SpatioTemporalNetwork.
    """

    def __init__(
        self,
        experiment_id,
        nclasses,
        input_shape,
        seed=None,
        train=True,
        task="letter_recognition",
        outtime=320,
        my_dir="/media/data4/sebastian/part6_trained_models/",
        training_seed=None,
    ):
        super(LinearModel, self).__init__()

        # Set the random seed for reproducibility
        if training_seed is not None:
            self.training_seed = training_seed
        else:
            self.training_seed = seed
        if self.training_seed is not None:
            self._set_seed(self.training_seed)

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.input_shape = input_shape
        self.seed = seed
        self.task = task
        self.outtime = outtime

        # Make model name
        self.name = f"linear_seed_{seed}_{training_seed}"

        MODELS_DIR = os.path.join(os.path.dirname(CUR_DIR), my_dir)
        exp_dir = os.path.join(MODELS_DIR, f"experiment_{self.experiment_id}")
        self.model_path = os.path.join(exp_dir, self.name)
        print("model path model def", self.model_path)
        breakpoint()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.is_training = True

        # linear layer
        self.fc = nn.Linear(input_shape[0] * input_shape[1], self.nclasses)

    def _set_seed(self, seed):
        """Set the random seed for reproducibility."""
        print("setting seed to ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def forward(self, X):
        batch_size = X.shape[0]

        # Reshape
        score = X
        score = score.permute(0, 3, 1, 2)  # [B, Time, Channels, Muscles]
        score = score.reshape(batch_size, self.outtime, -1)  # Flatten spatial dims

        # Apply the linear fully connected layer
        score = self.fc(score)
        net = OrderedDict()
        net["score"] = score

        # normalize
        probabilities = nn.functional.softmax(score, dim=1)

        return score, probabilities, net
