"""
This file contains all helper functions used to train the models
Adapted from https://github.com/amathislab/DeepDraw/blob/master/code/nn_train_utils.py
"""

import copy
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import interp1d


class Dataset:
    """Defines a dataset object with simple routines to generate batches."""

    def __init__(
        self,
        path_to_data=None,
        data=None,
        dataset_type="train",
        key="spindle_firing",
        fraction=None,
        task="letter_recognition",
        n_out_time=320,
        aclass=None,
        verbatim=True,
        need_muscles=False,
        new_size=None,
    ):
        """Set up the `Dataset` object.

        Arguments
        ---------
        path_to_data : str, absolute location of the dataset file.
        dataset_type : {'train', 'test'} str, type of data that will be used along with the model.
        key : {'endeffector_coords', 'joint_coords', 'muscle_coords', 'spindle_firing'} str

        """

        start_time = time.time()

        self.path_to_data = path_to_data
        self.dataset_type = dataset_type
        self.key = key
        self.train_data = self.train_labels = None
        self.val_data = self.val_labels = None
        self.test_data = self.test_labels = None
        self.task = task
        self.n_out_time = n_out_time
        self.need_muscles = need_muscles
        self.new_size = new_size

        if task == "letter_recognition":
            self.ground_truth = "label"
        elif (
            task == "letter_reconstruction"
            or task == "letter_reconstruction_joints"
            or task == "letter_reconstruction_joints_vel"
        ):
            self.ground_truth = "endeffector_coords"
        elif task == "elbow_flex" or task == "elbow_flex_joints":
            self.ground_truth = "end_effector_coord"

        self.make_data(data, aclass=aclass)

        # For when I want to use only a fraction of the dataset to train!
        if fraction is not None:
            random_idx = np.random.permutation(self.train_data.shape[0])
            subset_num = int(fraction * random_idx.size)
            self.train_data = self.train_data[random_idx[:subset_num]]
            self.train_labels = self.train_labels[random_idx[:subset_num]]

        if verbatim:
            print(f"Dataset -> loaded in {round(time.time()-start_time,2)} seconds")

    def make_data(self, mydata, aclass=None):
        """Load train/val or test splits into the `Dataset` instance.

        Returns
        -------
        if dataset_type == 'train' : loads train and val splits.
        if dataset_type == 'test' : loads the test split.

        """

        # Load and shuffle dataset randomly before splitting
        if self.path_to_data is not None:

            with h5py.File(self.path_to_data, "r") as datafile:

                # for k,v in datafile.items():
                #     print(k,v)

                # inputs to model
                data = datafile[self.key][()]
                data = (  # check permutation
                    torch.from_numpy(data).permute(0, 3, 1, 2).type(torch.FloatTensor)
                )  # make shape (batch_size, #channels, space, time)

                # retreive muscle info for plotting
                if self.key == "spindle_FR" and self.need_muscles:
                    self.muscles = (
                        torch.from_numpy(datafile["spindle_info"][()])
                        .permute(0, 3, 2, 1)
                        .type(torch.FloatTensor)
                    )
                    optimal_lengths = torch.from_numpy(
                        datafile["optimal_muscle_lengths"][()]
                    ).type(torch.FloatTensor)
                    self.muscles_normed = (self.muscles / optimal_lengths).permute(
                        0, 1, 3, 2
                    )
                    self.muscles = self.muscles.permute(0, 1, 3, 2)

                # ground truth labels of the letter
                if self.task == "letter_recognition":
                    labels = torch.from_numpy(datafile[self.ground_truth][()] - 1).type(
                        torch.LongTensor
                    )

                # trajectory of the letter
                elif self.task == "letter_reconstruction":
                    labels = (
                        torch.from_numpy(datafile[self.ground_truth][()])
                        .permute(0, 2, 1)
                        .type(torch.FloatTensor)
                    )  # make shape (batch_size, time, space)

                    # only extract one specific letter
                    if aclass is not None:
                        labels_number = torch.from_numpy(
                            datafile["label"][()] - 1
                        ).type(torch.LongTensor)
                        flag = 0
                        for i, value in enumerate(labels_number):
                            if value == aclass:
                                if flag == 0:
                                    flag = 1
                                    new_labels = labels[i].unsqueeze(0)
                                    new_data = data[i].unsqueeze(0)
                                elif flag == 1:
                                    new_labels = torch.cat(
                                        (new_labels, labels[i].unsqueeze(0))
                                    )
                                    new_data = torch.cat(
                                        (new_data, data[i].unsqueeze(0))
                                    )
                        labels = new_labels
                        data = new_data

                elif self.task == "letter_reconstruction_joints":

                    # stack the joints and coordinates
                    coords = (
                        torch.from_numpy(datafile[self.ground_truth][()])
                        .permute(0, 2, 1)
                        .type(torch.FloatTensor)
                    )  # make shape (N, time, 3)
                    joints = (
                        torch.from_numpy(datafile["joint_coords"][()])
                        .permute(0, 2, 1)
                        .type(torch.FloatTensor)
                    )  # make shape (N, time, 4)
                    labels = torch.cat((coords, joints), dim=2)

                    # only extract one specific letter
                    if aclass is not None:
                        labels_number = torch.from_numpy(
                            datafile["label"][()] - 1
                        ).type(torch.LongTensor)
                        flag = 0
                        for i, value in enumerate(labels_number):
                            if value == aclass:
                                if flag == 0:
                                    flag = 1
                                    new_labels = labels[i].unsqueeze(0)
                                    new_data = data[i].unsqueeze(0)
                                elif flag == 1:
                                    new_labels = torch.cat(
                                        (new_labels, labels[i].unsqueeze(0))
                                    )
                                    new_data = torch.cat(
                                        (new_data, data[i].unsqueeze(0))
                                    )
                        labels = new_labels
                        data = new_data

                # for elbow trajectory
                elif self.task == "elbow_flex" or self.task == "elbow_flex_joints":
                    labels = torch.from_numpy(datafile[self.ground_truth][()]).type(
                        torch.FloatTensor
                    )
        else:
            data = mydata["data"]
            labels = mydata["label"][()] - 1

        # For training data, create training and validation splits
        if self.dataset_type == "train":
            self.train_data, self.train_labels, self.val_data, self.val_labels = (
                train_val_split(data, labels)
            )
            if self.need_muscles:
                (
                    self.train_muscles,
                    self.train_muscles_normed,
                    self.val_muscles,
                    self.val_muscles_normed,
                ) = train_val_split(self.muscles, self.muscles_normed)

        # For test data, do not split
        elif self.dataset_type == "test":
            self.test_data, self.test_labels = data, labels

    def next_trainbatch(self, batch_size, step=0, flag=False):
        """Returns a new batch of training data.

        Arguments
        ---------
        batch_size : int, size of training batch.
        step : int, step index in the epoch.

        Returns
        -------
        2-tuple of batch of training data and correspondig labels.

        """
        if step == 0:
            shuffle_idx = np.random.permutation(self.train_data.shape[0])
            self.train_data[:] = self.train_data[shuffle_idx]
            self.train_labels[:] = self.train_labels[shuffle_idx]
            if self.key == "spindle_FR" and self.need_muscles:
                self.train_muscles = self.train_muscles[shuffle_idx]

        mybatch_data = self.train_data[batch_size * step : batch_size * (step + 1)]
        mybatch_labels = self.train_labels[batch_size * step : batch_size * (step + 1)]
        # breakpoint()
        # mybatch_data, mybatch_labels = self.resize(mybatch_data, mybatch_labels)

        if flag:
            return (
                self.train_muscles[batch_size * step : batch_size * (step + 1)],
                mybatch_data,
                mybatch_labels,
            )
        else:
            return (mybatch_data, mybatch_labels)

    def next_valbatch(self, batch_size, type="val", step=0, flag=False):
        """Returns a new batch of validation or test data.

        Arguments
        ---------
        type : {'val', 'test'} str, type of data to return.

        """

        if type == "val":
            mybatch_data = self.val_data[batch_size * step : batch_size * (step + 1)]
            mybatch_labels = self.val_labels[
                batch_size * step : batch_size * (step + 1)
            ]
        elif type == "test":
            mybatch_data = self.test_data[batch_size * step : batch_size * (step + 1)]
            mybatch_labels = self.test_labels[
                batch_size * step : batch_size * (step + 1)
            ]

        # mybatch_data, mybatch_labels = self.resize(mybatch_data, mybatch_labels)

        if flag:
            return (
                self.muscles[batch_size * step : batch_size * (step + 1)],
                self.muscles_normed[batch_size * step : batch_size * (step + 1)],
                mybatch_data,
                mybatch_labels,
            )
        else:
            return (mybatch_data, mybatch_labels)

    def resize(self, batch_x, batch_y):
        if self.new_size is not None:
            true_timestamps = np.arange(batch_x.shape[-1])
            new_timestamps = np.linspace(0, true_timestamps[-1], self.new_size)
            batch_x = torch.from_numpy(
                interp1d(true_timestamps, batch_x, axis=-1)(new_timestamps)
            ).type(torch.FloatTensor)
            if self.task != "letter_recognition":
                batch_y = torch.from_numpy(
                    interp1d(true_timestamps, batch_y, axis=1)(new_timestamps)
                ).type(torch.FloatTensor)
        return batch_x, batch_y


class Trainer:
    """Trains a `Model` object with the given `Dataset` object."""

    def __init__(self, model=None, dataset=None, global_step=None, device="cpu"):
        """Set up the `Trainer`.

        Arguments
        ---------
        model : an instance of `ConvModel`, `AffineModel` or `RecurrentModel` to be trained.
        dataset : an instance of `Dataset`, containing the train/val data splits.

        """

        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.log_dir = model.model_path
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.graph = None
        self.best_loss = np.inf
        self.validation_accuracy = 0
        self.summary = {
            "path to train data": dataset.path_to_data,
            "time": [],
            "validation loss": [],
            "validation accuracy": [],
            "training loss": [],
            "training accuracy": [],
        }

    def load(self, retrain=True):
        """
        loads the model parameters into self.model for inference.
        make sure to have initialized self.model
        """

        self.model.load_state_dict(
            torch.load(
                os.path.join(self.log_dir, "model.ckpt"),
                map_location=torch.device(self.device),
            )
        )
        if not retrain:
            self.model.eval()

        path_to_config_file = os.path.join(self.model.model_path, "config.yaml")
        with open(path_to_config_file, "r") as myfile:
            # model_config = yaml.load(myfile)
            model_config = yaml.full_load(myfile)
        # check if the model is trained on normalized data
        # otherwise use default from data
        if (
            "train_mean" in model_config
            and "train_std" in model_config
            and "label_mean" in model_config
            and "label_std" in model_config
        ):
            self.train_data_mean = model_config["train_mean"]
            self.train_data_std = model_config["train_std"]
            self.label_mean = model_config["label_mean"]
            self.label_max = model_config["label_std"]
            # if the values are lists, make into tensors
            if isinstance(self.train_data_mean, list):
                self.train_data_mean = torch.tensor(self.train_data_mean)
            if isinstance(self.train_data_std, list):
                self.train_data_std = torch.tensor(self.train_data_std)
            if isinstance(self.label_mean, list):
                self.label_mean = torch.tensor(self.label_mean)
            if isinstance(self.label_max, list):
                self.label_max = torch.tensor(self.label_max)
        self.summary["time"] = model_config["time"]
        self.summary["validation loss"] = model_config["validation loss"]
        self.summary["validation accuracy"] = model_config["validation accuracy"]
        self.summary["training loss"] = model_config["training loss"]
        self.summary["training accuracy"] = model_config["training accuracy"]

    def save(self):
        """saves the model parameters. make sure to remeber the class initialization"""
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "model.ckpt"))

    def train(
        self,
        num_epochs=10,
        learning_rate=0.0005,
        batch_size=256,
        val_steps=100,
        early_stopping_epochs=5,
        end_training_num=2,
        early_stop_min_epoch=40,
        retrain=False,
        normalize=False,
        verbose=True,
        save_rand=False,
    ):
        """Train the `Model` object.

        Arguments
        ---------
        num_epochs : int, number of epochs to train for.
        learning_rate : float, learning rate for Adam Optimizer.
        batch_size : int, size of batch to train on.
        val_steps : int, number of batches after which to perform validation.
        early_stopping_epochs : int, number of steps for early stopping criterion.
        end_training_num : int, number of times the learning rate is divided before stopping training
        early_stop_min_epoch : int, the minimum number of epochs needed to have gone through before stopping
        retrain : bool, train already existing model vs not.
        normalize : bool, whether to normalize training data or not.
        verbose : bool, print progress on screen.
        save_rand : does not train. saves the model and generates config file.

        """
        # for early stopping
        if batch_size > self.dataset.train_data.shape[0]:
            batch_size = self.dataset.val_data.shape[0]
        steps_per_epoch = self.dataset.train_data.shape[0] // batch_size
        max_iter = num_epochs * steps_per_epoch
        self.batch_size = batch_size
        if val_steps > early_stopping_epochs * steps_per_epoch:
            val_steps = early_stopping_epochs * steps_per_epoch
        early_stopping_steps = early_stopping_epochs * steps_per_epoch // val_steps

        if normalize:
            # breakpoint()
            # normalize by channel. values are tensors of shape (Batch, Channel, Muscles, Time) not yet in device
            self.train_data_mean = self.dataset.train_data.mean(
                dim=[0, 3], keepdim=True
            )
            # self.train_data_std = torch.abs(self.dataset.train_data).max(dim=0, keepdim=True).values.max(dim=3, keepdim=True).values
            self.train_data_std = self.dataset.train_data.std(dim=[0, 3], keepdim=True)
            self.label_mean = self.dataset.train_labels.mean(dim=[0, 1], keepdim=True)
            # self.label_max = torch.abs(self.dataset.train_labels).max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values
            self.label_max = self.dataset.train_labels.std(dim=[0, 1], keepdim=True)
            # save params as dict with lists for the config
            train_params = {
                "train_mean": self.train_data_mean.cpu().numpy().tolist(),
                "train_std": self.train_data_std.cpu().numpy().tolist(),
                "label_mean": self.label_mean.cpu().numpy().tolist(),
                "label_std": self.label_max.cpu().numpy().tolist(),
            }
        else:
            self.train_data_mean = 0
            self.train_data_std = 1
            self.label_mean = 0
            self.label_max = 1
            train_params = {
                "train_mean": self.train_data_mean,
                "train_std": self.train_data_std,
                "label_mean": self.label_mean,
                "label_std": self.label_max,
            }

        val_params = {"validation_loss": 1e10, "validation_accuracy": 0}

        self.learning_rate = learning_rate

        # Define criterion for training
        if self.dataset.ground_truth == "label":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif (
            self.dataset.ground_truth == "endeffector_coords"
            or self.dataset.ground_truth == "end_effector_coord"
        ):
            self.criterion = torch.nn.MSELoss()
        elif self.dataset.ground_truth == "labels":
            # use scaled loss later
            # self.criterion = torch.nn.MSELoss()
            # self.criterion = coord_angles_loss
            pass

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        not_improved = 0
        end_training = 0
        val_params = {}

        if retrain:
            self.load(retrain)

        if save_rand:
            self.save()
            self.model.is_training = False
            make_config_file(self.model, train_params, val_params, self.summary)
            return

        start_time = time.time()

        for self.global_step in range(max_iter):

            # get a batch of data
            batch_X, batch_y = self.dataset.next_trainbatch(
                batch_size, self.global_step % steps_per_epoch
            )

            # normalize
            epsilon = 1e-8  # Small constant to prevent division by zero
            if type(self.train_data_mean) == torch.Tensor:  # Normalize by tensor
                batch_X = (
                    batch_X.to(self.device) - self.train_data_mean.to(self.device)
                ) / (self.train_data_std.to(self.device) + epsilon)
                batch_y = (
                    batch_y.to(self.device) - self.label_mean.to(self.device)
                ) / (self.label_max.to(self.device) + epsilon)
            else:  # Or float
                batch_X = (batch_X.to(self.device) - self.train_data_mean) / (
                    self.train_data_std + epsilon
                )
                batch_y = (batch_y.to(self.device) - self.label_mean) / (
                    self.label_max + epsilon
                )

            # print(f"Batch_X mean: {batch_X.mean().item()}, std: {batch_X.std().item()}")

            # train step
            self.optimizer.zero_grad()
            scores, pred, _ = self.model(batch_X)

            if self.dataset.ground_truth == "labels":
                loss = coord_angles_loss(scores, batch_y)
            else:
                loss = self.criterion(scores, batch_y)
            loss.backward()
            self.optimizer.step()

            # Validate/save periodically
            if self.global_step % val_steps == 0:
                # breakpoint()

                # Summarize, print progress
                loss_val, acc_val = self.save_summary(
                    loss.item(), (scores, pred), batch_y, self.global_step // val_steps
                )
                if verbose and self.global_step != 0:
                    frac_left = 1 - self.global_step / max_iter
                    time_elapsed = time.time() - start_time
                    hours_left = (
                        (max_iter * time_elapsed / self.global_step) * frac_left / 3600
                    )
                    if hours_left > 1:
                        print(
                            f"Step : {self.global_step}/{max_iter} ({round((1-frac_left)*100, 2)}%), {round(hours_left,2)} hours left, Validation Accuracy : {round(acc_val, 6)}, Validation Loss : {loss_val}, lr={learning_rate}, NI={not_improved}/{early_stopping_steps}, S={end_training}/{end_training_num}"
                        )
                    else:
                        minutes_left = hours_left * 60
                        if minutes_left > 1:
                            print(
                                f"Step : {self.global_step}/{max_iter} ({round((1-frac_left)*100, 2)}%), {round(minutes_left,2)} minutes left, Validation Accuracy : {round(acc_val, 6)}, Validation Loss : {loss_val}, lr={learning_rate}, NI={not_improved}/{early_stopping_steps}, S={end_training}/{end_training_num}"
                            )
                        else:
                            print(
                                f"Step : {self.global_step}/{max_iter} ({round((1-frac_left)*100, 2)}%), {round(minutes_left*60,2)} seconds left, Validation Accuracy : {round(acc_val, 6)}, Validation Loss : {loss_val}, lr={learning_rate}, NI={not_improved}/{early_stopping_steps}, S={end_training}/{end_training_num}"
                            )

                # early stopping and patience
                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    self.validation_accuracy = acc_val
                    self.save()
                    val_params = {
                        "validation_loss": float(self.best_loss),
                        "validation_accuracy": float(acc_val),
                    }
                    not_improved = 0
                    make_config_file(self.model, train_params, val_params, self.summary)
                else:
                    not_improved += 1

                if not_improved >= early_stopping_steps:
                    learning_rate /= 4
                    if not verbose:
                        print(learning_rate)
                    not_improved = 0
                    end_training += 1
                    self.load()

                if end_training == end_training_num:
                    if self.global_step < early_stop_min_epoch * steps_per_epoch:
                        end_training = 1
                        not_improved = 0
                    else:
                        break

        # finalize training
        print(f"Trainer -> trained in {round(time.time()-start_time,2)} seconds")
        self.model.is_training = False
        make_config_file(self.model, train_params, val_params, self.summary)
        self.plot_history()

    def save_summary(self, train_loss, train_pred, batch_y, n_iter):
        """Create and save summaries for training and validation."""

        # compute accuracy for training
        if self.model.task == "letter_recognition":
            train_accuracy = torch.mean(
                (batch_y == torch.argmax(train_pred[1], dim=1).squeeze(0)).type(
                    torch.FloatTensor
                )
            )
        elif (
            self.model.task == "letter_reconstruction"
            or self.model.task == "elbow_flex"
        ):
            train_accuracy = (
                torch.mean(
                    torch.norm(
                        torch.subtract(
                            train_pred[0].reshape(-1, 3), batch_y.reshape(-1, 3)
                        ),
                        dim=1,
                    ).type(torch.FloatTensor)
                )
                .detach()
                .numpy()
            )
        elif (
            self.model.task == "letter_reconstruction_joints"
            or self.model.task == "elbow_flex_joints"
            or self.model.task == "letter_reconstruction_joints_vel"
        ):
            train_accuracy = (
                torch.mean(
                    torch.norm(
                        torch.subtract(
                            train_pred[0][:, :, :3].reshape(-1, 3),
                            batch_y[:, :, :3].reshape(-1, 3),
                        ),
                        dim=1,
                    ).type(torch.FloatTensor)
                )
                .detach()
                .numpy()
            )

        # compute validation loss and accuracy
        validation_loss, validation_accuracy = self.eval()

        # keep track of vlaues for plotting
        self.summary["time"].append(int(n_iter))
        self.summary["validation loss"].append(float(validation_loss))
        self.summary["validation accuracy"].append(float(validation_accuracy))
        self.summary["training loss"].append(float(train_loss))
        self.summary["training accuracy"].append(float(train_accuracy))

        return validation_loss, validation_accuracy

    def eval(self):
        """Evaluate validation performance.

        Returns
        -------
        validation_loss : float, loss on the entire validation data
        validation_accuracy : float, accuracy on the validation data

        """
        # how many to go through dataset once
        num_iter = self.dataset.val_data.shape[0] // self.batch_size
        # num_iter = (self.dataset.val_end_idx - self.dataset.train_end_idx) // self.batch_size

        # store info
        acc_val = np.zeros(num_iter)
        loss_val = np.zeros(num_iter)

        # go through data
        for i in range(num_iter):

            # forward pass
            batch_X, batch_y = self.dataset.next_valbatch(self.batch_size, step=i)
            # normalize
            epsilon = 1e-8  # Small constant to prevent division by zero
            if type(self.train_data_mean) == torch.Tensor:  # Normalize by tensor
                batch_X = (
                    batch_X.to(self.device) - self.train_data_mean.to(self.device)
                ) / (self.train_data_std.to(self.device) + epsilon)
                batch_y = (
                    batch_y.to(self.device) - self.label_mean.to(self.device)
                ) / (self.label_max.to(self.device) + epsilon)
            else:  # Or float
                batch_X = (batch_X.to(self.device) - self.train_data_mean) / (
                    self.train_data_std + epsilon
                )
                batch_y = (batch_y.to(self.device) - self.label_mean) / (
                    self.label_max + epsilon
                )

            scores, prob, _ = self.model(batch_X)
            if self.dataset.ground_truth == "labels":
                loss = coord_angles_loss(scores, batch_y)
            else:
                loss = self.criterion(scores, batch_y)
            loss_val[i] = loss.item()

            # compute accuracy
            if self.model.task == "letter_recognition":
                acc_val[i] = torch.mean(
                    (batch_y == torch.argmax(prob, dim=1).squeeze(0)).type(
                        torch.FloatTensor
                    )
                )
            elif (
                self.model.task == "letter_reconstruction"
                or self.model.task == "elbow_flex"
            ):
                acc_val[i] = torch.mean(
                    torch.norm(
                        torch.subtract(scores.reshape(-1, 3), batch_y.reshape(-1, 3)),
                        dim=1,
                    )
                )
            elif (
                self.model.task == "letter_reconstruction_joints"
                or self.model.task == "elbow_flex_joints"
            ):
                acc_val[i] = torch.mean(
                    torch.norm(
                        torch.subtract(
                            scores[:, :, :3].reshape(-1, 3),
                            batch_y[:, :, :3].reshape(-1, 3),
                        ),
                        dim=1,
                    )
                )

        return loss_val.mean(), acc_val.mean()

    def plot_history(self):
        new_time = adjust_retraining(self.summary["time"])
        fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
        ax1.plot(new_time, self.summary["training loss"], label="training loss")
        ax2.plot(new_time, self.summary["training accuracy"], label="training accuracy")
        ax1.plot(new_time, self.summary["validation loss"], label="validation loss")
        ax2.plot(
            new_time, self.summary["validation accuracy"], label="validation accuracy"
        )
        ax1.legend(loc="best")
        ax2.legend(loc="best")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.set_xlabel("Iterations")
        ax2.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy or Error")
        plt.savefig(os.path.join(self.model.model_path, "plot.png"))
        # plt.show()


def adjust_retraining(time_array):
    return [i for i in range(len(time_array))]


def train_val_split(data, labels):
    num_train = int(0.9 * data.shape[0])
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:], labels[num_train:]

    return (train_data, train_labels, val_data, val_labels)


def make_config_file(model, train_params, val_params, summary):
    """Make a configuration file for the given model, created after training.

    Given a `ConvModel`, `AffineModel` or `RecurrentModel` instance, generates a
    yaml file to save the configuration of the model.

    """

    mydict = copy.copy(model.__dict__)

    # Convert to python native types for better readability
    for key, value in mydict.items():
        mydict[key] = str(value)

    # Save yaml file in the model's path
    path_to_yaml_file = os.path.join(model.model_path, "config.yaml")
    with open(path_to_yaml_file, "w") as myfile:
        yaml.dump(mydict, myfile, default_flow_style=False)
        yaml.dump(train_params, myfile, default_flow_style=False)
        yaml.dump(val_params, myfile, default_flow_style=False)
        yaml.dump(summary, myfile, default_flow_style=False)

    return


def coord_angles_loss(scores, batch_y):
    # Split outputs and ground truth into two groups
    scores_coords = scores[:, :3]  # First 3 outputs: x, y, z
    scores_angles = scores[:, 3:]  # Last 4 outputs: angles
    batch_y_coords = batch_y[:, :3]
    batch_y_angles = batch_y[:, 3:]

    # Compute individual losses
    # Compute per-batch standard deviations
    xyz_std = torch.std(batch_y_coords, dim=0) + 1e-8  # Avoid division by zero
    angle_std = torch.std(batch_y_angles, dim=0) + 1e-8

    # Normalize dynamically
    loss_coords = torch.mean(((scores_coords - batch_y_coords) / xyz_std) ** 2)
    loss_angles = torch.mean(((scores_angles - batch_y_angles) / angle_std) ** 2)

    # loss_coords = torch.nn.functional.mse_loss(
    #     scores_coords, batch_y_coords
    # )
    # angle_diff = scores_angles - batch_y_angles
    # loss_angles = torch.mean(1 - torch.cos(angle_diff))  # Angular loss
    loss = loss_coords + loss_angles
    return loss
