import copy
import csv
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

from train.train_model_utils import Dataset


class SpindleDataset(Dataset):
    def __init__(
        self,
        path_to_data=None,
        data=None,
        dataset_type="train",
        key="spindle_firing",
        fraction=None,
        task="letter_recognition",
        n_out_time=1152,
        aclass=None,
        verbatim=True,
        need_muscles=False,
        new_size=None,
        start_end_idx=None,
    ):
        self.start_end_idx = start_end_idx
        super().__init__(
            path_to_data,
            data,
            dataset_type,
            key,
            fraction,
            task,
            n_out_time,
            aclass,
            verbatim,
            need_muscles,
            new_size,
        )
        # self.ground_truth = "labels" # uncomment to use scaled loss for angles and coords

    def make_data(self, mydata, aclass=None, start_end_idx=None):
        if self.start_end_idx is not None and start_end_idx is None:
            start_end_idx = self.start_end_idx

        # Load and shuffle dataset randomly before splitting
        if self.path_to_data is not None:
            with h5py.File(self.path_to_data, "r") as datafile:
                if start_end_idx is None:
                    data = torch.from_numpy(datafile["data"][()])
                    labels = torch.from_numpy(datafile["labels"][()])
                else:
                    print("loading data between", start_end_idx)
                    data = torch.from_numpy(
                        datafile["data"][start_end_idx[0] : start_end_idx[1]]
                    )
                    labels = torch.from_numpy(
                        datafile["labels"][start_end_idx[0] : start_end_idx[1]]
                    )
            # if task is letter_reconstruction_joints_vel add to labels the velocity of the joints
            if self.task == "letter_reconstruction_joints_vel":
                if self.n_out_time == 1152:
                    sample_rate = 240
                dt = 1 / sample_rate
                # for each label in labels, get the corresponding joint velocity
                velocities = np.gradient(labels, dt, axis=1)
                labels = np.concatenate((labels, velocities), axis=2)
                labels = torch.from_numpy(labels)
                # breakpoint()

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
                # breakpoint()
                self.test_data, self.test_labels = data, labels
        else:
            data = mydata["data"]
            labels = mydata["label"][()] - 1

        return data, labels


def adjust_retraining(time_array):
    return [i for i in range(len(time_array))]


def train_val_split(data, labels):
    num_train = int(0.9 * data.shape[0])
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:], labels[num_train:]

    return (train_data, train_labels, val_data, val_labels)
