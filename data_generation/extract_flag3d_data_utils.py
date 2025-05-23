"""
This file contains all the code needed to extract, visualize, and convert the 
FLAG3D data to the data needed for the spatiotemporal model as used in ```extract_flag3d_data.py```.
The extraction is adapted from https://github.com/AndyTang15/FLAG3D/blob/main/gcn-c3d-mindspore/dataset/
and the conversion from https://github.com/amathislab/DeepDraw/blob/master/dataset/
"""

import os
import pickle

import h5py
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import scipy
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from utils.visualize_sample import rotx, roty, rotz, transform


class GenSkeFeat:
    """Unified interface for generating multi-stream skeleton features."""

    def transform(self, results: dict) -> dict:
        if "keypoint_score" in results and "keypoint" in results:
            results["keypoint"] = np.concatenate(
                [results["keypoint"], results["keypoint_score"][..., None]], -1
            )
        return results


class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = False,
        seed: int = 255,
    ) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False
                )
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)]
                )
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_test_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for testing clips.
        """

        np.random.seed(self.seed)
        all_inds = []
        for i in range(self.num_clips):
            if num_frames < clip_len:
                start_ind = (
                    i
                    if num_frames < self.num_clips
                    else i * num_frames // self.num_clips
                )
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False
                )
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)]
                )
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`UniformSampleFrames`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results["total_frames"]

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get("start_index", 0)
        inds = inds + start_index

        if "keypoint" in results:
            kp = results["keypoint"]
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int64)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results["frame_inds"] = inds.astype(
            np.int32
        )  # 添加索引到字典中 # Add index to dictionary
        results["clip_len"] = self.clip_len
        results["frame_interval"] = None
        results["num_clips"] = self.num_clips
        return results


class PoseDecode:
    """Load and decode pose with given indices."""

    def _load_kp(self, kp: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoints according to sampled indexes."""
        return kp[:, frame_inds].astype(np.float32)

    def _load_kpscore(self, kpscore: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoint scores according to sampled indexes."""
        return kpscore[:, frame_inds].astype(np.float32)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PoseDecode`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if "total_frames" not in results:
            results["total_frames"] = results["keypoint"].shape[1]

        if "frame_inds" not in results:
            results["frame_inds"] = np.arange(results["total_frames"])

        if results["frame_inds"].ndim != 1:
            results["frame_inds"] = np.squeeze(results["frame_inds"])

        offset = results.get("offset", 0)
        frame_inds = results["frame_inds"] + offset

        if "keypoint_score" in results:
            results["keypoint_score"] = self._load_kpscore(
                results["keypoint_score"], frame_inds
            )

        results["keypoint"] = self._load_kp(results["keypoint"], frame_inds)

        return results


class FormatGCNInput:
    """Format final skeleton shape."""

    def __init__(self, num_person: int = 1, mode: str = "zero") -> None:
        self.num_person = num_person  # 人数
        assert mode in ["zero", "loop"]
        self.mode = mode  # zero padding

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`FormatGCNInput`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results["keypoint"]

        cur_num_person = keypoint.shape[0]

        # FLAG3D是一个人的，所以这里没用上 # FLAG3D belongs to one person, so it is not used here.
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros((pad_dim,) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == "loop" and cur_num_person == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif cur_num_person > self.num_person:
            keypoint = keypoint[: self.num_person]

        # 根据num_clips改变下形状 # Change the shape according to num_clips
        M, T, V, C = keypoint.shape
        nc = results.get("num_clips", 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results["keypoint"] = np.ascontiguousarray(keypoint)
        return results


class Collect:
    """Collect keypoint and label"""

    def __init__(self):
        self.keys = ["keypoint", "label"]

    def transform(self, results: dict) -> dict:
        results_back = {}
        for key in self.keys:
            results_back[key] = results[key]

        return results_back


class ToTensor:
    """ToTensor"""

    def __init__(self):
        self.keys = ["keypoint"]

    def transform(self, results: dict) -> dict:
        results[self.keys[0]] = torch.Tensor(results["keypoint"])
        return results


class Arm:
    """
    class that converts coordinates to joint angles
    """

    def __init__(self, q=None, q0=None, L=None):
        # Initial joint angles
        self.q = np.array([10, 10, 10, 10]) if q is None else q
        # Default arm position, set to last position in the trajectory.
        self.q0 = np.array([50, 40, 20, 75]) if q0 is None else q0
        # Lengths of humerus and radius (in cm).
        self.L = np.array([33, 26]) if L is None else L

        # Maximum and minimum angles. [in degrees]
        self.min_angles = np.array([-95, 0, -90, 0])
        self.max_angles = np.array([130, 180, 120, 130])

    def get_xyz(self, q=None):
        """Implements forward kinematics:
        Returns the end-effector coordinates (euclidean) for a given set of joint
        angle values.
        Inputs : q: joint angles (in degrees)
        Returns :
            new_hand_loc: xyz coordinates of the end-effector
            link_pos: xyz coordinates of each joint (shoulder, elbow, hand)
        """
        if q is None:
            q = self.q

        # Define rotation matrices about the shoulder and elbow.
        # Translations for the shoulder frame will be introduced later.
        def shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot):
            return (
                roty(elv_angle)
                .dot(rotz(shoulder_elv))
                .dot(roty(-elv_angle))
                .dot(roty(shoulder_rot))
            )

        def elbow_rotation(elbow_flexion):
            return rotx(elbow_flexion)

        # Unpack variables
        elv_angle, shoulder_elv, shoulder_rot, elbow_flexion = q
        upperarm_length, forearm_length = self.L

        # Define initial joint locations:
        origin = np.array([0, 0, 0])
        elbow = np.array([0, -upperarm_length, 0])
        hand = np.array([0, -forearm_length, 0])

        new_elbow_loc = shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot).dot(
            elbow
        )
        new_hand_loc = shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot).dot(
            elbow_rotation(elbow_flexion).dot(hand) + elbow
        )

        link_pos = np.column_stack((origin, new_elbow_loc, new_hand_loc))

        return new_hand_loc, link_pos

    def inv_kin(self, elbow, hand):
        """Implements inverse kinematics:
        Given an xyz position of the hand, return a set of joint angles (q)
        using constraint based minimization. Constraint is to match hand xyz and
        minimize the distance of each joint from it's default position (q0).
        Inputs :
        Returns :
        """

        def distance_to_default(q, *args):
            return np.linalg.norm(q - np.asarray(self.q0))

        def pos_constraint(q, elbow, hand):
            """
            this is the function that the inverse kinematics will try to make 0
            change this function as needed
            """
            return np.linalg.norm(
                self.get_xyz(q=q)[1][:, 1:] - np.column_stack((elbow, hand))
            )

        def joint_limits_upper_constraint(q, *args):
            return self.max_angles - q

        def joint_limits_lower_constraint(q, *args):
            return q - self.min_angles

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.q,
            eqcons=[pos_constraint],
            ieqcons=[joint_limits_upper_constraint, joint_limits_lower_constraint],
            args=(
                elbow,
                hand,
            ),
            iprint=0,
        )


class BuildDatasetForFLAG3D:
    """
    class that contains all the necessary functions to convert the data set
    """

    def __init__(
        self,
        dataset_dir,
        dataset_type="all",
        clip_len=1380,
        num_clips=1,
        path_to_osim=None,
        path_to_hdf5=None,
        augment=1,
        upperarm_length=33,
        augment_trajectory=True,
        max_jerk=1,
        max_tracing_error=1e-2,
        num_div=3,
        increase_fps_by_a_factor_of=1,
        target_len=576,
        start_indx=0,
        end_indx=7200,
    ):
        """
        Loads the FLAG3D data set from the pickle file and transforms it to a list ```self.dataset```
        of length 7200 (size of dataset). Each item of the list is a dictionnary containing a
        label ```label``` and torch tensor ```keypoints``` of shape (clip_len, 25, 3) containing the cartesian
        coordinates (x,y,z) of 25 keypoints in ```clip_len``` time steps. Note that these values are normalized.

        Arguments
            dataset_dir: string, file path to dataset pickle file
            dataset_type: {"train", "test", "all"}, splits the dataset according to a predetermined partition unless "all"
            clip_len: int > 0, length of clip
            path_to_osim: string, file path to opensim model
            path_to_hdf5: string, folder path to save intermediate values
            augment: int > 0, number of times to traverse dataset
            upperarm_length: int > 0, for proper scaling
            augment_trajectory: bool, whether to pass through the dataset by running the augment function or only the raw data
            num_div: int > 0, in how many parts to split the frames
            increase_fps_by_a_factor_of: float > 0, by how much to interpolate
            target_len: int > 0, number of frames to fit the new clip length (should be larger than clip_len/num_div) with padding - only used if augment_trajectory is True
            start_indx: 0 < int < end_indx, where to start in the dataset the inverse kin and osim
            end_indx: start_indx < int < 7200, where to end in the dataset the inverse kin and osim
        """

        # save the key variables
        self.path_to_osim = path_to_osim
        self.path_to_hdf5 = path_to_hdf5
        self.augment = augment
        self.upperarm_length = upperarm_length
        self.augment_trajectory = augment_trajectory
        self.max_jerk = max_jerk
        self.max_tracing_error = max_tracing_error
        self.num_clips = num_clips
        self.dataset_type = dataset_type
        self.dataset_dir = dataset_dir
        self.num_div = num_div
        self.div_length = clip_len * increase_fps_by_a_factor_of // num_div
        self.target_len = target_len
        self.start_indx = start_indx
        self.end_indx = end_indx

        # these are fixed values (number of possible labels and number of keypoints per sample)
        self.CLASS_NUM = 60
        self.KEYPOINT_NUM = 25
        self.CLIP_LEN = clip_len * increase_fps_by_a_factor_of

        try:
            with open(self.path_to_hdf5 + "config.pkl", "rb") as myfile:
                self.dataset_len = pickle.load(myfile)[0]
        except:
            pass

    def __getitem__(self, index):
        return (
            ToTensor.transform(self.dataset[index])["keypoint"],
            ToTensor.transform(self.dataset[index])["label"],
        )

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.CLASS_NUM

    def extract_flag3d_key_points(self):
        """
        reads the FLAG3D keypoints and reformats them to the appropriate hdf5 file format for this project
        """

        # set up all the transforms
        self.GenSkeFeat = GenSkeFeat()  # (1, 1380, 25, 3)
        self.UniformSampleFrames = UniformSampleFrames(
            self.CLIP_LEN, self.num_clips, test_mode=self.dataset_type == "test"
        )
        self.PoseDecode = PoseDecode()  # (1, 500, 25, 3)
        self.FormatGCNInput = FormatGCNInput()  # (1, 1, 500, 25, 3)
        self.Collect = Collect()  # (1, 1, 500, 25, 3)
        self.ToTensor = ToTensor()  # (1, 1, 500, 25, 3)

        # load the raw data
        with open(self.dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset)

        # split if needed
        if self.dataset_type == "test":
            self.dataset_len = len(self.dataset["split"]["val"])
            self.dataset = self.dataset["annotations"][
                len(self.dataset["split"]["train"]) :
            ]
        elif self.dataset_type == "train":
            self.dataset_len = len(self.dataset["split"]["train"])
            self.dataset = self.dataset["annotations"][: self.dataset_len]
        elif self.dataset_type == "all":
            self.dataset_len = len(self.dataset["split"]["train"]) + len(
                self.dataset["split"]["val"]
            )
            self.dataset = self.dataset["annotations"]

        #### added by florian ####
        if not os.path.exists(self.path_to_hdf5):
            os.makedirs(self.path_to_hdf5)

        # save critical parameters
        with open(self.path_to_hdf5 + "config.pkl", "wb") as myfile:
            pickle.dump((len(self), self.augment, self.CLIP_LEN), myfile)

        # convert the raw data to the proper format
        for i in range(self.dataset_len):

            print(f"\r{i+1}/{self.dataset_len}", end="")

            self.dataset[i] = self.GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = self.PoseDecode.transform(self.dataset[i])
            self.dataset[i] = self.FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = self.Collect.transform(self.dataset[i])
            self.dataset[i] = self.ToTensor.transform(self.dataset[i])

        print()
        print("Dataset loaded")

    def save_key_points(self):

        # go through each datapoint
        for extra in range(self.augment):  # 1):#
            for i in range(len(self)):  # 1):#
                for num in range(self.num_div):

                    print(
                        f"\r{extra*len(self)+1+i*self.num_div+num}/{(self.augment)*len(self)*self.num_div}",
                        end="",
                    )

                    # temporary arrays for data storage
                    end_idx = (
                        (num + 1) * self.div_length
                        if (num + 1) * self.div_length < self.CLIP_LEN
                        else self.CLIP_LEN
                    )
                    storage_length = (
                        self.div_length
                        if (num + 1) * self.div_length < self.CLIP_LEN
                        else self.CLIP_LEN - num * self.div_length
                    )

                    # extract the key data
                    label = self.dataset[i]["label"]
                    shoulder_elbow_wrist = (
                        self.dataset[i]["keypoint"][0, 0]
                        .permute(1, 0, 2)[8:11]
                        .cpu()
                        .detach()
                        .numpy()[:, num * self.div_length : end_idx]
                    )

                    # center the arm
                    for j in range(storage_length):
                        shoulder_elbow_wrist[:, j] -= shoulder_elbow_wrist[0, j]

                    # rotate it
                    shoulder_elbow_wrist = shoulder_elbow_wrist.dot(rotz(90))

                    # scale it
                    norm_upper_arm_length = np.sqrt(
                        (shoulder_elbow_wrist[1, 0, 0] - shoulder_elbow_wrist[0, 0, 0])
                        ** 2
                        + (
                            shoulder_elbow_wrist[1, 0, 1]
                            - shoulder_elbow_wrist[0, 0, 1]
                        )
                        ** 2
                        + (
                            shoulder_elbow_wrist[1, 0, 2]
                            - shoulder_elbow_wrist[0, 0, 2]
                        )
                        ** 2
                    )
                    shoulder_elbow_wrist *= self.upperarm_length / norm_upper_arm_length

                    # change the speed of the motion
                    if self.augment_trajectory:
                        shoulder_elbow_wrist = augment_traj(
                            shoulder_elbow_wrist,
                            target_len=self.target_len,
                            speed_range=(1.0, 1.0),
                        )

                    # Slow down the motion 0.5x

                    # save the key points
                    endeffector_coords = shoulder_elbow_wrist[2].transpose((1, 0))
                    elbow_coords = shoulder_elbow_wrist[1].transpose((1, 0))
                    shoulder_coords = shoulder_elbow_wrist[0].transpose((1, 0))

                    # save all information in a temporary file
                    with h5py.File(
                        self.path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "w"
                    ) as myfile:
                        myfile.create_dataset("label", data=label)
                        myfile.create_dataset(
                            "endeffector_coords_flag3d", data=endeffector_coords
                        )
                        myfile.create_dataset("elbow_coords_flag3d", data=elbow_coords)
                        myfile.create_dataset("shoulder_coords", data=shoulder_coords)

        print()
        print("Coordinates extracted")

    def compute_joint_angles(self):
        """
        computes the joint angles after ```extract_flag3d_key_points``` has been run
        saves them in the same file
        """

        # skipped = []
        # go through each datapoint
        for extra in range(self.augment):  # 1):#
            for i in range(self.start_indx, self.end_indx):  # 1):#
                for num in range(self.num_div):

                    print(
                        f"\r{extra*len(self)+1+i*self.num_div+num}/{(self.augment)*len(self)*self.num_div}",
                        end="",
                    )

                    with h5py.File(
                        self.path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "r"
                    ) as myfile:
                        endeffector_coords = myfile["endeffector_coords_flag3d"][
                            ()
                        ].transpose((1, 0))
                        elbow_coords = myfile["elbow_coords_flag3d"][()].transpose(
                            (1, 0)
                        )
                        shoulder_coords = myfile["shoulder_coords"][()].transpose(
                            (1, 0)
                        )

                    # compute the critical data points
                    joint_coords, _, new_coords = convert_to_joint_angles(
                        np.stack(
                            (shoulder_coords, elbow_coords, endeffector_coords), axis=0
                        )
                    )
                    # if compute_jerk(joint_coords) > self.max_jerk or check_angles(joint_coords, clip=False):
                    #     skipped.append([extra, i])
                    #     continue
                    print(f"Jerk: {compute_jerk(new_coords)}")

                    # save all information in a temporary file
                    with h5py.File(
                        self.path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "a"
                    ) as myfile:
                        myfile.create_dataset("joint_coords", data=joint_coords)
                        myfile.create_dataset(
                            "endeffector_coords", data=new_coords[2].transpose((1, 0))
                        )
                        myfile.create_dataset(
                            "elbow_coords", data=new_coords[1].transpose((1, 0))
                        )

        # # save critical parameters
        # with open(self.path_to_hdf5+"skipped.pkl", "wb") as myfile:
        #     pickle.dump(skipped, myfile)

        print()
        print("Joint angles computed")

    def compute_muscle_kin(self):
        """
        computes the muscle lengths after the ```compute_joint_angles``` has been run
        saves them along with new joint coordinates in the same file
        """

        # with open(self.path_to_hdf5+"skipped.pkl", "rb") as myfile:
        #     skipped = pickle.load(myfile)

        # load the open sim model
        osim_model = osim.Model(self.path_to_osim)

        # go through each datapoint
        for extra in range(self.augment):  # 1):#
            for i in range(self.start_indx, self.end_indx):  # 1):#
                for num in range(self.num_div):
                    # if i <4000: continue

                    print(
                        f"\r{extra*len(self)+1+i*self.num_div+num}/{(self.augment)*len(self)*self.num_div}",
                        end="",
                    )

                    # if [extra, i] in skipped: continue

                    with h5py.File(
                        self.path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "r"
                    ) as myfile:
                        joint_coords = myfile["joint_coords"][()]

                    muscle_lengths, wrist, elbow = convert_to_muscle_lengths(
                        osim_model, joint_coords
                    )

                    # if compute_jerk(muscle_lengths) > self.max_jerk:
                    #     skipped.append([extra, i])
                    #     continue

                    # save all information in a temporary file
                    with h5py.File(
                        self.path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "a"
                    ) as myfile:
                        if "muscle_lengths" in myfile.keys():
                            del myfile["muscle_lengths"]
                            del myfile["endeffector_coords_opensim"]
                            del myfile["elbow_coords_opensim"]
                        myfile.create_dataset("muscle_lengths", data=muscle_lengths)
                        myfile.create_dataset(
                            "endeffector_coords_opensim", data=wrist.transpose((1, 0))
                        )
                        myfile.create_dataset(
                            "elbow_coords_opensim", data=elbow.transpose((1, 0))
                        )

        # save critical parameters
        # with open(self.path_to_hdf5+"skipped.pkl", "wb") as myfile:
        #     pickle.dump(skipped, myfile)

        print()
        print("muscle kinematics computed")

    def visualize_exercise(
        self,
        idx=0,
        path_to_save=None,
        is_static=True,
        show_plt=True,
        dim=1,
        use_labels=False,
    ):
        """
        plots one sample from the dataset for visualization

        Arguments
            idx: 0 <= int < 7200, which sample to plot
            path_to_save: string file pathto save the figure (make sure .png) or animation (make sure .gif), None will not save
            is_static: bool, whether to animate or plot a static image
            show_plt: bool, whether to show the plot
            dim: {1, 2}, for colour scheme
        """

        # set up the 3d plot to full screen
        fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
        fig.set_size_inches(23, 13)

        # plot the index number of the key points
        if use_labels:
            labels = []
            for j in range(25):
                labels.append(
                    ax.text(
                        self.dataset[idx]["keypoint"][0, 0, 0, j, 0],
                        self.dataset[idx]["keypoint"][0, 0, 0, j, 1],
                        self.dataset[idx]["keypoint"][0, 0, 0, j, 2],
                        j,
                    )
                )

        # for .png
        if is_static:

            # plot the keypoints and change the colour across both dimensions
            if dim == 2:
                for i in range(self.CLIP_LEN):
                    for j in range(25):
                        ax.scatter(
                            self.dataset[idx]["keypoint"][0, 0, i, j, 0],
                            self.dataset[idx]["keypoint"][0, 0, i, j, 1],
                            self.dataset[idx]["keypoint"][0, 0, i, j, 2],
                            color=((j + 1) / 25, 0, (i + 1) / self.CLIP_LEN),
                        )

            # plot the keypoints and change the colour across only the time dimension
            elif dim == 1:
                for i in range(self.CLIP_LEN):
                    ax.scatter(
                        self.dataset[idx]["keypoint"][0, 0, i, :, 0],
                        self.dataset[idx]["keypoint"][0, 0, i, :, 1],
                        self.dataset[idx]["keypoint"][0, 0, i, :, 2],
                        color=(0, 0, (i + 1) / self.CLIP_LEN),
                    )

            # save if needed
            if path_to_save is not None:
                plt.savefig(path_to_save)

        # for the animation .gif
        else:

            # set up the plot
            (line1,) = ax.plot(
                [], [], [], color="blue", linestyle="", marker="o", markersize=10
            )

            def animation(i):

                # update the points
                line1.set_data_3d(
                    np.array(self.dataset[idx]["keypoint"][0, 0, i, :, 0]),
                    np.array(self.dataset[idx]["keypoint"][0, 0, i, :, 1]),
                    np.array(self.dataset[idx]["keypoint"][0, 0, i, :, 2]),
                )

                # update the label locations
                if use_labels:
                    for j in range(25):
                        labels[j].set_position(
                            (
                                self.dataset[idx]["keypoint"][0, 0, i, j, 0],
                                self.dataset[idx]["keypoint"][0, 0, i, j, 1],
                                self.dataset[idx]["keypoint"][0, 0, i, j, 2],
                            )
                        )

            # finalize the plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=30, azim=140)

            # animate
            anim = ani.FuncAnimation(
                fig, animation, repeat=True, frames=self.CLIP_LEN, interval=1000
            )

            # save if needed
            if path_to_save is not None:
                anim.save(path_to_save, writer=ani.PillowWriter(fps=120))

        if show_plt:
            plt.show()


def augment_traj(trajectory, target_len, speed_range=[0.8, 1.5]):
    """
    randomize the length of the exercise motion and the initial start point

    Arguments
        trajectory: numpy array of shape (num_joints=3, time_steps=320, coordinates=3) with the
            trajectories of the relevant joints in cartesian coordinates
        speed_range: list of two ints, the first being the lower bound and the second being the upper bound
        target_len: int, target length to pad the beginning and end of the current traj. Make sure that the
            lower bound of speed will not make the new trajectory greater than this number. For example,
            if the traj is 920 time steps long and the target_len is 1152, then the lowest speed should be about
            0.8 (920/0.8=1150).

    Return
        new_traj: same as the input trajectory but at a different speed
    """

    # find a random speed
    speed = np.random.uniform(speed_range[0], speed_range[1])

    # change speed of movement
    true_timestamps = np.arange(trajectory.shape[1])
    n_timestamps_new = int(true_timestamps.size / speed)
    new_timestamps = np.linspace(0, true_timestamps[-1], n_timestamps_new)
    func = interp1d(true_timestamps, trajectory, axis=1)
    new_traj = func(new_timestamps)
    new_len = new_traj.shape[1]

    # if the new length is smaller than the original length
    if new_len < target_len:

        # find a new start point
        start_pt = np.random.randint(0, target_len - new_len)

        # if the new start point is not at t=0, pad the beginning
        if start_pt != 0:
            beginning = np.ones((3, start_pt, 3))
            beginning[0] *= trajectory[0, 0]
            beginning[1] *= trajectory[1, 0]
            beginning[2] *= trajectory[2, 0]
            new_traj = np.concatenate((beginning, new_traj), axis=1)

        # if there is still space left to fill up until the end of the motion, pad the ending
        if new_traj.shape[1] < target_len:
            ending = np.ones((3, target_len - new_traj.shape[1], 3))
            ending[0] *= trajectory[0, -1]
            ending[1] *= trajectory[1, -1]
            ending[2] *= trajectory[2, -1]
            new_traj = np.concatenate((new_traj, ending), axis=1)

    return new_traj


def convert_to_joint_angles(trajectories):
    """
    converts the shoulder, elbow, and wrist trajectories to joint angles

    Arguments
        trajectories: numpy array of shape (num_joints=3, time_steps=320, coordinates=3) with the
            trajectories of the relevant joints in cartesian coordinates

    Returns
        joint_trajectory: numpy array of shape (4, time_steps=320) with the joint angles of
            'elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion' in degrees
        error: float, distance between the real end effector location and that calcualted
            by the inverse kinematics
        coords: the coordinates computed by the forward kinematics of shoulder, elbow, and wrist
    """

    myarm = Arm()
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Inverse Kinematics to obtain joint configurations for the character trajectory
    traj_length = trajectories.shape[1]
    joint_trajectory = np.zeros((4, traj_length))

    # For each point in the trajectory derive the joint angle configuration
    # After finding the joint configuration for a particular point, change q0
    # to the current joint configuration!
    # error = 0.
    # coords = np.zeros((3,traj_length,3))
    # for i in range(traj_length):
    #     dest_elbow = (shoulder_to_world.T).dot(trajectories[1,i])
    #     dest_wrist = (shoulder_to_world.T).dot(trajectories[2,i])
    #     joint_config = myarm.inv_kin(elbow=dest_elbow, hand=dest_wrist)
    #     myarm.q0 = joint_config
    #     joint_trajectory[:, i] = joint_config
    #     error += np.linalg.norm(dest_wrist - coords[0])

    ## Florian ##
    from joblib import Parallel, delayed

    # Precompute transposed transformation matrix
    shoulder_to_world_T = shoulder_to_world.T
    coords = np.zeros((3, traj_length, 3))
    joint_trajectory = np.zeros((myarm.q0.size, traj_length))

    # Precompute all destination coordinates
    dest_elbows = shoulder_to_world_T @ trajectories[1, :, :].T
    dest_wrists = shoulder_to_world_T @ trajectories[2, :, :].T

    # Define a function to process each trajectory point
    def compute_trajectory_point(i):
        dest_elbow = dest_elbows[:, i]
        dest_wrist = dest_wrists[:, i]

        # Compute inverse kinematics for the current elbow and wrist destination
        joint_config = myarm.inv_kin(elbow=dest_elbow, hand=dest_wrist)

        # Calculate the error
        point_error = np.linalg.norm(dest_wrist - coords[0, i])

        return joint_config, point_error

    # Run computations in parallel
    results = Parallel(n_jobs=19)(
        delayed(compute_trajectory_point)(i) for i in range(traj_length)
    )

    # Unpack results back into arrays
    error = sum(point_error for _, point_error in results)
    for i, (joint_config, _) in enumerate(results):
        joint_trajectory[:, i] = joint_config

    # with open("/media/data4/sebastian/FLAG3D_cleaned/temp/file.pkl", 'wb') as myfile:
    #     pickle.dump(joint_trajectory, myfile)

    # with open("/media/data4/sebastian/FLAG3D_cleaned/temp/file.pkl", 'rb') as myfile:
    #     joint_trajectory = pickle.load(myfile)

    joint_trajectory_2 = check_angles(
        savgol_filter(
            np.stack(
                (
                    median_filter(joint_trajectory[0], 5),
                    median_filter(joint_trajectory[1], 5),
                    median_filter(joint_trajectory[2], 5),
                    median_filter(joint_trajectory[3], 5),
                ),
                axis=0,
            ),
            5,
            2,
            axis=1,
        )
    )
    for i in range(traj_length):
        coords[:, i] = shoulder_to_world.dot(
            myarm.get_xyz(joint_trajectory_2[:, i])[1]
        ).T

    return joint_trajectory_2, error, coords


def check_angles(angles, clip=True):
    """
    clamps the angles according to opensim constraints

    Arguments
        angles: numpy array of shape (4, time_steps) containing the
            joint angles in the correct order
        clip: bool, if True will clip the joint angles in the correct range,
                    if False will only indicate if the anlges should be clipped

    Return
        angles: clipped angles if clip else bool of whether they should be clipped
    """

    shoulder_elv_angle = [-95, 130]
    shoulder_shoulder_elv = [0, 180]
    shoulder_shoulder_rot = [-90, 120]
    elbow_flexion = [0, 130]

    if clip:
        angles[0] = np.clip(angles[0], shoulder_elv_angle[0], shoulder_elv_angle[1])
        angles[1] = np.clip(
            angles[1], shoulder_shoulder_elv[0], shoulder_shoulder_elv[1]
        )
        angles[2] = np.clip(
            angles[2], shoulder_shoulder_rot[0], shoulder_shoulder_rot[1]
        )
        angles[3] = np.clip(angles[3], elbow_flexion[0], elbow_flexion[1])
        return angles
    else:
        if (
            angles[0] < shoulder_elv_angle[0]
            or angles[0] > shoulder_elv_angle[1]
            or angles[1] < shoulder_shoulder_elv[0]
            or angles[1] > shoulder_shoulder_elv[1]
            or angles[2] < shoulder_shoulder_rot[0]
            or angles[2] > shoulder_shoulder_rot[1]
            or angles[3] < elbow_flexion[0]
            or angles[3] > elbow_flexion[1]
        ):
            return True
        else:
            return False


def convert_to_muscle_lengths(model, joint_trajectory):
    """
    Compute muscle configurations of a given opensim model to given coordinate (joint angle) trajectories.

    Arguments
        model: opensim model object, the MoBL Dynamic Arm.
        joint_trajectory: np.array, shape=(4, T) joint angles at each of T time points

    Returns
        muscle_length_configurations: np.array, shape=(25, T) muscle lengths at each of T time points
    """

    # initialize the open sim model
    init_state = model.initSystem()
    model.equilibrateMuscles(init_state)

    # Prepare for simulation
    num_coords, num_timepoints = joint_trajectory.shape
    muscle_set = model.getMuscles()  # returns a Set<Muscles> object
    num_muscles = muscle_set.getSize()

    # Set the order of the coordinates
    coord_order = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]

    # get the markers
    names_marker_osim = []
    if model.getMarkerSet().getSize() == 0:
        print("Need to add markers to model")
    for i in range(model.getMarkerSet().getSize()):
        names_marker_osim.append(model.getMarkerSet().get(i).getName())

    # for simplicity
    ind_shoulder = names_marker_osim.index("Shoulder")
    ind_elbow = names_marker_osim.index("Elbow")
    ind_wrist = names_marker_osim.index("Wrist")

    # For each time step of the trajectory, compute equibrium muscle states
    # Create a dictionary of muscle configurations
    muscle_config = {}
    for i in range(num_muscles):
        muscle_config[muscle_set.get(i).getName()] = np.zeros(num_timepoints)

    wrist = np.zeros((num_timepoints, 3))
    elbow = np.zeros((num_timepoints, 3))
    shoulder = np.zeros((num_timepoints, 3))

    # perform the simulation
    for timepoint in range(num_timepoints):

        # compute the muscle lengths
        for i in range(num_coords):
            # model.getCoordinateSet().get(coord_order[i]).setLocked(init_state, False)
            model.getCoordinateSet().get(coord_order[i]).setValue(
                init_state, np.pi * (joint_trajectory[i, timepoint] / 180)
            )

        model.equilibrateMuscles(init_state)

        # get the muscle lengths and store them appropriately
        for muscle_num in range(num_muscles):
            muscle = muscle_set.get(muscle_num)
            name = muscle.getName()
            muscle_config[name][timepoint] = (
                muscle.getFiberLength(init_state) * 1000
            )  # change to mm

            # fix issue with ECRL
            # if name == "ECRL":
            #     if muscle_config[name][timepoint] > 0.3 * 1000:
            #         muscle_config[name][timepoint] -= ECRL_CORRECTION
            #     elif muscle_config[name][timepoint] > 0.2 * 1000:
            #         muscle_config[name][timepoint] -= ECRL_CORRECTION * 0.47

        wrist[timepoint:,] = np.array(
            [
                model.getMarkerSet().get(ind_wrist).getLocationInGround(init_state)[i]
                for i in range(3)
            ]
        )
        elbow[timepoint:,] = np.array(
            [
                model.getMarkerSet().get(ind_elbow).getLocationInGround(init_state)[i]
                for i in range(3)
            ]
        )
        shoulder[timepoint:,] = np.array(
            [
                model.getMarkerSet()
                .get(ind_shoulder)
                .getLocationInGround(init_state)[i]
                for i in range(3)
            ]
        )

    # Delete muscles that are not of interest to us
    shoulder_muscles = [
        "CORB",
        "DELT1",
        "DELT2",
        "DELT3",
        "INFSP",
        "LAT1",
        "LAT2",
        "LAT3",
        "PECM1",
        "PECM2",
        "PECM3",
        "SUBSC",
        "SUPSP",
        "TMAJ",
        "TMIN",
    ]
    elbow_muscles = [
        "ANC",
        "BIClong",
        "BICshort",
        "BRA",
        "BRD",
        "ECRL",
        "PT",
        "TRIlat",
        "TRIlong",
        "TRImed",
    ]
    # muscle_config_delete_later = {}
    for i in list(muscle_config.keys()):
        if not (i in shoulder_muscles or i in elbow_muscles):
            del muscle_config[i]
        else:
            muscle_config[i] = savgol_filter(median_filter(muscle_config[i], 7), 5, 2)
            # muscle_config_delete_later[i] = muscle_config[i]
            # diff1 = 0
            # diff2 = 0
            # for time in range(num_timepoints):
            #     # if time == 164 and i == 'LAT3': breakpoint()
            #     if time < num_timepoints - 1:
            #         if muscle_config[i][time] * 1.3 < muscle_config[i][time+1]:
            #             if diff1 != 0: diff1 = muscle_config[i][time+1] - muscle_config[i][time]
            #             muscle_config[i][time+1] -= diff1
            #         elif muscle_config[i][time+1] * 1.3 < muscle_config[i][time]:
            #             if diff2 != 0: diff1 = muscle_config[i][time] - muscle_config[i][time+1]
            #             muscle_config[i][time] -= diff2

    # convert to a numpy array
    muscle_length_configurations = make_muscle_matrix(muscle_config)
    # muscle_length_configurations_delete_later = make_muscle_matrix(muscle_config_delete_later)

    # recenter and scale
    wrist -= shoulder
    elbow -= shoulder
    shoulder -= shoulder
    wrist = rotz(180).dot(transform().dot(wrist.T)).T * 116.6
    elbow = rotz(180).dot(transform().dot(elbow.T)).T * 116.6

    # # plot to ensure in correct position
    # fig, ax = plt.subplots(1)
    # for muscle in muscle_length_configurations:
    #     ax.plot(muscle)
    # fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
    # fig.set_size_inches(23,13)
    # line1, = ax.plot([],[],[], color="blue", linewidth=10,label="opensim")
    # def animation(i):
    #     line1.set_data_3d(np.stack((wrist[i,0],elbow[i,0],shoulder[i,0])),
    #                       np.stack((wrist[i,1],elbow[i,1],shoulder[i,1])),
    #                       np.stack((wrist[i,2],elbow[i,2],shoulder[i,2])))
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)
    # ax.set_zlim(-50, 50)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=30, azim=140)
    # ax.legend()
    # anim = ani.FuncAnimation(fig, animation, repeat=True, frames=320, interval=1)
    # fig1, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax14),(ax8,ax9,ax10,ax15),(ax11,ax12,ax13,ax16)) = plt.subplots(4,4)
    # fig1.set_size_inches(23,13)
    # ax1.plot(joint_trajectory[0], "red",label='shoulder_elv_angle, raw')
    # ax2.plot(joint_trajectory[1], "blue",label='shoulder_shoulder_elv, raw')
    # ax3.plot(joint_trajectory[2], "green",label='shoulder_shoulder_rot, raw')
    # ax4.plot(joint_trajectory[3], "brown",label='elbow_flexion, raw')
    # ax5.plot(shoulder[:,0], "red",label="x-coord, opensim")
    # ax6.plot(shoulder[:,1], "blue",label="y-coord, opensim")
    # ax7.plot(shoulder[:,2], "green",label="z-coord, opensim")
    # ax8.plot(elbow[:,0], "red",label="x-coord, opensim")
    # ax9.plot(elbow[:,1], "blue",label="y-coord, opensim")
    # ax10.plot(elbow[:,2], "green",label="z-coord, opensim")
    # ax11.plot(wrist[:,0], "red",label="x-coord, opensim")
    # ax12.plot(wrist[:,1], "blue",label="y-coord, opensim")
    # ax13.plot(wrist[:,2], "green",label="z-coord, opensim")
    # # ax5.plot(coords[2,:,0], "pink",label="x-coord, forward_kin")
    # # ax6.plot(coords[2,:,1], "cyan",label="y-coord, forward_kin")
    # # ax7.plot(coords[2,:,2], "lime",label="z-coord, forward_kin")
    # # ax8.plot(coords[1,:,0], "pink",label="x-coord, forward_kin")
    # # ax9.plot(coords[1,:,1], "cyan",label="y-coord, forward_kin")
    # # ax10.plot(coords[1,:,2], "lime",label="z-coord, forward_kin")
    # # ax11.plot(coords[0,:,0], "pink",label="x-coord, forward_kin")
    # # ax12.plot(coords[0,:,1], "cyan",label="y-coord, forward_kin")
    # # ax13.plot(coords[0,:,2], "lime",label="z-coord, forward_kin")
    # ax1.legend()
    # ax1.set_title("joint_angle (deg) shoulder_elv_angle")
    # ax2.legend()
    # ax2.set_title("joint_angle (deg) shoulder_shoulder_elv")
    # ax3.legend()
    # ax3.set_title("joint_angle (deg) shoulder_shoulder_rot")
    # ax4.legend()
    # ax4.set_title("joint_angle (deg) elbow_flexion")
    # ax5.legend()
    # ax5.set_title("shoulder - x")
    # ax6.legend()
    # ax6.set_title("shoulder - y")
    # ax7.legend()
    # ax7.set_title("shoulder - z")
    # ax8.legend()
    # ax8.set_title("elbow - x")
    # ax9.legend()
    # ax9.set_title("elbow - y")
    # ax10.legend()
    # ax10.set_title("elbow - z")
    # ax11.legend()
    # ax11.set_title("wrist - x")
    # ax12.legend()
    # ax12.set_title("wrist - y")
    # ax13.legend()
    # ax13.set_title("wrist - z")
    # fig2, axs = plt.subplots(5,5)
    # fig2.set_size_inches(23,13)
    # order = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1',
    #         'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 'BICshort',
    #         'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']
    # for i in range(5):
    #     for j in range(5):
    #         axs[i][j].plot(muscle_length_configurations_delete_later[i*5+j],label="raw")
    #         axs[i][j].plot(muscle_length_configurations[i*5+j], label="filtered")
    #         axs[i][j].set_title(order[i*5+j])
    #         axs[i][j].legend()
    # fig2.tight_layout()
    # plt.show()

    return muscle_length_configurations, wrist, elbow


def make_muscle_matrix(muscle_config):
    # Arrange muscles configurations in a 25xT matrix, given a dictionary of muscle configurations.
    order = [
        "CORB",
        "DELT1",
        "DELT2",
        "DELT3",
        "INFSP",
        "LAT1",
        "LAT2",
        "LAT3",
        "PECM1",
        "PECM2",
        "PECM3",
        "SUBSC",
        "SUPSP",
        "TMAJ",
        "TMIN",
        "ANC",
        "BIClong",
        "BICshort",
        "BRA",
        "BRD",
        "ECRL",
        "PT",
        "TRIlat",
        "TRIlong",
        "TRImed",
    ]
    mconf = [muscle_config[i] for i in order]
    return np.asarray(mconf)


def combine_data_set(path_to_hdf5, start, end):
    """
    once the appropriate functions in ```BuildDatasetForFLAG3D``` has been run in the indicated order,
    this function will group all the data into one hdf5 file

    Arguments
        path_to_hdf5: string folder path to the data
    """

    # retreive the key information
    with open(path_to_hdf5 + "config.pkl", "rb") as myfile:
        len_self, augment, frames = pickle.load(myfile)

    # create the data arrays
    # label = np.zeros(len_self*augment)
    # endeffector_coords = np.zeros((len_self*augment*(3), 3, frames))
    # elbow_coords = np.zeros((len_self*augment*(3), 3, frames))
    # shoulder_coords = np.zeros((len_self*augment*(3), 3, frames))
    # joint_coords = np.zeros((len_self*augment*(3), 4, frames))
    # muscle_lengths = np.zeros((len_self*augment*(3), 25, frames))

    label = []
    endeffector_coords = []
    elbow_coords = []
    shoulder_coords = []
    joint_coords = []
    muscle_lengths = []

    # go through each file in the path ```path_to_hdf5```
    for extra in range(augment):  # 1):#
        for i in range(start, end):  # 1):#
            for num in range(4):

                # print progress
                # print(f"\r{(num+i*(3)+1+len_self*extra)}/{len_self*augment*(3)}", end="")

                # extract and save the key datapoints
                with h5py.File(path_to_hdf5 + f"{extra}_{i}_{num}.hdf5", "r") as myfile:
                    if "muscle_lengths" not in myfile.keys():
                        print(f"Skipping {extra}_{i}_{num}.hdf5")
                        continue
                    print(f"Reading {extra}_{i}_{num}.hdf5")
                    label.append(myfile["label"][()])
                    endeffector_coords.append(myfile["endeffector_coords"][()])
                    elbow_coords.append(myfile["elbow_coords"][()])
                    shoulder_coords.append(myfile["shoulder_coords"][()])
                    joint_coords.append(myfile["joint_coords"][()])
                    muscle_lengths.append(myfile["muscle_lengths"][()])

    label = np.array(label)
    endeffector_coords = np.array(endeffector_coords)
    elbow_coords = np.array(elbow_coords)
    shoulder_coords = np.array(shoulder_coords)
    joint_coords = np.array(joint_coords)
    muscle_lengths = np.array(muscle_lengths)

    # compute the velocities
    vel_inputs = np.gradient(muscle_lengths, 1 / 60, axis=-1)
    print(f"Max vel: {np.max(vel_inputs)}")
    spindle_info = np.stack((muscle_lengths, vel_inputs), axis=-1)

    # save as one large file
    with h5py.File(path_to_hdf5 + "all.hdf5", "w") as myfile:
        myfile.create_dataset("label", data=label)
        myfile.create_dataset("endeffector_coords", data=endeffector_coords)
        myfile.create_dataset("elbow_coords", data=elbow_coords)
        myfile.create_dataset("shoulder_coords", data=shoulder_coords)
        myfile.create_dataset("joint_coords", data=joint_coords)
        myfile.create_dataset("muscle_lengths", data=muscle_lengths)
        myfile.create_dataset("spindle_info", data=spindle_info)


def fix_muscles(folderpath, path_to_osim, state="r"):

    with h5py.File(folderpath + "all.hdf5", "r") as myfile:
        joint_coords = myfile["joint_coords"][()]

    if state == "w":
        osim_model = osim.Model(path_to_osim)
        for i in range(7200):
            with h5py.File(folderpath + f"fix_muscle_{i}.hdf5", "w") as myfile:
                myfile.create_dataset(
                    "muscle_lengths",
                    data=convert_to_muscle_lengths(osim_model, joint_coords[i]),
                )

    elif state == "r":
        muscle_lengths = np.zeros((joint_coords.shape[0], 25, joint_coords.shape[2]))
        for i in range(joint_coords.shape[0]):
            with h5py.File(folderpath + f"fix_muscle_{i}.hdf5", "r") as myfile:
                muscle_lengths[i] = myfile["muscle_lengths"][()]
        vel_inputs = np.gradient(muscle_lengths, 0.015, axis=-1)
        spindle_info = np.stack((muscle_lengths, vel_inputs), axis=-1)
        with h5py.File(folderpath + "all.hdf5", "a") as myfile:
            del myfile["muscle_lengths"]
            del myfile["spindle_info"]
            myfile.create_dataset("muscle_lengths", data=muscle_lengths)
            myfile.create_dataset("spindle_info", data=spindle_info)


def split_data_set(folderpath):
    """
    splits the combined dataset after running ```combine_data_set```
    into training-validation and testing portions
    """

    # load the data
    with h5py.File(folderpath + "flag3d_raw.hdf5", "r") as myfile:
        label = myfile["label"][()]
        endeffector_coords = myfile["endeffector_coords"][()]
        elbow_coords = myfile["elbow_coords"][()]
        shoulder_coords = myfile["shoulder_coords"][()]
        joint_coords = myfile["joint_coords"][()]
        muscle_lengths = myfile["muscle_lengths"][()]
        spindle_info = myfile["spindle_info"][()]

    total_length = len(label)

    # shuffle the data
    shuffle_idx = np.random.permutation(total_length)
    label = label[shuffle_idx]
    endeffector_coords = endeffector_coords[shuffle_idx]
    elbow_coords = elbow_coords[shuffle_idx]
    shoulder_coords = shoulder_coords[shuffle_idx]
    joint_coords = joint_coords[shuffle_idx]
    muscle_lengths = muscle_lengths[shuffle_idx]
    spindle_info = spindle_info[shuffle_idx]

    # split and save the datasets
    TRAIN_VAL = 0.8  # 72+8 - testing is 20
    train_split_idx = int(total_length * TRAIN_VAL)
    with h5py.File(folderpath + "flag3d_raw_train.hdf5", "w") as myfile:
        myfile.create_dataset("label", data=label[:train_split_idx])
        myfile.create_dataset(
            "endeffector_coords", data=endeffector_coords[:train_split_idx]
        )
        myfile.create_dataset("elbow_coords", data=elbow_coords[:train_split_idx])
        myfile.create_dataset("shoulder_coords", data=shoulder_coords[:train_split_idx])
        myfile.create_dataset("joint_coords", data=joint_coords[:train_split_idx])
        myfile.create_dataset("muscle_lengths", data=muscle_lengths[:train_split_idx])
        myfile.create_dataset("spindle_info", data=spindle_info[:train_split_idx])
    with h5py.File(folderpath + "flag3d_raw_test.hdf5", "w") as myfile:
        myfile.create_dataset("label", data=label)
        myfile.create_dataset(
            "endeffector_coords", data=endeffector_coords[train_split_idx:]
        )
        myfile.create_dataset("elbow_coords", data=elbow_coords[train_split_idx:])
        myfile.create_dataset("shoulder_coords", data=shoulder_coords[train_split_idx:])
        myfile.create_dataset("joint_coords", data=joint_coords[train_split_idx:])
        myfile.create_dataset("muscle_lengths", data=muscle_lengths[train_split_idx:])
        myfile.create_dataset("spindle_info", data=spindle_info[train_split_idx:])


def compute_jerk(joint_trajectory, sampling_rate=120):
    """Compute the jerk in joint space for the obtained joint configurations.

    Returns
    -------
    jerk : np.array, [T,] array of jerk for a given trajectory

    """
    joint_vel = np.gradient(joint_trajectory, 1 / sampling_rate, axis=1)
    joint_acc = np.gradient(joint_vel, 1, axis=1)
    joint_jerk = np.gradient(joint_acc, 1, axis=1)
    jerk = np.linalg.norm(joint_jerk, axis=0)
    return np.max(jerk)
