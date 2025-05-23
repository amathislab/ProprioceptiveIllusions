import os
import numpy as np
import h5py
import opensim as osim
import argparse
from scipy.signal import savgol_filter
from tqdm import tqdm
from data_generation.extract_flag3d_data_utils import BuildDatasetForFLAG3D, convert_to_joint_angles, convert_to_muscle_lengths

def compute_jerk(joint_trajectory, sampling_rate=120):
    """Compute the jerk in joint space for the obtained joint configurations.

    Returns
    -------
    jerk : np.array, [T,] array of jerk for a given trajectory

    """
    joint_vel = np.gradient(joint_trajectory, 1/sampling_rate, axis=1)
    joint_acc = np.gradient(joint_vel, 1, axis=1)
    joint_jerk = np.gradient(joint_acc, 1, axis=1)
    jerk = np.linalg.norm(joint_jerk, axis=1)
    return np.max(jerk)

if __name__=="__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--start_indx", type=int, default=0)
    parser.add_argument("--end_indx", type=int, default=7200)
    params = parser.parse_args()

    path_to_hdf5 = os.path.join(params.dir, "FLAG3D/")
    os.makedirs(path_to_hdf5, exist_ok=True)

    path_to_keypoints = os.path.join(params.dir, "flag3d_keypoint.pkl")
    path_to_model = os.path.join(params.dir, 'MOBL_ARMS_41_seb_writing_pos.osim')

    osim_model = osim.Model(path_to_model)

    dataset_generator = BuildDatasetForFLAG3D(dataset_dir=path_to_keypoints, 
                                                dataset_type="all", 
                                                path_to_osim=path_to_model,
                                                path_to_hdf5=path_to_hdf5,
                                                start_indx=params.start_indx,
                                                end_indx=params.end_indx,
                                                clip_len=1152,
                                                augment_trajectory=False,
                                                num_div=4,
                                                target_len=288)
    
    dataset_generator.extract_flag3d_key_points()
    dataset_generator.save_key_points()

    accepted_files = []

    for extra in range(dataset_generator.augment):
        for i in tqdm(range(params.start_indx, params.end_indx)):
            for num in range(dataset_generator.num_div):

                print(f"\r{extra*len(dataset_generator)+1+i*dataset_generator.num_div+num}/{(dataset_generator.augment)*len(dataset_generator)*dataset_generator.num_div}")

                with h5py.File(dataset_generator.path_to_hdf5+f"{extra}_{i}_{num}.hdf5", "r") as myfile:
                    endeffector_coords = myfile["endeffector_coords_flag3d"][()].transpose((1,0))
                    elbow_coords = myfile["elbow_coords_flag3d"][()].transpose((1,0))
                    shoulder_coords = myfile["shoulder_coords"][()].transpose((1,0))

                # smooth the data
                smooth_elbow_coords = savgol_filter(elbow_coords.T, 25, 1).T
                smooth_endeffector_coords = savgol_filter(endeffector_coords.T, 25, 1).T
                joint_coords, _, new_coords = convert_to_joint_angles(np.stack((shoulder_coords, smooth_elbow_coords, smooth_endeffector_coords), axis=0))
                smooth_jerk = compute_jerk(new_coords, sampling_rate=60)
                if smooth_jerk > 700:
                    print(f"Skipped! Jerk: {smooth_jerk}")
                    continue

                muscle_lengths, wrist, elbow = convert_to_muscle_lengths(osim_model, joint_coords)
                muscle_vel = np.gradient(muscle_lengths, 1/60, axis=-1)
                if np.max(np.abs(muscle_vel)) > 1300:
                    print(f"Skipped! Max muscle velocity: {np.max(muscle_vel)}")
                    continue 

                # save all information in a temporary file
                with h5py.File(dataset_generator.path_to_hdf5+f"{extra}_{i}_{num}.hdf5", "a") as myfile:
                    myfile.create_dataset("joint_coords", data=joint_coords)
                    myfile.create_dataset("endeffector_coords", data=new_coords[2].transpose((1,0)))
                    myfile.create_dataset("elbow_coords", data=new_coords[1].transpose((1,0)))
                    myfile.create_dataset("muscle_lengths", data=muscle_lengths)
                    myfile.create_dataset("endeffector_coords_opensim", data=wrist.transpose((1,0)))
                    myfile.create_dataset("elbow_coords_opensim", data=elbow.transpose((1,0)))

                accepted_files.append(dataset_generator.path_to_hdf5+f"{extra}_{i}_{num}.hdf5")

    np.save(os.path.join(path_to_hdf5, f"accepted_files_{params.start_indx}_{params.end_indx}.npy"), accepted_files)