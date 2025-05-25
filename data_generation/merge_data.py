import argparse
import h5py
import numpy as np


def merge_data(path_1, path_2, num_samples):
    endeffector_coords = []
    joint_coords = []
    spindle_info = []
    with h5py.File(path_1, 'r') as f:
        endeffector_coords.append(f['endeffector_coords'][()])
        joint_coords.append(f['joint_coords'][()])
        spindle_info.append(f['spindle_info'][()])
    with h5py.File(path_2, 'r') as f:
        endeffector_coords.append(f['endeffector_coords'][:num_samples])
        joint_coords.append(f['joint_coords'][:num_samples])
        spindle_info.append(f['spindle_info'][:num_samples])

    endeffector_coords = np.concatenate(endeffector_coords, axis=0)
    joint_coords = np.concatenate(joint_coords, axis=0)
    spindle_info = np.concatenate(spindle_info, axis=0)

    return endeffector_coords, joint_coords, spindle_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--flag3d", type=str)
    parser.add_argument("--pcr", type=str)
    parser.add_argument("--num_samples", type=int, default=50_000)

    params = parser.parse_args()

    endeffector_coords, joint_coords, spindle_info = merge_data(params.flag3d, params.pcr, params.num_samples)

    with h5py.File(params.save_path, 'w') as f:
        f.create_dataset('endeffector_coords', data=endeffector_coords)
        f.create_dataset('joint_coords', data=joint_coords)
        f.create_dataset('spindle_info', data=spindle_info)
        
    print("done!")
    