import argparse
import numpy as np
import h5py
from scipy.signal import savgol_filter


def smooth_data(input_path, output_path):
    with h5py.File(input_path, 'r') as f:
        spindles = f['spindle_info'][()]

    velocity = spindles[:, :, :, 1]
    acceleration = np.gradient(velocity, 1/240, axis=2)
    smoothed_velocity = savgol_filter(velocity, 31, 1, axis=2)
    del velocity
    smoothed_acceleration = np.gradient(smoothed_velocity, 1/240, axis=2)
    del acceleration
    new_spindles = np.stack([spindles[:, :, :, 0], smoothed_velocity], axis=3)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spindle_info', data=new_spindles)
        f.create_dataset('muscle_lengths', data=spindles[:, :, :, 0])
        f.create_dataset('muscle_velocities', data=smoothed_velocity)
        f.create_dataset('muscle_accelerations', data=smoothed_acceleration)

    with h5py.File(input_path, 'r') as f:
        endeffector_coords = f['endeffector_coords'][()]
        joint_coords = f['joint_coords'][()]

    with h5py.File(output_path, 'a') as f:
        f.create_dataset('endeffector_coords', data=endeffector_coords)
        f.create_dataset('joint_coords', data=joint_coords)

    with h5py.File(output_path, 'r') as f:
        print(list(f.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    params = parser.parse_args()

    smooth_data(params.input_path, params.output_path)
