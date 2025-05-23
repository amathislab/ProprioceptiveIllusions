import argparse
import numpy as np
import h5py

def check_velocity(input_path, output_path):
    with h5py.File(input_path, 'r') as f:
        spindle_info = f['spindle_info'][()]
        velocity = spindle_info[:, :, :, 1]

    remove_indices = []
        
    # Check if the velocity is within the specified range
    for i in range(velocity.shape[0]):
        if np.max(np.abs(velocity[i])) > 1300:
            print(f"Skipped! Max muscle velocity: {np.max(np.abs(velocity[i]))}")
            remove_indices.append(i)
        
    # Remove the indices from the spindle_info array
    spindle_info = np.delete(spindle_info, remove_indices, axis=0)
    print(f"Removed {len(remove_indices)} samples due to high velocity.")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spindle_info', data=spindle_info)
        
    # Add the rest of the data
    with h5py.File(input_path, 'r') as f:
        endeffector_coords = f['endeffector_coords'][()]
        joint_coords = f['joint_coords'][()]

    with h5py.File(output_path, 'a') as f:
        f.create_dataset('endeffector_coords', data=endeffector_coords)
        f.create_dataset('joint_coords', data=joint_coords)
        print(list(f.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    params = parser.parse_args()

    check_velocity(params.input_path, params.output_path)