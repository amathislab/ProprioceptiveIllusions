import os
import argparse
import h5py
from data_generation.extract_flag3d_data_utils import combine_data_set, split_data_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--start_indx", type=int, default=0)
    parser.add_argument("--end_indx", type=int, default=7200)

    params = parser.parse_args()

    path_to_hdf5 = os.path.join(params.dir, "FLAG3D/")

    combine_data_set(path_to_hdf5, params.start_indx, params.end_indx)

    split_data_set(path_to_hdf5)

    with h5py.File(os.path.join(params.dir, "FLAG3D/flag3d_raw_test.hdf5")) as f:
        print(f.keys())
        print(f['spindle_info'].shape)
    
    print("done!")