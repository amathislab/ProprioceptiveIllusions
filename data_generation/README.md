# Generating the ProprioceptiveIllusions dataset

### Step 1: Process FLAG3D Data
- Download the FLAG3D data from the [FLAG3D website](https://www.flag3d.org/).
- Place the `flag3d_keypoint.pkl` file in the same `raw` directory as downloaded from Zenodo (hereby denoted as <your_directory>).
- Run the following scripts:
```bash
python -m data_generation.filter_extract_flag3d_data --dir <your_directory>
python -m data_generation.combine_flag3d_data
```
- These scripts will generate the `flag3d_raw_train.hdf5` and `flag3d_raw_test.hdf5` files in the `FLAG3D` directory.

NOTE: The above scripts require a significant amount of time to run. As an alternative, we provide the processed data in the `FLAG3D` directory for your convenience. If you choose to use the provided data, you may simply continue with the following steps.

- Upsample the data to 240Hz by running the following script:
```bash
python -m data_generation.upsample_data \
    --input_file <your_directory>/FLAG3D/flag3d_raw_train.hdf5 \
    --output_file <your_directory>/FLAG3D/flag3d_upsampled_train.hdf5
python -m data_generation.upsample_data \
    --input_file <your_directory>/FLAG3D/flag3d_raw_test.hdf5 \
    --output_file <your_directory>/FLAG3D/flag3d_upsampled_test.hdf5
```

### Step 2: Process PCR Data
- Place the PCR data in a directory named `PCR`, in the same directory as FLAG3D (WARNING: the PCR dataset is *very* large).
- Upsample the data to 240Hz by running the following script:
```bash
python -m data_generation.upsample_data \
    --input_file <your_directory>/PCR/pcr_dataset_train.hdf5 \
    --output_file <your_directory>/PCR/pcr_upsampled_train.hdf5
python -m data_generation.upsample_data \
    --input_file <your_directory>/PCR/pcr_dataset_test.hdf5 \
    --output_file <your_directory>/PCR/pcr_upsampled_test.hdf5
```

NOTE: The above scripts require a significant amount of time to run. As an alternative, we provide the processed data in the `PCR` directory for your convenience. If you choose to use the provided data, you may simply continue with the following steps.

### Step 3: Combine FLAG3D and PCR Data
- Run the following script:
```bash
python -m data_generation.merge_data \
    --save_path <your_directory>/flag_pcr_train.hdf5 \
    --flag3d <your_directory>/FLAG3D/flag3d_upsampled_train.hdf5 \
    --pcr <your_directory>/PCR/pcr_upsampled_train.hdf5
python -m data_generation.merge_data \
    --save_path <your_directory>/flag_pcr_test.hdf5 \
    --flag3d <your_directory>/FLAG3D/flag3d_upsampled_test.hdf5 \
    --pcr <your_directory>/PCR/pcr_upsampled_test.hdf5 \
    --num_samples 5000
```

### Step 4: Remove PCR sequences that exceed the physiological velocity limits
- After generating the datasets, we found that some sequences in the PCR test set exceeded the physiological limits of joint velocities. To remove these sequences, run the following script:
```bash
python -m data_generation.clean_data \
    --input_path <your_directory>/flag_pcr_test.hdf5 \
    --output_path <your_directory>/flag_pcr_test_cleaned.hdf5
```

### Step 4: Add temporal smoothing and calculate acceleration
- Run the following script:
```bash
python -m data_generation.smooth_data \
    --input_path <your_directory>/flag_pcr_train.hdf5 \
    --output_path <your_directory>/flag_pcr_train_smoothed.hdf5
python -m data_generation.smooth_data \
    --input_path <your_directory>/flag_pcr_test.hdf5 \
    --output_path <your_directory>/flag_pcr_test_smoothed.hdf5
```

### Step 5: Remove intermediate files and rename the final dataset
- Run the following commands:
```bash
rm <your_directory>/flag_pcr_train.hdf5
rm <your_directory>/flag_pcr_test.hdf5
rm <your_directory>/flag_pcr_test_cleaned.hdf5
mv <your_directory>/flag_pcr_train_smoothed.hdf5 <your_directory>/flag_pcr_train.hdf5
mv <your_directory>/flag_pcr_test_smoothed.hdf5 <your_directory>/flag_pcr_test.hdf5
```