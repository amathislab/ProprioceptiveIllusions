#!/bin/bash
set -euo pipefail

# Default value
data_dir=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if data_dir is set
if [[ -z "$data_dir" ]]; then
    echo "Error: --data_dir must be provided"
    exit 1
fi

# Logging
echo "Using data directory: $data_dir"

# if /data in data_dir base_dir is base of data_dir otherwise use data_dir
if [[ "$data_dir" == *"/data"* ]]; then
    base_dir=$(dirname "$data_dir")
else
    base_dir="$data_dir"
fi
MODEL_BASE_PATH="${base_dir}/trained_models" 

SEEDS=(0 1 2 3 4)
TRAINING_SEEDS=(0 1 2 9) # for models from the paper
# TRAINING_SEEDS=(0 1 2 3) # for models trained

# Muscles to vibrate
MUSCLES_TO_VIB_LIST=(
    "TRIlat"
    "TRIlong"
    "TRImed"
    "BICshort"
    "BIClong"
)
## To vibrate individual channels for each model
CHANNEL_INDICES_LIST=(
    "0"
    "1"
    "2"
    "3"
    "4"
)
VIB_FREQS="0 20 40 60 80 100 110 130 150 170 190"
N_AFFS=(5)
VIB_START=200
VIB_END=900
I_A_SAMPLED_COEFF_PATH_BASE="data/extended_spindle_coefficients/i_a/linear/"
I_A_COEFF_PATH="data/extended_spindle_coefficients/i_a/linear/coefficients.csv"
TASK="letter_reconstruction_joints"
model_prefix="optimized_linear_extended"
vib_exp_id="vib_vary_ia_subChannels"

for N_AFF in "${N_AFFS[@]}"; do
    # Loop through each seed and muscle group
    for SEED in "${SEEDS[@]}"; do
        for TRAINING_SEED in "${TRAINING_SEEDS[@]}"; do
            MODEL_PATH="${MODEL_BASE_PATH}/experiment_causal_flag-pcr_${model_prefix}_${N_AFF}_${N_AFF}_${TASK}/spatiotemporal_4_${SEED}_${TRAINING_SEED}"
            DATA_PATH="${data_dir}/${model_prefix}_${SEED}_${N_AFF}_${N_AFF}_ES3D.hdf5"
            I_A_SAMPLED_COEFF_PATH="${I_A_SAMPLED_COEFF_PATH_BASE}sampled_coefficients_i_a_${N_AFF}_${SEED}.csv"

            # check that model path exists
            if [ ! -d "$MODEL_PATH" ]; then
                echo "Model path does not exist: $MODEL_PATH"
                continue
            fi
            for MUSCLES_TO_VIB in "${MUSCLES_TO_VIB_LIST[@]}"; do
                for channel_indices in "${CHANNEL_INDICES_LIST[@]}"; do
                    echo "Vibrating channel_indices: $channel_indices"
                    # Generate a save path dynamically based on seed and muscles to vibrate
                    SAVE_PATH="${MODEL_PATH}/test/ES3D/${vib_exp_id}/$(echo ${MUSCLES_TO_VIB} | tr ' ' '_')"
                    mkdir -p "$SAVE_PATH"
                        echo "Testing model at path: $MODEL_PATH with muscles: $MUSCLES_TO_VIB"
                        # Run the Python script
                        python test_vibrations/test_model_vib_multipleFs.py \
                            --model_path "$MODEL_PATH" \
                            --data_path "$DATA_PATH" \
                            --save_path "$SAVE_PATH" \
                            --vib_freqs $VIB_FREQS \
                            --vib_start $VIB_START \
                            --vib_end $VIB_END \
                            --muscles_to_vib $MUSCLES_TO_VIB \
                            --i_a_sampled_coeff_path "$I_A_SAMPLED_COEFF_PATH" \
                            --i_a_coeff_path "$I_A_COEFF_PATH" \
                            --channel_indices $channel_indices 
                            # --save_plots
                done
            done
        done
    done
done

echo "Testing completed for all seed and muscle combinations."
