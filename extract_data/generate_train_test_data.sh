#!/bin/bash

# Default base directory
BASE_DIR="/media/data16/adriana/ProprioPerception"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

run_generation() {
    local config_path="$1"
    shift
    local -a seeds=()
    while [[ "$1" != "--" ]]; do
        seeds+=("$1")
        shift
    done
    shift # skip the "--" marker

    for input_file in "$@"; do
        echo "Processing $input_file with config $config_path and seeds ${seeds[*]}"
        python extract_data/generate_train_test_data.py \
            --config_path="$config_path" \
            --seeds "${seeds[@]}" \
            --input_file="$input_file" \
            --output_dir="$BASE_DIR"
    done
}

INPUTS_FILES=(
    "${BASE_DIR}/flag_pcr_training.hdf5"
    "${BASE_DIR}/flag_pcr_test.hdf5"
    "${BASE_DIR}/EF3D.hdf5"
    "${BASE_DIR}/ES3D.hdf5"
)
### --- Data for model for Fig 3 --- ###
SEEDS_FIG3=(0)
CONFIG_FIG3="extract_data/configs/train_test_data_spindles.yaml"

run_generation "$CONFIG_FIG3" "${SEEDS_FIG3[@]}" -- "${INPUTS_FILES[@]}"

### --- Data for rest of models --- ###
# SEEDS_EXTENDED=(0 1 2 3 4)
SEEDS_EXTENDED=(0 1)
CONFIG_EXTENDED="extract_data/configs/train_test_data_spindles_extended.yaml"

run_generation "$CONFIG_EXTENDED" "${SEEDS_EXTENDED[@]}" -- "${INPUTS_FILES[@]}"

