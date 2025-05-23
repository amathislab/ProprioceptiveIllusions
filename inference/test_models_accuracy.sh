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
echo "Starting model testing..."

# Run test for model (for Fig 3)
SEEDS=(0)
echo "Running test with SEEDS=${SEEDS[*]} and n_aff=5 (Fig 3 model)"
python inference/test_model.py \
    --data_dir "$data_dir" \
    --seeds "${SEEDS[@]}" \
    --training_seeds "${SEEDS[@]}" \
    --n_aff 5 \
    --base_config inference/configs/test_spindles_fig3.yaml

# Run extended test (for rest of paper)
SEEDS=(0 1 2 3 4)
TRAIN_SEEDS=(0 1 2 3)
SEEDS=(0)
TRAIN_SEEDS=(0 1)
echo "Running extended test with SEEDS=${SEEDS[*]} and n_aff=5"
python inference/test_model.py \
    --data_dir "$data_dir" \
    --seeds "${SEEDS[@]}" \
    --training_seeds "${TRAIN_SEEDS[@]}" \
    --n_aff 5 \
    --base_config inference/configs/test_spindles_extended.yaml

echo "All training runs completed successfully."