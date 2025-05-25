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
echo "Starting model training..."

# Run first training (for Fig 3)
SEEDS=(0)
echo "Running training with SEEDS=${SEEDS[*]} and n_aff=5 (Fig 3 model)"
python train/train_model.py \
    --data_dir "$data_dir" \
    --seeds "${SEEDS[@]}" \
    --training_seeds "${SEEDS[@]}" \
    --n_aff 5 \
    --base_config train/configs/train_spindles.yaml

# Run extended training (for rest of paper)
SEEDS=(0 1 2 3 4)
TRAIN_SEEDS=(0 1 2 3)
# only subset of seeds for training
# SEEDS=(0 1)
# TRAIN_SEEDS=(0 1)
echo "Running extended training with SEEDS=${SEEDS[*]} and n_aff=5"
python train/train_model.py \
    --data_dir "$data_dir" \
    --seeds "${SEEDS[@]}" \
    --training_seeds "${TRAIN_SEEDS[@]}" \
    --n_aff 5 \
    --base_config train/configs/train_spindles_extended.yaml

echo "All training runs completed successfully."