#!/bin/bash
set -e

# Create data directory
DATA_DIR="/root/autodl-tmp/Reasoning360/data/noise_math"
mkdir -p "$DATA_DIR"

echo "Preparing data for Noise Math training..."

# Convert test data
python /root/autodl-tmp/Noise_math_data/rl/convert_data.py \
    --input /root/autodl-tmp/Noise_math_data/data/testdata/gsm8k-test.jsonl \
    --output "$DATA_DIR/test.parquet"

# For demonstration, use the same file for training if no specific training file is identified yet
# Or replace with the actual training file path if known (e.g., subset_100.jsonl or generated data)
TRAIN_INPUT="/root/autodl-tmp/Noise_math_data/data/subset_100.jsonl"
if [ ! -f "$TRAIN_INPUT" ]; then
    echo "Warning: $TRAIN_INPUT not found, using gsm8k-test.jsonl for training as well."
    TRAIN_INPUT="/root/autodl-tmp/Noise_math_data/data/testdata/gsm8k-test.jsonl"
fi

python /root/autodl-tmp/Noise_math_data/rl/convert_data.py \
    --input "$TRAIN_INPUT" \
    --output "$DATA_DIR/train.parquet"

echo "Data preparation complete. Parquet files saved to $DATA_DIR"
