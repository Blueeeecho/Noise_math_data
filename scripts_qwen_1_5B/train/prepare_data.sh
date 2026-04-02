#!/bin/bash
set -e

# Create data directory
DATA_DIR="${1:-/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed}"
PROMPT_VERSION="${2:-case_1}"
mkdir -p "$DATA_DIR"

echo "Preparing data for Noise Math training using internal converter..."

# Input data path
INPUT_FILE="/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Ours/all_backward_data.jsonl"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# For demonstration, we split the same file into train/test or just use it for both
# In a real scenario, you should have separate train.jsonl and test.jsonl

echo "Converting $INPUT_FILE to $DATA_DIR/train.parquet"
python /export/home/asifali/Noise_math_data/examples/noise_math/convert_data_noise.py \
    --input "$INPUT_FILE" \
    --output "$DATA_DIR/train.parquet" \
    --prompt_version "$PROMPT_VERSION"

echo "Converting $INPUT_FILE to $DATA_DIR/test.parquet"
python /export/home/asifali/Noise_math_data/examples/noise_math/convert_data_noise.py \
    --input "$INPUT_FILE" \
    --output "$DATA_DIR/test.parquet" \
    --prompt_version "$PROMPT_VERSION"

echo "Data preparation complete. Parquet files saved to $DATA_DIR using prompt version $PROMPT_VERSION"
