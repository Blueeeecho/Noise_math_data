#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Reasoning360

# ==============================================================================
# 1. Choose your Model Path
# ==============================================================================
# You can easily paste your RL checkpoint path here, or use base/sft model paths.
# Examples:
# MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct" (Base)
# MODEL_PATH="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/sft_model" (SFT)
# MODEL_PATH="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/checkpoints/qwen2.5-0.5b-grpo-test/global_step_10" (RL)

MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/Reasoning360/examples/noise_math/Output/checkpoints/qwen2.5-0.5b-grpo-test/global_step_10"}

# Extract a short name for the model to use in the output directory
MODEL_NAME=$(basename "$MODEL_PATH")
if [[ "$MODEL_PATH" == *"global_step_"* ]]; then
    # If it's a checkpoint, include the parent directory name (e.g., qwen2.5-0.5b-grpo-test_global_step_10)
    PARENT_DIR=$(basename $(dirname "$MODEL_PATH"))
    MODEL_NAME="${PARENT_DIR}_${MODEL_NAME}"
fi

# ==============================================================================
# 2. Choose your Test Dataset
# ==============================================================================
# You can change the filename here to evaluate on different datasets in test_data/
TEST_FILE_NAME=${TEST_FILE_NAME:-"gsm8k-test.jsonl"}
DATA_PATH="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/test_data/${TEST_FILE_NAME}"

# ==============================================================================
# 3. Dynamic Output Directory
# ==============================================================================
# Results will be saved clearly categorized by model and dataset
TEST_NAME="${TEST_FILE_NAME%.*}" # Remove .jsonl extension
OUTPUT_DIR="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/eval_results/${MODEL_NAME}/${TEST_NAME}"

echo "========================================================"
echo "Starting Evaluation..."
echo "Model:    $MODEL_PATH"
echo "Dataset:  $DATA_PATH"
echo "Output:   $OUTPUT_DIR"
echo "========================================================"

python /root/autodl-tmp/Reasoning360/examples/noise_math/eval_model.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.6

echo "Evaluation completed. Results and process saved to $OUTPUT_DIR"
