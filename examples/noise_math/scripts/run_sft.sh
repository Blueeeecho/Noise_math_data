#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Reasoning360

# Paths
MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct"
DATA_PATH="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Ours/all_backward_data.jsonl"
OUTPUT_DIR="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/sft_model"

echo "Starting SFT Training..."

python /root/autodl-tmp/Reasoning360/examples/noise_math/sft_train.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --bf16 True \
    --use_wandb False \
    --gradient_checkpointing True \
    --max_seq_length 1024

echo "SFT Training completed. Model saved to $OUTPUT_DIR"
