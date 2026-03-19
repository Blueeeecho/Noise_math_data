#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

# Ensure data exists
if [ ! -f "/root/autodl-tmp/Reasoning360/data/noise_math/train.parquet" ]; then
    echo "Data not found. Running prepare_data.sh..."
    bash /root/autodl-tmp/Reasoning360/examples/noise_math/prepare_data.sh
fi

echo "Starting Local PPO Training on Single GPU (4090D)..."

# Use python -m to run the module
# We need to set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Reasoning360

python -m verl.trainer.main_ppo \
    config=/root/autodl-tmp/Reasoning360/examples/noise_math/config/ppo_local_4090.yaml \
    trainer.project_name='noise_math_local' \
    trainer.experiment_name='qwen2.5-0.5b-ppo-test' \
    trainer.total_epochs=1 2>&1 | tee local_training.log
