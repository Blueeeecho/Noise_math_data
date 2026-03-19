#!/bin/bash
set -e

# Example script for 4x A100 environment
# You might need to adjust paths and environment variables based on your cluster setup

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set NPROC_PER_NODE to 4 for 4 GPUs
export NPROC_PER_NODE=4

# Ensure PYTHONPATH includes the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Remote PPO Training on 4x A100..."

# Update the config path to where you deploy it on the remote server
CONFIG_PATH="./config/ppo_remote_a100.yaml"

python3 -m verl.trainer.main_ppo \
    config=$CONFIG_PATH \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    +trainer.val_before_train=False \
    2>&1 | tee remote_training.log
