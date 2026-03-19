#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Reasoning360

# Ensure data exists
if [ ! -f "/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Processed/train.parquet" ]; then
    echo "Data not found. Running prepare_data.sh..."
    bash /root/autodl-tmp/Reasoning360/examples/noise_math/scripts/prepare_data.sh
fi

echo "Starting Local PPO Training on Single GPU (4090D)..."

# Default model path is the output of SFT training. 
# If SFT hasn't been run, fallback to the original base model.
BASE_MODEL_PATH="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/sft_model"
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "SFT model not found at $BASE_MODEL_PATH. Falling back to base Qwen model."
    BASE_MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct"
fi

# Using standard verl PPO trainer logic with CLI overrides
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Processed/train.parquet" \
    data.val_files="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Processed/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$BASE_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path="$BASE_MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=False \
    custom_reward_function.path="/root/autodl-tmp/Reasoning360/examples/noise_math/reward_noise.py" \
    custom_reward_function.name="compute_reward" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='noise_math_local' \
    trainer.experiment_name='qwen2.5-0.5b-ppo-test' \
    trainer.default_local_dir="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/checkpoints" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    ray_init.num_cpus=8
