#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/Reasoning360"
export PYTHONUNBUFFERED=1

CASE_NAME="${CASE_NAME:-case_2}"
PROMPT_VERSION="${PROMPT_VERSION:-case_2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="/root/autodl-tmp/Reasoning360/examples/noise_math/Output/local_0.5b/${CASE_NAME}/${MODEL_NAME}"
JOB_ROOT="${OUTPUT_ROOT}/run_${RUN_TAG}"
DATA_DIR="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Processed/local_0.5b_${CASE_NAME}"
EVAL_DATA="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/test_data/gsm8k-test.jsonl"
EVAL_OUTPUT_DIR="${JOB_ROOT}/offline_eval/gsm8k-test"

mkdir -p "${JOB_ROOT}" "${DATA_DIR}" "${EVAL_OUTPUT_DIR}"

export INPUT_FILE="/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Ours/all_backward_data.jsonl"
export CONVERT_SCRIPT="/root/autodl-tmp/Reasoning360/examples/noise_math/convert_data_noise.py"
bash /root/autodl-tmp/Reasoning360/scripts_qwen_1_5B/train/prepare_data.sh "${DATA_DIR}" "${PROMPT_VERSION}"

python3 /root/autodl-tmp/Reasoning360/examples/noise_math/scripts/prepare_test_eval_data.py \
    --test_data_dir "/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/test_data" \
    --output_path "${DATA_DIR}/test_eval.parquet" \
    --prompt_version "${PROMPT_VERSION}"

echo "Starting local 0.5B GRPO training."
echo "Validation and checkpoint saving are enforced once per epoch inside ray_trainer.py."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test_eval.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=False \
    custom_reward_function.path="/root/autodl-tmp/Reasoning360/examples/noise_math/reward_noise.py" \
    custom_reward_function.name="compute_reward" \
    +custom_reward_function.reward_kwargs.reward_mode=step_rule \
    +custom_reward_function.reward_kwargs.global_fail_reward=-0.5 \
    +custom_reward_function.reward_kwargs.step_acc_weight=0.7 \
    +custom_reward_function.reward_kwargs.step_good_weight=0.4 \
    +custom_reward_function.reward_kwargs.step_bad_weight=0.3 \
    +custom_reward_function.reward_kwargs.step_fmt_weight=0.2 \
    +custom_reward_function.reward_kwargs.step_norm_min=3 \
    +custom_reward_function.reward_kwargs.require_reasoning=False \
    +custom_reward_function.reward_kwargs.require_source=False \
    +custom_reward_function.reward_kwargs.bad_on_unused_var=True \
    +custom_reward_function.reward_kwargs.bad_on_duplicate_var=True \
    +custom_reward_function.reward_kwargs.bad_on_missing_dependency=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='noise_math_local_0_5b' \
    trainer.experiment_name="local_0_5b_${CASE_NAME}_${RUN_TAG}" \
    trainer.default_local_dir="${JOB_ROOT}/checkpoints" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    +trainer.validation_share_to_all_logs=0 \
    +trainer.validation_shared_datasets="gsm8k-test" \
    +trainer.validation_include_step_rubric=1 \
    ray_init.num_cpus=8

BEST_MODEL_PATH="${JOB_ROOT}/checkpoints/best_checkpoint/actor"
if [ -d "${BEST_MODEL_PATH}" ]; then
    EVAL_MODEL_PATH="${BEST_MODEL_PATH}"
else
    LATEST_STEP_DIR="$(find "${JOB_ROOT}/checkpoints" -maxdepth 1 -type d -name 'global_step_*' | sort | tail -n 1)"
    if [ -z "${LATEST_STEP_DIR}" ]; then
        echo "No checkpoint found for evaluation."
        exit 1
    fi
    EVAL_MODEL_PATH="${LATEST_STEP_DIR}/actor"
fi

python3 /root/autodl-tmp/Reasoning360/examples/noise_math/eval_model.py \
    --model_path "${EVAL_MODEL_PATH}" \
    --data_path "${EVAL_DATA}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.6 \
    --prompt_version "${PROMPT_VERSION}"

echo "Local 0.5B training and evaluation completed."
echo "Training output: ${JOB_ROOT}"
echo "Offline eval output: ${EVAL_OUTPUT_DIR}"
