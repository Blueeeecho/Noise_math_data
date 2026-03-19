#!/bin/bash -l

#SBATCH -J MTMT-A100 #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 128  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit
nvidia-smi
source activate Reasoning360

export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES

#### MY Parameters
export USE_Thinking=1
#export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

#export RAY_TMPDIR="/export/home/asifali/HF_cache/RAY_TMP"
#mkdir -p RAY_TMPDIR

# ===============================
# Force ALL temp/cache off /tmp
# ===============================

# temp dirs (your existing block)
LOCAL_BASE="/var/tmp/$USER/${SLURM_JOB_ID}"
export RAY_TMPDIR="$LOCAL_BASE/ray"
export TMPDIR="$LOCAL_BASE/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
chmod 700 "$LOCAL_BASE" "$RAY_TMPDIR" "$TMPDIR"
export RAY_DISABLE_DASHBOARD=1

# ensure we don't connect to an old cluster
unset RAY_ADDRESS RAY_HEAD_IP RAY_PORT

# run-once guard (pick one)
if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
  echo "Skipping on SLURM_LOCALID=${SLURM_LOCALID}"
  exit 0
fi

# cleanup only once (same process)
${CONDA_BIN_PATH}ray stop -f || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
sleep 3

ulimit -n 1048576 2>/dev/null || true
echo "NOFILE=$(ulimit -n)"

# =======================================================================
# ==================== All INPUTS =================================
TRAIN_TEMP=$1
TEST_TEMP=$2
SYSTEM_NAME=$3


echo "Submitted job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SYSTEM_NAME=$SYSTEM_NAME"

# ===============================================================
#SYSTEM_NAME="Reasoning360_sys_B_v2"
export PYTHONPATH="/export/home/asifali/${SYSTEM_NAME}:${PYTHONPATH:-}"
echo "Python Path = ${PYTHONPATH}"



# =================== User-Configurable Settings ===================
# --- Execution Environment ---
NUM_GPUS=4 # Set the number of GPUs to use on this node
gpu_memory_utilization=0.8
# --- Resuming & Logging ---
RESUME_CKPT_DIR_NAME=""  # Fill in the W&B experiment name to resume from, otherwise leave empty to start from scratch
WANDB_PROJECT="EXP1_Noisy_data_A100" # Your wandb project name

# --- External Services ---
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"

# =================== Environment Setup ==============================
#export NCCL_NVLS_ENABLE=1
#export NCCL_IB_DISABLE=0
#export NCCL_P2P_DISABLE=0
#export CUDA_LAUNCH_BLOCKING=0
#export NCCL_DEBUG=WARN
#export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset CUDA_LAUNCH_BLOCKING
unset CUDA_DEVICE_MAX_CONNECTIONS
# =====================================================================

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

#unset LD_LIBRARY_PATH
export DEBUG_CODE=0
export USE_NL=0 # Using NL Prompts
export TEST_SCORE_METHOD='gt'
export TRAIN_SCORE_METHOD=${SCORING_METHOD}
export Z3_CLUE_GATE=0.7
export ACC_W=${ACC_W}
export Z3_W=${Z3_W}
export SWITCH_EPOCH=${SWITCH_EPOCH}

# export CUDA_LAUNCH_BLOCKING=1 # Uncomment for easier debugging of CUDA errors


# ======================================================================


# Ensure data exists
if [ ! -f "/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed/train.parquet" ]; then
    echo "Data not found. Running prepare_data.sh..."
    bash /export/home/asifali/Noise_math_data/scripts_qwen_1_5B/train/prepare_data.sh
fi

echo "Starting GRPO Training on 4 x A100 GPUs"

# Using standard verl PPO trainer logic with CLI overrides for GRPO
# Key changes for GRPO:
# 1. algorithm.adv_estimator=grpo
# 2. actor_rollout_ref.rollout.n=4 (Group size > 1)
# 3. No critic configs (GRPO doesn't use a learned critic)

# Default model path is the output of SFT training. 
# If SFT hasn't been run, fallback to the original base model.
BASE_MODEL_PATH="/export/home/asifali/Noise_math_data/examples/noise_math/Output/sft_model"
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "SFT model not found at $BASE_MODEL_PATH. Falling back to base Qwen model."
    BASE_MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct"
fi

# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0


clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 1))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10


#train_prompt_bsz=32  # on-policy model update batchsize: train_prompt_bsz * rollout.n
#gen_prompt_bsz=$((train_prompt_bsz * 1))
#n_resp_per_prompt=4
#train_prompt_mini_bsz=4  # model grad update batchsize


train_prompt_bsz=600  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))


# Algorithm
top_p=0.9
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Training config
# NOTE: sp_size and gen_tp are parallelism settings.
# sp_size: Sequence Parallelism size.
# gen_tp: Tensor Parallelism size for vLLM generation.
# For a 32B model on 8 GPUs, TP=2 is a reasonable starting point. Adjust if you have memory issues.
sp_size=1
gen_tp=${NUM_GPUS}
gen_max_num_seqs=1024
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward, but note memory overflow
offload=True


echo "Starting training..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed/train.parquet" \
    data.val_files="/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed/test.parquet" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$BASE_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMP} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEST_TEMP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    custom_reward_function.path="/export/home/asifali/Noise_math_data/examples/noise_math/reward_noise.py" \
    custom_reward_function.name="compute_reward" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='noise_math_local_grpo' \
    trainer.experiment_name='qwen2.5-1.5b-grpo-test' \
    trainer.default_local_dir="/export/home/asifali/Noise_math_data/examples/noise_math/Output/checkpoints" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=500