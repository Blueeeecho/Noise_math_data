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
source activate noisymaths


export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES


export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"


export PYTHONPATH="/export/home/asifali/Noise_math_data/:${PYTHONPATH:-}"
echo "Python Path = ${PYTHONPATH}"


# =================== User-Configurable Settings ===================
# --- Execution Environment ---
NUM_GPUS=4 # Set the number of GPUs to use on this node
gpu_memory_utilization=0.8
# --- Resuming & Logging ---

WANDB_PROJECT="NOISY_MATHS_A100" 

# --- External Services ---
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"

# =================== Environment Setup ===================
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
# ==============================================================

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1


# Setup directories - FIXED BASE_DIR
BASE_DIR="/export/home/asifali/Noise_math_data"

# Fallback for local testing if the export path doesn't exist
if [ ! -d "$BASE_DIR" ]; then
    BASE_DIR=$(dirname $(readlink -f "$0"))/..
    echo "Warning: Remote path not found, falling back to local: $BASE_DIR"
fi

DATA_DIR="$BASE_DIR/data"
SFT_DIR="$BASE_DIR/sft"
RL_DIR="$BASE_DIR/rl"
EVAL_DIR="$BASE_DIR/eval"
ANALYSIS_DIR="$BASE_DIR/analysis"
OUTPUT_DIR="$BASE_DIR/output_backward"


# Python executable for Verl environment
mkdir -p "$OUTPUT_DIR"


echo "=== Step 1: Prepare Backward Data ==="
# Use absolute path
python "$BASE_DIR/scripts/prepare_backward_data.py"

echo "=== Step 2: SFT Backward Test ==="
MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model path $MODEL_PATH not found."
    MODEL_PATH=$(find /home/wwq416/snap/wwq/model -maxdepth 3 -name "config.json" | head -n 1 | xargs dirname)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model found. Cannot run training."
        exit 1
    fi
    echo "Found model at: $MODEL_PATH"
fi
# Re-enforce remote path if it was overwritten by local check logic in a real remote run
if [ -d "/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct" ]; then
    MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct"
fi

python "$SFT_DIR/sft_train.py" \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$OUTPUT_DIR/train.jsonl" \
    --test_file "$OUTPUT_DIR/test.jsonl" \
    --output_dir "$OUTPUT_DIR/sft_model" \
    --use_lora True \
    --lora_r 16 \
    --max_length 256 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --save_strategy "no" \
    --logging_steps 1 \
    --use_wandb False \
    --eval_on_test_after_train True \
    --gen_max_new_tokens 128


echo "=== Step 3: RL Backward Test ==="

# Convert data using absolute paths
python "$RL_DIR/convert_data.py" --input "$OUTPUT_DIR/train.jsonl" --output "$OUTPUT_DIR/rl_train.parquet"
python "$RL_DIR/convert_data.py" --input "$OUTPUT_DIR/test.jsonl" --output "$OUTPUT_DIR/rl_test.parquet"

# Run RL
REWARD_FN_PATH="$RL_DIR/reward_fn.py"


python "$RL_DIR/train_verl.py" \
    data.train_files="$OUTPUT_DIR/rl_train.parquet" \
    data.val_files="$OUTPUT_DIR/rl_test.parquet" \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    ++actor_rollout_ref.model.lora_adapter_path="$OUTPUT_DIR/sft_model" \
    ++actor_rollout_ref.model.lora_rank=16 \
    ++actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    ++actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name="compute_reward" \
    reward.reward_model.n_gpus_per_node=1 \
    algorithm.adv_estimator=grpo \
    trainer.project_name=backward_test \
    trainer.experiment_name=run1 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.total_epochs=5 \
    trainer.total_training_steps=10 \
    hydra.run.dir="$OUTPUT_DIR/rl_run"


# REMOVED cd "$BASE_DIR/scripts"

echo "=== Step 4: Eval Backward Test ==="
python "$EVAL_DIR/eval_model.py" \
    --model_path "$OUTPUT_DIR/sft_model" \
    --base_model_path "$MODEL_PATH" \
    --data_path "$OUTPUT_DIR/test.jsonl" \
    --output_dir "$OUTPUT_DIR/eval_result" \
    --max_tokens 256

echo "=== Step 5: Analysis Backward Test ==="
python "$ANALYSIS_DIR/analyze_results.py" \
    --results_file "$OUTPUT_DIR/eval_result/results.jsonl" \
    --output_report "$OUTPUT_DIR/analysis_report.md"

echo "=== Backward Test Completed Successfully ==="
