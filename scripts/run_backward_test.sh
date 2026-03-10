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


# Setup directories
BASE_DIR=$(dirname $(readlink -f "$0"))/..
DATA_DIR="$BASE_DIR/data"
SFT_DIR="$BASE_DIR/sft"
RL_DIR="$BASE_DIR/rl"
EVAL_DIR="$BASE_DIR/eval"
ANALYSIS_DIR="$BASE_DIR/analysis"
OUTPUT_DIR="$BASE_DIR/output_backward"


# Python executable for Verl environment
mkdir -p "$OUTPUT_DIR"
cd "$BASE_DIR/scripts"

echo "=== Step 1: Prepare Backward Data ==="
python3 prepare_backward_data.py

echo "=== Step 2: SFT Backward Test ==="
MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct

if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model path $MODEL_PATH not found."
    MODEL_PATH=$(find /home/wwq416/snap/wwq/model -maxdepth 3 -name "config.json" | head -n 1 | xargs dirname)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model found. Cannot run training."
        exit 1
    fi
    echo "Found model at: $MODEL_PATH"
fi
MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct

# Run SFT with small batch size and few steps/epochs for quick verification
python3 "$SFT_DIR/sft_train.py" \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$OUTPUT_DIR/train.jsonl" \
    --test_file "$OUTPUT_DIR/test.jsonl" \
    --output_dir "$OUTPUT_DIR/sft_model" \
    --use_lora True \
    --lora_r 16 \
    --max_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --save_strategy "no" \
    --logging_steps 1 \
    --use_wandb False \
    --eval_on_test_after_train True \
    --gen_max_new_tokens 128


echo "=== Step 3: RL Backward Test ==="
cd "$RL_DIR"
# Convert data
$PYTHON_VERL convert_data.py --input "$OUTPUT_DIR/train.jsonl" --output "$OUTPUT_DIR/rl_train.parquet"
$PYTHON_VERL convert_data.py --input "$OUTPUT_DIR/test.jsonl" --output "$OUTPUT_DIR/rl_test.parquet"

# Run RL
REWARD_FN_PATH=$(readlink -f "$RL_DIR/reward_fn.py")

# Use conservative settings to avoid OOM
# Removed 'attn_implementation=eager' to allow SDPA (memory efficient)
# train_batch_size = 8 (must be divisible by n_gpus * micro_batch)
# ppo_mini_batch_size = 4
$PYTHON_VERL train_verl.py \
    data.train_files="$OUTPUT_DIR/rl_train.parquet" \
    data.val_files="$OUTPUT_DIR/rl_test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_adapter_path="$OUTPUT_DIR/sft_model" \
    actor_rollout_ref.model.lora_rank=16 \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    +actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name="compute_reward" \
    reward.reward_model.n_gpus_per_node=1 \
    algorithm.adv_estimator=grpo \
    trainer.project_name=backward_test \
    trainer.experiment_name=run1 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    hydra.run.dir="$OUTPUT_DIR/rl_run"

cd "$BASE_DIR/scripts"

echo "=== Step 4: Eval Backward Test ==="
$PYTHON_VERL "$EVAL_DIR/eval_model.py" \
    --model_path "$OUTPUT_DIR/sft_model" \
    --base_model_path "$MODEL_PATH" \
    --data_path "$OUTPUT_DIR/test.jsonl" \
    --output_dir "$OUTPUT_DIR/eval_result" \
    --max_tokens 256

echo "=== Step 5: Analysis Backward Test ==="
python3 "$ANALYSIS_DIR/analyze_results.py" \
    --results_file "$OUTPUT_DIR/eval_result/results.jsonl" \
    --output_report "$OUTPUT_DIR/analysis_report.md"

echo "=== Backward Test Completed Successfully ==="
