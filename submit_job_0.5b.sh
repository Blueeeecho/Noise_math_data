#!/bin/bash

ACRONYM="${ACRONYM:-case_2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SYSTEM_NAME_LIST=("Noise_math_data")
MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT_A100="${SCRIPT_DIR}/scripts_qwen_1_5B/train/run_${ACRONYM}.sh"
SLURM_LOG_DIR="/export/home/asifali/Noise_math_data/all_logs"
mkdir -p "$SLURM_LOG_DIR"

for i in "${!TRAIN_TEMP_LIST[@]}"; do
    TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
    TEST_TEMP=${TEST_TEMP_LIST[$i]}
    SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}

    echo "Submitting ${ACRONYM} for ${MODEL_NAME}: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, TOTAL_EPOCHS=$TOTAL_EPOCHS"
    sbatch \
        --output="${SLURM_LOG_DIR}/%j-%x.out" \
        --error="${SLURM_LOG_DIR}/%j-%x.err" \
        --export="ALL,CASE_NAME=${ACRONYM},PROMPT_VERSION=case_2,BASE_MODEL_PATH_OVERRIDE=${MODEL_PATH},MODEL_NAME=${MODEL_NAME},TOTAL_EPOCHS=${TOTAL_EPOCHS},SHARE_EVAL_JSONL_TO_ALL_LOGS=1,SHARED_EVAL_DATASETS=gsm8k-test,SHARED_ALL_LOG_DIR=${SLURM_LOG_DIR},INCLUDE_STEP_SCORE_RUBRIC=1" \
        "$SLURM_SCRIPT_A100" "$TRAIN_TEMP" "$TEST_TEMP" "$SYSTEM_NAME"
done

echo "All 0.5B jobs submitted."
