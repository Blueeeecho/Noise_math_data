#!/bin/bash

ACRONYM="${ACRONYM:-case_6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SYSTEM_NAME_LIST=("Noise_math_data")
MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PROMPT_VERSION="${ACRONYM}"
if [ "${ACRONYM}" = "case_5" ] || [ "${ACRONYM}" = "case_6" ]; then
    DEFAULT_PROMPT_VERSION="case_4"
fi
PROMPT_VERSION="${PROMPT_VERSION:-${DEFAULT_PROMPT_VERSION}}"
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
        --export="ALL,CASE_NAME=${ACRONYM},PROMPT_VERSION=${PROMPT_VERSION},BASE_MODEL_PATH_OVERRIDE=${MODEL_PATH},MODEL_NAME=${MODEL_NAME},TOTAL_EPOCHS=${TOTAL_EPOCHS},SHARE_EVAL_JSONL_TO_ALL_LOGS=1,SHARED_EVAL_DATASETS=global_training_summary.csv,SHARED_ALL_LOG_DIR=${SLURM_LOG_DIR},INCLUDE_STEP_SCORE_RUBRIC=1" \
        "$SLURM_SCRIPT_A100" "$TRAIN_TEMP" "$TEST_TEMP" "$SYSTEM_NAME"
done

echo "All 3B jobs submitted."
