#!/bin/bash

ACRONYM="case_3"
TOTAL_EPOCHS=10
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SYSTEM_NAME_LIST=("Noise_math_data")
MODEL_PATH_LIST=(
    "/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct"
    "/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct"
    "/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT_A100="${SCRIPT_DIR}/scripts_qwen_1_5B/train/run_${ACRONYM}.sh"
SLURM_LOG_DIR="/export/home/asifali/Noise_math_data/all_logs"
mkdir -p "$SLURM_LOG_DIR"

PREV_JOB_ID=""

for model_path in "${MODEL_PATH_LIST[@]}"; do
    MODEL_NAME="$(basename "$model_path")"

    for i in "${!TRAIN_TEMP_LIST[@]}"; do
        TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
        TEST_TEMP=${TEST_TEMP_LIST[$i]}
        SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}

        echo "Submitting ${ACRONYM} for ${MODEL_NAME}: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, TOTAL_EPOCHS=$TOTAL_EPOCHS"

        SBATCH_ARGS=(
            --parsable
            --output="${SLURM_LOG_DIR}/%j-%x.out"
            --error="${SLURM_LOG_DIR}/%j-%x.err"
            --export="ALL,CASE_NAME=${ACRONYM},PROMPT_VERSION=case_2,BASE_MODEL_PATH_OVERRIDE=${model_path},MODEL_NAME=${MODEL_NAME},TOTAL_EPOCHS=${TOTAL_EPOCHS}"
        )

        if [ -n "$PREV_JOB_ID" ]; then
            SBATCH_ARGS+=(--dependency="afterok:${PREV_JOB_ID}")
        fi

        JOB_ID_RAW=$(sbatch "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT_A100" "$TRAIN_TEMP" "$TEST_TEMP" "$SYSTEM_NAME")
        JOB_ID="${JOB_ID_RAW%%;*}"
        echo "Submitted ${MODEL_NAME} as job ${JOB_ID}"
        PREV_JOB_ID="$JOB_ID"
    done
done

echo "All case_3 loop jobs submitted."
