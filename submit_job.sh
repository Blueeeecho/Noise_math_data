#!/bin/bash

#echo " I am only this..!"

ACRONYM="case_1"
echo "Submitting Training job A100 + GT"
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SYSTEM_NAME_LIST=("Noise_math_data")

SLURM_SCRIPT_A100="./scripts_qwen_1_5B/train/run_${ACRONYM}.sh"
SLURM_LOG_DIR="/export/home/asifali/Noise_math_data/examples/noise_math/Output/${ACRONYM}/slurm"
mkdir -p "$SLURM_LOG_DIR"

for i in "${!TRAIN_TEMP_LIST[@]}"; do
    TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
    TEST_TEMP=${TEST_TEMP_LIST[$i]}
    SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}

    echo "Submitting job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP"
    sbatch \
        --output="${SLURM_LOG_DIR}/%j-%x.out" \
        --error="${SLURM_LOG_DIR}/%j-%x.err" \
        $SLURM_SCRIPT_A100 $TRAIN_TEMP $TEST_TEMP $SYSTEM_NAME

done
echo "All jobs submitted A100."
