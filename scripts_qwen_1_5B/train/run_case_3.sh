#!/bin/bash -l

#SBATCH -J MTMT-A100
#SBATCH -p gpu-A100
#SBATCH --gres gpu:4
#SBATCH -c 128
#SBATCH --mem 256GB
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com

export CASE_NAME="case_3"
export PROMPT_VERSION="case_2"

bash /export/home/asifali/Noise_math_data/scripts_qwen_1_5B/train/run_exp1.sh "$@"
