# Math-Noise (A-new-1) Training Pipeline & Code Guide

This guide details the training process, code structure, and usage of the refactored Math-Noise project (`A-new-1`). This project aims to train models with "Backward Reasoning" capabilities by combining SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning using the Verl framework).

## 1. Project Structure Overview

```
A-new-1/
├── data/               # Data processing related
├── sft/                # SFT training code
├── rl/                 # RL training code (Verl adapter)
├── eval/               # Model evaluation code
├── analysis/           # Results analysis code
├── scripts/            # Run scripts
├── config/             # Configuration files (Hydra)
└── README.md           # Project introduction
```

## 2. Environment Preparation

This project depends on the following core libraries:
- **Verl**: For distributed reinforcement learning (PPO/GRPO).
- **vLLM**: For efficient inference and RL rollout generation.
- **TRL / Transformers / PEFT**: For SFT training.
- **Flash Attention 2**: (Optional) Accelerates training.

Please ensure you are running in the `verl` virtual environment.

## 3. Training Pipeline Details

The entire training process is divided into four main stages: **Data Preparation -> SFT -> RL -> Evaluation & Analysis**.

### Stage 1: Data Preparation

Goal: Generate datasets conforming to the "Backward Reasoning" format.

- **Core Concepts**:
  - **Backward Reasoning**: The model does not give the answer directly but first performs `[Goal Analysis]`, then `[Backward Execution]`, and finally provides the `[Final Answer]`.
- **Script**: `scripts/gen_dummy_data.py` (Example)
  - This script generates data containing the `backward_reasoning` field.
  - **Format**:
    ```json
    {
      "question": "...",
      "answer": "...",
      "backward_reasoning": "[Goal Analysis]\n...\n[Backward Execution]\n...\n[Final Answer]\n..."
    }
    ```

### Stage 2: Supervised Fine-Tuning (SFT)

Goal: Teach the model the output format and basic logic of "Backward Reasoning".

- **Script**: `sft/sft_train.py`
- **Core Logic**:
  - Uses `trl.SFTTrainer` for LoRA fine-tuning.
  - When loading data, it automatically uses the `backward_reasoning` field as the training target (response).
  - **Key Parameters**:
    - `--use_peft`: Enable LoRA.
    - `--lora_rank`: LoRA rank (e.g., 64).
    - `--target_modules`: Specify modules for LoRA application (e.g., q_proj, k_proj, etc.).
- **Output**: Saves LoRA Adapter to `output/sft_model`.

### Stage 3: Reinforcement Learning (RL - PPO/GRPO)

Goal: Further optimize the model's reasoning ability and correctness via a reward function.

- **Framework**: **Verl**
- **Script**: `rl/train_verl.py`
  - This is a wrapper script responsible for setting environment variables and calling the Verl training entry point.
  - **Monkey Patch**: To support custom reward functions, we dynamically replace Verl's default scoring logic (`verl.utils.reward_score.default_compute_score` -> `rl.reward_fn.compute_reward`) at runtime.
- **Reward Function**: `rl/reward_fn.py`
  - **Format Reward**: Checks if the output contains `[Goal Analysis]`, `[Backward Execution]`, `[Final Answer]` tags.
  - **Process Reward**: Checks if numerical calculations (`<<...>>`) in the reasoning process match the Ground Truth.
  - **Outcome Reward**: Checks if the final answer is correct.
- **Data Conversion**: `rl/convert_data.py`
  - Converts SFT format data to the Parquet/Arrow format required by Verl, including `data_source` and `reward_model` fields.
- **Configuration**:
  - Uses the Hydra configuration system (`config/`).
  - Overrides default configurations (like batch size, gpu utilization) via command line arguments in `run_dry_run.sh`.
  - **Rollout**: Uses vLLM for generation.

### Stage 4: Evaluation & Analysis

Goal: Verify model performance on the test set.

- **Evaluation Script**: `eval/eval_model.py`
  - Loads the Base Model + SFT/RL Adapter.
  - Uses vLLM for batch generation.
  - Forces the model to use the System Prompt: `"You are a backward reasoning expert..."`.
  - Parses the `[Final Answer]` in the output and compares it with the standard answer.
- **Analysis Script**: `analysis/analyze_results.py`
  - Calculates Accuracy.
  - Generates an analysis report in Markdown format.

## 4. How to Run

We provide a one-click test script `scripts/run_dry_run.sh` to verify the complete pipeline.

### Steps:

1. **Enter the script directory**:
   ```bash
   cd A-new-1/scripts
   ```

2. **Configure Model Path**:
   Modify the `MODEL_PATH` variable in `run_dry_run.sh` to point to your local model (e.g., Qwen2.5-0.5B).

3. **Execute Dry Run**:
   ```bash
   bash run_dry_run.sh
   ```

### Script Content:
1. **Generate Dummy Data**: `gen_dummy_data.py` generates a small number of samples.
2. **SFT Training**: Runs 10 steps of fine-tuning to verify SFT code.
3. **RL Training**: Runs 2 Steps of PPO/GRPO to verify Verl integration and reward functions.
4. **Evaluation**: Loads the trained Adapter for inference.
5. **Analysis**: Outputs a report.

## 5. FAQ

- **Q: What if I encounter CUDA OOM (Out of Memory)?**
  - A: Lower `gpu_memory_utilization` (e.g., 0.5 -> 0.3 -> 0.1) and `batch_size` in `run_dry_run.sh`.
  
- **Q: Verl error `AssertionError: only support equal chunk`?**
  - A: This is because the data size is too small (Batch Size < Worker Count). In Dry Run, ensure `train_batch_size` is at least `rollout.n` * `n_gpus`.

- **Q: How to view WandB logs?**
  - A: Set `--use_wandb True` in the script arguments. It is disabled by default in Dry Run to avoid polluting the project.
