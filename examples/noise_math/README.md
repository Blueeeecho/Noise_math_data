# Noise Math Training with Reasoning360 (Verl)

This project adapts the **Noise Math** training pipeline to fully leverage the **Reasoning360 (Verl)** framework. It supports both **PPO** and **GRPO** reinforcement learning algorithms, optimized for mathematical reasoning tasks with custom reward functions (Format, Process, and Outcome rewards).

This implementation is fully integrated into Reasoning360, utilizing its distributed training capabilities (FSDP, Hybrid Engine) while maintaining the specific logic required for noise-robust math reasoning.

## Directory Structure

All relevant files are located in `/root/autodl-tmp/Reasoning360/examples/noise_math/`:

```
noise_math/
├── dataset/                # Data storage
│   ├── Ours/               # Raw input data (e.g., all_backward_data.jsonl)
│   └── Processed/          # Generated Parquet files for training
├── Output/                 # Output directory for checkpoints and logs
│   ├── sft_model/          # SFT model weights
│   ├── checkpoints/        # RL saved model weights
│   └── eval_results/       # Evaluation results
├── scripts/                # Execution scripts
│   ├── run_sft.sh          # Run SFT training
│   ├── prepare_data.sh     # RL Data preparation (JSONL -> Parquet)
│   ├── run_local.sh        # Local PPO training script (Single GPU / 4090D)
│   ├── run_local_grpo.sh   # Local GRPO training script (Single GPU / 4090D)
│   ├── run_eval.sh         # Evaluation script
│   ├── run_a100_ppo.sh     # Remote PPO training script (Multi-GPU / 4x A100)
│   └── run_a100_grpo.sh    # Remote GRPO training script (Multi-GPU / 4x A100)
├── sft_train.py            # SFT training code
├── convert_data_noise.py   # Data conversion utility (adds 'ability', 'extra_info')
├── reward_noise.py         # Custom reward function implementation
├── eval_model.py           # Evaluation code
├── README.md               # This file
└── README_CN.md            # Chinese documentation
```

## Prerequisites

Ensure you are in the `Reasoning360` conda environment. This project requires specific versions of dependencies to ensure compatibility with the Verl framework.

```bash
conda activate Reasoning360
# Ensure transformers is compatible (v5.x is not supported)
uv pip install "transformers==4.52.4"
# Ensure trl is compatible
uv pip install "trl<=0.9.6"
```

## Data Preparation & Format

Reasoning360 (Verl) requires data to be in a specific **Parquet** format. We provide a script to convert your raw JSONL data into this format.

### 1. Raw Data Format (Before Processing)
Place your raw JSONL data in `dataset/Ours/all_backward_data.jsonl`. The raw data usually contains `question`, `answer`, and `backward_reasoning` fields.

**Example of Raw JSONL Entry:**
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
  "backward_reasoning": "[Goal Analysis]\nTarget: Var{Total_Clips_Sold}\nPlan: ...\n[Final Answer]\n72"
}
```

### 2. Run Data Conversion
Execute the preparation script:
```bash
bash scripts/prepare_data.sh
```

### 3. Processed Data Format (After Processing)
The script `convert_data_noise.py` converts the JSONL into `dataset/Processed/train.parquet` and `test.parquet`. The resulting Parquet file contains the following required columns:

- `data_source`: Set to `"math_noise"` (Used by the reward manager to route to the correct reward function).
- `prompt`: A list of dictionaries following the chat template.
  ```json
  [{"role": "user", "content": "Natalia sold clips..."}]
  ```
- `response`: The ground truth string (used as SFT target or reference).
- `ability`: Set to `"math"`.
- `reward_model`: A nested dictionary containing the ground truth used for reward calculation.
  ```json
  {
    "style": "rule",
    "ground_truth": {
      "gold_answer": "[Goal Analysis]...",
      "gold_chain": "[Goal Analysis]..."
    }
  }
  ```
- `extra_info`: Metadata such as split name and index.

## Complete Training Pipeline (SFT -> RL -> Evaluation)

We provide a complete three-step training pipeline:
1. **SFT (Supervised Fine-Tuning)**: Initial alignment on the math reasoning data.
2. **RL (Reinforcement Learning)**: Optimization using PPO or GRPO with custom math rewards.
3. **Evaluation**: Assessing the model's accuracy on the dataset.

All generated model checkpoints and outputs will be saved in `/root/autodl-tmp/Reasoning360/examples/noise_math/Output/`.

### Step 1: Run SFT Training
Execute the SFT script to train the base model (`Qwen2.5-0.5B-Instruct`).
```bash
bash scripts/run_sft.sh
```
*   **Input**: Raw JSONL data from `dataset/Ours/all_backward_data.jsonl`.
*   **Output**: SFT model saved to `Output/sft_model`.

### Step 2: Run RL Training (PPO or GRPO)
The RL scripts are configured to automatically use the `Output/sft_model` as the base model. If you skipped Step 1, it will fallback to the raw base model.

**Option A: Run PPO Training**
```bash
bash scripts/run_local.sh
```

**Option B: Run GRPO Training** (Saves VRAM, recommended)
```bash
bash scripts/run_local_grpo.sh
```

### Step 3: Run Evaluation
After training, evaluate your final checkpoint (e.g., global_step_10). You can change the target model path by modifying the script or passing an environment variable.
```bash
MODEL_PATH="Output/checkpoints/qwen2.5-0.5b-grpo-test/global_step_10" bash scripts/run_eval.sh
```
*   **Output**: Evaluation results and metrics are saved to `Output/eval_results/`.

### Remote Training (Multi-GPU)
For 4x A100 setups, we provide dedicated scripts with optimized hyperparameters for both algorithms:

**Run PPO:**
```bash
bash scripts/run_a100_ppo.sh
```

**Run GRPO:**
```bash
bash scripts/run_a100_grpo.sh
```

---

## Custom Reward Function
The reward logic is implemented in `reward_noise.py` and includes three components:
1.  **Format Reward**: Checks if the output follows the required structure (e.g., `[Goal Analysis]`, `[Final Answer]`).
2.  **Process Reward**: Matches intermediate calculation steps (`<<...>>`) against the ground truth chain.
3.  **Outcome Reward**: Compares the final numerical answer.

## Troubleshooting

*   **OOM (Out of Memory)**: Reduce `actor_rollout_ref.rollout.gpu_memory_utilization` in the script.
*   **Flash Attention Error**: Ensure `use_remove_padding=False` is set in the script.
*   **Disk Full**: Checkpoints are saved to `Output/checkpoints/`. Ensure you have enough disk space before training.
