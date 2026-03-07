# Math-Noise Project Refactor (A-new-1)

This project is a refactor of the `A-new` codebase, aiming to provide a cleaner code structure and adapt the **Verl** framework for reinforcement learning training (supporting GRPO/PPO).

## Directory Structure

```
A-new-1/
├── data/               # Data processing related
│   ├── testdata        # Test datasets
│   ├── generate_data.py       # Unified data generation script (Backward / Program)
│   └── ...
├── sft/                # Supervised Fine-Tuning (SFT)
│   └── sft_train.py           # TRL-based SFT training script
├── rl/                 # Reinforcement Learning (Verl)
│   ├── convert_data.py        # Convert SFT data to Verl Parquet format
│   ├── reward_fn.py           # Custom reward functions (Format, Process, Outcome)
│   ├── train_verl.py          # Verl training entry point (Monkey-patch supports custom rewards)
│   └── run_verl.sh            # Script to launch Verl training
├── eval/               # Evaluation
│   └── eval_model.py          # vLLM-based model evaluation script
├── scripts/            # Run scripts
│   └── run_pipeline.sh        # One-click script to run the full pipeline
└── README.md           # This file
```

## Workflow

### 1. Data Generation
Generate Backward Reasoning or Python Reasoning data.
```bash
# Generate Backward data
python3 data/generate_data.py --mode backward --input raw_data.jsonl --output data/backward.jsonl

# Generate Program data
python3 data/generate_data.py --mode program_reason --input raw_data.jsonl --output data/program.jsonl
```

### 2. SFT Training
Perform Supervised Fine-Tuning on the base model.
```bash
python3 sft/sft_train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/backward.jsonl \
    --output_dir output/sft_model \
    --use_lora True
```

### 3. RL Training (Verl Framework)
Use Verl for GRPO/PPO training.
This step automatically converts JSONL data to Parquet format and invokes custom reward functions.

```bash
# Enter rl directory or use script
cd rl
bash run_verl.sh
```
`run_verl.sh` internally calls `train_verl.py`, configuring key parameters such as:
- `algorithm.adv_estimator=grpo` (Use GRPO)
- `reward.reward_manager.reward_fn_key=math_noise` (Use custom reward)
- `actor.model.path` points to the SFT model

### 4. Evaluation
Perform fast evaluation using vLLM.
```bash
python3 eval/eval_model.py \
    --model_path output/rl_model \
    --data_path data/test.jsonl \
    --output_dir output/eval_result
```
