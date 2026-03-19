# 基于 Reasoning360 (Verl) 的 Noise Math 训练项目

本项目将 **Noise Math** 训练流程（源自 `Noise_math_data`）完整迁移并适配到了 **Reasoning360 (Verl)** 框架中。项目支持 **PPO** 和 **GRPO** 两种强化学习算法，并针对数学推理任务定制了包含格式、过程和结果奖励的复合奖励函数。

该实现与 Reasoning360 框架深度集成，充分利用了其分布式训练能力（FSDP、混合引擎），同时保留了处理抗噪数学推理所需的特定逻辑。

## 目录结构

所有相关文件均位于 `/root/autodl-tmp/Reasoning360/examples/noise_math/`：

```
noise_math/
├── dataset/                # 数据存储
│   ├── Ours/               # 原始输入数据存放路径 (如 all_backward_data.jsonl)
│   └── Processed/          # 转换后用于训练的 Parquet 格式数据
├── Output/                 # 训练输出目录
│   ├── sft_model/          # SFT 微调后的模型权重
│   ├── checkpoints/        # RL 强化学习保存的模型权重及训练中间参数
│   └── eval_results/       # 模型评估结果
├── scripts/                # 执行脚本
│   ├── run_sft.sh          # SFT 监督微调脚本
│   ├── prepare_data.sh     # RL 数据准备脚本 (JSONL -> Parquet)
│   ├── run_local.sh        # 本地 PPO 训练脚本 (单卡 / 4090D)
│   ├── run_local_grpo.sh   # 本地 GRPO 训练脚本 (单卡 / 4090D)
│   ├── run_eval.sh         # 模型评估脚本
│   ├── run_a100_ppo.sh     # 远程 PPO 训练脚本 (多卡 / 4x A100)
│   └── run_a100_grpo.sh    # 远程 GRPO 训练脚本 (多卡 / 4x A100)
├── sft_train.py            # SFT 训练代码
├── convert_data_noise.py   # 数据转换工具 (自动添加 'ability', 'extra_info' 等字段)
├── reward_noise.py         # 自定义奖励函数实现
├── eval_model.py           # 模型评估代码
├── README.md               # 英文版说明文档
└── README_CN.md            # 本文件 (中文说明文档)
```

## 前置要求

### 环境配置
请确保你已激活 `Reasoning360` conda 环境。本项目对依赖包的版本有特定要求，以确保与 Verl 框架兼容：

```bash
conda activate Reasoning360
# 必须使用兼容的 transformers 版本 (不支持 v5.x)
uv pip install "transformers==4.52.4"
# 必须安装兼容的 trl 库
uv pip install "trl<=0.9.6"
```

### 模型准备
请确保基础模型（如 `Qwen2.5-0.5B-Instruct`）已下载到本地。
脚本中的默认路径为：`/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct`

## 数据准备与格式详情

Reasoning360 (Verl) 框架要求训练数据必须为特定的 **Parquet** 格式。我们提供了一个脚本，用于将原始的 JSONL 数据转换为所需的格式。

### 1. 处理前的数据格式 (原始数据)
请将原始 JSONL 数据放置在 `/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Ours/all_backward_data.jsonl`。
原始数据是一个 JSON Lines (`.jsonl`) 文件，每行是一个 JSON 对象。转换脚本支持包含 `question` + `backward_reasoning` 或 `question` + `answer` 的数据格式。

**处理前的原始 JSONL 数据样例：**
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
  "backward_reasoning": "[Goal Analysis]\nTarget: Var{Total_Clips_Sold}\nPlan: ...\n[Final Answer]\n72"
}
```

### 2. 运行数据转换
执行数据准备脚本，它将读取 `dataset/Ours/all_backward_data.jsonl` 并在 `dataset/Processed/` 目录下生成 `train.parquet` 和 `test.parquet`。
```bash
# 运行数据准备脚本
bash scripts/prepare_data.sh
```

### 3. 处理后的数据格式 (Parquet格式)
`convert_data_noise.py` 脚本会将上述 JSONL 数据转换为 Verl 框架所需的 Parquet 格式。Parquet 文件中的每一行对应一个训练样本，并包含特定的结构化列。

**处理后的数据记录样例 (以 JSON 形式概念化展示)：**
```json
{
  "data_source": "math_noise",
  "prompt": [
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    }
  ],
  "ability": "math",
  "response": "[Goal Analysis]\nTarget: Var{Total_Clips_Sold}\nPlan: ...\n[Final Answer]\n72",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "gold_answer": "[Goal Analysis]\nTarget: Var{Total_Clips_Sold}\nPlan: ...\n[Final Answer]\n72",
      "gold_chain": "[Goal Analysis]\nTarget: Var{Total_Clips_Sold}\nPlan: ...\n[Final Answer]\n72"
    }
  },
  "extra_info": {
    "split": "train",
    "index": 0
  }
}
```

**处理后各字段说明：**
- `data_source`: 设置为 `"math_noise"`（供奖励函数管理器路由到正确的自定义奖励函数）。
- `prompt`: 标准的对话模板格式，包含 `role` 和 `content` 字典的列表。
- `ability`: 设置为 `"math"`。
- `response`: 标准答案字符串（可作为模型的 SFT 目标或参考）。
- `reward_model`: 一个嵌套字典，包含了计算奖励所需的基础标准数据。其中的 `gold_answer` 和 `gold_chain` 会被自定义奖励函数用于计算结果奖励和过程奖励。
- `extra_info`: 包含数据集切分名称、索引等元数据信息。

## 完整训练工作流 (SFT -> RL -> Eval)

为了保持最初的研究流程，本项目支持从 **监督微调 (SFT)** 到 **强化学习 (RL)** 再到 **模型评估 (Evaluation)** 的完整三步走流水线。所有生成的模型参数及中间结果都会保存在 `Output/` 目录下。

### 1. 执行 SFT 监督微调
首先执行 SFT 脚本，利用原始 JSONL 数据对基础模型（`Qwen2.5-0.5B-Instruct`）进行微调。
```bash
bash scripts/run_sft.sh
```
*   **输入**: `/root/autodl-tmp/Reasoning360/examples/noise_math/dataset/Ours/all_backward_data.jsonl`
*   **输出**: 训练好的 SFT 模型将保存到 `Output/sft_model` 目录。

### 2. 执行强化学习 (PPO / GRPO)
强化学习脚本已被修改为**优先自动读取** `Output/sft_model` 作为初始 Actor 模型。如果 SFT 未执行，脚本将自动回退（Fallback）到未微调的原始模型。

**方案 A: 本地运行 PPO 训练**
```bash
bash scripts/run_local.sh
```

**方案 B: 本地运行 GRPO 训练** (更节省显存)
```bash
bash scripts/run_local_grpo.sh
```

### 3. 模型评估 (Evaluation)
RL 训练完成后，使用评估脚本对最终的 checkpoint 进行准确率测试。
可以通过环境变量 `MODEL_PATH` 动态指定需要测试的模型路径。
```bash
MODEL_PATH="Output/checkpoints/qwen2.5-0.5b-grpo-test/global_step_10" bash scripts/run_eval.sh
```
*   **输出**: 评估结果和准确率指标（Metrics）将保存在 `Output/eval_results/` 目录下。

### 远程训练 (多卡 / 4x A100)
针对 4 张 A100 显卡的集群环境，我们提供了专门优化过超参数的独立脚本，方便你直接进行分布式训练。

**运行 PPO 算法**：
```bash
bash scripts/run_a100_ppo.sh
```

**运行 GRPO 算法**：
```bash
bash scripts/run_a100_grpo.sh
```

---

## 自定义奖励函数
奖励逻辑在 `reward_noise.py` 中实现，包含三个部分：
1.  **格式奖励 (Format Reward)**: 检查输出是否符合特定结构（如 `[Goal Analysis]`, `[Final Answer]`）。
2.  **过程奖励 (Process Reward)**: 匹配中间计算步骤（`<<...>>`）与标准答案的一致性。
3.  **结果奖励 (Outcome Reward)**: 比较最终数值答案的准确性。

## 故障排除指南

*   **OOM (显存不足)**: 尝试降低脚本中的 `actor_rollout_ref.rollout.gpu_memory_utilization` (例如降至 0.5) 或者减小 `rollout.n` (组大小)。
*   **Flash Attention 报错**: 确保脚本中设置了 `use_remove_padding=False`。
*   **磁盘空间不足**: 检查模型输出目录 `Output/checkpoints/` 以及 `/tmp/ray` 目录。如果遇到磁盘写满错误，请清理旧的 Checkpoint 或临时文件。
