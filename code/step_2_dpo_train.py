import os
import sys
import logging
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType

# 基础日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Args
# =========================
@dataclass
class ScriptArguments:
    # --- 路径配置 ---
    # [修改] 默认指向你 SFT 后的 Python 模型
    model_name_or_path: str = field(
        default="/root/autodl-tmp/A-new/output/model/sft/qwen2.5-1.5b-In/sft_train_test_py", 
        metadata={"help": "SFT 后的模型路径 (如果是 LoRA SFT，请确保已 Merge 或在此处正确加载)"}
    )
    # [修改] 指向转换好格式的 DPO 数据
    data_path: str = field(
        default="/root/autodl-tmp/A-new/data/dpo/new_1.5b_instruct/hard_dpo_all_end.jsonl",
        metadata={"help": "包含 chosen/rejected 列的 JSONL 数据"}
    )

    output_dir: str = field(
        default="/root/autodl-tmp/A-new/output/model/DPO/1.5B_instruct/new_end_3_hard_lr_2e_6",
        metadata={"help": "DPO 模型输出目录"}
    )
    
    # --- WandB ---
    use_wandb: bool = field(default=True)
    wandb_project: str = field(default="Robust-Reasoning-DPO")
    wandb_run_name: str = field(default="dpo_family_v1_1.5b_new_hard_lr_2e_6")



    # --- 训练超参 ---
    learning_rate: float = field(default=2e-6, metadata={"help": "DPO 学习率 (0.5B/1.5B 建议 5e-7 ~ 1e-6)"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "显存不够就调小，配合梯度累积"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "保持 global batch size 足够大"})
    num_train_epochs: int = field(default=2, metadata={"help": "DPO 很容易过拟合，1 个 epoch 通常足够"})
    beta: float = field(default=0.1, metadata={"help": "DPO KL 惩罚系数 (0.1 是标准值)"})
    
    # --- LoRA ---
    use_lora: bool = field(default=True, metadata={"help": "DPO 建议使用 LoRA 以节省显存并稳定训练"})

    # --- Resume ---
    resume_training: bool = field(default=False)
    resume_from_checkpoint: str = field(default="")


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # --- WandB Setup ---
    if script_args.use_wandb:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ["WANDB_RUN_NAME"] = script_args.wandb_run_name

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # DPO 必须左侧 Padding
    tokenizer.padding_side = "left"

    # Template 兜底
    if not tokenizer.chat_template:
        logger.warning("Tokenizer has no chat_template, using fallback template.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # 2. 模型加载
    # [核心修改] 强制不使用 flash_attention_2
    # 优先尝试 "sdpa" (PyTorch 2.1+ 自带加速)，如果报错则改为 "eager" (最慢但最兼容)
    try:
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_impl = "sdpa" 
        else:
            attn_impl = "eager"
    except:
        attn_impl = "eager"
        
    logger.info(f"Loading model with attn_implementation='{attn_impl}' (No FA2)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, # 强制指定
        trust_remote_code=True
    )

    # 3. LoRA 配置
    peft_config = None
    if script_args.use_lora:
        logger.info("Using LoRA for DPO...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    # 4. 数据集
    dataset = load_dataset("json", data_files=script_args.data_path, split="train")
    
    # 5. Config
    dpo_config = DPOConfig(
        output_dir=script_args.output_dir,
        beta=script_args.beta,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        num_train_epochs=script_args.num_train_epochs,
        gradient_checkpointing=True,
        bf16=True,
        max_length=2048,
        max_prompt_length=1024,
        report_to="wandb" if script_args.use_wandb else "none",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False,
        dataset_num_proc=1,
    )

    # 6. Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training (No FA2)...")
    
    resume_arg = None
    if script_args.resume_training:
        if script_args.resume_from_checkpoint:
            resume_arg = script_args.resume_from_checkpoint
        else:
            resume_arg = True
        
    trainer.train(resume_from_checkpoint=resume_arg)
    
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)

if __name__ == "__main__":
    main()