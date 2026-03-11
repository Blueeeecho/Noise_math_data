import os
import sys
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from collections import defaultdict, Counter

import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from tqdm import tqdm

# Try wandb
try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_file: str = field(
        metadata={"help": "Path to training data (jsonl)."}
    )
    test_file: str = field(
        default=None,
        metadata={"help": "Path to test data (jsonl) for evaluation."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    
    use_wandb: bool = field(default=True)
    wandb_project: str = field(default="SFT-Training")
    wandb_run_name: str = field(default="sft_run")
    wandb_entity: Optional[str] = field(default=None)
    
    # Eval options
    eval_on_test_after_train: bool = field(default=False)
    gen_max_new_tokens: int = field(default=1024)
    gen_batch_size: int = field(default=8)
    strict_xml_only: bool = field(default=False)

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "train.log"))
        ]
    )

# =========================
# Evaluation Helpers
# =========================
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"\[Final Answer\]\s*(.*)", re.DOTALL | re.IGNORECASE)

def extract_answer(text: str, strict_xml_only: bool = True):
    if not text: return None
    ans = None
    # 1. XML
    all_ans = ANSWER_RE.findall(text)
    if all_ans: ans = all_ans[-1].strip()
    
    # 2. [Final Answer]
    if not ans:
        m = FINAL_ANSWER_RE.search(text)
        if m: ans = m.group(1).strip()
    
    if not ans: return None
    
    ans2 = ans.replace(",", "")
    try:
        return float(ans2)
    except:
        return None if strict_xml_only else ans

def is_correct(pred, gt, tol=1e-6):
    if pred is None or gt is None: return False
    if isinstance(pred, float) and isinstance(gt, float):
        return (math.isfinite(pred) and math.isfinite(gt) and abs(pred - gt) <= tol)
    return str(pred).strip() == str(gt).strip()

@torch.no_grad()
def evaluate_generation(model, tokenizer, dataset, args, device):
    model.eval()
    total, correct = 0, 0
    results = []
    
    # Prepare batch
    idxs = list(range(len(dataset)))
    
    def build_prompt(messages):
        # Apply template to all but last message (assistant)
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    for start in tqdm(range(0, len(idxs), args.gen_batch_size), desc="Evaluating"):
        batch_ids = idxs[start:start + args.gen_batch_size]
        batch = [dataset[i] for i in batch_ids]
        
        input_texts = [build_prompt(ex["messages"]) for ex in batch]
        enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        gen_ids = model.generate(
            **enc,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode only new tokens
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i, (ex, p_len) in enumerate(zip(batch, prompt_lens)):
            new_ids = gen_ids[i, p_len:]
            pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            
            gt_text = ex["messages"][-1]["content"]
            gt = extract_answer(gt_text, strict_xml_only=args.strict_xml_only)
            pred = extract_answer(pred_text, strict_xml_only=args.strict_xml_only)
            
            ok = is_correct(pred, gt)
            total += 1
            if ok: correct += 1
            
            results.append({
                "prompt": input_texts[i],
                "pred_text": pred_text,
                "gt_text": gt_text,
                "pred_val": pred,
                "gt_val": gt,
                "correct": ok
            })
            
    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "total": total, "correct": correct}, results


def main():
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, sft_config = parser.parse_args_into_dataclasses()
    
    setup_logging(sft_config.output_dir)
    logger.info(f"Script Args: {script_args}")
    
    # WandB Init
    if script_args.use_wandb and wandb:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ["WANDB_RUN_NAME"] = script_args.wandb_run_name
        if script_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = script_args.wandb_entity
        sft_config.report_to = ["wandb"]
    else:
        sft_config.report_to = []

    set_seed(sft_config.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        padding_side="left",  # Ensure left padding for generation
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    data_files = {"train": script_args.train_file}
    if script_args.test_file:
        data_files["test"] = script_args.test_file
        
    dataset = load_dataset("json", data_files=data_files)
    
    # Format check helper
    def ensure_format(ex):
        messages = ex.get("messages")
        
        # Handle raw backward data
        if not messages and "question" in ex and "backward_reasoning" in ex:
             messages = [
                 {"role": "user", "content": ex["question"]},
                 {"role": "assistant", "content": ex["backward_reasoning"]}
             ]

        # Handle prompt/completion
        if not messages and "prompt" in ex and "completion" in ex:
             messages = [
                 {"role": "user", "content": ex["prompt"]},
                 {"role": "assistant", "content": ex["completion"]}
             ]
             
        # Fallback for standard GSM8K style
        if not messages and "question" in ex and "answer" in ex:
             messages = [
                 {"role": "user", "content": ex["question"]},
                 {"role": "assistant", "content": ex["answer"]}
             ]
        
        if messages:
            # Apply chat template to create 'text' field which SFTTrainer expects by default
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                return {"messages": messages, "text": text}
            except Exception:
                # Fallback if template fails or messages invalid
                return {"messages": messages}
        return ex

    dataset = dataset.map(ensure_format)

    # LoRA
    peft_config = None
    if script_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_cache=False if sft_config.gradient_checkpointing else True
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {sft_config.output_dir}")
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    
    # Post-training Evaluation
    if script_args.eval_on_test_after_train and "test" in dataset:
        logger.info("Running generation-based evaluation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        metrics, results = evaluate_generation(model, tokenizer, dataset["test"], script_args, device)
        
        logger.info(f"Test Metrics: {metrics}")
        with open(os.path.join(sft_config.output_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        with open(os.path.join(sft_config.output_dir, "test_results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
                
        if wandb.run:
            wandb.log({"test/accuracy": metrics["accuracy"]})

if __name__ == "__main__":
    main()
