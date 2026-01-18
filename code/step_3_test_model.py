import os
import sys
import logging
import torch
import json
import time
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List

import re
import math
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 模型和数据路径
MODEL_PATH = "/root/autodl-tmp/A-new/output/model/SFT/sft-mix-qwen2.5-0.5b-Instruct-checkpoints"
TEST_DATA_PATH = "/root/autodl-tmp/A-new/data/test_data/gsm8k-test.jsonl"
REPORT_PATH = "/root/autodl-tmp/A-new/output/model/SFT/test_mix_report.json"

# 评估参数
GEN_MAX_NEW_TOKENS = 128
GEN_BATCH_SIZE = 8
ANSWER_TOL = 1e-6

# 正则表达式提取答案
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text: str):
    if text is None:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    try:
        ans2 = ans.replace(",", "")
        return float(ans2)
    except Exception:
        return ans


def extract_gt_from_target(target: str):
    return extract_answer(target)


def is_correct(pred, gt, tol=1e-6):
    if pred is None or gt is None:
        return False
    if isinstance(pred, float) and isinstance(gt, float):
        return (math.isfinite(pred) and math.isfinite(gt) and abs(pred - gt) <= tol)
    return str(pred).strip() == str(gt).strip()


def ensure_messages_format(example: Dict[str, Any],
                           data_format: str = "prompt_target",
                           add_system_prompt: bool = True,
                           system_prompt: str = "You are a helpful assistant. Always answer in the following format ONLY:") -> Dict[str, Any]:
    """
    Convert prompt/target -> messages or keep messages.
    """
    if data_format == "messages":
        if "messages" not in example:
            raise ValueError("data_format=messages 但样本缺少 messages 字段")
        return example

    # prompt_target
    if "prompt" not in example or "target" not in example:
        raise ValueError("data_format=prompt_target 但样本缺少 prompt/target 字段")

    msgs = []
    if add_system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": example["prompt"]})
    msgs.append({"role": "assistant", "content": example["target"]})

    example["messages"] = msgs
    return example


@torch.no_grad()
def evaluate_by_kind_subtype(model, tokenizer, dataset, max_new_tokens=128, batch_size=8, device=None, tol=1e-6):
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据一致性检查
    logger.info("开始评测前的数据一致性检查...")
    sample = dataset[0] if len(dataset) > 0 else {}
    required_fields = ["messages", "target"]
    for field in required_fields:
        if field not in sample:
            logger.warning(f"评测数据缺少必要字段: {field}")
    if "messages" in sample and (not isinstance(sample["messages"], list) or len(sample["messages"]) < 2):
        logger.warning("评测数据的messages格式不正确")
    logger.info(f"数据一致性检查完成，共 {len(dataset)} 个样本")

    total = 0
    correct = 0

    kind_tot = Counter()
    kind_cor = Counter()
    subtype_tot = Counter()
    subtype_cor = Counter()

    fam = defaultdict(list)  # seed_id -> list[(kind, ok)]
    detailed_logs = []  # 详细日志列表

    def build_input_text(messages):
        # 只使用用户消息，不包含system prompt和assistant回复
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        # 只添加生成提示，不包含完整对话
        return tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)

    idxs = list(range(len(dataset)))
    for start in tqdm(range(0, len(idxs), batch_size), desc="Eval (generate)"):
        batch_ids = idxs[start:start + batch_size]
        batch = [dataset[i] for i in batch_ids]

        input_texts = [build_input_text(ex["messages"]) for ex in batch]
        enc = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 2048),
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=256,  # 增加生成长度
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # 解码时只获取生成的部分，不包含输入
        out_texts = []
        for i in range(len(gen)):
            # 只解码生成的新标记，不包含输入标记
            generated_tokens = gen[i, len(enc["input_ids"][i]):]
            out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            out_texts.append(out_text)

        for ex, out in zip(batch, out_texts):
            meta = ex.get("meta", {}) or {}
            kind = meta.get("kind", "unknown")
            subtype = meta.get("subtype", "unknown")
            sid = str(meta.get("seed_id", "unknown_seed"))
            
            # 获取用户输入（原始prompt）
            user_input = ex.get("prompt", "")
            # 获取真实target
            target = ex.get("target", "")
            
            # 提取答案
            gt = extract_gt_from_target(target)
            pred = extract_answer(out)
            ok = is_correct(pred, gt, tol=tol)

            total += 1
            correct += int(ok)

            kind_tot[kind] += 1
            kind_cor[kind] += int(ok)

            subtype_tot[subtype] += 1
            subtype_cor[subtype] += int(ok)

            fam[sid].append((kind, ok))
            
            # 保存详细日志
            detailed_logs.append({
                "sample_id": total,
                "seed_id": sid,
                "kind": kind,
                "subtype": subtype,
                "user_input": user_input,
                "model_output": out,
                "ground_truth": target,
                "predicted_answer": pred,
                "true_answer": gt,
                "is_correct": ok,
                "timestamp": datetime.now().isoformat()
            })

    report = {
        "overall": {"total": total, "correct": correct, "acc": correct / max(total, 1)},
        "by_kind": {
            k: {"total": kind_tot[k], "correct": kind_cor[k], "acc": kind_cor[k] / max(kind_tot[k], 1)}
            for k in kind_tot
        },
        "by_subtype": {
            s: {"total": subtype_tot[s], "correct": subtype_cor[s], "acc": subtype_cor[s] / max(subtype_tot[s], 1)}
            for s in subtype_tot
        },
        "families": {
            "n_families": len(fam),
            "seed_correct_rate": sum(1 for sid in fam if any(k == "seed" and ok for k, ok in fam[sid])) / max(len(fam), 1),
        },
        "detailed_logs": detailed_logs  # 包含详细日志
    }
    return report


def main():
    logger.info("开始加载模型和分词器...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",  # 修复右填充问题
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    
    # 加载模型
    _use_cuda = torch.cuda.is_available()
    if _use_cuda and torch.cuda.is_bf16_supported():
        _dtype = torch.bfloat16
    elif _use_cuda:
        _dtype = torch.float16
    else:
        _dtype = torch.float32
    
    _attn = "eager"
    if _use_cuda:
        try:
            import flash_attn  # noqa: F401
            _attn = "flash_attention_2"
        except Exception:
            _attn = "sdpa"
    
    logger.info(f"模型配置：dtype={_dtype}, attn_implementation={_attn}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=_dtype,
        attn_implementation=_attn,
        trust_remote_code=True,
        use_cache=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"模型已加载到 {device}")
    
    # 加载测试数据集
    logger.info(f"加载测试数据集：{TEST_DATA_PATH}")
    ds = load_dataset("json", data_files={"test": TEST_DATA_PATH})
    test_dataset = ds["test"]
    logger.info(f"测试数据集大小：{len(test_dataset)}")
    
    # 转换为messages格式
    def _map_to_messages(ex):
        return ensure_messages_format(ex)
    
    test_dataset = test_dataset.map(_map_to_messages, desc="Converting to messages format", num_proc=1)
    
    # 运行评估
    logger.info("开始评估模型...")
    test_report = evaluate_by_kind_subtype(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_new_tokens=256,  # 使用固定的大生成长度
        batch_size=GEN_BATCH_SIZE,
        device=device,
        tol=ANSWER_TOL,
    )
    
    # 准备保存路径和文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(TEST_DATA_PATH).replace(".jsonl", "")
    output_dir = "/root/autodl-tmp/A-new/output/result"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细日志到单独文件
    detailed_logs_path = os.path.join(
        output_dir, f"{current_time}_model_test_{dataset_name}_detailed.jsonl"
    )
    with open(detailed_logs_path, "w", encoding="utf-8") as f:
        for log_entry in test_report["detailed_logs"]:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"详细日志已保存到：{detailed_logs_path}")
    
    # 移除详细日志以减小报告文件大小
    report_without_logs = test_report.copy()
    report_without_logs.pop("detailed_logs")
    
    # 保存评估报告
    report_path = os.path.join(
        output_dir, f"{current_time}_model_test_{dataset_name}_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_without_logs, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存到：{report_path}")
    
    # 打印评估结果
    logger.info("\n=== 评估结果 ===")
    logger.info(f"整体准确率: {test_report['overall']['acc']:.4f}")
    logger.info("\n按类型准确率:")
    for kind, stats in test_report['by_kind'].items():
        logger.info(f"  {kind}: {stats['acc']:.4f} ({stats['correct']}/{stats['total']})")
    
    logger.info("\n按子类型准确率 (前10个):")
    sorted_subtypes = sorted(test_report['by_subtype'].items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    for subtype, stats in sorted_subtypes:
        logger.info(f"  {subtype}: {stats['acc']:.4f} ({stats['correct']}/{stats['total']})")
    
    logger.info(f"\n家族统计:")
    logger.info(f"  家族数量: {test_report['families']['n_families']}")
    logger.info(f"  种子样本正确率: {test_report['families']['seed_correct_rate']:.4f}")
    
    logger.info("\n评估完成！")


if __name__ == "__main__":
    main()
