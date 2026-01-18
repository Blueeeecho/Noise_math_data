import os
import sys
import json
import math
import re
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

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

# try import wandb (optional)
try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


# =========================
# Args
# =========================
@dataclass
class ScriptArguments:
    # --- Model ---
    model_name_or_path: str = field(
        default="/root/autodl-tmp/model/Qwen2.5-1.5B-Instruct",
        metadata={"help": "模型路径（本地或HF）"},
    )

    # --- Data (train/test only) ---
    train_path: str = field(
        default="/root/autodl-tmp/A-new/data/sft_data_pure/train_sft_py.jsonl",
        metadata={"help": "训练集 jsonl 路径"},
    )
    test_path: str = field(
        default="/root/autodl-tmp/A-new/data/sft_data_pure/test_sft_py.jsonl",
        metadata={"help": "测试集 jsonl 路径（只做评测/统计）"},
    )

    # --- Output ---
    save_dir: str = field(
        default="/root/autodl-tmp/A-new/output/model/sft/qwen2.5-1.5b-In/sft_train_test_py",
        metadata={"help": "输出目录"},
    )

    # --- WandB ---
    use_wandb: bool = field(default=True, metadata={"help": "是否开启 wandb"})
    wandb_project: str = field(default="Robust-Reasoning-Python", metadata={"help": "WandB project"})
    wandb_run_name: str = field(default="sft_1.5b_train_test_v2_py", metadata={"help": "WandB run name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "WandB entity (optional)"})

    # --- Data format compatibility ---
    # messages: {"messages":[{role,content},...], "meta":..., ...}
    # prompt_target: {"prompt": "...", "target": "...", "meta":..., ...}
    data_format: str = field(
        default="messages",
        metadata={"help": "messages 或 prompt_target"},
    )
    add_system_prompt: bool = field(
        default=False,
        metadata={"help": "是否为每条样本额外添加 system（默认 False，避免改变分布）"},
    )
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"help": "system prompt 内容（仅 add_system_prompt=True 时生效）"},
    )

    # --- Train mode ---
    use_lora: bool = field(default=False, metadata={"help": "是否使用 LoRA"})

    # --- Optional checkpoint cleanup ---
    keep_top_k: int = field(
        default=0,
        metadata={"help": "训练结束后按 eval_loss 保留 top-k ckpt；0=不清理（train/test-only 场景通常关掉）"},
    )

    # --- Generation eval ---
    eval_on_test_after_train: bool = field(default=True, metadata={"help": "训练后是否跑生成式 test 评测"})
    gen_max_new_tokens: int = field(default=1024, metadata={"help": "生成评测 max_new_tokens"})
    gen_batch_size: int = field(default=8, metadata={"help": "生成评测 batch_size"})
    answer_tol: float = field(default=1e-6, metadata={"help": "数值容差"})
    strict_xml_only: bool = field(
        default=False,
        metadata={"help": "True=必须有<answer>且float可解析才算有效；False=允许字符串答案"},
    )


# =========================
# Logging
# =========================
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_file = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(fh)

    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"日志将写入: {log_file}")


# =========================
# Checkpoint cleanup
# =========================
def keep_top_k_checkpoints(
    output_dir: str,
    k: int = 3,
    metric_name: str = "eval_loss",
    greater_is_better: bool = False,
):
    if k <= 0:
        return

    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        logger.warning(f"未找到 {state_path}，无法进行 top-{k} checkpoint 清理。")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    records = []
    for entry in log_history:
        if metric_name in entry and "step" in entry:
            records.append((entry["step"], entry[metric_name]))

    if not records:
        logger.warning(f"log_history 中未找到 {metric_name} 记录，跳过 top-{k} 清理。")
        return

    if len(records) <= k:
        logger.info(f"eval 次数为 {len(records)} <= {k}，无需清理 checkpoint。")
        return

    records_sorted = sorted(records, key=lambda x: x[1], reverse=greater_is_better)
    best_steps = [r[0] for r in records_sorted[:k]]
    best_ckpt_names = {f"checkpoint-{step}" for step in best_steps}

    logger.info(f"根据 {metric_name} 选出的 top-{k} checkpoint: {best_ckpt_names}")

    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path) and name.startswith("checkpoint-"):
            if name not in best_ckpt_names:
                logger.info(f"删除非最优 checkpoint: {name}")
                shutil.rmtree(full_path, ignore_errors=True)

    logger.info(f"top-{k} checkpoint 清理完成。")


# =========================
# Family stats
# =========================
def compute_family_stats(dataset, name: str) -> Dict[str, Any]:
    fam2cnt = Counter()
    kind_cnt = Counter()
    subtype_cnt = Counter()

    n = len(dataset)
    for ex in dataset:
        meta = ex.get("meta", {}) or {}
        sid = str(meta.get("seed_id", "unknown_seed"))
        fam2cnt[sid] += 1
        kind_cnt[meta.get("kind", "unknown")] += 1
        subtype_cnt[meta.get("subtype", "unknown")] += 1

    size_dist = Counter(fam2cnt.values())
    return {
        "split": name,
        "samples": n,
        "families": len(fam2cnt),
        "family_size_distribution": dict(sorted(size_dist.items())),
        "kind_distribution": dict(kind_cnt),
        "subtype_top20": subtype_cnt.most_common(20),
    }


def log_family_stats(ds_dict: DatasetDict):
    for split_name in ds_dict.keys():
        st = compute_family_stats(ds_dict[split_name], split_name)
        logger.info(f"==== FAMILY STATS [{split_name}] ====")
        logger.info(json.dumps(st, ensure_ascii=False, indent=2))


# =========================
# Data conversion
# =========================
def ensure_messages_format(
    example: Dict[str, Any],
    data_format: str,
    add_system_prompt: bool,
    system_prompt: str,
) -> Dict[str, Any]:
    """
    Convert prompt/target -> messages or keep messages.
    """
    if data_format == "messages":
        if "messages" not in example:
            raise ValueError("data_format=messages 但样本缺少 messages 字段")
        return example

    if data_format != "prompt_target":
        raise ValueError(f"未知 data_format={data_format}，仅支持 messages / prompt_target")

    if "prompt" not in example or "target" not in example:
        raise ValueError("data_format=prompt_target 但样本缺少 prompt/target 字段")

    msgs = []
    if add_system_prompt and system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    msgs.append({"role": "user", "content": example["prompt"]})
    msgs.append({"role": "assistant", "content": example["target"]})
    example["messages"] = msgs
    return example


def messages_to_prompt_completion(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build 'prompt' and 'completion' columns for TRL SFTTrainer.
    prompt = chat_template(messages[:-1]) with add_generation_prompt=True
    completion = messages[-1].content + eos
    """
    msgs = example.get("messages", None)
    if not isinstance(msgs, list) or len(msgs) < 2:
        raise ValueError("messages 缺失或格式不合法")

    if msgs[-1].get("role") != "assistant":
        raise ValueError("messages 最后一条不是 assistant，无法拆 completion")

    prompt_msgs = msgs[:-1]
    completion_text = msgs[-1].get("content", "")

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    if tokenizer.eos_token and not completion_text.endswith(tokenizer.eos_token):
        completion_text = completion_text + tokenizer.eos_token

    example["prompt"] = prompt_text
    example["completion"] = completion_text
    return example


# =========================
# Generation-based evaluation (robust)
# =========================
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text: str, strict_xml_only: bool = True):
    """
    - 取最后一个 <answer>...</answer>
    - strict_xml_only=True: 必须能转 float，否则 None
      strict_xml_only=False: 不能转 float 就返回字符串
    """
    if not text:
        return None
    all_ans = ANSWER_RE.findall(text)
    if not all_ans:
        return None
    ans = all_ans[-1].strip()
    ans2 = ans.replace(",", "")
    try:
        return float(ans2)
    except Exception:
        return None if strict_xml_only else ans


def is_correct(pred, gt, tol=1e-6):
    if pred is None or gt is None:
        return False
    if isinstance(pred, float) and isinstance(gt, float):
        return (math.isfinite(pred) and math.isfinite(gt) and abs(pred - gt) <= tol)
    return str(pred).strip() == str(gt).strip()


@torch.no_grad()
def evaluate_by_kind_subtype(
    model,
    tokenizer,
    dataset,
    max_new_tokens=1024,
    batch_size=8,
    device=None,
    tol=1e-6,
    strict_xml_only=True,
):
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    total = 0
    correct = 0

    kind_tot = Counter()
    kind_cor = Counter()
    subtype_tot = Counter()
    subtype_cor = Counter()

    fam = defaultdict(list)  # seed_id -> list[(kind, ok)]

    def build_input_text(messages):
        # 输入：system+user（以及可能的数据内前置消息），不包含最后的 assistant label
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

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

        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 只 decode 新生成部分，避免 prompt 中出现 <answer> 干扰
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        pred_texts = []
        for i in range(gen_ids.size(0)):
            new_ids = gen_ids[i, prompt_lens[i]:]
            pred_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        for ex, pred_text in zip(batch, pred_texts):
            meta = ex.get("meta", {}) or {}
            kind = meta.get("kind", "unknown")
            subtype = meta.get("subtype", "unknown")
            sid = str(meta.get("seed_id", "unknown_seed"))

            gt_text = ex["messages"][-1].get("content", "")
            gt = extract_answer(gt_text, strict_xml_only=strict_xml_only)
            pred = extract_answer(pred_text, strict_xml_only=strict_xml_only)
            ok = is_correct(pred, gt, tol=tol)

            total += 1
            correct += int(ok)

            kind_tot[kind] += 1
            kind_cor[kind] += int(ok)

            subtype_tot[subtype] += 1
            subtype_cor[subtype] += int(ok)

            fam[sid].append((kind, ok))

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
            "seed_correct_rate": (
                sum(1 for sid in fam if any(k == "seed" and ok for k, ok in fam[sid])) / max(len(fam), 1)
            ),
        },
    }
    return report


# =========================
# Main
# =========================
def main():
    parser = HfArgumentParser((ScriptArguments, SFTConfig))

    # 无命令行参数时：给一套合理默认超参
    if len(sys.argv) == 1:
        script_args, sft_config = parser.parse_args_into_dataclasses(args=[])

        sft_config.output_dir = script_args.save_dir
        sft_config.logging_steps = 10

        sft_config.save_strategy = "steps"
        sft_config.save_steps = 2000
        sft_config.save_total_limit = None

        # ✅ 只有 train/test，没有 dev；这里不做训练中 eval（避免误用 test 作为 eval）
        sft_config.evaluation_strategy = "no"
        sft_config.do_eval = False

        sft_config.per_device_train_batch_size = 4
        sft_config.gradient_accumulation_steps = 4
        sft_config.num_train_epochs = 3
        sft_config.learning_rate = 2e-5

        sft_config.no_cuda = False
        sft_config.bf16 = True
        sft_config.fp16 = False

        sft_config.gradient_checkpointing = True

        sft_config.weight_decay = 0.1
        sft_config.warmup_ratio = 0.03
        sft_config.lr_scheduler_type = "cosine"
        sft_config.max_grad_norm = 1.0

        sft_config.max_seq_length = 1024
        sft_config.packing = False

        sft_config.dataset_text_field = None
        if hasattr(sft_config, "assistant_only_loss"):
            sft_config.assistant_only_loss = False

        # train/test-only：不 load_best_model_at_end
        sft_config.load_best_model_at_end = False

        # 保留 meta / messages -> prompt/completion 映射时需要
        sft_config.remove_unused_columns = False

    else:
        script_args, sft_config = parser.parse_args_into_dataclasses()
        if not getattr(sft_config, "output_dir", None):
            sft_config.output_dir = script_args.save_dir
        if getattr(sft_config, "max_seq_length", None) is None:
            sft_config.max_seq_length = 1024
        if getattr(sft_config, "packing", None) is None:
            sft_config.packing = False
        if getattr(sft_config, "dataset_text_field", None) is None:
            sft_config.dataset_text_field = None
        if getattr(sft_config, "remove_unused_columns", None) is None:
            sft_config.remove_unused_columns = False

        # 强制 train/test-only 不在训练中用 test 做 eval
        sft_config.evaluation_strategy = "no"
        sft_config.do_eval = False
        sft_config.load_best_model_at_end = False

    # WandB env + TRL report_to
    if script_args.use_wandb and wandb is not None:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ["WANDB_NAME"] = script_args.wandb_run_name
        if script_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = script_args.wandb_entity
        sft_config.report_to = ["wandb"]
        sft_config.run_name = script_args.wandb_run_name
    else:
        sft_config.report_to = []
        os.environ["WANDB_DISABLED"] = "true"

    # logging
    setup_logging(sft_config.output_dir)
    logger.info(f"ScriptArguments: {script_args}")
    logger.info(f"SFTConfig: {sft_config}")

    # seed
    set_seed(42)
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    logger.info(f"Versions - torch: {torch.__version__}, transformers: {transformers.__version__}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")

    # model dtype & attention
    logger.info("正在加载模型...")
    _use_cuda = torch.cuda.is_available()
    if _use_cuda and torch.cuda.is_bf16_supported() and getattr(sft_config, "bf16", False):
        _dtype = torch.bfloat16
    elif _use_cuda and getattr(sft_config, "fp16", False):
        _dtype = torch.float16
    elif _use_cuda:
        _dtype = torch.float32
    else:
        _dtype = torch.float32

    _attn = "eager"
    if _use_cuda:
        try:
            import flash_attn  # noqa: F401
            _attn = "flash_attention_2"
        except Exception:
            _attn = "sdpa"

    logger.info(f"模型配置：dtype={_dtype}, attn_implementation={_attn}, gradient_checkpointing={getattr(sft_config, 'gradient_checkpointing', False)}")

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=_dtype,
        attn_implementation=_attn,
        trust_remote_code=True,
        use_cache=False if getattr(sft_config, "gradient_checkpointing", False) else True,
    )

    # LoRA
    peft_config = None
    if script_args.use_lora:
        logger.info("检测到 LoRA 模式开启")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    # load dataset: train/test
    if not os.path.exists(script_args.train_path):
        raise FileNotFoundError(f"train_path not found: {script_args.train_path}")
    if not os.path.exists(script_args.test_path):
        raise FileNotFoundError(f"test_path not found: {script_args.test_path}")

    data_files = {"train": script_args.train_path, "test": script_args.test_path}
    ds = load_dataset("json", data_files=data_files)
    logger.info(f"数据集加载完成：train={len(ds['train'])}, test={len(ds['test'])}")

    # convert to messages (if needed)
    def _map_to_messages(ex):
        return ensure_messages_format(
            ex,
            data_format=script_args.data_format,
            add_system_prompt=script_args.add_system_prompt,
            system_prompt=script_args.system_prompt,
        )

    ds = ds.map(_map_to_messages, desc="Converting to messages format", num_proc=1)

    # family stats
    log_family_stats(ds)

    train_dataset = ds["train"]
    test_dataset = ds["test"]

    # sanity: show first 2 formatted messages
    logger.info("*** Data Sanity Check (First 2 Train Examples, messages->chat_template) ***")
    for i in range(min(2, len(train_dataset))):
        sample = train_dataset[i]
        formatted = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        logger.info(f"Sample {i}:\n{formatted}\n{'=' * 40}")

    # build prompt/completion for trainer
    def _to_prompt_completion(ex):
        return messages_to_prompt_completion(ex, tokenizer)

    ds_pc = DatasetDict({
        "train": train_dataset.map(_to_prompt_completion, desc="Build prompt/completion (train)", num_proc=1),
        "test": test_dataset.map(_to_prompt_completion, desc="Build prompt/completion (test)", num_proc=1),
    })

    # drop messages from ds_pc to save RAM
    for split_name in list(ds_pc.keys()):
        cols = ds_pc[split_name].column_names
        drop_cols = [c for c in ["messages"] if c in cols]
        if drop_cols:
            ds_pc[split_name] = ds_pc[split_name].remove_columns(drop_cols)

    train_dataset_pc = ds_pc["train"]

    # sanity prompt/completion
    logger.info("*** Data Sanity Check (First 2 Train Examples, prompt/completion) ***")
    for i in range(min(2, len(train_dataset_pc))):
        logger.info(f"[prompt]\n{train_dataset_pc[i]['prompt']}\n")
        logger.info(f"[completion]\n{train_dataset_pc[i]['completion']}\n{'=' * 40}")

    # force single-proc preprocess inside TRL if supported
    if hasattr(sft_config, "dataset_num_proc"):
        sft_config.dataset_num_proc = 1
    if hasattr(sft_config, "num_proc"):
        sft_config.num_proc = 1

    # init trainer (compat TRL versions: processing_class vs tokenizer)
    logger.info("初始化 SFTTrainer ...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset_pc,
            eval_dataset=None,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset_pc,
            eval_dataset=None,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    train_dataloader = trainer.get_train_dataloader()
    logger.info(
        f"Train dataloader steps per epoch: {len(train_dataloader)}, "
        f"num_train_epochs: {getattr(sft_config, 'num_train_epochs', 'NA')}"
    )

    # train
    logger.info("开始训练流程...")
    train_result = trainer.train()

    # save model/tokenizer
    logger.info(f"训练结束，保存模型至 {sft_config.output_dir}")
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    # save metrics/state
    metrics = getattr(train_result, "metrics", {}) or {}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # save args/config
    with open(os.path.join(sft_config.output_dir, "script_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(script_args), f, indent=2, ensure_ascii=False)

    try:
        cfg = sft_config.to_dict()
        with open(os.path.join(sft_config.output_dir, "sft_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存 sft_config.json 失败: {e}")
        try:
            json_str = sft_config.to_json_string()
            with open(os.path.join(sft_config.output_dir, "sft_config.json"), "w", encoding="utf-8") as f:
                f.write(json_str)
        except Exception as e2:
            logger.error(f"使用 to_json_string() 保存 sft_config.json 也失败: {e2}")

    # optional: clean checkpoints
    if script_args.keep_top_k and script_args.keep_top_k > 0:
        keep_top_k_checkpoints(
            output_dir=sft_config.output_dir,
            k=script_args.keep_top_k,
            metric_name="eval_loss",
            greater_is_better=False,
        )

    # generation-based test report by kind/subtype
    if script_args.eval_on_test_after_train:
        logger.info("开始 test 生成式评测（按 kind/subtype 聚合）...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        test_report = evaluate_by_kind_subtype(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,  # 原始 test_dataset（含 messages/meta）
            max_new_tokens=script_args.gen_max_new_tokens,
            batch_size=script_args.gen_batch_size,
            device=device,
            tol=script_args.answer_tol,
            strict_xml_only=script_args.strict_xml_only,
        )

        report_path = os.path.join(sft_config.output_dir, "test_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"test_report.json 已写入：{report_path}")
        logger.info(json.dumps(test_report, ensure_ascii=False, indent=2))

        # wandb log (optional)
        if script_args.use_wandb and wandb is not None and wandb.run is not None:
            wandb.log({
                "test/acc": test_report["overall"]["acc"],
                "test/seed_correct_rate": test_report["families"]["seed_correct_rate"],
            })
            for k in ["no_op", "sem", "seed"]:
                if k in test_report.get("by_kind", {}):
                    wandb.log({f"test/kind_{k}_acc": test_report["by_kind"][k]["acc"]})

    # print wandb url if any
    try:
        if wandb is not None and wandb.run is not None:
            logger.info(f"W&B run URL: {wandb.run.url}")
    except Exception:
        pass

    logger.info("所有步骤完成。")


if __name__ == "__main__":
    main()