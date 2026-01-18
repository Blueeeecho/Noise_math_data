import os
import re
import json
import math
import signal
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/A-new/output/model/sft_train_test_py"

# 你的 jsonl：每行是 {"messages":[...], "meta": {...}}
LABELED_DATA_PATH = "/root/autodl-tmp/A-new/data/sft_data_pure/train_sft_py.jsonl"

OUTPUT_DPO_PATH = "/root/autodl-tmp/A-new/data/dpo/0.5B_instruct/train_from_messages_vllm.jsonl"

BATCH_SIZE = 16
MAX_INPUT_LEN = 1024
MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.7  # 用采样诱导错误
TOP_P = 0.9
# =========================================


# ----------- 解析工具 -----------
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.DOTALL | re.IGNORECASE)
ANS_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def extract_python_code(text: str) -> Optional[str]:
    m = PLAN_RE.search(text or "")
    return m.group(1).strip() if m else None

def extract_answer_number(text: str) -> Optional[float]:
    """从 <answer> 里解析数字，作为标准答案（gt）。"""
    m = ANS_RE.search(text or "")
    if not m:
        return None
    s = m.group(1).strip()
    s = s.replace(",", "")
    # 允许 "4", "4.0", "-3", "2/3"(不支持)等；这里只做 float
    try:
        return float(s)
    except:
        return None


# ----------- 安全执行 solve() -----------
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("timeout")

def execute_generated_code(code_str: str, timeout_sec: int = 2) -> Tuple[Optional[float], str]:
    """
    执行 <plan> 中的 def solve(): 并返回 solve() 的数值结果。
    返回: (value or None, status)
    status in {"success","no_code","format_error","no_solve_func","exec_error","timeout","nan_inf"}
    """
    if not code_str:
        return None, "no_code"
    if "def solve():" not in code_str:
        return None, "format_error"

    # 超时保护（Linux 可用）
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)

    local_env = {}
    try:
        # 允许基本运算；如你需要更严格可替换 __builtins__
        exec(code_str, {}, local_env)
        if "solve" not in local_env:
            return None, "no_solve_func"
        out = local_env["solve"]()

        val = float(out)
        if math.isnan(val) or math.isinf(val):
            return None, "nan_inf"
        return val, "success"

    except TimeoutException:
        return None, "timeout"
    except Exception:
        return None, "exec_error"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def build_prompt_messages(rec: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    从你的 {"messages":[...]} 里抽取 system+user 作为 prompt。
    假设结构为 system -> user -> assistant（你示例就是这样）
    """
    msgs = rec.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    prompt = []
    for m in msgs:
        role = m.get("role")
        if role in ("system", "user"):
            prompt.append({"role": role, "content": m.get("content", "")})
    if len(prompt) < 2:
        return None
    return prompt


def get_assistant_content(rec: Dict[str, Any]) -> Optional[str]:
    msgs = rec.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return None


def validate_chosen(rec: Dict[str, Any], tol: float = 1e-4) -> Tuple[bool, Optional[float], str]:
    """
    chosen 校验策略：
    - 标准答案 gt 来自 labeled assistant 的 <answer>
    - 执行 labeled assistant 的 <plan> solve()
    - solve() 输出与 gt 一致 才通过
    注意：你说不考虑 <answer> 的“生成端”，但这里用它当“标准答案来源”。
    """
    chosen_text = get_assistant_content(rec)
    if not chosen_text:
        return False, None, "no_assistant"

    gt = extract_answer_number(chosen_text)
    if gt is None:
        return False, None, "no_gt_answer_tag"

    code = extract_python_code(chosen_text)
    pred, status = execute_generated_code(code)
    if status != "success" or pred is None:
        return False, gt, f"chosen_exec_{status}"

    if abs(pred - gt) > tol:
        return False, gt, "chosen_wrong_vs_gt"

    return True, gt, "ok"


def main():
    os.makedirs(os.path.dirname(OUTPUT_DPO_PATH), exist_ok=True)

    print(f"[1] Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2] Loading vLLM model from: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        # 你多卡可加 tensor_parallel_size=N
    )

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        # 可选：如果你希望生成到 </answer> 就停
        stop=["</answer>"]
    )

    print(f"[3] Loading labeled data: {LABELED_DATA_PATH}")
    raw_recs = []
    with open(LABELED_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_recs.append(json.loads(line))

    print(f"Total records loaded: {len(raw_recs)}")

    # 过滤：只有 chosen 执行正确、且与标准答案一致的才保留
    tasks = []
    bad_stats = defaultdict(int)

    for rec in raw_recs:
        ok, gt, reason = validate_chosen(rec)
        if not ok:
            bad_stats[reason] += 1
            continue

        prompt_msgs = build_prompt_messages(rec)
        chosen_text = get_assistant_content(rec)
        meta = rec.get("meta", {}) or {}

        if not prompt_msgs or not chosen_text:
            bad_stats["bad_prompt_or_chosen"] += 1
            continue

        tasks.append({
            "prompt_msgs": prompt_msgs,  # list[dict]
            "chosen_text": chosen_text,  # str (XML with reasoning/plan/answer)
            "gt": float(gt),
            "meta": {
                "seed_id": meta.get("seed_id"),
                "kind": meta.get("kind"),
            }
        })

    print(f"Valid chosen records after exec+gt check: {len(tasks)}")
    if bad_stats:
        print("Filtered-out reasons:", dict(bad_stats))

    # vLLM 批量生成 + 挖 rejected
    dpo_pairs = []
    kind_stats = defaultdict(int)

    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Mining rejected with vLLM"):
        batch = tasks[i:i + BATCH_SIZE]

        # chat template -> plain prompt string
        prompt_texts = []
        for item in batch:
            prompt_texts.append(
                tokenizer.apply_chat_template(
                    item["prompt_msgs"],
                    tokenize=False,
                    add_generation_prompt=True
                )
            )

        outputs = llm.generate(prompt_texts, sampling)

        # vLLM 输出顺序与输入对齐
        for item, out in zip(batch, outputs):
            gen_text = out.outputs[0].text  # 生成的 assistant 内容（不含 prompt 部分）
            gt = item["gt"]

            # 判 rejected：执行失败 or 数值不等于 gt
            code = extract_python_code(gen_text)
            pred, status = execute_generated_code(code)

            is_rejected = False
            if status != "success" or pred is None:
                is_rejected = True
            else:
                if abs(pred - gt) > 1e-4:
                    is_rejected = True

            if is_rejected:
                dpo_pairs.append({
                    "prompt": item["prompt_msgs"],
                    "chosen": [{"role": "assistant", "content": item["chosen_text"]}],
                    "rejected": [{"role": "assistant", "content": gen_text}],
                    "meta": item["meta"]
                })
                kind_stats[item["meta"].get("kind", "unknown")] += 1

    print(f"[4] Mined DPO pairs: {len(dpo_pairs)}")
    print("Pairs distribution by kind:", dict(kind_stats))

    # 保存：trl 通常不需要 meta；你也可以保留
    with open(OUTPUT_DPO_PATH, "w", encoding="utf-8") as f:
        for p in dpo_pairs:
            out_p = {k: v for k, v in p.items() if k != "meta"}
            f.write(json.dumps(out_p, ensure_ascii=False) + "\n")

    print(f"[5] Saved to: {OUTPUT_DPO_PATH}")


if __name__ == "__main__":
    main()