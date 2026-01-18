import argparse
import json
import os
import re
import math
import signal
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM not installed. Please `pip install vllm`.")
    raise


# =========================
# Regex extractors
# =========================
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.DOTALL | re.IGNORECASE)
ANS_RE  = re.compile(r"<answer>\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*</answer>", re.DOTALL | re.IGNORECASE)

PROBLEM_PREFIX_RE = re.compile(r"Problem:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/A-new/output/model/sft/qwen2.5-1.5b-In/sft_train_test_py", help="SFT 模型路径")
    p.add_argument("--data_path", type=str, default="/root/autodl-tmp/A-new/data/sft_data_pure/all_sft_all.jsonl", help="输入 jsonl（兼容 messages 格式或 question/gold 格式）")
    p.add_argument("--output_dpo_path", type=str, default="/root/autodl-tmp/A-new/data/dpo/new_1.5b_instruct/hard_dpo_all_end.jsonl", help="输出 DPO jsonl")
    p.add_argument("--batch_size", type=int, default=200, help="vLLM batch size")
    p.add_argument("--n_samples", type=int, default=16, help="best-of-N：每题采样候选数")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_pairs_per_item", type=int, default=2, help="每题最多输出多少条 (chosen, rejected) pair")
    p.add_argument("--max_wrong_candidates", type=int, default=8, help="每题最多考虑多少个 hard wrong（用于挑选 rejected）")
    p.add_argument("--exec_timeout", type=int, default=2, help="执行 solve() 的超时秒数")
    p.add_argument("--tol", type=float, default=1e-6, help="数值判等容忍度")
    p.add_argument("--stop_on_answer", action="store_true", help="是否在 </answer> 停止生成（减少废话）")
    p.add_argument("--require_plan_solve", action="store_true", help="强制候选必须含 <plan> 且 def solve() 才参与评估（推荐开）")
    return p.parse_args()

# =========================
# Utils: numeric parse
# =========================
def _to_float_maybe(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    s = str(x).strip()
    if not s:
        return None
    # handle GSM8K-like "#### 42"
    if "####" in s:
        s = s.split("####")[-1].strip()
    s = s.replace(",", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None


def extract_gold_from_assistant_xml(assistant_text: str) -> Optional[float]:
    if not assistant_text:
        return None
    m = ANS_RE.search(assistant_text)
    if not m:
        return None
    return _to_float_maybe(m.group(1))


def extract_plan_code(text: str) -> Optional[str]:
    if not text:
        return None
    m = PLAN_RE.search(text)
    return m.group(1).strip() if m else None


# =========================
# Safe execute solve() with timeout
# =========================
class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("timeout")


def execute_solve_from_code(code_str: str, timeout_sec: int) -> Tuple[Optional[float], str]:
    """
    Execute code_str containing def solve(): ...
    Return (pred_value, status)
    status: success/no_code/format_error/no_solve_func/exec_error/timeout/nan_inf
    """
    if not code_str:
        return None, "no_code"
    if "def solve():" not in code_str:
        return None, "format_error"

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)

    local_env = {}
    try:
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


def is_close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


# =========================
# Data adapters (compat with your earlier messages format)
# =========================
DEFAULT_SYSTEM_PROMPT = (
    "You are a reasoning engine. Solve the math problem by strictly generating the XML output: "
    "<reasoning>, <plan>, and <answer>."
)


def get_prompt_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return prompt messages in ChatML: [{"role":"system"...},{"role":"user"...}]
    Support:
      - item["messages"] exists: use system+user from it
      - else: use DEFAULT_SYSTEM_PROMPT + item["question"]
    """
    if "messages" in item and isinstance(item["messages"], list):
        sys_msg = None
        user_msg = None
        for m in item["messages"]:
            if m.get("role") == "system" and sys_msg is None:
                sys_msg = m.get("content", "")
            if m.get("role") == "user" and user_msg is None:
                user_msg = m.get("content", "")
        if sys_msg is None:
            sys_msg = DEFAULT_SYSTEM_PROMPT
        if user_msg is None:
            # fallback to question
            q = item.get("question", "")
            user_msg = f"Problem:\n{q}"
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    # question/gold format
    q = item.get("question", "")
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{q}"}]


def get_question_text(item: Dict[str, Any]) -> str:
    """
    Best-effort: return a plain question string.
    """
    if "question" in item and item["question"]:
        return str(item["question"])
    # parse from messages user content
    if "messages" in item and isinstance(item["messages"], list):
        for m in item["messages"]:
            if m.get("role") == "user":
                s = m.get("content", "")
                if not s:
                    continue
                mm = PROBLEM_PREFIX_RE.search(s)
                if mm:
                    return mm.group(1).strip()
                return s.strip()
    return ""


def get_gold_answer(item: Dict[str, Any]) -> Optional[float]:
    """
    Gold answer priority:
      1) item["gold"] or item["answer"] (top-level)
      2) if messages-format and assistant exists: parse <answer> from assistant XML
    """
    v = _to_float_maybe(item.get("gold"))
    if v is not None:
        return v
    v = _to_float_maybe(item.get("answer"))
    if v is not None:
        return v
    # from assistant
    if "messages" in item and isinstance(item["messages"], list):
        for m in item["messages"]:
            if m.get("role") == "assistant":
                return extract_gold_from_assistant_xml(m.get("content", ""))
    return None


def get_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta


# =========================
# Candidate evaluation
# =========================
def evaluate_candidate(text: str, gold: float, exec_timeout: int, tol: float,
                       require_plan_solve: bool) -> Tuple[Optional[float], str, bool]:
    """
    Return (pred, status, is_correct)
    Only correctness based on executing solve() output vs gold.
    """
    if not text:
        return None, "empty", False

    if require_plan_solve:
        if "<plan>" not in text.lower() or "def solve():" not in text:
            return None, "missing_plan_or_solve", False

    code = extract_plan_code(text)
    pred, status = execute_solve_from_code(code, timeout_sec=exec_timeout)
    if status != "success" or pred is None:
        return None, status, False

    return pred, status, is_close(pred, gold, tol)


# =========================
# Main pipeline: Best-of-N self DPO
# =========================
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_dpo_path), exist_ok=True)

    # Load tokenizer & vLLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    stop = ["<|im_end|>"]
    if args.stop_on_answer:
        stop = ["</answer>", "<|im_end|>"]

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop,
    )

    # Load data
    data: List[Dict[str, Any]] = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Prepare prompts & golds (filter missing gold)
    items = []
    prompts = []
    missing_gold = 0

    for item in data:
        gold = get_gold_answer(item)
        if gold is None:
            missing_gold += 1
            continue

        prompt_msgs = get_prompt_messages(item)
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

        items.append({
            "raw": item,
            "gold": float(gold),
            "prompt_msgs": prompt_msgs,
            "prompt_text": prompt_text,
            "question": get_question_text(item),
            "meta": get_meta(item),
        })
        prompts.append(prompt_text)

    print(f"Loaded {len(data)} lines. Usable (has gold): {len(items)}. Missing gold: {missing_gold}")

    # Streaming write
    out_f = open(args.output_dpo_path, "w", encoding="utf-8")

    # Stats
    total_pairs = 0
    coverage_items = 0
    kind_cov = defaultdict(int)
    status_counter = defaultdict(int)
    skipped_no_correct = 0
    skipped_no_wrong = 0

    try:
        # Batch generation
        for start in tqdm(range(0, len(items), args.batch_size), desc=f"Best-of-{args.n_samples} mining"):
            batch_items = items[start:start + args.batch_size]
            batch_prompts = [x["prompt_text"] for x in batch_items]

            outputs = llm.generate(batch_prompts, sampling_params)

            for bi, out in enumerate(outputs):
                it = batch_items[bi]
                gold = it["gold"]

                # Collect runnable correct/wrong only (hard negatives)
                correct_cands = []
                wrong_cands = []

                # Evaluate candidates
                for cand in out.outputs:
                    text = cand.text

                    pred, status, ok = evaluate_candidate(
                        text=text,
                        gold=gold,
                        exec_timeout=args.exec_timeout,
                        tol=args.tol,
                        require_plan_solve=args.require_plan_solve or True,  # default enforce
                    )

                    # Record statuses for debug
                    status_counter[status] += 1

                    # Keep ONLY runnable candidates (status=success)
                    if status == "success":
                        if ok:
                            correct_cands.append((text, pred))
                        else:
                            wrong_cands.append((text, pred))

                # Only build self-correction DPO when both exist
                if not correct_cands:
                    skipped_no_correct += 1
                    continue
                if not wrong_cands:
                    skipped_no_wrong += 1
                    continue

                # Choose 1 correct path (chosen)
                # Strategy: prefer shortest correct (often cleaner), tie by first
                correct_cands.sort(key=lambda x: len(x[0]))
                chosen_text = correct_cands[0][0]

                # Choose up to max_pairs_per_item wrong paths as rejected
                # Strategy: pick "hard" wrongs (closest numeric error), then shortest
                wrong_cands_scored = []
                for t, p in wrong_cands[:max(args.max_wrong_candidates, 1)]:
                    wrong_cands_scored.append((abs(float(p) - gold), len(t), t))
                wrong_cands_scored.sort(key=lambda x: (x[0], x[1]))

                num_written_for_item = 0
                for _, _, rejected_text in wrong_cands_scored:
                    if num_written_for_item >= args.max_pairs_per_item:
                        break

                    dpo_record = {
                        "prompt": it["prompt_msgs"],
                        "chosen": [{"role": "assistant", "content": chosen_text}],
                        "rejected": [{"role": "assistant", "content": rejected_text}],
                        # 你若希望保留 meta 便于分析，可取消注释：
                        # "meta": it["meta"],
                    }
                    out_f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                    out_f.flush()

                    total_pairs += 1
                    num_written_for_item += 1

                if num_written_for_item > 0:
                    coverage_items += 1
                    k = (it["meta"] or {}).get("kind", "unknown")
                    kind_cov[k] += 1

        # Report
        print("\n===== Mining Done =====")
        print(f"Output: {args.output_dpo_path}")
        print(f"Coverage (items with >=1 pair): {coverage_items}/{len(items)} = {coverage_items / max(1,len(items)):.3f}")
        print(f"Total DPO pairs written: {total_pairs}")
        print(f"Skipped (no correct candidate in best-of-N): {skipped_no_correct}")
        print(f"Skipped (no wrong candidate among runnable): {skipped_no_wrong}")

        # Status summary (top)
        top_status = sorted(status_counter.items(), key=lambda x: x[1], reverse=True)[:15]
        print("Top candidate statuses:", top_status)

        # Kind coverage summary
        if kind_cov:
            print("Coverage by kind (items hit at least once):", dict(kind_cov))

    finally:
        out_f.close()


if __name__ == "__main__":
    main()