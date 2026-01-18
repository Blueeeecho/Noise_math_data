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
MODEL_PATH = "/root/autodl-tmp/A-new/output/model/sft/qwen2.5-1.5b-In/sft_train_test_py"

# 你的 jsonl：每行是 {"messages":[...], "meta": {...}}
LABELED_DATA_PATH = "/root/autodl-tmp/A-new/data/sft_data_pure/train_sft_py.jsonl"

OUTPUT_DPO_PATH = "/root/autodl-tmp/A-new/data/dpo/1.5B_instruct/hard_train_from_messages_vllm_all.jsonl"

BATCH_SIZE = 32
MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.7
TOP_P = 0.9

# --- 负例挖掘策略（核心）---
N_SAMPLES_PER_ROUND = 4          # 每题每轮采样 K 次（vLLM 的 n=K）
MAX_ROUNDS_PER_BATCH = 3         # 若没挖到足够 rejected，最多再跑几轮
MIN_REJECTED_PER_ITEM = 1        # 每题至少挖到多少 rejected（尽量保证覆盖）
MAX_REJECTED_PER_ITEM = 1        # 每题最多保留多少 rejected（控制规模/重复性）
PREFER_RUNNABLE_WRONG = True     # 优先收集 “能跑但数值错” 的 rejected
ALLOW_NONRUNNING_FALLBACK = True # 若本轮没有 runnable-wrong，可用 1 个非运行类错误做兜底
EXEC_TIMEOUT_SEC = 2             # 执行 solve() 超时秒数
TOL = 1e-4                       # 数值对比阈值
# =========================================


# ----------- 解析工具 -----------
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.DOTALL | re.IGNORECASE)
ANS_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def extract_python_code(text: str) -> Optional[str]:
    m = PLAN_RE.search(text or "")
    return m.group(1).strip() if m else None

def extract_answer_number(text: str) -> Optional[float]:
    """从 <answer> 里解析数字，作为标准答案(gt)。"""
    m = ANS_RE.search(text or "")
    if not m:
        return None
    s = m.group(1).strip().replace(",", "")
    try:
        return float(s)
    except:
        return None


# ----------- 安全执行 solve() -----------
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("timeout")

def execute_generated_code(code_str: str, timeout_sec: int = EXEC_TIMEOUT_SEC) -> Tuple[Optional[float], str]:
    """
    执行 <plan> 中的 def solve(): 并返回 solve() 的数值结果。
    返回: (value or None, status)
    status in:
      {"success","no_code","format_error","no_solve_func","exec_error","timeout","nan_inf"}
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


# ----------- 数据结构适配（你的 messages+meta）-----------
def build_prompt_messages(rec: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    从 {"messages":[...]} 里抽取 system+user 作为 prompt。
    """
    msgs = rec.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    prompt = []
    for m in msgs:
        role = m.get("role")
        if role in ("system", "user"):
            prompt.append({"role": role, "content": m.get("content", "")})
    return prompt if len(prompt) >= 2 else None

def get_assistant_content(rec: Dict[str, Any]) -> Optional[str]:
    msgs = rec.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return None


def validate_chosen(rec: Dict[str, Any], tol: float = TOL) -> Tuple[bool, Optional[float], str]:
    """
    chosen 校验（强制）：
    - gt 来自 labeled assistant 的 <answer>
    - 执行 labeled assistant 的 <plan> solve()
    - solve() 输出与 gt 一致 -> 才允许作为 chosen
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

    return True, float(gt), "ok"


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(OUTPUT_DPO_PATH)

    print(f"[1] Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2] Loading vLLM model from: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        # 多卡可加 tensor_parallel_size=N
    )

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        n=N_SAMPLES_PER_ROUND,
        # 可选：stop=["</answer>"]
    )

    print(f"[3] Loading labeled data: {LABELED_DATA_PATH}")
    raw_recs = []
    with open(LABELED_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_recs.append(json.loads(line))

    print(f"Total records loaded: {len(raw_recs)}")

    # 过滤：chosen 必须 exec 正确且与 gt 一致
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
            "prompt_msgs": prompt_msgs,
            "chosen_text": chosen_text,
            "gt": float(gt),
            "meta": {
                "seed_id": meta.get("seed_id"),
                "kind": meta.get("kind"),
            }
        })

    print(f"Valid chosen records after exec+gt check: {len(tasks)}")
    if bad_stats:
        print("Filtered-out reasons:", dict(bad_stats))

    # ========== 实时写入：打开输出文件（覆盖旧文件）==========
    out_f = open(OUTPUT_DPO_PATH, "w", encoding="utf-8")

    # 统计
    kind_stats = defaultdict(int)
    status_stats = defaultdict(int)
    coverage_hit = 0  # 多少题至少挖到 1 个 rejected
    total_items = len(tasks)

    try:
        # ========== 批量挖掘 ==========
        for bi in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Mining rejected (multi-sample, streaming write)"):
            batch = tasks[bi:bi + BATCH_SIZE]
            if not batch:
                continue

            # 每题收集的 rejected 数量
            collected = [0] * len(batch)
            any_hit = [False] * len(batch)  # 是否至少命中 1 个 rejected（用于 coverage）

            # 候选索引：仍需要挖的题
            remaining = set(range(len(batch)))

            for round_idx in range(MAX_ROUNDS_PER_BATCH):
                if not remaining:
                    break

                # 只对 remaining 构造 prompt
                idx_list = sorted(list(remaining))
                prompt_texts = []
                for j in idx_list:
                    prompt_texts.append(
                        tokenizer.apply_chat_template(
                            batch[j]["prompt_msgs"],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    )

                outputs = llm.generate(prompt_texts, sampling)

                # 对每个 remaining 样本处理其 n 个候选
                for local_pos, out in enumerate(outputs):
                    j = idx_list[local_pos]  # batch 内索引
                    item = batch[j]
                    gt = item["gt"]

                    # 若已收满则跳过
                    if collected[j] >= MAX_REJECTED_PER_ITEM:
                        remaining.discard(j)
                        continue

                    # 将候选分两类：runnable-wrong 优先；其他错误作为兜底
                    runnable_wrong_texts = []
                    fallback_texts = []  # 非运行成功类错误（format/exec/timeout等）

                    for cand in out.outputs:
                        gen_text = cand.text

                        code = extract_python_code(gen_text)
                        pred, status = execute_generated_code(code)

                        # 统计 status（帮助你看错误组成）
                        status_stats[status] += 1

                        if status == "success" and pred is not None and abs(pred - gt) > TOL:
                            runnable_wrong_texts.append(gen_text)

                    # 选择写入哪些 rejected
                    picked = []

                    if PREFER_RUNNABLE_WRONG and runnable_wrong_texts:
                        picked.extend(runnable_wrong_texts)

                    # # 若本轮没有 runnable-wrong，并且允许兜底，则拿 1 个 fallback
                    # if (collected[j] == 0) and (not picked) and ALLOW_NONRUNNING_FALLBACK and fallback_texts:
                    #     picked.append(fallback_texts[0])

                    # 写入（实时）
                    for gen_text in picked:
                        if collected[j] >= MAX_REJECTED_PER_ITEM:
                            break

                        dpo_record = {
                            "prompt": item["prompt_msgs"],
                            "chosen": [{"role": "assistant", "content": item["chosen_text"]}],
                            "rejected": [{"role": "assistant", "content": gen_text}],
                        }

                        out_f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                        out_f.flush()  # 实时写入落盘

                        collected[j] += 1
                        if not any_hit[j]:
                            any_hit[j] = True
                            kind = item["meta"].get("kind", "unknown")
                            kind_stats[kind] += 1

                    # 若达到最低需求就可以停止继续挖该题（也可继续直到 MAX）
                    if collected[j] >= MIN_REJECTED_PER_ITEM:
                        # 如果你希望“满足最少就停”，启用下面这一行：
                        # remaining.discard(j)
                        pass

                    if collected[j] >= MAX_REJECTED_PER_ITEM:
                        remaining.discard(j)

                # 一轮结束后：把已满足 MIN 的样本移出 remaining（省算力）
                for j in list(remaining):
                    if collected[j] >= MIN_REJECTED_PER_ITEM:
                        remaining.discard(j)

            # 统计 coverage：本 batch 有多少题至少挖到 1 个 rejected
            coverage_hit += sum(1 for x in any_hit if x)

        # ========== 结束统计 ==========
        print("\n[Done]")
        print(f"Output written to: {OUTPUT_DPO_PATH}")
        print(f"Coverage (items with >=1 rejected): {coverage_hit}/{total_items} = {coverage_hit/ max(1,total_items):.3f}")
        print("Pairs distribution by kind (counted by items hit at least once):", dict(kind_stats))
        # status_stats 会很大，这里只打印前几名
        top_status = sorted(status_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top exec statuses:", top_status)

    finally:
        out_f.close()


if __name__ == "__main__":
    main()