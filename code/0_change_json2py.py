import json
import argparse
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

# -------------------------
# Regex extractors
# -------------------------
PLAN_RE = re.compile(r"<plan>\s*(\{.*?\})\s*</plan>", re.DOTALL)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# -------------------------
# Helpers
# -------------------------
def to_py_literal(x: Any) -> str:
    """Python字面量输出（给 vars 用）。"""
    if isinstance(x, str):
        return repr(x)
    return repr(x)

def to_expr_token(x: Any) -> str:
    """
    args 通常是变量名字符串；若不是字符串（如数字/None/bool/list/dict），转为 Python 字面量。
    """
    if isinstance(x, str):
        return x
    return to_py_literal(x)

def safe_join(op_symbol: str, args: List[Any], neutral: str) -> str:
    """把 args 拼成 a + b + c；为空返回 neutral。"""
    if not args:
        return neutral
    tokens = [to_expr_token(a) for a in args]
    return f" {op_symbol} ".join(tokens)

def extract_plan_reason_answer(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """从文本中提取 plan_json_str、reasoning、answer。"""
    plan_match = PLAN_RE.search(text)
    reason_match = REASONING_RE.search(text)
    ans_match = ANS_RE.search(text)

    plan_str = plan_match.group(1) if plan_match else None
    reasoning = reason_match.group(1) if reason_match else ""
    answer = ans_match.group(1) if ans_match else None
    return plan_str, reasoning, answer

# -------------------------
# Core conversion
# -------------------------
def json_plan_to_python(plan_json: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    将 plan_json 转为 Python solve() 代码。
    返回 (py_code, warnings)
    """
    warnings: List[str] = []
    lines: List[str] = ["def solve():"]

    # 1) Vars（排序保证稳定）
    vars_dict = plan_json.get("vars", {})
    if isinstance(vars_dict, dict) and vars_dict:
        for k in sorted(vars_dict.keys()):
            v = vars_dict[k]
            lines.append(f"    {k} = {to_py_literal(v)}")

    lines.append("")

    # 2) Steps
    steps = plan_json.get("steps", [])
    last_out = None
    if not isinstance(steps, list):
        warnings.append("steps is not a list; ignored")
        steps = []

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings.append(f"step[{idx}] is not a dict; skipped")
            continue

        out_var = step.get("out")
        op = step.get("op")
        args = step.get("args", [])

        # 兼容旧格式 a/b -> args
        if (not args) and ("a" in step):
            args = [step.get("a")]
            if "b" in step:
                args.append(step.get("b"))

        # 规范化 args
        if args is None:
            args = []
        if not isinstance(args, list):
            args = [args]

        if not out_var or not isinstance(out_var, str):
            warnings.append(f"step[{idx}] missing/invalid out; skipped")
            continue
        if not op or not isinstance(op, str):
            warnings.append(f"step[{idx}] missing/invalid op; treat as unknown")
            op = "unknown"

        op_l = op.lower()
        if op_l == "add":
            expr = safe_join("+", args, neutral="0")
        elif op_l == "sub":
            expr = safe_join("-", args, neutral="0")
        elif op_l == "mul":
            expr = safe_join("*", args, neutral="1")
        elif op_l == "div":
            if len(args) >= 2:
                expr = f"{to_expr_token(args[0])} / {to_expr_token(args[1])}"
                if len(args) > 2:
                    warnings.append(f"step[{idx}] div has >2 args; extra ignored")
            elif len(args) == 1:
                expr = f"{to_expr_token(args[0])}"
                warnings.append(f"step[{idx}] div has 1 arg; used identity fallback")
            else:
                expr = "0"
                warnings.append(f"step[{idx}] div has 0 args; used 0 fallback")
        else:
            tokens = ", ".join(to_expr_token(a) for a in args)
            expr = f"unknown_op({tokens})"
            warnings.append(f"step[{idx}] unknown op='{op}' -> unknown_op(...)")

        # evidence 作为注释（仅取第一条，截断）
        evidence = step.get("evidence", [])
        comment = ""
        if isinstance(evidence, list) and evidence:
            ev0 = str(evidence[0]).replace("\n", " ").strip()
            if ev0:
                if len(ev0) > 200:
                    ev0 = ev0[:200] + "..."
                comment = f"  # {ev0}"

        lines.append(f"    {out_var} = {expr}{comment}")
        last_out = out_var

    # 3) Final
    final_var = plan_json.get("final", None)
    if isinstance(final_var, str) and final_var.strip():
        final_expr = final_var.strip()
    else:
        if last_out:
            final_expr = last_out
            warnings.append("final missing; fallback to last step out")
        else:
            final_expr = "None"
            warnings.append("final missing and no steps; return None")

    lines.append("")
    lines.append(f"    return {final_expr}")

    return "\n".join(lines), warnings

def convert_assistant_content(content: str) -> Tuple[str, bool, List[str]]:
    """
    将 assistant 的 content 中 plan JSON 转为 python。
    返回 (new_content, changed, warnings)
    """
    plan_str, reasoning, answer = extract_plan_reason_answer(content)
    if not plan_str or answer is None:
        return content, False, ["missing plan or answer; skipped"]

    try:
        plan_json = json.loads(plan_str)
    except Exception as e:
        return content, False, [f"plan json loads failed: {repr(e)}"]

    py_code, warnings = json_plan_to_python(plan_json)

    # 保持 reasoning/answer 原样，plan 替换为 python
    new_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<plan>\n{py_code}\n</plan>\n"
        f"<answer>{answer}</answer>"
    )
    return new_content, True, warnings

# -------------------------
# Record processing (messages schema)
# -------------------------
def process_record(record: Dict[str, Any], store_backup: bool = False) -> Tuple[Dict[str, Any], Counter, List[str]]:
    """
    处理单条 record：遍历 messages，把 role==assistant 的 content 里的 plan 转为 python。
    返回 (new_record, stats_counter, warnings_list)
    """
    stats = Counter()
    warnings_all: List[str] = []

    msgs = record.get("messages")
    if not isinstance(msgs, list):
        stats["record_skipped_no_messages"] += 1
        return record, stats, ["record has no messages list"]

    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            stats["message_skipped_not_dict"] += 1
            continue
        role = msg.get("role")
        if role != "assistant":
            stats["message_skipped_not_assistant"] += 1
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            stats["assistant_skipped_empty_content"] += 1
            continue

        if store_backup and "content_backup" not in msg:
            msg["content_backup"] = content

        new_content, changed, warns = convert_assistant_content(content)
        if changed:
            msg["content"] = new_content
            stats["assistant_converted"] += 1
        else:
            stats["assistant_skipped_no_change"] += 1

        warnings_all.extend([f"messages[{i}]: {w}" for w in warns])

    # 可选：写回少量 warnings 便于排查（避免爆炸）
    if warnings_all:
        record["_convert_plan_warnings"] = warnings_all[:50]
        if len(warnings_all) > 50:
            record["_convert_plan_warnings_truncated"] = True

    return record, stats, warnings_all

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/root/autodl-tmp/A-new/data/sft_data_pure/train_sft.jsonl", help="输入 jsonl")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/A-new/data/sft_data_pure/train_sft_py.jsonl", help="输出 jsonl")
    parser.add_argument("--store-backup", action="store_true", help="为 assistant message 额外保存 content_backup")
    parser.add_argument("--verbose", action="store_true", help="打印一些失败/跳过原因统计")
    args = parser.parse_args()

    total_records = 0
    agg = Counter()
    error_top = Counter()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total_records += 1

            try:
                rec = json.loads(line)
            except Exception as e:
                agg["json_load_failed"] += 1
                error_top[f"line {line_no} json_load_failed: {repr(e)}"] += 1
                continue

            new_rec, stats, warns = process_record(rec, store_backup=args.store_backup)
            agg.update(stats)

            # 收集一些常见错误信息（只挑关键几类）
            for w in warns:
                if "failed" in w or "json loads failed" in w:
                    error_top[w] += 1

            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(f"Processed records: {total_records}")
    print(f"Assistant messages converted: {agg.get('assistant_converted', 0)}")
    print(f"Assistant skipped (no change): {agg.get('assistant_skipped_no_change', 0)}")
    print(f"JSON load failed lines: {agg.get('json_load_failed', 0)}")

    if args.verbose:
        print("\n[Verbose stats]")
        for k, v in agg.most_common():
            if k in ("assistant_converted", "assistant_skipped_no_change", "json_load_failed"):
                continue
            print(f"  {k}: {v}")

        if error_top:
            print("\nTop errors/warnings:")
            for msg, cnt in error_top.most_common(10):
                print(f"  ({cnt}) {msg}")

if __name__ == "__main__":
    main()