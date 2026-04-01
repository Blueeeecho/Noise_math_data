import ast
import re
from collections import Counter


STEP_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s*(Define|Derive|Calculate)\s+Var\{([^}]+)\}", re.IGNORECASE | re.MULTILINE)
VAR_RE = re.compile(r"Var\{([^}]+)\}")


def extract_calcs(text):
    if not text:
        return []
    matches = re.findall(r"<<(.*?)>>", text)
    results = []
    for m in matches:
        content = m.strip()
        if "=" in content:
            content = content.split("=")[-1].strip()
        val_str = content.replace(",", "").replace("$", "")
        try:
            val = float(val_str)
            results.append(f"{val:.4f}")
        except Exception:
            results.append(content.replace(" ", ""))
    return results


def extract_final_answer(text):
    if not text:
        return None
    m = re.search(r"\[Final Answer\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_number(text):
    if not text:
        return None
    clean = text.replace(",", "").replace("$", "")
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", clean)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def check_format(text):
    if not text:
        return False
    required = ["[Goal Analysis]", "[Backward Execution]", "[Final Answer]"]
    for tag in required:
        if tag not in text:
            return False
    return True


def extract_target_var(text):
    if not text:
        return None
    match = re.search(r"Target:\s*Var\{([^}]+)\}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_backward_execution(text):
    if not text:
        return ""
    match = re.search(r"\[Backward Execution\](.*?)(?=\[Final Answer\]|\Z)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def split_steps(backward_text):
    if not backward_text:
        return []
    matches = list(STEP_HEADER_RE.finditer(backward_text))
    if not matches:
        return []
    steps = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(backward_text)
        step_text = backward_text[start:end].strip()
        steps.append(step_text)
    return steps


def parse_step_header(step_text):
    if not step_text:
        return None
    match = STEP_HEADER_RE.search(step_text)
    if not match:
        return None
    return {
        "step_id": match.group(1).strip(),
        "action": match.group(2).strip(),
        "var_name": match.group(3).strip(),
    }


def extract_calc_block(step_text):
    if not step_text:
        return ""
    match = re.search(r"\[Calc\]\s*:?\s*(.*?)(?=\n\s*\[[A-Za-z ]+\]\s*:|\Z)", step_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_var_refs(text):
    if not text:
        return []
    return [ref.strip() for ref in VAR_RE.findall(text)]


def _normalize_expr(expr):
    expr = expr.strip().replace(",", "").replace("$", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError("unsupported expression")


def safe_eval_numeric(expr):
    if not expr:
        return None
    expr = _normalize_expr(expr)
    if re.search(r"[A-Za-z_]", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        return float(_safe_eval_node(tree))
    except Exception:
        return None


def check_calc_correct(calc_text):
    calc_segments = re.findall(r"<<(.*?)>>", calc_text or "")
    if not calc_segments:
        return "missing"
    saw_correct = False
    saw_unverifiable = False
    for segment in calc_segments:
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        if not parts:
            return "incorrect"
        values = [safe_eval_numeric(part) for part in parts]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            saw_unverifiable = True
            continue
        if len(parts) == 1:
            saw_correct = True
            continue
        if any(value is None for value in values):
            saw_unverifiable = True
            continue
        base = values[0]
        if all(abs(base - value) < 1e-6 for value in values[1:]):
            saw_correct = True
        else:
            return "incorrect"
    if saw_correct:
        return "correct"
    if saw_unverifiable:
        return "unverifiable"
    return "incorrect"


def compute_legacy_process_reward(response_str, gold_chain):
    if not gold_chain:
        return 0.0
    model_calcs = extract_calcs(response_str)
    gold_calcs = extract_calcs(gold_chain)
    if not gold_calcs:
        return 0.0
    model_cnt = Counter(model_calcs)
    gold_cnt = Counter(gold_calcs)
    intersection = model_cnt & gold_cnt
    matched = sum(intersection.values())
    total = sum(gold_cnt.values())
    if total <= 0:
        return 0.0
    return matched / total


def classify_step(
    step_text,
    target_var,
    previous_text,
    future_text,
    seen_vars,
    is_last_step,
    require_reasoning,
    require_source,
    bad_on_unused_var,
    bad_on_duplicate_var,
    bad_on_missing_dependency,
):
    header = parse_step_header(step_text)
    calc_text = extract_calc_block(step_text)
    has_reasoning = bool(re.search(r"\[Reasoning\]", step_text, re.IGNORECASE))
    has_source = bool(re.search(r"\[Source\]", step_text, re.IGNORECASE))
    has_calc = bool(calc_text) and "<<" in calc_text and ">>" in calc_text
    if not header or not has_calc:
        return "bad", {
            "var_name": header["var_name"] if header else None,
            "calc_status": "missing",
            "is_useful": False,
        }
    if require_reasoning and not has_reasoning:
        return "bad", {
            "var_name": header["var_name"],
            "calc_status": "missing_reasoning",
            "is_useful": False,
        }
    if require_source and not has_source:
        return "bad", {
            "var_name": header["var_name"],
            "calc_status": "missing_source",
            "is_useful": False,
        }
    var_name = header["var_name"]
    calc_status = check_calc_correct(calc_text)
    if calc_status in {"incorrect", "missing"}:
        return "bad", {
            "var_name": var_name,
            "calc_status": calc_status,
            "is_useful": False,
        }
    previous_refs = set(extract_var_refs(previous_text))
    future_refs = set(extract_var_refs(future_text))
    is_goal_var = bool(target_var) and var_name == target_var
    is_useful = is_goal_var or var_name in future_refs or var_name in previous_refs
    calc_refs = {ref for ref in extract_var_refs(calc_text) if ref != var_name}
    if bad_on_missing_dependency and is_last_step and is_goal_var and any(ref not in seen_vars for ref in calc_refs):
        return "bad", {
            "var_name": var_name,
            "calc_status": calc_status,
            "is_useful": is_useful,
        }
    if bad_on_duplicate_var and var_name in seen_vars and not is_goal_var:
        return "bad", {
            "var_name": var_name,
            "calc_status": calc_status,
            "is_useful": is_useful,
        }
    if bad_on_unused_var and not is_useful:
        return "bad", {
            "var_name": var_name,
            "calc_status": calc_status,
            "is_useful": is_useful,
        }
    if calc_status == "correct" and is_useful:
        return "good", {
            "var_name": var_name,
            "calc_status": calc_status,
            "is_useful": is_useful,
        }
    return "neutral", {
        "var_name": var_name,
        "calc_status": calc_status,
        "is_useful": is_useful,
    }


def compute_step_rule_process_score(
    response_str,
    step_norm_min,
    require_reasoning,
    require_source,
    bad_on_unused_var,
    bad_on_duplicate_var,
    bad_on_missing_dependency,
):
    target_var = extract_target_var(response_str)
    backward_text = extract_backward_execution(response_str)
    steps = split_steps(backward_text)
    step_count = len(steps)
    good_count = 0
    bad_count = 0
    neutral_count = 0
    seen_vars = set()
    if not steps:
        z = max(step_norm_min, 0)
        return {
            "good_count": 0,
            "bad_count": 1,
            "neutral_count": 0,
            "step_count": 0,
            "z": z,
            "good_ratio": 0.0,
            "bad_ratio": 1.0 / z,
        }
    for idx, step_text in enumerate(steps):
        previous_text = "\n".join(steps[:idx])
        future_text = "\n".join(steps[idx + 1 :])
        label, meta = classify_step(
            step_text=step_text,
            target_var=target_var,
            previous_text=previous_text,
            future_text=future_text,
            seen_vars=seen_vars,
            is_last_step=idx == len(steps) - 1,
            require_reasoning=require_reasoning,
            require_source=require_source,
            bad_on_unused_var=bad_on_unused_var,
            bad_on_duplicate_var=bad_on_duplicate_var,
            bad_on_missing_dependency=bad_on_missing_dependency,
        )
        if label == "good":
            good_count += 1
        elif label == "bad":
            bad_count += 1
        else:
            neutral_count += 1
        if meta.get("var_name"):
            seen_vars.add(meta["var_name"])
    z = max(step_norm_min, step_count)
    return {
        "good_count": good_count,
        "bad_count": bad_count,
        "neutral_count": neutral_count,
        "step_count": step_count,
        "z": z,
        "good_ratio": good_count / z,
        "bad_ratio": bad_count / z,
    }


def build_reward_result(score, **extra):
    result = {"score": score}
    result.update(extra)
    return result


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def compute_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    w_format=0.5,
    w_process=1.0,
    w_outcome=2.5,
    enable_format=True,
    enable_process=True,
    enable_outcome=True,
    reward_mode="legacy_overlap",
    global_fail_reward=-0.5,
    step_acc_weight=0.7,
    step_good_weight=0.4,
    step_bad_weight=0.3,
    step_fmt_weight=0.2,
    step_norm_min=3,
    require_reasoning=False,
    require_source=False,
    bad_on_unused_var=True,
    bad_on_duplicate_var=True,
    bad_on_missing_dependency=True,
    **kwargs,
):
    response_str = solution_str or ""
    enable_format = as_bool(enable_format)
    enable_process = as_bool(enable_process)
    enable_outcome = as_bool(enable_outcome)
    require_reasoning = as_bool(require_reasoning)
    require_source = as_bool(require_source)
    bad_on_unused_var = as_bool(bad_on_unused_var)
    bad_on_duplicate_var = as_bool(bad_on_duplicate_var)
    bad_on_missing_dependency = as_bool(bad_on_missing_dependency)
    w_format = as_float(w_format, 0.5)
    w_process = as_float(w_process, 1.0)
    w_outcome = as_float(w_outcome, 2.5)
    global_fail_reward = as_float(global_fail_reward, -0.5)
    step_acc_weight = as_float(step_acc_weight, 0.7)
    step_good_weight = as_float(step_good_weight, 0.4)
    step_bad_weight = as_float(step_bad_weight, 0.3)
    step_fmt_weight = as_float(step_fmt_weight, 0.2)
    step_norm_min = max(as_int(step_norm_min, 3), 1)
    gold_chain = ground_truth.get("gold_chain", "")
    gold_ans_text = ground_truth.get("gold_answer", "")
    if not gold_ans_text and gold_chain:
        gold_ans_text = extract_final_answer(gold_chain)
    pred_ans_text = extract_final_answer(response_str)
    pred_val = parse_number(pred_ans_text)
    gold_val = parse_number(gold_ans_text)
    final_answer_parsed = int(pred_val is not None and gold_val is not None)
    r_acc = 0.0
    if enable_outcome and final_answer_parsed and abs(pred_val - gold_val) < 1e-6:
        r_acc = 1.0
    format_ok = check_format(response_str)
    r_format = -1.0
    if enable_format and format_ok:
        r_format = 1.0
    legacy_process_score = 0.0
    if enable_process and gold_chain:
        legacy_process_score = compute_legacy_process_reward(response_str, gold_chain)
    if reward_mode == "step_rule":
        if not format_ok or not final_answer_parsed:
            return build_reward_result(
                global_fail_reward,
                reward_mode=reward_mode,
                r_acc=r_acc,
                r_fmt=0.0,
                legacy_process_score=legacy_process_score,
                step_process_score=0.0,
                step_good_count=0,
                step_bad_count=0,
                step_neutral_count=0,
                step_count=0,
                step_norm_z=max(int(step_norm_min), 0),
                global_format_pass=int(format_ok),
                final_answer_parsed=final_answer_parsed,
            )
        step_metrics = compute_step_rule_process_score(
            response_str=response_str,
            step_norm_min=step_norm_min,
            require_reasoning=require_reasoning,
            require_source=require_source,
            bad_on_unused_var=bad_on_unused_var,
            bad_on_duplicate_var=bad_on_duplicate_var,
            bad_on_missing_dependency=bad_on_missing_dependency,
        )
        step_process_score = step_metrics["good_ratio"] - step_metrics["bad_ratio"]
        total_reward = (
            step_acc_weight * r_acc
            + step_good_weight * step_metrics["good_ratio"]
            - step_bad_weight * step_metrics["bad_ratio"]
            + step_fmt_weight * 1.0
        )
        return build_reward_result(
            total_reward,
            reward_mode=reward_mode,
            r_acc=r_acc,
            r_fmt=1.0,
            legacy_process_score=legacy_process_score,
            step_process_score=step_process_score,
            step_good_count=step_metrics["good_count"],
            step_bad_count=step_metrics["bad_count"],
            step_neutral_count=step_metrics["neutral_count"],
            step_count=step_metrics["step_count"],
            step_norm_z=step_metrics["z"],
            global_format_pass=1,
            final_answer_parsed=final_answer_parsed,
        )
    total_reward = (w_format * r_format) + (w_process * legacy_process_score) + (w_outcome * r_acc)
    return build_reward_result(
        total_reward,
        reward_mode=reward_mode,
        r_acc=r_acc,
        r_fmt=1.0 if format_ok else 0.0,
        legacy_process_score=legacy_process_score,
        step_process_score=0.0,
        step_good_count=0,
        step_bad_count=0,
        step_neutral_count=0,
        step_count=0,
        step_norm_z=0,
        global_format_pass=int(format_ok),
        final_answer_parsed=final_answer_parsed,
    )
