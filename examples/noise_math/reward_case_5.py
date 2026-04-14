import ast
import re
from collections import Counter, defaultdict


STEP_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s*(Define|Derive|Calculate)\s+Var\{([^}]+)\}", re.IGNORECASE | re.MULTILINE)
VAR_RE = re.compile(r"Var\{([^}]+)\}")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "with",
    "if",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "from",
    "then",
    "than",
    "into",
    "onto",
    "their",
    "there",
    "they",
    "them",
    "have",
    "has",
    "had",
    "will",
    "would",
    "should",
    "could",
    "each",
    "every",
    "only",
    "same",
    "following",
    "format",
    "question",
    "answer",
}


def extract_calcs(text):
    if not text:
        return []
    matches = re.findall(r"<<(.*?)>>", text)
    results = []
    for match in matches:
        content = match.strip()
        if "=" in content:
            content = content.split("=")[-1].strip()
        value_str = content.replace(",", "").replace("$", "")
        try:
            value = float(value_str)
            results.append(f"{value:.4f}")
        except Exception:
            results.append(content.replace(" ", ""))
    return results


def extract_final_answer(text):
    if not text:
        return None
    match = re.search(r"\[Final Answer\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
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


def extract_block(step_text, tag):
    if not step_text:
        return ""
    match = re.search(
        rf"\[{re.escape(tag)}\]\s*:?\s*(.*?)(?=\n\s*\[[A-Za-z ]+\]\s*:|\Z)",
        step_text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_calc_block(step_text):
    return extract_block(step_text, "Calc")


def extract_source_block(step_text):
    return extract_block(step_text, "Source")


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


def normalize_token(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_current_question(extra_info):
    if not isinstance(extra_info, dict):
        return ""
    direct = extra_info.get("current_question") or extra_info.get("question") or ""
    if direct:
        return str(direct).strip()
    prompt_text = extra_info.get("prompt_text") or ""
    if not prompt_text:
        return ""
    matches = list(
        re.finditer(
            r"Question:\s*(.*?)(?=(?:\nassistant\n|\nuser\n|\Z))",
            str(prompt_text),
            re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def extract_question_numbers(question_text):
    return {match.strip() for match in re.findall(r"-?\d+(?:\.\d+)?", question_text or "")}


def extract_question_keywords(question_text):
    tokens = re.findall(r"[A-Za-z][A-Za-z_-]*", (question_text or "").lower())
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def analyze_source_grounding(source_text, question_text, seen_vars):
    source_text = source_text or ""
    source_lower = normalize_token(source_text)
    source_numbers = extract_question_numbers(source_text)
    question_numbers = extract_question_numbers(question_text)
    question_keywords = extract_question_keywords(question_text)
    source_tokens = set(re.findall(r"[A-Za-z][A-Za-z_-]*", source_lower))

    number_hits = sorted(question_numbers & source_numbers)
    keyword_hits = sorted(question_keywords & source_tokens)
    prev_var_hits = []
    for var_name in seen_vars:
        var_lower = var_name.lower()
        if f"var{{{var_lower}}}" in source_lower or var_lower in source_tokens:
            prev_var_hits.append(var_name)

    return {
        "grounded": bool(number_hits or keyword_hits or prev_var_hits),
        "question_number_hits": number_hits,
        "question_keyword_hits": keyword_hits,
        "prev_var_hits": prev_var_hits,
    }


def classify_step_case5(
    step_text,
    target_var,
    question_text,
    previous_text,
    future_text,
    seen_vars,
    var_history,
    require_source_grounding,
    bad_on_duplicate_goal_without_new_dependency,
    bad_on_idle_chain,
    bad_on_invalid_dependency,
):
    header = parse_step_header(step_text)
    reasoning_text = extract_block(step_text, "Reasoning")
    source_text = extract_source_block(step_text)
    calc_text = extract_calc_block(step_text)
    has_reasoning = bool(reasoning_text)
    has_source = bool(source_text)
    has_calc = bool(calc_text) and "<<" in calc_text and ">>" in calc_text

    if not header:
        return "bad", {
            "reason": "invalid_step_title",
            "var_name": None,
            "calc_status": "missing",
            "has_reasoning": has_reasoning,
            "has_source": has_source,
            "has_calc": has_calc,
            "is_goal_var": False,
            "is_useful": False,
            "source_grounded": False,
            "source_grounded_by": [],
            "step_title": step_text.strip().splitlines()[0].strip() if step_text.strip() else "",
        }

    var_name = header["var_name"]
    is_goal_var = bool(target_var) and var_name == target_var
    calc_status = check_calc_correct(calc_text)
    calc_refs = [ref for ref in extract_var_refs(calc_text) if ref != var_name]
    future_refs = set(extract_var_refs(future_text))
    source_grounding = analyze_source_grounding(source_text, question_text, seen_vars)
    question_grounded = bool(
        source_grounding["question_number_hits"] or source_grounding["question_keyword_hits"]
    )
    prev_var_grounded = bool(source_grounding["prev_var_hits"])
    grounded = source_grounding["grounded"]
    invalid_dependency = any(ref not in seen_vars for ref in calc_refs)

    calc_signature = normalize_token(calc_text)
    source_signature = normalize_token(source_text)
    seen_signatures = var_history.get(var_name, [])
    duplicate_same_content = any(sig == (calc_signature, source_signature) for sig in seen_signatures)

    contributes = False
    if var_name not in seen_vars and (var_name in future_refs or is_goal_var or question_grounded or prev_var_grounded or calc_refs):
        contributes = True
    elif is_goal_var and (question_grounded or prev_var_grounded or bool(calc_refs)):
        contributes = True

    repeated_goal_without_new_dependency = (
        is_goal_var
        and var_name in seen_vars
        and not question_grounded
        and not prev_var_grounded
        and not calc_refs
    )
    idle_chain = duplicate_same_content or (
        var_name in seen_vars
        and not question_grounded
        and not prev_var_grounded
        and not calc_refs
        and var_name not in future_refs
    )
    source_self_talk = require_source_grounding and (not grounded) and bool(source_text.strip())

    reason = "neutral_step"
    label = "neutral"

    if not has_reasoning or not has_source or not has_calc:
        reason = "missing_required_block"
        label = "bad"
    elif calc_status in {"incorrect", "missing", "unverifiable"}:
        reason = calc_status
        label = "bad"
    elif source_self_talk:
        reason = "ungrounded_source"
        label = "bad"
    elif bad_on_invalid_dependency and invalid_dependency:
        reason = "invalid_dependency"
        label = "bad"
    elif bad_on_duplicate_goal_without_new_dependency and repeated_goal_without_new_dependency:
        reason = "duplicate_goal_without_new_dependency"
        label = "bad"
    elif bad_on_idle_chain and idle_chain:
        reason = "idle_chain"
        label = "bad"
    elif grounded and contributes:
        reason = "good_step"
        label = "good"
    elif grounded and not contributes:
        reason = "weak_contribution"
        label = "neutral"
    else:
        reason = "neutral_step"
        label = "neutral"

    source_grounded_by = []
    if source_grounding["question_number_hits"]:
        source_grounded_by.append("question_number")
    if source_grounding["question_keyword_hits"]:
        source_grounded_by.append("question_keyword")
    if source_grounding["prev_var_hits"]:
        source_grounded_by.append("previous_var")

    return label, {
        "reason": reason,
        "var_name": var_name,
        "calc_status": calc_status,
        "has_reasoning": has_reasoning,
        "has_source": has_source,
        "has_calc": has_calc,
        "is_goal_var": is_goal_var,
        "is_useful": contributes,
        "source_grounded": grounded,
        "source_grounded_by": source_grounded_by,
        "question_number_hits": source_grounding["question_number_hits"],
        "question_keyword_hits": source_grounding["question_keyword_hits"],
        "prev_var_hits": source_grounding["prev_var_hits"],
        "step_title": step_text.strip().splitlines()[0].strip() if step_text.strip() else "",
    }


def compute_step_rule_process_score(
    response_str,
    extra_info,
    step_norm_min,
    require_source_grounding,
    bad_on_duplicate_goal_without_new_dependency,
    bad_on_idle_chain,
    bad_on_invalid_dependency,
):
    target_var = extract_target_var(response_str)
    backward_text = extract_backward_execution(response_str)
    steps = split_steps(backward_text)
    question_text = extract_current_question(extra_info)
    step_count = len(steps)
    good_count = 0
    bad_count = 0
    neutral_count = 0
    seen_vars = set()
    var_history = defaultdict(list)
    step_details = []

    if not steps:
        z = max(step_norm_min, 0)
        return {
            "good_count": 0,
            "bad_count": 1,
            "neutral_count": 0,
            "step_count": 0,
            "z": z,
            "good_ratio": 0.0,
            "bad_ratio": 1.0 / z if z else 1.0,
            "step_details": [],
            "question_text": question_text,
        }

    for idx, step_text in enumerate(steps):
        previous_text = "\n".join(steps[:idx])
        future_text = "\n".join(steps[idx + 1 :])
        label, meta = classify_step_case5(
            step_text=step_text,
            target_var=target_var,
            question_text=question_text,
            previous_text=previous_text,
            future_text=future_text,
            seen_vars=seen_vars,
            var_history=var_history,
            require_source_grounding=require_source_grounding,
            bad_on_duplicate_goal_without_new_dependency=bad_on_duplicate_goal_without_new_dependency,
            bad_on_idle_chain=bad_on_idle_chain,
            bad_on_invalid_dependency=bad_on_invalid_dependency,
        )
        if label == "good":
            good_count += 1
        elif label == "bad":
            bad_count += 1
        else:
            neutral_count += 1
        if meta.get("var_name"):
            seen_vars.add(meta["var_name"])
            calc_signature = normalize_token(extract_calc_block(step_text))
            source_signature = normalize_token(extract_source_block(step_text))
            var_history[meta["var_name"]].append((calc_signature, source_signature))
        step_details.append(
            {
                "index": idx + 1,
                "label": label,
                "reason": meta["reason"],
                "var_name": meta["var_name"],
                "calc_status": meta["calc_status"],
                "is_goal_var": meta["is_goal_var"],
                "is_useful": meta["is_useful"],
                "has_reasoning": meta["has_reasoning"],
                "has_source": meta["has_source"],
                "has_calc": meta["has_calc"],
                "source_grounded": meta["source_grounded"],
                "source_grounded_by": meta["source_grounded_by"],
                "question_number_hits": meta["question_number_hits"],
                "question_keyword_hits": meta["question_keyword_hits"],
                "prev_var_hits": meta["prev_var_hits"],
                "step_title": meta["step_title"],
            }
        )

    z = max(step_norm_min, step_count)
    return {
        "good_count": good_count,
        "bad_count": bad_count,
        "neutral_count": neutral_count,
        "step_count": step_count,
        "z": z,
        "good_ratio": good_count / z,
        "bad_ratio": bad_count / z,
        "step_details": step_details,
        "question_text": question_text,
    }


def analyze_response_for_display(
    response_str,
    ground_truth,
    extra_info=None,
    step_norm_min=3,
    require_source_grounding=True,
    bad_on_duplicate_goal_without_new_dependency=True,
    bad_on_idle_chain=True,
    bad_on_invalid_dependency=True,
    **kwargs,
):
    step_metrics = compute_step_rule_process_score(
        response_str=response_str,
        extra_info=extra_info or {},
        step_norm_min=max(int(step_norm_min), 1),
        require_source_grounding=as_bool(require_source_grounding),
        bad_on_duplicate_goal_without_new_dependency=as_bool(bad_on_duplicate_goal_without_new_dependency),
        bad_on_idle_chain=as_bool(bad_on_idle_chain),
        bad_on_invalid_dependency=as_bool(bad_on_invalid_dependency),
    )
    gold_ans_text = ground_truth.get("gold_answer", "")
    if not gold_ans_text and ground_truth.get("gold_chain"):
        gold_ans_text = extract_final_answer(ground_truth.get("gold_chain", ""))
    pred_ans_text = extract_final_answer(response_str)
    pred_val = parse_number(pred_ans_text)
    gold_val = parse_number(gold_ans_text)
    return {
        "target_var": extract_target_var(response_str),
        "format_pass": bool(check_format(response_str)),
        "final_answer_parsed": bool(pred_val is not None and gold_val is not None),
        "final_answer_correct": bool(pred_val is not None and gold_val is not None and abs(pred_val - gold_val) < 1e-6),
        "step_norm_min": max(int(step_norm_min), 1),
        "question_text": step_metrics["question_text"],
        "step_details": step_metrics["step_details"],
        "score_rubric": {
            "good_step": "结构完整、计算正确、Source grounded，并且该步骤对当前题目标求解有真实推进",
            "bad_step": "缺字段、算错、Source 脱离题干/前序变量、重复刷目标变量、空转链或使用未定义依赖",
            "neutral_step": "结构与计算基本可解析，但 grounded 或目标推进不足，暂不奖励也不强惩罚",
        },
    }


def build_reward_result(score, **extra):
    result = {"score": score}
    result.update(extra)
    return result


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
    require_source_grounding=True,
    bad_on_duplicate_goal_without_new_dependency=True,
    bad_on_idle_chain=True,
    bad_on_invalid_dependency=True,
    **kwargs,
):
    response_str = solution_str or ""
    extra_info = extra_info or {}
    enable_format = as_bool(enable_format)
    enable_process = as_bool(enable_process)
    enable_outcome = as_bool(enable_outcome)
    w_format = as_float(w_format, 0.5)
    w_process = as_float(w_process, 1.0)
    w_outcome = as_float(w_outcome, 2.5)
    global_fail_reward = as_float(global_fail_reward, -0.5)
    step_acc_weight = as_float(step_acc_weight, 0.7)
    step_good_weight = as_float(step_good_weight, 0.4)
    step_bad_weight = as_float(step_bad_weight, 0.3)
    step_fmt_weight = as_float(step_fmt_weight, 0.2)
    step_norm_min = max(as_int(step_norm_min, 3), 1)
    require_source_grounding = as_bool(require_source_grounding)
    bad_on_duplicate_goal_without_new_dependency = as_bool(bad_on_duplicate_goal_without_new_dependency)
    bad_on_idle_chain = as_bool(bad_on_idle_chain)
    bad_on_invalid_dependency = as_bool(bad_on_invalid_dependency)

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
            extra_info=extra_info,
            step_norm_min=step_norm_min,
            require_source_grounding=require_source_grounding,
            bad_on_duplicate_goal_without_new_dependency=bad_on_duplicate_goal_without_new_dependency,
            bad_on_idle_chain=bad_on_idle_chain,
            bad_on_invalid_dependency=bad_on_invalid_dependency,
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
