import ast
import re
from collections import Counter, defaultdict


STEP_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s*(Define|Derive|Calculate)\s+Var\{([^}]+)\}", re.IGNORECASE | re.MULTILINE)
NATURAL_STEP_HEADER_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.\s+(?!Define\b|Derive\b|Calculate\s+Var\{)(.+\S.*)$",
    re.IGNORECASE | re.MULTILINE,
)
VAR_RE = re.compile(r"Var\{([^}]+)\}")
INLINE_EQUATION_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:\s*%\s*)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?(?:\s*%\s*)?)+\s*=\s*-?\d+(?:\.\d+)?)"
)

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


def normalize_number_token(token):
    try:
        value = float(str(token).replace(",", "").replace("$", "").strip())
    except Exception:
        return str(token).strip()
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


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
            results.append(normalize_number_token(value))
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
    clean = str(text).replace(",", "").replace("$", "")
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
    return all(tag in text for tag in required)


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


def has_math_signal(text):
    text = text or ""
    if re.search(r"\d", text) and (re.search(r"[+\-*/=]", text) or "<<" in text or ">>" in text):
        return True
    if NATURAL_STEP_HEADER_RE.search(text):
        return True
    return False


def split_steps_strict(backward_text):
    if not backward_text:
        return []
    matches = list(STEP_HEADER_RE.finditer(backward_text))
    if not matches:
        return []
    steps = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(backward_text)
        steps.append(backward_text[start:end].strip())
    return steps


def split_steps_natural(backward_text):
    if not backward_text:
        return []
    matches = list(NATURAL_STEP_HEADER_RE.finditer(backward_text))
    if matches:
        steps = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(backward_text)
            steps.append(backward_text[start:end].strip())
        return [step for step in steps if step]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", backward_text) if part.strip()]
    candidates = [part for part in paragraphs if has_math_signal(part)]
    if len(candidates) >= 2:
        return candidates
    if len(candidates) == 1 and len(candidates[0].splitlines()) >= 2:
        return candidates
    return []


def parse_steps_dual_mode(backward_text, step_parser_mode="dual", enable_natural_step_parser=True):
    step_parser_mode = str(step_parser_mode or "dual").strip().lower()
    if step_parser_mode not in {"strict", "natural", "dual"}:
        step_parser_mode = "dual"

    if step_parser_mode in {"strict", "dual"}:
        strict_steps = split_steps_strict(backward_text)
        if strict_steps:
            return strict_steps, "strict", "matched_strict_step_headers"
        if step_parser_mode == "strict":
            return [], "failed", "strict_parser_found_no_steps"

    if enable_natural_step_parser and step_parser_mode in {"natural", "dual"}:
        natural_steps = split_steps_natural(backward_text)
        if natural_steps:
            return natural_steps, "natural", "matched_natural_step_boundaries"
        return [], "failed", "natural_parser_found_no_steps"

    return [], "failed", "natural_parser_disabled"


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


def parse_natural_step_header(step_text, index):
    first_line = (step_text or "").strip().splitlines()[0].strip() if (step_text or "").strip() else ""
    match = NATURAL_STEP_HEADER_RE.search(first_line)
    if match:
        return {
            "step_id": match.group(1).strip(),
            "title": match.group(2).strip(),
        }
    return {
        "step_id": str(index + 1),
        "title": first_line,
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


def extract_numbers(text):
    if not text:
        return set()
    clean = str(text).replace(",", "").replace("$", "")
    return {normalize_number_token(match) for match in re.findall(r"-?\d+(?:\.\d+)?", clean)}


def extract_calc_segments(calc_text):
    return [segment.strip() for segment in re.findall(r"<<(.*?)>>", calc_text or "")]


def extract_calc_operand_numbers(calc_text):
    operand_numbers = set()
    for segment in extract_calc_segments(calc_text):
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        expr_parts = parts[:-1] if len(parts) >= 2 else parts
        for part in expr_parts:
            operand_numbers.update(extract_numbers(part))
    return operand_numbers


def extract_calc_result_numbers(calc_text):
    result_numbers = set()
    for segment in extract_calc_segments(calc_text):
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        if not parts:
            continue
        value = safe_eval_numeric(parts[-1])
        if value is not None:
            result_numbers.add(normalize_number_token(value))
        else:
            result_numbers.update(extract_numbers(parts[-1]))
    return result_numbers


def calc_uses_operator(calc_text):
    for segment in extract_calc_segments(calc_text):
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        expr_parts = parts[:-1] if len(parts) >= 2 else parts
        for part in expr_parts:
            expr = _normalize_expr(part)
            if any(op in expr for op in ["+", "-", "*", "/", "%", "**"]):
                return True
    return False


def tokenize_identifier(text):
    normalized = re.sub(r"[_\d]+", " ", text or "")
    tokens = re.findall(r"[A-Za-z][A-Za-z_-]*", normalized.lower())
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


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
            return left ** right
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
    return extract_numbers(question_text)


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
        "strong_grounded": bool(number_hits or prev_var_hits),
        "question_number_hits": number_hits,
        "question_keyword_hits": keyword_hits,
        "prev_var_hits": prev_var_hits,
    }


def extract_inline_equation_analyses(text):
    analyses = []
    for segment in INLINE_EQUATION_RE.findall(text or ""):
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        if len(parts) < 2:
            continue
        left_expr = "=".join(parts[:-1])
        right_expr = parts[-1]
        left_val = safe_eval_numeric(left_expr)
        right_val = safe_eval_numeric(right_expr)
        analyses.append(
            {
                "raw": segment,
                "left_value": left_val,
                "right_value": right_val,
                "correct": left_val is not None and right_val is not None and abs(left_val - right_val) < 1e-6,
                "operand_numbers": extract_numbers(left_expr),
                "result_numbers": {normalize_number_token(right_val)} if right_val is not None else extract_numbers(right_expr),
            }
        )
    return analyses


def summarize_step_reasons(step_details):
    if not step_details:
        return ""
    counts = Counter(detail.get("reason", "unknown") for detail in step_details)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def strict_result_used_later(var_name, result_numbers, future_text, is_goal_var, is_last_step):
    future_refs = set(extract_var_refs(future_text))
    future_numbers = extract_numbers(future_text)
    if var_name and var_name in future_refs:
        return True
    if result_numbers & future_numbers:
        return True
    if is_goal_var and is_last_step:
        return True
    return False


def classify_step_strict(
    step_text,
    target_var,
    question_text,
    future_text,
    seen_vars,
    var_history,
    require_source_grounding,
    bad_on_duplicate_goal_without_new_dependency,
    bad_on_idle_chain,
    bad_on_invalid_dependency,
    is_last_step,
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
            "question_number_hits": [],
            "question_keyword_hits": [],
            "prev_var_hits": [],
            "calc_refs": [],
            "step_title": step_text.strip().splitlines()[0].strip() if step_text.strip() else "",
            "parser_mode": "strict",
            "constraint_mode": "strict",
            "result_numbers": set(),
        }

    var_name = header["var_name"]
    is_goal_var = bool(target_var) and var_name == target_var
    calc_status = check_calc_correct(calc_text)
    calc_has_target = bool(re.search(rf"Var\{{\s*{re.escape(var_name)}\s*\}}\s*=", calc_text or "", re.IGNORECASE))
    calc_refs = [ref for ref in extract_var_refs(calc_text) if ref != var_name]
    source_grounding = analyze_source_grounding(source_text, question_text, seen_vars)
    grounded = source_grounding["grounded"]
    strong_grounded = source_grounding["strong_grounded"]
    invalid_dependency = any(ref not in seen_vars for ref in calc_refs)
    source_numbers = extract_question_numbers(source_text)
    question_numbers = extract_question_numbers(question_text)
    operand_numbers = extract_calc_operand_numbers(calc_text)
    result_numbers = extract_calc_result_numbers(calc_text)
    unsupported_operand_numbers = sorted(operand_numbers - question_numbers - source_numbers)

    calc_signature = normalize_token(calc_text)
    source_signature = normalize_token(source_text)
    vh = var_history.get(
        var_name,
        {
            "signatures": [],
            "good_set": False,
            "source_numbers": set(),
            "source_vars": set(),
            "calc_refs": set(),
        },
    )
    duplicate_same_content = any(sig == (calc_signature, source_signature) for sig in vh.get("signatures", []))
    new_prev_var_hits = sorted(set(source_grounding["prev_var_hits"]) - set(vh.get("source_vars", set())))
    new_question_number_hits = sorted(set(source_grounding["question_number_hits"]) - set(vh.get("source_numbers", set())))
    new_calc_refs = sorted(set(calc_refs) - set(vh.get("calc_refs", set())))
    has_new_support = bool(new_prev_var_hits or new_question_number_hits or new_calc_refs)
    result_used_later = strict_result_used_later(var_name, result_numbers, future_text, is_goal_var, is_last_step)
    repeated_goal_without_new_dependency = is_goal_var and vh.get("good_set", False) and not has_new_support
    idle_chain = duplicate_same_content or (var_name in seen_vars and not has_new_support and not result_used_later)
    source_required_but_missing = require_source_grounding and not strong_grounded
    source_ok = strong_grounded if require_source_grounding else (grounded or bool(calc_refs) or bool(operand_numbers & question_numbers))

    label = "neutral"
    reason = "neutral_step"

    if not has_reasoning or not has_source or not has_calc:
        label = "bad"
        reason = "missing_required_block"
    elif calc_status == "incorrect":
        label = "bad"
        reason = "incorrect"
    elif calc_status == "missing":
        label = "bad"
        reason = "missing_calc"
    elif bad_on_invalid_dependency and invalid_dependency:
        label = "bad"
        reason = "invalid_dependency"
    elif unsupported_operand_numbers:
        label = "bad"
        reason = "out_of_scope"
    elif bad_on_duplicate_goal_without_new_dependency and repeated_goal_without_new_dependency:
        label = "bad"
        reason = "duplicate_goal_without_new_dependency"
    elif bad_on_idle_chain and idle_chain:
        label = "bad"
        reason = "idle_chain"
    elif is_goal_var and not calc_refs and not source_grounding["prev_var_hits"] and calc_status == "correct" and not vh.get("good_set", False):
        label = "bad"
        reason = "missing_key_dependency_for_goal"
    elif source_required_but_missing:
        label = "bad"
        reason = "ungrounded_source"
    elif calc_status == "correct" and calc_has_target and source_ok and result_used_later:
        label = "good"
        reason = "good_step"
    elif calc_status == "correct" and calc_has_target:
        label = "neutral"
        reason = "unused_result"
    elif calc_status == "unverifiable":
        label = "neutral"
        reason = "unverifiable"

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
        "is_useful": result_used_later,
        "source_grounded": grounded,
        "source_grounded_by": source_grounded_by,
        "question_number_hits": source_grounding["question_number_hits"],
        "question_keyword_hits": source_grounding["question_keyword_hits"],
        "prev_var_hits": source_grounding["prev_var_hits"],
        "calc_refs": calc_refs,
        "step_title": step_text.strip().splitlines()[0].strip() if step_text.strip() else "",
        "parser_mode": "strict",
        "constraint_mode": "strict",
        "result_numbers": result_numbers,
    }


def classify_step_natural(
    step_text,
    question_numbers,
    previous_result_numbers,
    future_text,
    is_last_step,
    final_answer_value,
    seen_step_signatures,
    seen_result_signatures,
    index,
):
    header = parse_natural_step_header(step_text, index)
    step_numbers = extract_numbers(step_text)
    equation_analyses = extract_inline_equation_analyses(step_text)
    incorrect_equation = any(not item["correct"] for item in equation_analyses)
    has_equation = bool(equation_analyses)
    operand_numbers = set().union(*(item["operand_numbers"] for item in equation_analyses)) if equation_analyses else set()
    result_numbers = set().union(*(item["result_numbers"] for item in equation_analyses)) if equation_analyses else set()
    direct_fact = False
    if not has_equation:
        q_hits = step_numbers & question_numbers
        prev_hits = step_numbers & previous_result_numbers
        if len(step_numbers) == 1 and (q_hits or prev_hits):
            direct_fact = True
            result_numbers = set(step_numbers)
    source_candidate_numbers = operand_numbers if has_equation else step_numbers
    source_hits = source_candidate_numbers & (question_numbers | previous_result_numbers)
    unsupported_source_numbers = sorted(source_candidate_numbers - question_numbers - previous_result_numbers - result_numbers)
    has_numeric_progress = bool(has_equation or direct_fact)
    future_numbers = extract_numbers(future_text)
    final_answer_token = normalize_number_token(final_answer_value) if final_answer_value is not None else None
    result_used_later = bool(result_numbers & future_numbers)
    if not result_used_later and is_last_step and final_answer_token:
        result_used_later = final_answer_token in result_numbers

    signature = normalize_token(step_text)
    result_signature = tuple(sorted(result_numbers))
    duplicate_same_content = signature in seen_step_signatures
    duplicate_same_result = bool(result_numbers) and result_signature in seen_result_signatures

    label = "neutral"
    reason = "neutral_step"
    calc_status = "missing"
    if has_equation:
        calc_status = "incorrect" if incorrect_equation else "correct"
    elif direct_fact:
        calc_status = "correct"
    elif step_numbers:
        calc_status = "unverifiable"

    if incorrect_equation:
        label = "bad"
        reason = "incorrect"
    elif unsupported_source_numbers:
        label = "bad"
        reason = "out_of_scope"
    elif duplicate_same_content or duplicate_same_result:
        label = "bad"
        reason = "duplicate_without_new_info"
    elif not has_numeric_progress and not step_numbers:
        label = "neutral"
        reason = "explanatory_only"
    elif has_numeric_progress and not source_hits and not direct_fact:
        label = "bad"
        reason = "missing_grounding"
    elif has_numeric_progress and not result_numbers:
        label = "bad"
        reason = "missing_numeric_result"
    elif has_numeric_progress and result_used_later:
        label = "good"
        reason = "good_step"
    elif has_numeric_progress:
        label = "neutral"
        reason = "unused_result"
    elif step_numbers:
        label = "neutral"
        reason = "weak_progress"

    return label, {
        "reason": reason,
        "var_name": None,
        "calc_status": calc_status,
        "has_reasoning": True,
        "has_source": bool(source_hits or direct_fact),
        "has_calc": bool(has_equation or direct_fact),
        "is_goal_var": is_last_step,
        "is_useful": result_used_later,
        "source_grounded": bool(source_hits or direct_fact),
        "source_grounded_by": [
            key
            for key, condition in (
                ("question_number", bool(source_candidate_numbers & question_numbers)),
                ("previous_result", bool(source_candidate_numbers & previous_result_numbers)),
                ("direct_fact", direct_fact),
            )
            if condition
        ],
        "question_number_hits": sorted(source_candidate_numbers & question_numbers),
        "question_keyword_hits": [],
        "prev_var_hits": sorted(source_candidate_numbers & previous_result_numbers),
        "calc_refs": [],
        "step_title": header["title"],
        "parser_mode": "natural",
        "constraint_mode": "natural",
        "result_numbers": result_numbers,
    }


def compute_step_rule_process_score(
    response_str,
    extra_info,
    step_norm_min,
    require_source_grounding,
    bad_on_duplicate_goal_without_new_dependency,
    bad_on_idle_chain,
    bad_on_invalid_dependency,
    step_parser_mode="dual",
    enable_natural_step_parser=True,
):
    target_var = extract_target_var(response_str)
    backward_text = extract_backward_execution(response_str)
    question_text = extract_current_question(extra_info)
    final_answer_text = extract_final_answer(response_str)
    final_answer_value = parse_number(final_answer_text)
    steps, parse_mode, parse_reason = parse_steps_dual_mode(
        backward_text,
        step_parser_mode=step_parser_mode,
        enable_natural_step_parser=enable_natural_step_parser,
    )
    parsed_step_count = len(steps)
    good_count = 0
    bad_count = 0
    neutral_count = 0
    seen_vars = set()
    previous_result_numbers = set()
    seen_step_signatures = set()
    seen_result_signatures = set()
    var_history = defaultdict(
        lambda: {
            "signatures": [],
            "good_set": False,
            "source_numbers": set(),
            "source_vars": set(),
            "calc_refs": set(),
        }
    )
    step_details = []

    if not steps:
        z = max(step_norm_min, 0)
        return {
            "good_count": 0,
            "bad_count": 0,
            "neutral_count": 0,
            "step_count": 0,
            "parsed_step_count": 0,
            "step_parse_failed": 1,
            "step_parse_mode": "failed",
            "step_constraint_mode": "failed",
            "step_parse_reason": parse_reason,
            "step_constraint_hits": "parse_failed:1",
            "z": z,
            "good_ratio": 0.0,
            "bad_ratio": 0.0,
            "step_details": [],
            "question_text": question_text,
        }

    question_numbers = extract_question_numbers(question_text)
    for idx, step_text in enumerate(steps):
        future_text = "\n".join(steps[idx + 1 :])
        is_last_step = idx == len(steps) - 1
        if parse_mode == "strict":
            label, meta = classify_step_strict(
                step_text=step_text,
                target_var=target_var,
                question_text=question_text,
                future_text=future_text,
                seen_vars=seen_vars,
                var_history=var_history,
                require_source_grounding=require_source_grounding,
                bad_on_duplicate_goal_without_new_dependency=bad_on_duplicate_goal_without_new_dependency,
                bad_on_idle_chain=bad_on_idle_chain,
                bad_on_invalid_dependency=bad_on_invalid_dependency,
                is_last_step=is_last_step,
            )
        else:
            label, meta = classify_step_natural(
                step_text=step_text,
                question_numbers=question_numbers,
                previous_result_numbers=previous_result_numbers,
                future_text=future_text,
                is_last_step=is_last_step,
                final_answer_value=final_answer_value,
                seen_step_signatures=seen_step_signatures,
                seen_result_signatures=seen_result_signatures,
                index=idx,
            )

        if label == "good":
            good_count += 1
        elif label == "bad":
            bad_count += 1
        else:
            neutral_count += 1

        seen_step_signatures.add(normalize_token(step_text))
        result_signature = tuple(sorted(meta.get("result_numbers", set())))
        if result_signature:
            seen_result_signatures.add(result_signature)
        previous_result_numbers.update(meta.get("result_numbers", set()))

        if meta.get("var_name"):
            seen_vars.add(meta["var_name"])
            calc_signature = normalize_token(extract_calc_block(step_text))
            source_signature = normalize_token(extract_source_block(step_text))
            vrec = var_history[meta["var_name"]]
            vrec["signatures"].append((calc_signature, source_signature))
            vrec["source_numbers"].update(meta.get("question_number_hits", []))
            vrec["source_vars"].update(meta.get("prev_var_hits", []))
            vrec["calc_refs"].update(meta.get("calc_refs", []))
            if meta.get("is_goal_var") and label == "good":
                vrec["good_set"] = True
            var_history[meta["var_name"]] = vrec

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
                "parser_mode": meta["parser_mode"],
                "constraint_mode": meta["constraint_mode"],
                "result_numbers": sorted(meta.get("result_numbers", set())),
            }
        )

    step_count = len(steps)
    z = max(step_norm_min, step_count)
    return {
        "good_count": good_count,
        "bad_count": bad_count,
        "neutral_count": neutral_count,
        "step_count": step_count,
        "parsed_step_count": step_count,
        "step_parse_failed": 0,
        "step_parse_mode": parse_mode,
        "step_constraint_mode": parse_mode,
        "step_parse_reason": parse_reason,
        "step_constraint_hits": summarize_step_reasons(step_details),
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
    step_parser_mode="dual",
    enable_natural_step_parser=True,
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
        step_parser_mode=step_parser_mode,
        enable_natural_step_parser=as_bool(enable_natural_step_parser),
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
        "step_parse_mode": step_metrics["step_parse_mode"],
        "step_parse_failed": step_metrics["step_parse_failed"],
        "step_parse_reason": step_metrics["step_parse_reason"],
        "parsed_step_count": step_metrics["parsed_step_count"],
        "step_details": step_metrics["step_details"],
        "score_rubric": {
            "good_step": "strict: 结构完整、数值正确、来源可追踪、结果被后续使用或为最终步；natural: 有明确数值推进、来源明确、结果被后续使用或为最终步",
            "bad_step": "数值错误、题外数字、未定义依赖、无依据跳步、重复空转或严格结构缺关键块",
            "neutral_step": "可解析但推进不足、结果暂未被后续消费、或缺少可验证算式但无明显错误",
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
    step_parser_mode="dual",
    enable_natural_step_parser=True,
    strict_step_preferred=True,
    strict_format_term=1.0,
    natural_format_term=0.5,
    failed_format_term=-1.0,
    enforce_explicit_step_constraints=True,
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
    enable_natural_step_parser = as_bool(enable_natural_step_parser)
    strict_step_preferred = as_bool(strict_step_preferred)
    strict_format_term = as_float(strict_format_term, 1.0)
    natural_format_term = as_float(natural_format_term, 0.5)
    failed_format_term = as_float(failed_format_term, -1.0)

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
        parser_mode = "strict" if strict_step_preferred and str(step_parser_mode).strip().lower() == "strict" else step_parser_mode
        step_metrics = compute_step_rule_process_score(
            response_str=response_str,
            extra_info=extra_info,
            step_norm_min=step_norm_min,
            require_source_grounding=require_source_grounding,
            bad_on_duplicate_goal_without_new_dependency=bad_on_duplicate_goal_without_new_dependency,
            bad_on_idle_chain=bad_on_idle_chain,
            bad_on_invalid_dependency=bad_on_invalid_dependency,
            step_parser_mode=parser_mode,
            enable_natural_step_parser=enable_natural_step_parser,
        )
        step_process_score = step_metrics["good_ratio"] - step_metrics["bad_ratio"]
        if not format_ok:
            format_term = failed_format_term
        elif step_metrics["step_parse_mode"] == "strict":
            format_term = strict_format_term
        elif step_metrics["step_parse_mode"] == "natural":
            format_term = natural_format_term
        else:
            format_term = failed_format_term
        total_reward = (
            step_acc_weight * r_acc
            + step_good_weight * step_metrics["good_ratio"]
            - step_bad_weight * step_metrics["bad_ratio"]
            + step_fmt_weight * format_term
        )
        return build_reward_result(
            total_reward,
            reward_mode=reward_mode,
            r_acc=r_acc,
            r_fmt=format_term,
            legacy_process_score=legacy_process_score,
            step_process_score=step_process_score,
            step_good_count=step_metrics["good_count"],
            step_bad_count=step_metrics["bad_count"],
            step_neutral_count=step_metrics["neutral_count"],
            step_count=step_metrics["step_count"],
            parsed_step_count=step_metrics["parsed_step_count"],
            step_parse_failed=step_metrics["step_parse_failed"],
            step_parse_mode=step_metrics["step_parse_mode"],
            step_parse_reason=step_metrics["step_parse_reason"],
            step_constraint_mode=step_metrics["step_constraint_mode"],
            step_constraint_hits=step_metrics["step_constraint_hits"],
            step_norm_z=step_metrics["z"],
            global_format_pass=int(format_ok),
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
        parsed_step_count=0,
        step_parse_failed=0,
        step_parse_mode="legacy",
        step_parse_reason="legacy_reward_mode",
        step_constraint_mode="legacy",
        step_constraint_hits="",
        step_norm_z=0,
        global_format_pass=int(format_ok),
        final_answer_parsed=final_answer_parsed,
    )
