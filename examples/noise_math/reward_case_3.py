import ast
import re


STEP_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s*(Define|Derive|Calculate)\s+Var\{([^}]+)\}", re.IGNORECASE | re.MULTILINE)
VAR_RE = re.compile(r"Var\{([^}]+)\}")


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


def extract_final_answer(text):
    if not text:
        return None
    match = re.search(r"\[Final Answer\]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def check_format(text):
    if not text:
        return False
    for tag in ("[Goal Analysis]", "[Backward Execution]", "[Final Answer]"):
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
        steps.append(backward_text[start:end].strip())
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
    return [item.strip() for item in VAR_RE.findall(text)]


def normalize_expr(expr):
    expr = expr.strip().replace(",", "").replace("$", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
        left = safe_eval_node(node.left)
        right = safe_eval_node(node.right)
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
    expr = normalize_expr(expr)
    if re.search(r"[A-Za-z_]", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        return float(safe_eval_node(tree))
    except Exception:
        return None


def extract_numeric_value(text):
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*%|-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?", text.replace(",", ""))
    for candidate in reversed(matches):
        value = safe_eval_numeric(candidate)
        if value is not None:
            return value
    return None


def check_calc_correct(calc_text):
    segments = re.findall(r"<<(.*?)>>", calc_text or "")
    if not segments:
        return False
    for segment in segments:
        parts = [part.strip() for part in segment.split("=") if part.strip()]
        if not parts:
            return False
        values = [safe_eval_numeric(part) for part in parts]
        if any(value is None for value in values):
            return False
        base = values[0]
        if any(abs(base - value) >= 1e-6 for value in values[1:]):
            return False
    return True


def compute_step_components(step_text, target_var, previous_vars, future_text):
    header = parse_step_header(step_text)
    calc_text = extract_calc_block(step_text)
    struct_score = int(header is not None and bool(calc_text) and "<<" in calc_text and ">>" in calc_text)
    if not header:
        return {
            "struct": struct_score,
            "calc": 0,
            "closure": 0,
            "useful": 0,
            "var_name": None,
            "step_score": 0.0,
        }
    current_var = header["var_name"]
    calc_score = int(struct_score == 1 and check_calc_correct(calc_text))
    calc_refs = [item for item in extract_var_refs(calc_text) if item != current_var]
    closure_score = int(struct_score == 1 and all(ref in previous_vars for ref in calc_refs))
    future_refs = set(extract_var_refs(future_text))
    useful_score = int(current_var == target_var or current_var in future_refs)
    step_score = (
        0.20 * struct_score
        + 0.35 * calc_score
        + 0.25 * closure_score
        + 0.20 * useful_score
    )
    return {
        "struct": struct_score,
        "calc": calc_score,
        "closure": closure_score,
        "useful": useful_score,
        "var_name": current_var,
        "step_score": step_score,
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
    global_fail_reward=-0.5,
    acc_weight=0.55,
    fmt_weight=0.15,
    step_weight=0.30,
    step_norm_min=3,
    **kwargs,
):
    response_str = solution_str or ""
    global_fail_reward = as_float(global_fail_reward, -0.5)
    acc_weight = as_float(acc_weight, 0.55)
    fmt_weight = as_float(fmt_weight, 0.15)
    step_weight = as_float(step_weight, 0.30)
    step_norm_min = max(as_int(step_norm_min, 3), 1)

    gold_chain = ground_truth.get("gold_chain", "")
    gold_answer_text = ground_truth.get("gold_answer", "")
    if not gold_answer_text and gold_chain:
        gold_answer_text = extract_final_answer(gold_chain)
    elif gold_answer_text and "[Final Answer]" in str(gold_answer_text):
        gold_answer_text = extract_final_answer(str(gold_answer_text))

    pred_answer_text = extract_final_answer(response_str)
    pred_value = extract_numeric_value(pred_answer_text)
    gold_value = extract_numeric_value(gold_answer_text)
    final_answer_parsed = int(pred_value is not None and gold_value is not None)

    format_ok = check_format(response_str)
    target_var = extract_target_var(response_str)
    steps = split_steps(extract_backward_execution(response_str))
    structure_parseable = int(format_ok and bool(target_var) and len(steps) > 0 and final_answer_parsed == 1)
    if not structure_parseable:
        return build_reward_result(
            global_fail_reward,
            reward_mode="case_3",
            r_acc=0.0,
            r_fmt=0.0,
            r_step=0.0,
            step_count=len(steps),
            step_norm_z=max(step_norm_min, len(steps)),
            step_score_sum=0.0,
            struct_count=0,
            calc_count=0,
            closure_count=0,
            useful_count=0,
            global_format_pass=int(format_ok),
            final_answer_parsed=final_answer_parsed,
        )

    r_acc = 1.0 if abs(pred_value - gold_value) < 1e-6 else 0.0
    r_fmt = 1.0

    previous_vars = set()
    step_score_sum = 0.0
    struct_count = 0
    calc_count = 0
    closure_count = 0
    useful_count = 0

    for idx, step_text in enumerate(steps):
        future_text = "\n".join(steps[idx + 1 :])
        metrics = compute_step_components(step_text, target_var, previous_vars, future_text)
        step_score_sum += metrics["step_score"]
        struct_count += metrics["struct"]
        calc_count += metrics["calc"]
        closure_count += metrics["closure"]
        useful_count += metrics["useful"]
        if metrics["var_name"]:
            previous_vars.add(metrics["var_name"])

    step_norm_z = max(step_norm_min, len(steps))
    r_step = step_score_sum / step_norm_z
    total_reward = acc_weight * r_acc + fmt_weight * r_fmt + step_weight * r_step

    return build_reward_result(
        total_reward,
        reward_mode="case_3",
        r_acc=r_acc,
        r_fmt=r_fmt,
        r_step=r_step,
        step_count=len(steps),
        step_norm_z=step_norm_z,
        step_score_sum=step_score_sum,
        struct_count=struct_count,
        calc_count=calc_count,
        closure_count=closure_count,
        useful_count=useful_count,
        global_format_pass=1,
        final_answer_parsed=final_answer_parsed,
    )
