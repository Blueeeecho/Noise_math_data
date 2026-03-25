import re
import math
from collections import Counter

def extract_calcs(text):
    """Extract content inside <<...>> tags."""
    if not text: return []
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
        except:
            results.append(content.replace(" ", ""))
    return results

def extract_final_answer(text):
    """Extract content after [Final Answer]."""
    if not text: return None
    m = re.search(r"\[Final Answer\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def parse_number(text):
    if not text: return None
    clean = text.replace(",", "").replace("$", "")
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", clean)
    if nums:
        try: return float(nums[-1])
        except: return None
    return None

def check_format(text):
    if not text: return False
    required = ["[Goal Analysis]", "[Backward Execution]", "[Final Answer]"]
    for tag in required:
        if tag not in text: return False
    return True

def compute_reward(data_source, solution_str, ground_truth, extra_info=None,
                   w_format=0.5, w_process=1.0, w_outcome=2.5,
                   enable_format=True, enable_process=True, enable_outcome=True, **kwargs):
    
    response_str = solution_str
    
    # 1. Format Reward
    r_format = -1.0
    if enable_format and check_format(response_str):
        r_format = 1.0
        
    # 2. Process Reward (<<>> Overlap)
    r_process = 0.0
    gold_chain = ground_truth.get("gold_chain", "")
    if enable_process and gold_chain:
        model_calcs = extract_calcs(response_str)
        gold_calcs = extract_calcs(gold_chain)
        
        if gold_calcs:
            model_cnt = Counter(model_calcs)
            gold_cnt = Counter(gold_calcs)
            intersection = model_cnt & gold_cnt
            matched = sum(intersection.values())
            total = sum(gold_cnt.values())
            if total > 0:
                r_process = matched / total

    # 3. Outcome Reward
    r_outcome = 0.0
    gold_ans_text = ground_truth.get("gold_answer", "")
    # If not in dict, try extracting from chain
    if not gold_ans_text and gold_chain:
         gold_ans_text = extract_final_answer(gold_chain)
         
    if enable_outcome and gold_ans_text:
        pred_ans_text = extract_final_answer(response_str)
        pred_val = parse_number(pred_ans_text)
        gold_val = parse_number(gold_ans_text)
        
        if pred_val is not None and gold_val is not None:
            if abs(pred_val - gold_val) < 1e-6:
                r_outcome = 1.0

    total_reward = (w_format * r_format) + (w_process * r_process) + (w_outcome * r_outcome)
    return total_reward
