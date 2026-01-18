import os
import sys
import json
import re
import math
import ast
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# vLLM
from vllm import LLM, SamplingParams

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================
# Paths (edit here)
# =========================
MODEL_PATH = "/root/autodl-tmp/A-new/output/model/sft_train_test_py"
TEST_DATA_PATH = "/root/autodl-tmp/A-new/data/test_data/gsm8k-test.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/A-new/output/result"


# =========================
# Eval params
# =========================
GEN_MAX_TOKENS = 1024          # vLLM uses max_tokens
GEN_BATCH_SIZE = 128            # vLLM: 先把 batch 拉大，吞吐会明显更高（按显存调）
ANSWER_TOL = 1e-6

SYSTEM_PROMPT = (
    "You are a reasoning engine. Solve the math problem by strictly generating the XML output: "
    "<reasoning>, <plan>, and <answer>."
)

# vLLM performance knobs (adjust if needed)
TENSOR_PARALLEL_SIZE = 1       # 多卡可设 >1
GPU_MEMORY_UTILIZATION = 0.9  # 0.85~0.95 之间调
MAX_MODEL_LEN = 2048           # 评测输入最大长度（越大越占显存）
STOP_AT_ANSWER_END = True      # 生成到 </answer> 就停，加速显著

# =========================
# Answer extraction
# =========================
GSM8K_ANSWER_RE = re.compile(r"####\s*(.*?)$", re.MULTILINE)
XML_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

XML_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
XML_PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.DOTALL)


def extract_gsm8k_answer(text: str):
    if text is None:
        return None
    m = GSM8K_ANSWER_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    try:
        return float(ans.replace(",", ""))
    except Exception:
        return ans


def extract_xml_answer(text: str):
    """取最后一个 <answer>...</answer>"""
    if text is None:
        return None
    all_ans = XML_ANSWER_RE.findall(text)
    if not all_ans:
        return None
    ans = all_ans[-1].strip()
    try:
        return float(ans.replace(",", ""))
    except Exception:
        return ans


def extract_answer(text: str):
    """
    通用答案提取：
    1) <answer>（优先，因为你现在目标是 XML）
    2) ####
    3) 最后一个数字兜底
    """
    if text is None:
        return None

    ans = extract_xml_answer(text)
    if ans is not None:
        return ans

    ans = extract_gsm8k_answer(text)
    if ans is not None:
        return ans

    numbers = re.findall(r"\b-?\d+(?:\.\d+)?\b", text)
    if numbers:
        try:
            return float(numbers[-1])
        except Exception:
            return numbers[-1]
    return None


def extract_gt_from_gsm8k(answer_text: str):
    ans = extract_gsm8k_answer(answer_text)
    if ans is not None:
        return ans
    numbers = re.findall(r"\b-?\d+(?:\.\d+)?\b", answer_text or "")
    if numbers:
        try:
            return float(numbers[-1])
        except Exception:
            return numbers[-1]
    return None


def is_correct(pred, gt, tol=1e-6):
    if pred is None or gt is None:
        return False
    if isinstance(pred, float) and isinstance(gt, float):
        return (math.isfinite(pred) and math.isfinite(gt) and abs(pred - gt) <= tol)
    return str(pred).strip() == str(gt).strip()


# =========================
# XML + Python plan checks
# =========================
def check_xml_structure(text: str) -> Dict[str, Any]:
    """
    检查是否包含三段 XML 标签，并给出基础结构信息。
    不要求严格 XML 语法，但会检查标签存在、顺序大致合理。
    """
    if not text:
        return {"has_reasoning": False, "has_plan": False, "has_answer": False, "order_ok": False}

    r_pos = text.find("<reasoning>")
    p_pos = text.find("<plan>")
    a_pos = text.find("<answer>")
    has_reasoning = r_pos != -1 and text.find("</reasoning>") != -1
    has_plan = p_pos != -1 and text.find("</plan>") != -1
    has_answer = a_pos != -1 and text.find("</answer>") != -1
    order_ok = (r_pos != -1 and p_pos != -1 and a_pos != -1 and r_pos < p_pos < a_pos)
    return {
        "has_reasoning": has_reasoning,
        "has_plan": has_plan,
        "has_answer": has_answer,
        "order_ok": order_ok,
    }


def extract_plan_code(text: str) -> Optional[str]:
    """抽取 <plan>...</plan> 中的内容（最后一个 plan）"""
    if not text:
        return None
    all_plans = XML_PLAN_RE.findall(text)
    if not all_plans:
        return None
    return all_plans[-1].strip()


def check_python_plan(plan_code: Optional[str]) -> Dict[str, Any]:
    """
    检查 plan 是否是“合理的 Python solve()”：
    - 包含 def solve():
    - ast.parse 通过
    - compile 通过
    不执行代码（安全）。
    """
    if not plan_code:
        return {"plan_present": False, "plan_ok": False, "plan_error": "missing <plan> content"}

    if "def solve" not in plan_code:
        return {"plan_present": True, "plan_ok": False, "plan_error": "no 'def solve' found"}

    try:
        tree = ast.parse(plan_code)
    except Exception as e:
        return {"plan_present": True, "plan_ok": False, "plan_error": f"ast.parse failed: {repr(e)}"}

    try:
        compile(plan_code, filename="<plan>", mode="exec")
    except Exception as e:
        return {"plan_present": True, "plan_ok": False, "plan_error": f"compile failed: {repr(e)}"}

    # 进一步：是否真的定义了 solve
    has_solve = any(
        isinstance(node, ast.FunctionDef) and node.name == "solve"
        for node in tree.body
    )
    if not has_solve:
        return {"plan_present": True, "plan_ok": False, "plan_error": "no function named solve() defined"}

    return {"plan_present": True, "plan_ok": True, "plan_error": ""}


# =========================
# vLLM inference
# =========================
def build_chat_prompt(tokenizer: AutoTokenizer, question: str, system_prompt: str) -> str:
    """
    贴合你训练的 messages：system + user("Problem:\n...")
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem:\n{question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_generate(
    llm: LLM,
    prompts: List[str],
    max_tokens: int,
    stop_at_answer_end: bool = True,
) -> List[str]:
    """
    返回每个 prompt 的生成文本（不包含 prompt 本身）。
    如果 stop_at_answer_end=True：遇到 </answer> 即停止，并把 </answer> 补回（便于解析）。
    """
    if stop_at_answer_end:
        # vLLM stop 会截掉 stop 字符串本身，所以我们后面补回
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=["</answer>"],
        )
        outputs = llm.generate(prompts, sampling_params)
        texts = []
        for out in outputs:
            t = out.outputs[0].text
            # 补回终止符，保证 XML 可解析
            if "</answer>" not in t:
                t = t + "</answer>"
            texts.append(t)
        return texts

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


# =========================
# Eval
# =========================
def evaluate_gsm8k_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    max_tokens: int,
    batch_size: int,
    tol: float,
    system_prompt: str,
    stop_at_answer_end: bool = True,
) -> Dict[str, Any]:
    logger.info(f"开始评测 GSM8K，共 {len(dataset)} 个样本")
    total, correct = 0, 0

    # 统计：格式/plan 正确率
    xml_ok_cnt = 0
    plan_ok_cnt = 0
    answer_tag_cnt = 0

    detailed_logs = []

    idxs = list(range(len(dataset)))
    for start in tqdm(range(0, len(idxs), batch_size), desc="Eval GSM8K (vLLM)"):
        batch_ids = idxs[start:start + batch_size]
        batch = [dataset[i] for i in batch_ids]

        prompts = [build_chat_prompt(tokenizer, ex.get("question", ""), system_prompt) for ex in batch]
        outs = run_vllm_generate(llm, prompts, max_tokens=max_tokens, stop_at_answer_end=stop_at_answer_end)

        for ex, prompt, out in zip(batch, prompts, outs):
            question = ex.get("question", "")
            gt_text = ex.get("answer", "")
            gt = extract_gt_from_gsm8k(gt_text)

            xml_info = check_xml_structure(out)
            plan_code = extract_plan_code(out)
            plan_info = check_python_plan(plan_code)

            pred = extract_answer(out)
            ok = is_correct(pred, gt, tol=tol)

            total += 1
            correct += int(ok)

            xml_ok_cnt += int(xml_info["has_reasoning"] and xml_info["has_plan"] and xml_info["has_answer"] and xml_info["order_ok"])
            plan_ok_cnt += int(plan_info["plan_ok"])
            answer_tag_cnt += int(xml_info["has_answer"])

            detailed_logs.append({
                "sample_id": total,
                "question": question,
                "prompt_preview": prompt[-400:],   # 防止文件爆炸：只存末尾一截
                "model_output": out,
                "ground_truth": gt_text,
                "predicted_answer": pred,
                "true_answer": gt,
                "is_correct": ok,
                "xml_check": xml_info,
                "python_plan_check": plan_info,
                "plan_code_excerpt": (plan_code[:400] + "...") if plan_code and len(plan_code) > 400 else (plan_code or ""),
                "timestamp": datetime.now().isoformat(),
            })

    report = {
        "overall": {"total": total, "correct": correct, "acc": correct / max(total, 1)},
        "format_metrics": {
            "xml_full_ok_rate": xml_ok_cnt / max(total, 1),
            "answer_tag_rate": answer_tag_cnt / max(total, 1),
            "python_plan_ok_rate": plan_ok_cnt / max(total, 1),
        },
        "detailed_logs": detailed_logs,
    }
    return report


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",   # 对 vLLM 无所谓，但你 chat_template 可能依赖 pad 设置
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer eos_token={tokenizer.eos_token} pad_token={tokenizer.pad_token}")

    logger.info("加载 vLLM 模型...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        # dtype 可不写，让 vLLM 自己选；如需强制可加 dtype="bfloat16"/"float16"
    )

    logger.info(f"加载测试集：{TEST_DATA_PATH}")
    ds = load_dataset("json", data_files={"test": TEST_DATA_PATH})
    test_dataset = ds["test"]
    logger.info(f"测试集大小：{len(test_dataset)}")

    logger.info("开始评测（vLLM）...")
    test_report = evaluate_gsm8k_vllm(
        llm=llm,
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_tokens=GEN_MAX_TOKENS,
        batch_size=GEN_BATCH_SIZE,
        tol=ANSWER_TOL,
        system_prompt=SYSTEM_PROMPT,
        stop_at_answer_end=STOP_AT_ANSWER_END,
    )

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(TEST_DATA_PATH).replace(".jsonl", "")

    detailed_logs_path = os.path.join(OUTPUT_DIR, f"{current_time}_vllm_{dataset_name}_detailed.jsonl")
    with open(detailed_logs_path, "w", encoding="utf-8") as f:
        for log_entry in test_report["detailed_logs"]:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"详细日志已保存到：{detailed_logs_path}")

    report_without_logs = dict(test_report)
    report_without_logs.pop("detailed_logs", None)

    report_path = os.path.join(OUTPUT_DIR, f"{current_time}_vllm_{dataset_name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_without_logs, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存到：{report_path}")

    logger.info("\n=== 评估结果 ===")
    logger.info(f"整体准确率: {test_report['overall']['acc']:.4f}")
    logger.info(f"总样本数: {test_report['overall']['total']}")
    logger.info(f"正确样本数: {test_report['overall']['correct']}")
    logger.info("--- 格式指标 ---")
    logger.info(f"XML三段完整率: {test_report['format_metrics']['xml_full_ok_rate']:.4f}")
    logger.info(f"<answer>出现率: {test_report['format_metrics']['answer_tag_rate']:.4f}")
    logger.info(f"Python plan 可编译率: {test_report['format_metrics']['python_plan_ok_rate']:.4f}")
    logger.info("\n评估完成！")


if __name__ == "__main__":
    main()