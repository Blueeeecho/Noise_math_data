import argparse
import json
import os
import re
import math
from tqdm import tqdm
from transformers import AutoTokenizer

# 尝试导入 vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("Error: vLLM is not installed. Please install via `pip install vllm`.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Python-Reasoning Model on Math Datasets")
    
    # 模型路径
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/A-new/output/model/sft/qwen2.5-1.5b-In/sft_train_test_py", 
                        help="SFT/DPO Base Model Path (must be the Python-tuned model)")
    parser.add_argument("--adapter_path", type=str, default="/root/autodl-tmp/A-new/output/model/DPO/1.5B_instruct/new_end_3_hard_lr_2e_6/checkpoint-1476", 
                        help="LoRA Adapter Path (Optional)")

    # 数据路径
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/math/data/test_data/gsmplus_testmini-without-null.jsonl", 
                        help="Path to test data (jsonl)")
    parser.add_argument("--output_root", type=str, default="/root/autodl-tmp/A-new/eval_results/dpo/1.5B_instruct/new_end_3_hard_lr_2e_6-1476", 
                        help="Output directory")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Python code can be long")
    parser.add_argument("--batch_size", type=int, default=200, help="vLLM chunk size")
    
    return parser.parse_args()

# ================= 核心适配函数 =================

def extract_answer_xml(text):
    """
    针对 Python/XML 格式的专用提取器
    修正：兼容 vLLM 因 stop token 导致的 </answer> 缺失
    """
    if not text: return None
    text_str = str(text).strip()

    # 1. 优先尝试：标准完整匹配 <answer>...</answer>
    # (针对那些还没遇到 stop token 就生成完的情况)
    match_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text_str, re.DOTALL)
    if match_xml:
        return match_xml.group(1).strip().replace(',', '')

    # 2. 【核心修正】：截断匹配
    # 如果找不到闭合标签，说明在 </answer> 处停止了。
    # 我们直接提取 <answer> 之后的所有内容，或者提取紧跟的数字。
    match_truncated = re.search(r"<answer>\s*(-?[\d,]+(\.\d+)?)", text_str, re.DOTALL)
    if match_truncated:
        return match_truncated.group(1).replace(',', '')

    # 3. 兼容 Replay Buffer 的 #### 格式
    match_hash = re.search(r'####\s*(-?[\d,]+(\.\d+)?)', text_str)
    if match_hash:
        return match_hash.group(1).replace(',', '')

    # 4. 兼容 \boxed{}
    match_box = re.search(r'\\boxed\{([^}]+)\}', text_str)
    if match_box:
        return match_box.group(1).replace(',', '')

    return None

def is_correct(pred, gold):
    """
    数值比较，容忍度 1e-6
    """
    if not pred or not gold: return False
    
    # 尝试转 float 对比
    try:
        pred_f = float(pred)
        gold_f = float(gold)
        return abs(pred_f - gold_f) < 1e-6
    except:
        # 如果转 float 失败，退化为字符串清理后对比
        def clean(s): return re.sub(r'[^0-9a-zA-Z\.\-]', '', str(s))
        return clean(pred) == clean(gold)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    # 自动适配常见数据集字段
                    q = item.get('question') or item.get('input') or item.get('problem')
                    
                    # 尝试获取答案
                    a = item.get('answer') or item.get('output') or item.get('solution') or item.get('gold')
                    
                    # 针对 GSM8K/ASDiv 标准数据，通常答案在 #### 后面，需要预处理一下 gold
                    if a and isinstance(a, str) and '####' in a:
                        a = a.split('####')[-1].strip()
                    
                    if q and a:
                        data.append({"id": idx, "question": q, "gold": a})
                except: continue
    return data

# ================= 主流程 =================

def main():
    args = parse_args()

    # 1. 准备输出目录
    model_name = os.path.basename(os.path.normpath(args.base_model_path))
    adapter_suffix = f"_{os.path.basename(os.path.normpath(args.adapter_path))}" if args.adapter_path else ""
    save_dir = os.path.join(args.output_root, f"{model_name}{adapter_suffix}")
    
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "eval_details.jsonl")
    result_file = os.path.join(save_dir, "result_summary.txt")

    print(f"Results will be saved to: {save_dir}")

    # 2. 加载 Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    # 3. 设置 Stop Tokens (关键优化)
    # 除了 EOS，我们最好在 </answer> 处停止，防止模型继续吐字
    stop_tokens = ["<|im_end|>", "<|endoftext|>", "</answer>"]
    
    # 4. 初始化 vLLM
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=args.base_model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_lora=bool(args.adapter_path),
        max_lora_rank=64,
        gpu_memory_utilization=0.9 # 显存优化
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0, # Greedy
        max_tokens=args.max_new_tokens,
        stop=stop_tokens
    )

    # LoRA request
    lora_request = None
    if args.adapter_path:
        print(f"Loading LoRA Adapter: {args.adapter_path}")
        lora_request = LoRARequest("eval_adapter", 1, args.adapter_path)

    # 5. 加载数据
    dataset = load_data(args.data_path)
    print(f"Loaded {len(dataset)} samples.")
    
    # 6. 构造 Prompt (【核心修正】: 使用 Python XML System Prompt)
    # 必须和你 SFT/DPO 训练时的 Prompt 保持 100% 一致！
    SYSTEM_PROMPT = "You are a reasoning engine. Solve the math problem by strictly generating the XML output: <reasoning>, <plan>, and <answer>."
    
    prompts = []
    for item in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{item['question']}"}
        ]
        # 使用 tokenizer 渲染
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 7. 批量推理
    chunk_size = args.batch_size
    total_samples = len(prompts)
    correct_count = 0
    total_processed = 0
    
    # 清空日志文件
    with open(log_file, 'w', encoding='utf-8') as f: pass

    print(f"Starting Inference ({total_samples} samples)...")

    for i in tqdm(range(0, total_samples, chunk_size), desc="Evaluating"):
        chunk_prompts = prompts[i : i + chunk_size]
        chunk_items = dataset[i : i + chunk_size]
        
        # vLLM Generate
        outputs = llm.generate(chunk_prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
        
        chunk_logs = []
        for j, output in enumerate(outputs):
            item = chunk_items[j]
            raw_response = output.outputs[0].text
            
            # 提取
            gold_val = extract_answer_xml(str(item['gold'])) # 有可能 item['gold'] 本身就是纯数字
            if not gold_val: 
                # 尝试从 item['gold'] 中处理 ####
                match = re.search(r'####\s*(-?[\d,]+(\.\d+)?)', str(item['gold']))
                gold_val = match.group(1).replace(',', '') if match else str(item['gold']).strip()

            pred_val = extract_answer_xml(raw_response)
            
            # 判定
            is_acc = is_correct(pred_val, gold_val)
            if is_acc: correct_count += 1
            total_processed += 1
            
            chunk_logs.append({
                "id": item['id'],
                "question": item['question'],
                "gold": item['gold'],
                "pred_full": raw_response,
                "pred_val": pred_val,
                "correct": is_acc
            })
            
        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            for log in chunk_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

    # 8. 结果汇总
    final_acc = correct_count / total_processed if total_processed > 0 else 0
    summary = f"""
    Evaluation Report
    =================
    Model: {args.base_model_path}
    Adapter: {args.adapter_path}
    Dataset: {args.data_path}
    System Prompt: {SYSTEM_PROMPT[:50]}...
    
    Accuracy: {final_acc:.4f} ({correct_count}/{total_processed})
    """
    print(summary)
    with open(result_file, 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main()