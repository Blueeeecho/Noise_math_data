import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("Error: vLLM is not installed.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, help="Base model path for LoRA adapter")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    return parser.parse_args()

def extract_answer(text):
    if not text: return None
    text_str = str(text).strip()

    # 1. [Final Answer] pattern (Highest Priority)
    match_final_tag = re.search(r'\[Final Answer\]\s*(.*)', text_str, re.DOTALL | re.IGNORECASE)
    if match_final_tag:
        content = match_final_tag.group(1)
        nums = re.findall(r'(-?\$?[\d,]+(?:\.\d+)?)', content)
        if nums:
             return nums[-1].replace('$', '').replace(',', '').strip('.')

    # 2. \boxed{} pattern
    match_box = re.search(r'\\boxed\{([^}]+)\}', text_str)
    if match_box:
        return match_box.group(1).strip().replace(',', '')

    # 3. XML <answer>... (Standard)
    match_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text_str, re.DOTALL | re.IGNORECASE)
    if match_xml:
        return match_xml.group(1).strip().replace(',', '')

    # 4. XML <answer>... (Truncated)
    match_truncated = re.search(r"<answer>\s*(-?[\d,]+(\.\d+)?)", text_str, re.DOTALL | re.IGNORECASE)
    if match_truncated:
        return match_truncated.group(1).replace(',', '')

    # 5. #### Marker
    match_hash = re.search(r'####\s*(-?[\d,]+(\.\d+)?)', text_str)
    if match_hash:
        return match_hash.group(1).replace(',', '')

    # 6. "The answer is X" pattern
    match_phrase = re.findall(r'(?:answer|result|value|total|count) is\s*:?\s*(-?\$?[\d,]+(\.\d+)?)', text_str, re.IGNORECASE)
    if match_phrase:
        return match_phrase[-1][0].replace('$', '').replace(',', '')

    # 7. Extract number after "=" sign
    match_eq = re.findall(r'=\s*(-?\$?[\d,]+(\.\d+)?)', text_str)
    if match_eq:
        return match_eq[-1][0].replace('$', '').replace(',', '').strip('.')

    # 8. Last Resort: Extract the very last number
    all_nums = re.findall(r'(-?\$?[\d,]+(?:\.\d+)?)', text_str)
    if all_nums:
        return all_nums[-1].replace('$', '').replace(',', '').strip('.')

    # 9. Ultimate Fallback: Extract the last standalone number found anywhere in the text
    ultimate_nums = re.findall(r'(-?\d+(?:\.\d+)?)', text_str)
    if ultimate_nums:
        return ultimate_nums[-1].strip('.')

    return None

def is_correct(pred, gold):
    if not pred or not gold: return False
    
    # Handle "None" or empty gold from dataset
    if str(gold).strip().lower() == "none":
        return False

    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except:
        # Normalize strings for comparison
        def clean(s): return re.sub(r'[^0-9a-zA-Z\.\-]', '', str(s))
        return clean(pred) == clean(gold)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                q = item.get('question') or item.get('input') or item.get('problem')
                
                # Try to get answer from various fields
                a = item.get('answer') or item.get('output') or item.get('solution') or item.get('gold')
                
                # If answer contains ####, split it
                if a and isinstance(a, str) and '####' in a:
                    a = a.split('####')[-1].strip()
                
                # Filter out invalid answers
                if a and str(a).strip().lower() != "none":
                     data.append({"question": q, "gold": a})
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue

    # Init vLLM
    print(f"Loading model from {args.model_path}")
    
    # Check if LoRA
    enable_lora = False
    model_load_path = args.model_path
    if args.base_model_path:
        print(f"Loading LoRA adapter from {args.model_path} on top of {args.base_model_path}")
        model_load_path = args.base_model_path
        enable_lora = True

    llm = LLM(
        model=model_load_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=enable_lora,
        max_lora_rank=64
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("adapter", 1, args.model_path)
    
    # Prompts
    prompts = []
    # We assume standard prompt format
    # Use tokenizer to apply template if possible, or manual
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
    
    for item in data:
        # Default system prompt for backward reasoning
        messages = [
            {"role": "system", "content": "You are a backward reasoning expert. Respond with EXACTLY: [Goal Analysis]... [Backward Execution]... [Final Answer]..."},
            {"role": "user", "content": item['question']}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    print(f"Generating for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    correct = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold = data[i]['gold']
        pred = extract_answer(generated_text)
        acc = is_correct(pred, gold)
        if acc: correct += 1
        
        results.append({
            "question": data[i]['question'],
            "gold": gold,
            "pred": pred,
            "generated": generated_text,
            "correct": acc
        })

    accuracy = correct / len(data)
    print(f"Accuracy: {accuracy:.4f}")
    
    with open(os.path.join(args.output_dir, "results.jsonl"), 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump({"accuracy": accuracy, "total": len(data)}, f)

if __name__ == "__main__":
    main()
