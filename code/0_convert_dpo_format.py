import json
import argparse
import os
from tqdm import tqdm

def convert_to_trl_format(input_path, output_path):
    print(f"Processing {input_path} ...")
    
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 1. 读取原始数据
                record = json.loads(line)
                
                # 原始字段提取 (假设格式为你提供的 list 结构)
                # prompt: [{"role": "system", ...}, {"role": "user", ...}]
                # chosen: [{"role": "assistant", ...}]
                # rejected: [{"role": "assistant", ...}]
                prompt_msgs = record.get('prompt', [])
                chosen_msgs = record.get('chosen', [])
                rejected_msgs = record.get('rejected', [])
                
                # 2. 拼接为全对话历史 (Full History)
                # TRL 要求 chosen/rejected 必须是完整的 conversation list
                # List 相加: [A, B] + [C] = [A, B, C]
                full_chosen = prompt_msgs + chosen_msgs
                full_rejected = prompt_msgs + rejected_msgs
                
                # 3. 构造新对象
                # 这种格式下，DPOTrainer 会自动把最后一条 Assistant 的回复作为 Label，
                # 并自动 Mask 掉前面的 User/System Prompt。
                new_entry = {
                    "chosen": full_chosen,
                    "rejected": full_rejected
                }
                
                # 4. 写入文件
                fout.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Conversion complete. Saved {count} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/root/autodl-tmp/A-new/data/dpo/1.5B_instruct/hard_train_from_messages_vllm_all.jsonl", help="原始 DPO 数据路径")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/A-new/data/dpo/1.5B_instruct/hard_train_from_messages_vllm_all_dpo_format.jsonl", help="转换后的标准格式路径")
    args = parser.parse_args()
    
    convert_to_trl_format(args.input, args.output)