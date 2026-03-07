import argparse
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file (SFT format)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    args = parser.parse_args()

    data = []
    with open(args.input, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Extract prompt (all messages except last assistant)
            # and ground truth (last assistant message)
            messages = item.get("messages", [])
            
            # Fallback for dummy data or raw data
            if not messages:
                if "question" in item and "backward_reasoning" in item:
                    messages = [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["backward_reasoning"]}
                    ]
                elif "question" in item and "answer" in item:
                    messages = [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]}
                    ]
            
            if not messages:
                continue
                
            prompt = messages[:-1]
            ground_truth = messages[-1]['content']
            
            # Extract gold answer for reward calculation
            # We store it in 'non_tensor_data' or similar custom column
            # Verl usually expects 'prompt' (list of dict) and 'response' (ground truth, string)
            
            entry = {
                "data_source": "math_noise",
                "prompt": prompt,
                "response": ground_truth, # SFT target
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "gold_answer": ground_truth, # We will parse this in reward_fn
                        "gold_chain": ground_truth
                    }
                }
            }
            data.append(entry)

    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet(args.output)
    print(f"Converted {len(df)} records to {args.output}")

if __name__ == "__main__":
    main()
