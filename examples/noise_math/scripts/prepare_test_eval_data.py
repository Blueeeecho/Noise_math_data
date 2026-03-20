import os
import json
import pandas as pd

def main():
    test_data_dir = "/export/home/asifali/Noise_math_data/examples/noise_math/dataset/test_data"
    output_path = "/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed/test_eval.parquet"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    
    if not os.path.exists(test_data_dir):
        print(f"Error: Directory {test_data_dir} does not exist.")
        return
        
    for filename in os.listdir(test_data_dir):
        if not filename.endswith(".jsonl"):
            continue
            
        dataset_name = os.path.splitext(filename)[0]
        file_path = os.path.join(test_data_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    # Extract question
                    q = item.get('question') or item.get('input') or item.get('problem')
                    
                    # Extract gold answer exactly as in eval_model.py
                    a = item.get('answer') or item.get('output') or item.get('solution') or item.get('gold')
                    
                    if a and isinstance(a, str) and '####' in a:
                        a = a.split('####')[-1].strip()
                        
                    if not a or str(a).strip().lower() == "none":
                        continue
                        
                    # Format prompt using the system prompt required for evaluation
                    prompt = [
                        {"role": "system", "content": "You are a backward reasoning expert. Respond with EXACTLY: [Goal Analysis]... [Backward Execution]... [Final Answer]..."},
                        {"role": "user", "content": q}
                    ]
                    
                    entry = {
                        "data_source": "math_noise",
                        "prompt": prompt,
                        "ability": "math",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": {
                                "gold_answer": a
                            }
                        },
                        "extra_info": {
                            "dataset": dataset_name,
                            "split": "test",
                            "index": len(all_data)
                        }
                    }
                    all_data.append(entry)
                except Exception as e:
                    print(f"Skipping line in {filename} due to error: {e}")
                    continue

    if not all_data:
        print("No valid test data found.")
        return
        
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path)
    print(f"Successfully converted {len(df)} test samples from {test_data_dir} into {output_path}")

if __name__ == "__main__":
    main()
