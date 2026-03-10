import json
import os
import sys

def main():
    input_path = "/export/home/asifali/Noise_math_data/data/all_backward_data.jsonl"
    
    if not os.path.exists(input_path):
        # Fallback to local if not found
        input_path = "/export/home/asifali/Noise_math_data/data/all_backward_data.jsonl"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    output_dir = "/export/home/asifali/Noise_math_data/output_backward"
    os.makedirs(output_dir, exist_ok=True)


    data = []
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100: # limit to 100 samples as requested
                break
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON at line {i+1}")
                
    print(f"Loaded {len(data)} samples (max 100).")
    
    # Split 90/10
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(test_data)} test samples to {test_path}")

if __name__ == "__main__":
    main()
