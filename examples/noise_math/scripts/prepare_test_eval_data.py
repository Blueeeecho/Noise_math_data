import os
import json
import argparse
import pandas as pd

PROMPT_PLACEHOLDER = "__QUESTION__"

SYSTEM_PROMPT_CASE_1 = """You are a backward reasoning expert.
Respond using this exact structure:
[Goal Analysis]
Target: Var{...}
Plan: ...

[Backward Execution]
1. Define/Derive/Calculate Var{...}
   [Reasoning]: ...
   [Source]: ...
   [Calc]: Var{...} = <<...>>

[Final Answer]
..."""

SYSTEM_PROMPT_CASE_2 = """You are a backward reasoning expert.

Solve each math word problem by generating a structured backward reasoning solution.

Requirements:
1. Your output must follow exactly this format:
   [Goal Analysis]
   Target: Var{...}
   Plan: ...

   [Backward Execution]
   1. Define/Derive/Calculate Var{...}:
      [Reasoning]: ...
      [Source]: ...
      [Calc]: Var{...} = <<...>>

   [Final Answer]
   ...

2. In [Goal Analysis], identify the target variable and briefly describe the plan.

3. In [Backward Execution]:
   - Use numbered steps.
   - Each step must define, derive, or calculate exactly one variable in the form Var{...}.
   - Step titles may use:
     - Define Var{...}:
     - Derive Var{...} (Sub-goal):
     - Calculate Var{...} (Goal):
   - Each step must contain [Reasoning], [Source], and [Calc].

4. In [Calc], <<...>> may contain either:
   - a directly extracted value, such as <<48>>
   - or an arithmetic computation, such as <<48 / 2 = 24>>

5. The final step in [Backward Execution] must calculate the target variable.

6. In [Final Answer], output only the final number.

Do not output anything outside this structure."""

ONE_SHOT_PROMPT_CASE_2 = """Example

Question:
Emma is packing fruit for a picnic. She has 4 bags. In each bag, she puts 3 apples and 2 oranges. If she packs everything, how many pieces of fruit does she pack in total?

[Goal Analysis]
Target: Var{Total_Fruit}
Plan: I need to calculate how many pieces of fruit are in each bag, then multiply by the total number of bags.

[Backward Execution]
1. Define Var{Fruit_Per_Bag}:
   [Reasoning]: First, I calculate how many pieces of fruit Emma puts in one bag. Each bag has 3 apples and 2 oranges, so I add them together.
   [Source]: "In each bag, she puts 3 apples and 2 oranges"
   [Calc]: Var{Fruit_Per_Bag} = <<3 + 2 = 5>>

2. Calculate Var{Total_Fruit} (Goal):
   [Reasoning]: Next, I multiply the number of fruit pieces in each bag by the total number of bags to get the total fruit Emma packs.
   [Source]: "She has 4 bags" and Var{Fruit_Per_Bag}
   [Calc]: Var{Total_Fruit} = <<4 * 5 = 20>>

[Final Answer]
20

Now solve the following problem in the same format.

Question:
__QUESTION__
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", default="/export/home/asifali/Noise_math_data/examples/noise_math/dataset/test_data")
    parser.add_argument("--output_path", default="/export/home/asifali/Noise_math_data/examples/noise_math/dataset/Processed/test_eval.parquet")
    parser.add_argument("--prompt_version", default="case_1", choices=["case_1", "case_2"])
    return parser.parse_args()


def build_prompt(question, prompt_version):
    question = (question or "").strip()
    if prompt_version == "case_2":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_CASE_2},
            {"role": "user", "content": ONE_SHOT_PROMPT_CASE_2.replace(PROMPT_PLACEHOLDER, question)},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_CASE_1},
        {"role": "user", "content": question or ""},
    ]

def main():
    args = parse_args()
    test_data_dir = args.test_data_dir
    output_path = args.output_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    skipped_missing_question = 0
    skipped_missing_answer = 0
    skipped_parse_error = 0
    
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_dir}")
        
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
                        
                    if not q or not str(q).strip():
                        skipped_missing_question += 1
                        continue

                    if not a or str(a).strip().lower() == "none":
                        skipped_missing_answer += 1
                        continue
                        
                    # Format prompt using the system prompt required for evaluation
                    prompt = build_prompt(q, args.prompt_version)
                    
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
                            "index": len(all_data),
                            "prompt_version": args.prompt_version,
                        }
                    }
                    all_data.append(entry)
                except Exception as e:
                    skipped_parse_error += 1
                    print(f"Skipping line in {filename} due to error: {e}")
                    continue

    if not all_data:
        raise RuntimeError(
            "No valid test data found. "
            f"missing_question={skipped_missing_question}, "
            f"missing_answer={skipped_missing_answer}, "
            f"parse_error={skipped_parse_error}"
        )
        
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path)
    print(
        f"Successfully converted {len(df)} test samples from {test_data_dir} into {output_path}. "
        f"Skipped missing_question={skipped_missing_question}, "
        f"missing_answer={skipped_missing_answer}, "
        f"parse_error={skipped_parse_error}"
    )

if __name__ == "__main__":
    main()
