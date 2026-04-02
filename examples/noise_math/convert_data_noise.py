import argparse
import json
import pandas as pd

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
{question}
"""


def get_prompt_messages(question, prompt_version):
    question = question or ""
    if prompt_version == "case_2":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_CASE_2},
            {"role": "user", "content": ONE_SHOT_PROMPT_CASE_2.format(question=question)},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_CASE_1},
        {"role": "user", "content": question},
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file (SFT format)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--prompt_version", default="case_1", choices=["case_1", "case_2"], help="Prompt template version")
    args = parser.parse_args()

    data = []
    with open(args.input, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            messages = item.get("messages", [])
            question = item.get("question")
            if not question and messages:
                for msg in messages:
                    if msg.get("role") == "user":
                        question = msg.get("content")
                        break

            ground_truth = item.get("backward_reasoning")
            if not ground_truth and messages:
                ground_truth = messages[-1]["content"]
            if not question or not ground_truth:
                continue

            prompt = get_prompt_messages(question, args.prompt_version)
            
            entry = {
                "data_source": "math_noise",
                "prompt": prompt,
                "ability": "math",
                "response": ground_truth,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "gold_answer": ground_truth,
                        "gold_chain": ground_truth
                    }
                },
                "extra_info": {
                    "split": "train" if "train" in args.output else "test",
                    "index": len(data),
                    "prompt_version": args.prompt_version,
                }
            }
            data.append(entry)

    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet(args.output)
    print(f"Converted {len(df)} records to {args.output}")

if __name__ == "__main__":
    main()
