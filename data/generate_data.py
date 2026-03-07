#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Data Generation Script for Math-Noise Project.
Supports two modes:
1. Backward Reasoning (--mode backward): Generates [Goal Analysis] -> [Backward Execution] -> [Final Answer]
2. Program Reasoning (--mode program): Generates <reason> -> <plan> (Python) -> <answer>

Usage:
  python3 generate_data.py --mode backward --input in.jsonl --output out_backward.jsonl
  python3 generate_data.py --mode program --input in.jsonl --output out_program.jsonl
"""

import os
import json
import time
import argparse
import re
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# DeepSeek(OpenAI-SDK-compatible)
from openai import OpenAI

# Ollama
try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def extract_gsm8k_final(answer_text: str) -> Optional[str]:
    """
    GSM8K answer usually ends with: '#### 72'
    Return the string after #### (trimmed).
    """
    if not isinstance(answer_text, str):
        return None
    m = re.search(r"####\s*([^\n\r]+)\s*$", answer_text.strip())
    if m:
        return m.group(1).strip()
    # fallback: last number token
    m2 = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text)
    return m2[-1] if m2 else None

def robust_json_extract(text: str) -> Dict[str, Any]:
    """
    Minimal JSON extraction for Program mode which returns JSON-wrapped XML.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Model output is not valid JSON.")


# =============================================================================
# MODE 1: BACKWARD REASONING CONSTANTS & LOGIC
# =============================================================================

BACKWARD_SYSTEM_PROMPT = r"""You are a Backward Reasoning Expert.
Your task is to solve math problems by breaking them down into a **Reasoning-Augmented Backward Chain**.

**Format Guidelines:**

1. **[Goal Analysis]**:
   - Explicitly state the `Target` variable.
   - Briefly outline the high-level plan (e.g., "To find A, I need to calculate B and C first").

2. **[Backward Execution]**:
   - Work BACKWARDS from the goal.
   - Use indentation (1, 1.1, 1.2) to show Sub-goals.
   - For EACH variable definition, you must provide:
     - `X. Define Var{Name}:` (Use meaningful names like Var{Cost}, Var{Area})
     - `   [Reasoning]:` 1-2 sentences explaining the logic. **Crucial:** Explain WHY you perform specific operations (e.g., "Subtracting the spilled amount to get actual volume").
     - `   [Source]:` Quote the exact numbers/text from the input.
     - `   [Calc]:` `Var{Name} = <<expression=result>>`

3. **[Final Answer]**:
   - Output only the final number.

**Strict Rules:**
- **Reasoning First**: The [Reasoning] block acts as a "thought scratchpad" to verify logic before calculation.
- **Variable Binding**: All calculations MUST be assigned to a `Var{Name}`.
- **Supervision Format**: All math MUST use `<<expression=result>>`.

Output ONLY the solution in the above format. No wrapper JSON is needed.
"""

BACKWARD_FEWSHOT_TEMPLATE = r"""**Example:**

**Input Problem:**
I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?

**Output:**

[Goal Analysis]
Target: Var{Total_Water}
Plan: I need to calculate the water content from the Pineapple drink and the Orange drink separately, then sum them up.

[Backward Execution]
1. Define Var{Water_Pineapple}:
   [Reasoning]: First, I calculate the water in the pineapple drink. The problem explicitly gives the volume and water ratio, so I can multiply them directly.
   [Source]: "15 liters" and "three-fifths water"
   [Calc]: Var{Water_Pineapple} = <<15 * 0.6 = 9>>

2. Derive Var{Water_Orange} (Sub-goal):
   [Reasoning]: I need the water content from the orange drink. However, the volume changed because some was spilled. I cannot use the initial 10L directly; I must calculate the actual added volume first.

   2.1 Define Var{Vol_Orange_Added}:
       [Reasoning]: The initial volume was 10L, but 1L was spilled. I need to subtract the spilled amount to determine what actually went into the mixture.
       [Source]: "10 liters" minus "spill one liter"
       [Calc]: Var{Vol_Orange_Added} = <<10 - 1 = 9>>

   2.2 Define Var{Ratio_Orange}:
       [Reasoning]: The water ratio is a property of the liquid. Even though volume was spilled, the concentration (ratio) remains the same.
       [Source]: "two-thirds water"
       [Calc]: Var{Ratio_Orange} = <<2/3>>

   2.3 Calculate Var{Water_Orange}:
       [Reasoning]: Now I can find the water content of the orange drink by multiplying the actual added volume by the water ratio.
       [Calc]: Var{Water_Orange} = <<9 * (2/3) = 6>>

3. Calculate Var{Total_Water} (Goal):
   [Reasoning]: Finally, I verify the total volume (9L orange + 15L pineapple = 24L) matches the problem description, and then sum the water components.
   [Source]: Sum of Var{Water_Pineapple} and Var{Water_Orange}
   [Calc]: Var{Total_Water} = <<9 + 6 = 15>>

[Final Answer]
15

---

**Task:**
**Input Problem:**
"""

def extract_backward_answer(text: str) -> Optional[str]:
    # Match: [Final Answer] \n 15
    m = re.search(r"\[Final Answer\]\s*([\d\.]+)", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # Fallback
    m_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    if m_xml:
        return m_xml.group(1).strip()
    return None


# =============================================================================
# MODE 2: PROGRAM REASONING CONSTANTS & LOGIC
# =============================================================================

PROGRAM_SYSTEM_PROMPT = r"""You are a dataset transformation engine.

You are given ONE GSM8K-style record with:
- question: a math word problem (plain text)
- answer: a worked solution ending with a gold final answer line: "#### <final>"

Your job:
Generate a new field called "solution" in XML format with EXACTLY:
<reason>...</reason><plan>...</plan>\n<answer>...</answer>

Hard constraints:
1) Output MUST be valid JSON with exactly ONE key:
   {"assistant_content":"<reason>...</reason><plan>...</plan>\n<answer>...</answer>"}
   No other keys. No markdown. No extra text.

2) The XML MUST contain ONLY:
   <reason> ... </reason>
   <plan> ... </plan>
   <answer> ... </answer>.

3) The <answer> content MUST be EXACTLY the same as the gold final answer from the input (the text after "####").
   Do NOT change formatting (e.g., keep "10" vs "10.0" as-is).

4) The <plan> must contain runnable Python code with:
   def solve():
       ...
   No imports. No input(). No reading files. Just compute and return the final answer.

5) Reasoning MUST be written as Python comments (# ...) inside solve().

6) Keep code minimal and correct.
   - Return the final result (should match <answer>).

7) The program result must match the gold final answer.

Return ONLY the JSON object described in (1).
"""

PROGRAM_FEWSHOT_TEMPLATE = r"""Here are examples of the transformation.

EXAMPLE_1_INPUT_RECORD:
{"question":"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?","answer":"Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"}

EXAMPLE_1_OUTPUT_JSON:
{"assistant_content":"<reason>\n1. Identify variables from the problem:\n   - Clips sold in April: 48 (assign to x0)\n   - 'Half as many' means divide by 2 (assign to x1)\n\n2. Logic plan:\n   - Step 1: Compute h0 = x0 / x1 to get clips sold in May.\n   - Step 2: Compute h1 = x0 + h0 to get total clips sold in April and May.\n</reason><plan>\ndef solve():\n   #1. Identify variables:\n    x0 = 48  # number of clips sold in April.\n    x1 = 2   # 'half' means divide by 2.\n\n    #2. Compute clips sold in May:\n    # May sales are half of April sales.\n    h0 = x0 / x1  # clips sold in May.\n\n    #3. Compute total clips sold in April and May:\n    # Add April sales and May sales.\n    h1 = x0 + h0  # total clips sold.\n\n    return h1\n</plan>\n<answer>72</answer>"}

Now transform the following record.

INPUT_RECORD:
"""

def extract_program_plan_and_run(xml_text: str) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(xml_text, str):
        return None, "assistant_content is not a string"

    m = re.search(r"<plan>\s*(.*?)\s*</plan>", xml_text, flags=re.S)
    if not m:
        return None, "Missing <plan> block."

    code = m.group(1)
    # minimal safety
    banned = ["import ", "open(", "subprocess", "socket", "requests", "__import__", "eval(", "exec("]
    if any(b in code.lower() for b in banned):
        return None, "Banned operation detected in generated code."

    g = {}
    try:
        exec(code, g, g)  # noqa: S102
        if "solve" not in g or not callable(g["solve"]):
            return None, "solve() not defined."
        res = g["solve"]()
        if isinstance(res, (int, float)):
            return float(res), None
        return None, f"solve() returned non-numeric type: {type(res)}"
    except Exception as e:
        return None, f"Execution error: {e}"

def extract_program_answer(xml_text: str) -> Optional[str]:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", xml_text, flags=re.S)
    if m:
        return m.group(1).strip()
    return None


# =============================================================================
# MAIN TRANSFORMER CLASS
# =============================================================================

class LLMTransformer:
    def __init__(
        self,
        mode: str,
        backend: str,
        model: str,
        temperature: float,
        max_tokens: int,
        retries: int,
        retry_sleep: float,
        ollama_url: str,
        deepseek_base_url: str,
    ):
        self.mode = mode
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.ollama_url = ollama_url.rstrip("/")
        self.deepseek_base_url = deepseek_base_url

        if backend == "deepseek":
            self.client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-f10c3f461bb94854a7fbd205aae77a68"),
                base_url=self.deepseek_base_url,
            )
        elif backend == "ollama":
            if requests is None:
                raise RuntimeError("Please install requests for ollama backend: pip3 install requests")
        else:
            raise ValueError("backend must be: deepseek | ollama")

    def _build_messages(self, record: Dict[str, Any]) -> list:
        question = record.get("question", "")
        
        if self.mode == "backward":
            user_prompt = BACKWARD_FEWSHOT_TEMPLATE + question
            system_prompt = BACKWARD_SYSTEM_PROMPT
        else: # program
            # For program mode, we include the full record (q+a) to let model see gold answer
            user_prompt = PROGRAM_FEWSHOT_TEMPLATE + json.dumps(record, ensure_ascii=False)
            system_prompt = PROGRAM_SYSTEM_PROMPT

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_llm(self, messages: list) -> str:
        if self.backend == "deepseek":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return response.choices[0].message.content
        else: # ollama
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            url = f"{self.ollama_url}/api/chat"
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

    def transform_one(self, record: Dict[str, Any]) -> Optional[str]:
        gold = extract_gsm8k_final(record.get("answer", ""))
        if gold is None:
            return None

        messages = self._build_messages(record)
        last_err = None

        for attempt in range(self.retries + 1):
            try:
                raw = self._call_llm(messages)
                ac = raw.strip()

                if self.mode == "backward":
                    # --- Backward Validation ---
                    if "[Goal Analysis]" not in ac: raise ValueError("Missing [Goal Analysis]")
                    if "[Backward Execution]" not in ac: raise ValueError("Missing [Backward Execution]")
                    if "[Final Answer]" not in ac: raise ValueError("Missing [Final Answer]")
                    
                    out_ans = extract_backward_answer(ac)
                    if out_ans is None: raise ValueError("Missing final answer value")
                    
                    # Numeric check
                    try:
                        if abs(float(out_ans) - float(gold)) > 1e-6:
                            raise ValueError(f"Mismatch: got {out_ans}, expected {gold}")
                    except:
                        if out_ans.strip() != str(gold).strip():
                            raise ValueError(f"Mismatch: got {out_ans}, expected {gold}")
                    
                    return ac

                else:
                    # --- Program Validation ---
                    obj = robust_json_extract(ac)
                    if "assistant_content" not in obj: raise ValueError("Missing JSON key 'assistant_content'")
                    content = obj["assistant_content"]
                    
                    if "<plan>" not in content or "<answer>" not in content:
                        raise ValueError("Missing <plan> or <answer> tags")
                    
                    out_ans = extract_program_answer(content)
                    if out_ans is None: raise ValueError("Missing <answer> content")
                    
                    if out_ans.strip() != str(gold).strip():
                        raise ValueError(f"Answer mismatch: got {out_ans}, expected {gold}")

                    # Run code
                    res, err = extract_program_plan_and_run(content)
                    if err: raise ValueError(err)
                    try:
                        if abs(res - float(gold)) > 1e-6:
                            raise ValueError(f"Code result {res} != expected {gold}")
                    except:
                         raise ValueError("Gold answer non-numeric")
                    
                    return content

            except Exception as e:
                last_err = str(e)
                if self.retry_sleep > 0:
                    time.sleep(self.retry_sleep)
                # We could add error feedback to prompt here if desired, 
                # but keeping it simple for now as per original split structure.

        print(f"[WARN] transform failed: {last_err}")
        return None


def process_line_task(line: str, transformer: LLMTransformer, keep_on_fail: bool):
    line = line.strip()
    if not line: return ('skip', '')

    try:
        record = json.loads(line)
    except:
        return ('failed', '')

    if "question" not in record or "answer" not in record:
        if keep_on_fail: return ('failed', json.dumps(record, ensure_ascii=False) + "\n")
        return ('failed', '')

    result = transformer.transform_one(record)

    if result is None:
        if keep_on_fail: return ('failed', json.dumps(record, ensure_ascii=False) + "\n")
        return ('failed', '')

    # Field name depends on mode to match original scripts
    if transformer.mode == "backward":
        record["backward_reasoning"] = result
    else:
        record["solution"] = result

    return ('ok', json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backward", "program"], required=True, help="Generation mode")
    ap.add_argument("--backend", choices=["deepseek", "ollama"], default="deepseek", help="Choose LLM backend")
    ap.add_argument("--model", default="deepseek-chat", help="LLM Model Name")
    ap.add_argument("--input", default="data/train.jsonl", help="Input JSONL path")
    ap.add_argument("--output", default="data/generated.jsonl", help="Output JSONL path")
    
    # Params
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_sleep", type=float, default=0.3)
    
    # Backend specifics
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--deepseek_base_url", default="https://api.deepseek.com")
    
    # Misc
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep_on_fail", action="store_true")
    ap.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()

    transformer = LLMTransformer(
        mode=args.mode,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        ollama_url=args.ollama_url,
        deepseek_base_url=args.deepseek_base_url,
    )

    print(f"Mode: {args.mode}")
    print(f"Reading: {args.input}")
    
    try:
        with open(args.input, "r", encoding="utf-8") as fin:
            lines = [line for line in fin if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return

    if args.limit:
        lines = lines[:args.limit]

    total = len(lines)
    print(f"Processing {total} records with {args.workers} workers...")

    # Ensure output dir
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ok = 0
    failed = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_line_task, line, transformer, args.keep_on_fail) for line in lines]
            for future in tqdm(futures, total=total, desc="Processing"):
                try:
                    status, output_str = future.result()
                    if status == 'ok':
                        ok += 1
                        fout.write(output_str)
                    elif status == 'failed':
                        failed += 1
                        if output_str: fout.write(output_str)
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Error: {e}")
                    failed += 1

    print(f"Done. Output: {args.output}")
    print(f"Stats: OK={ok}, Failed={failed}, Skipped={skipped}")

if __name__ == "__main__":
    main()
