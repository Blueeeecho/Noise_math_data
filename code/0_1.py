from __future__ import annotations
import os
import re
import json
import time
import math
import argparse
import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from fractions import Fraction
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

# --- Progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# --- Local LLM (vLLM)
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except Exception:
    LLM = None
    SamplingParams = None
    AutoTokenizer = None


# =============================================================================
# 1) Utils
# =============================================================================

def safe_text(x: Any) -> str:
    return "" if x is None else str(x)

def strip_final_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    parts = answer.split("####")
    return parts[0].strip()

def extract_gold_final(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    if "####" not in answer:
        return ""
    tail = answer.split("####")[-1].strip()
    tail = tail.splitlines()[0].strip()
    return tail

def normalize_score_map(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    mx = max(d.values())
    if mx <= 0:
        return {k: 0.0 for k in d}
    return {k: (v / mx) for k, v in d.items()}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / float(uni) if uni > 0 else 0.0

def weighted_jaccard(a: Set[str], b: Set[str], w: Dict[str, float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    uni = a | b
    num = sum(w.get(t, 1.0) for t in inter)
    den = sum(w.get(t, 1.0) for t in uni)
    return (num / den) if den > 0 else 0.0

def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("LLM output is not a string.")
    s = text.strip()
    if not s:
        raise ValueError("Empty LLM output.")
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    start = s.find("{")
    if start < 0:
        raise ValueError("No '{' found in output; cannot locate JSON object.")
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                return json.loads(candidate)
    raise ValueError("Unbalanced braces; failed to extract a complete JSON object.")


# =============================================================================
# 2) Loaders
# =============================================================================

def load_problem_nodes(problem_nodes_csv: str) -> pd.DataFrame:
    df = pd.read_csv(problem_nodes_csv)
    if "problem_id" not in df.columns or "question" not in df.columns:
        raise ValueError("problem_nodes.csv must contain columns: problem_id, question")
    df = df.sort_values("problem_id").reset_index(drop=True)
    if "answer" not in df.columns:
        df["answer"] = ""
    if "final_answer" not in df.columns:
        df["final_answer"] = ""
    return df

def load_problem_graph(problem_edges_csv: str,
                       min_edge_sim: float = 0.0) -> Dict[int, List[Tuple[int, float]]]:
    e = pd.read_csv(problem_edges_csv)
    req = {"src_problem_id", "dst_problem_id", "similarity"}
    if not req.issubset(set(e.columns)):
        raise ValueError(f"problem_edges.csv must contain columns: {sorted(list(req))}")

    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for _, r in e.iterrows():
        u = int(r["src_problem_id"])
        v = int(r["dst_problem_id"])
        w = float(r["similarity"])
        if w < min_edge_sim:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
    return dict(adj)

def load_op_vocab(operation_nodes_csv: str) -> Tuple[Dict[int, str], Dict[str, int], List[str]]:
    op_df = pd.read_csv(operation_nodes_csv)
    if "operation_id" not in op_df.columns or "operation_tag" not in op_df.columns:
        raise ValueError("operation_nodes.csv must contain columns: operation_id, operation_tag")
    opid2tag: Dict[int, str] = {}
    tag2opid: Dict[str, int] = {}
    tags: List[str] = []
    for _, r in op_df.iterrows():
        oid = int(r["operation_id"])
        tag = str(r["operation_tag"]).strip().upper()
        opid2tag[oid] = tag
        tag2opid[tag] = oid
        tags.append(tag)
    tags = sorted(set(tags))
    return opid2tag, tag2opid, tags

def load_problem_ops(problem_operation_membership_csv: str,
                     opid2tag: Dict[int, str]) -> Tuple[Dict[int, Set[str]], Dict[str, List[int]]]:
    m = pd.read_csv(problem_operation_membership_csv)
    req = {"problem_id", "operation_id"}
    if not req.issubset(set(m.columns)):
        raise ValueError("problem_operation_membership.csv must contain columns: problem_id, operation_id")

    pid2ops: Dict[int, Set[str]] = defaultdict(set)
    op2pids: Dict[str, List[int]] = defaultdict(list)

    for _, r in m.iterrows():
        pid = int(r["problem_id"])
        oid = int(r["operation_id"])
        tag = opid2tag.get(oid, None)
        if not tag:
            continue
        pid2ops[pid].add(tag)

    for pid, ops in pid2ops.items():
        for tag in ops:
            op2pids[tag].append(pid)

    for tag in list(op2pids.keys()):
        op2pids[tag] = sorted(set(op2pids[tag]))

    return dict(pid2ops), dict(op2pids)

def load_op_idf(op_df_stats_csv: Optional[str]) -> Dict[str, float]:
    if not op_df_stats_csv:
        return {}
    if not os.path.exists(op_df_stats_csv):
        return {}
    df = pd.read_csv(op_df_stats_csv)
    if "tag" not in df.columns or "df" not in df.columns:
        return {}
    w: Dict[str, float] = {}
    for _, r in df.iterrows():
        tag = str(r["tag"]).strip().upper()
        frac = float(r["df"])
        frac = max(0.0, min(1.0, frac))
        w[tag] = math.log(2.0 / (1.0 + frac))
    return w

def load_queries_from_jsonl(path: str,
                           start: int = 0,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj:
                continue
            gold_final = extract_gold_final(obj.get("answer", ""))
            if gold_final:
                obj["gold_final"] = gold_final
            out.append(obj)
    return out


# =============================================================================
# 3) vLLM Based Inference
# =============================================================================

class LocalChatLLM:
    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto", max_context: Optional[int] = None):
        self.model = LLM.from_pretrained(model_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_context = max_context
        self.model.eval()

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"

    @torch.no_grad()
    def generate_text(self, system: str, user: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, do_sample: bool = True) -> str:
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Use vLLM for generation with improved performance
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, do_sample=do_sample)
        result = self.model.generate(inputs['input_ids'], sampling_params)

        # Convert the result back to text
        return self.tokenizer.decode(result[0], skip_special_tokens=True)


def main():
    # Initialize model with vLLM for faster inference
    llm = LocalChatLLM(model_path="/path/to/your/model", device="cuda")

    # Sample query processing loop
    query_items = [{"question": "What is 5+5?"}, {"question": "How do you solve a quadratic equation?"}]
    for item in query_items:
        query_text = item['question']
        result = llm.generate_text(system="System prompt", user=query_text)
        print(f"Query: {query_text}\nResult: {result}\n")

if __name__ == "__main__":
    main()