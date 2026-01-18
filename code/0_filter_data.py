#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

HASH_ANS_RE = re.compile(r"####\s*([-+]?(\d+(\.\d+)?|\.\d+))\s*$")
ANS_RE = re.compile(r"<answer>\s*([-+]?(\d+(\.\d+)?|\.\d+))\s*</answer>")

def parse_gt_from_solution(sol: str) -> Optional[float]:
    if not isinstance(sol, str):
        return None
    m = HASH_ANS_RE.search(sol.strip())
    if not m:
        return None
    return float(m.group(1))

def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol

def is_pac_ok(label_obj: Dict[str, Any]) -> bool:
    return bool(isinstance(label_obj, dict) and label_obj.get("pac_ok") is True)

def get_pred_answer(label_obj: Dict[str, Any]) -> Optional[float]:
    if not isinstance(label_obj, dict):
        return None
    if "answer" in label_obj and isinstance(label_obj["answer"], (int, float)):
        return float(label_obj["answer"])
    t = label_obj.get("target", None)
    if isinstance(t, str):
        m = ANS_RE.search(t)
        if m:
            return float(m.group(1))
    return None

def get_pred_target_text(label_obj: Dict[str, Any]) -> Optional[str]:
    if isinstance(label_obj, dict):
        t = label_obj.get("target", None)
        if isinstance(t, str):
            return t
    return None

def iter_items(seed_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []

    # seed
    if "original_question" in seed_record and "original_solution" in seed_record and "original_target" in seed_record:
        items.append({
            "kind": "seed",
            "subtype": "seed",
            "question": seed_record.get("original_question", ""),
            "solution": seed_record.get("original_solution", ""),
            "label": seed_record.get("original_target", {}),
        })

    # no-op
    for v in seed_record.get("no_op_variants_labeled", []) or []:
        items.append({
            "kind": "no_op",
            "subtype": v.get("type", "no_op"),
            "question": v.get("question", ""),
            "solution": v.get("solution", ""),
            "label": v.get("target", {}),
            "raw_obj": v,
        })

    # semantic
    for v in seed_record.get("semantic_variants_labeled", []) or []:
        items.append({
            "kind": "sem",
            "subtype": v.get("type", "sem"),
            "question": v.get("question", ""),
            "solution": v.get("solution", ""),
            "label": v.get("target", {}),
            "raw_obj": v,
        })

    return items

def validate_item(item: Dict[str, Any], seed_gt: Optional[float]) -> Tuple[bool, str]:
    gt = parse_gt_from_solution(item.get("solution", ""))
    if gt is None:
        return False, "missing_gt"

    label = item.get("label", {})
    if not is_pac_ok(label):
        return False, "pac_fail"

    pred = get_pred_answer(label)
    if pred is None:
        return False, "missing_pred"

    if not almost_equal(pred, gt):
        return False, "answer_mismatch"

    if item["kind"] == "no_op" and seed_gt is not None:
        if not almost_equal(pred, seed_gt):
            return False, "noop_not_equal_seed"

    return True, "ok"

def init_stats() -> Dict[str, Any]:
    # nested dicts to hold counts
    return {
        "seed_records": 0,
        "items_total": 0,
        "items_by_kind": defaultdict(int),
        "items_by_kind_pass": defaultdict(int),
        "items_by_kind_fail": defaultdict(int),

        "items_by_subtype": defaultdict(int),
        "items_by_subtype_pass": defaultdict(int),
        "items_by_subtype_fail": defaultdict(int),

        "fail_reasons_total": defaultdict(int),
        "fail_reasons_by_kind": defaultdict(lambda: defaultdict(int)),    # kind -> reason -> cnt
        "fail_reasons_by_subtype": defaultdict(lambda: defaultdict(int))  # subtype -> reason -> cnt
    }

def to_plain(obj: Any) -> Any:
    # convert defaultdict to normal dict recursively for json dump
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/Users/wangwenqing/Desktop/math/output_labeled.jsonl", help="Input jsonl (seed-level labeled)")
    ap.add_argument("--out_seedlevel", type=str, default="/Users/wangwenqing/Desktop/math/filtered_seedlevel.jsonl")
    ap.add_argument("--out_sft", type=str, default="/Users/wangwenqing/Desktop/math/sft_flattened.jsonl")
    ap.add_argument("--out_stats", type=str, default="/Users/wangwenqing/Desktop/math/stats.json")
    ap.add_argument("--drop_failed_variants", action="store_true",
                    help="If set, remove failed variants from seed-level output; otherwise keep with status fields.")
    ap.add_argument("--tol", type=float, default=1e-6, help="Float tolerance for answer match")
    args = ap.parse_args()

    global almost_equal
    def almost_equal(a: float, b: float, tol: float = args.tol) -> bool:  # override with CLI tol
        return abs(a - b) <= tol

    stats = init_stats()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_seedlevel, "w", encoding="utf-8") as fseed, \
         open(args.out_sft, "w", encoding="utf-8") as fsft:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            stats["seed_records"] += 1

            seed_gt = parse_gt_from_solution(rec.get("original_solution", ""))

            items = iter_items(rec)
            stats["items_total"] += len(items)

            no_op_new, sem_new = [], []

            # validate each item
            for it in items:
                kind = it["kind"]
                subtype = it["subtype"]
                stats["items_by_kind"][kind] += 1
                stats["items_by_subtype"][subtype] += 1

                ok, reason = validate_item(it, seed_gt)

                if ok:
                    stats["items_by_kind_pass"][kind] += 1
                    stats["items_by_subtype_pass"][subtype] += 1

                    # write flattened SFT sample
                    target_text = get_pred_target_text(it["label"])
                    if target_text is not None:
                        prompt = f"Question: {it['question']}\nReturn <plan>...</plan> then <answer>...</answer>."
                        fsft.write(json.dumps({
                            "prompt": prompt,
                            "target": target_text,
                            "meta": {
                                "seed_id": rec.get("seed_id"),
                                "kind": kind,
                                "subtype": subtype
                            }
                        }, ensure_ascii=False) + "\n")
                else:
                    stats["items_by_kind_fail"][kind] += 1
                    stats["items_by_subtype_fail"][subtype] += 1
                    stats["fail_reasons_total"][reason] += 1
                    stats["fail_reasons_by_kind"][kind][reason] += 1
                    stats["fail_reasons_by_subtype"][subtype][reason] += 1

                # annotate seed-level output
                if kind == "seed":
                    rec.setdefault("original_target", {})
                    rec["original_target"]["_filter_ok"] = ok
                    rec["original_target"]["_filter_reason"] = reason

                elif kind == "no_op":
                    obj = dict(it.get("raw_obj", {}))
                    obj["_filter_ok"] = ok
                    obj["_filter_reason"] = reason
                    if (not args.drop_failed_variants) or ok:
                        no_op_new.append(obj)

                elif kind == "sem":
                    obj = dict(it.get("raw_obj", {}))
                    obj["_filter_ok"] = ok
                    obj["_filter_reason"] = reason
                    if (not args.drop_failed_variants) or ok:
                        sem_new.append(obj)

            rec["no_op_variants_labeled"] = no_op_new
            rec["semantic_variants_labeled"] = sem_new

            fseed.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # dump stats
    with open(args.out_stats, "w", encoding="utf-8") as fstat:
        json.dump(to_plain(stats), fstat, ensure_ascii=False, indent=2)

    # print summary
    print("==== FILTER SUMMARY ====")
    print(f"seed_records: {stats['seed_records']}")
    print(f"items_total:  {stats['items_total']}")
    print("items_by_kind:")
    for k in ["seed", "no_op", "sem"]:
        print(f"  {k}: total={stats['items_by_kind'].get(k,0)} "
              f"pass={stats['items_by_kind_pass'].get(k,0)} "
              f"fail={stats['items_by_kind_fail'].get(k,0)}")
    print("top_fail_reasons:")
    # show up to 10
    fr = dict(stats["fail_reasons_total"])
    for r, c in sorted(fr.items(), key=lambda x: -x[1])[:10]:
        print(f"  {r}: {c}")

    print(f"\nSeed-level output: {args.out_seedlevel}")
    print(f"SFT flattened output: {args.out_sft}")
    print(f"Stats json: {args.out_stats}")

if __name__ == "__main__":
    main()