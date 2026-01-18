#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import random
import logging
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("smart_split")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_list(path: str, items: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


def family_signature(samples: List[dict]) -> Tuple[Counter, Counter]:
    kinds = Counter()
    subtypes = Counter()
    for x in samples:
        meta = x.get("meta", {}) or {}
        kinds[meta.get("kind", "unknown")] += 1
        subtypes[meta.get("subtype", "unknown")] += 1
    return kinds, subtypes


def is_complete_family(kinds: Counter) -> bool:
    # 评测最推荐：seed + >=1 noop + >=1 sem
    return (kinds.get("seed", 0) >= 1) and (kinds.get("no_op", 0) >= 1) and (kinds.get("sem", 0) >= 1)


def is_strict_family(kinds: Counter, subtypes: Counter) -> bool:
    # 更严格：要求5种都齐（seed + 4 subtype）
    required_subtypes = {"seed", "no_op_rephrase", "no_op_distractor", "semantic_numerical", "semantic_logic"}
    present = {st for st, c in subtypes.items() if c > 0}
    return kinds.get("seed", 0) >= 1 and required_subtypes.issubset(present)


def take_from_pool(pool: List[str], k: int, taken: set) -> List[str]:
    out = []
    for sid in pool:
        if sid in taken:
            continue
        out.append(sid)
        taken.add(sid)
        if len(out) >= k:
            break
    return out


def collect(fam2samples: Dict[str, List[dict]], fam_set: set) -> List[dict]:
    out = []
    for sid in fam_set:
        out.extend(fam2samples[sid])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_sft", type=str, default="/Users/wangwenqing/Desktop/math/sft_flattened.jsonl", help="Flattened jsonl, e.g. sft_flattened.jsonl")
    ap.add_argument("--out_train", type=str, default="train.jsonl")
    ap.add_argument("--out_dev", type=str, default="dev.jsonl")
    ap.add_argument("--out_test", type=str, default="test.jsonl")
    ap.add_argument("--out_report", type=str, default="split_report.json")
    ap.add_argument("--out_dir_lists", type=str, default=".", help="Where to write *_families.txt files")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_ratio", type=float, default=0.05, help="Target ratio by families")
    ap.add_argument("--test_ratio", type=float, default=0.05, help="Target ratio by families")

    ap.add_argument("--prefer_strict", action="store_true",
                    help="dev/test 优先从 strict families (seed+4 subtypes) 抽取，不够再退回 complete")
    ap.add_argument("--min_eval_families", type=int, default=100,
                    help="dev/test 每个 split 至少这么多 families（防止太小）")
    ap.add_argument("--allow_relax", action="store_true",
                    help="如果 complete 不够，允许 seed+noop 或 seed+sem 这种 near-complete family 进入 eval")

    ap.add_argument("--log_prefix", type=str, default="split_report", help="log file prefix")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.log_prefix}.{ts}.log"
    logger = setup_logger(log_path)

    rng = random.Random(args.seed)

    # 1) group by seed_id
    fam2samples = defaultdict(list)
    for ex in read_jsonl(args.input_sft):
        meta = ex.get("meta", {}) or {}
        sid = meta.get("seed_id", None)
        if sid is None:
            sid = "__MISSING_SEED_ID__"
        fam2samples[str(sid)].append(ex)

    families = list(fam2samples.keys())
    n_families = len(families)
    n_samples = sum(len(v) for v in fam2samples.values())

    # 2) bucket families
    strict_fams, complete_fams = [], []
    near_seed_noop, near_seed_sem, others = [], [], []
    fam_sizes = Counter()

    for sid in families:
        samples = fam2samples[sid]
        fam_sizes[len(samples)] += 1
        kinds, subtypes = family_signature(samples)

        strict = is_strict_family(kinds, subtypes)
        complete = is_complete_family(kinds)

        if strict:
            strict_fams.append(sid)
        if complete:
            complete_fams.append(sid)
        else:
            if (kinds.get("seed", 0) >= 1) and (kinds.get("no_op", 0) >= 1):
                near_seed_noop.append(sid)
            elif (kinds.get("seed", 0) >= 1) and (kinds.get("sem", 0) >= 1):
                near_seed_sem.append(sid)
            else:
                others.append(sid)

    logger.info(f"Input samples={n_samples}, families={n_families}")
    logger.info(f"Family size distribution: {dict(sorted(fam_sizes.items()))}")
    logger.info(f"Buckets: strict={len(strict_fams)}, complete={len(complete_fams)}, "
                f"near_seed_noop={len(near_seed_noop)}, near_seed_sem={len(near_seed_sem)}, others={len(others)}")

    # 3) decide eval family counts
    target_dev = max(args.min_eval_families, int(round(n_families * args.dev_ratio)))
    target_test = max(args.min_eval_families, int(round(n_families * args.test_ratio)))
    logger.info(f"Target eval families: dev={target_dev}, test={target_test} (prefer_strict={args.prefer_strict})")

    # 4) build eval pools
    pool_primary = strict_fams if args.prefer_strict else complete_fams
    pool_fallback = complete_fams  # used when strict insufficient

    pool_primary = list(pool_primary)
    pool_fallback = list(pool_fallback)
    rng.shuffle(pool_primary)
    rng.shuffle(pool_fallback)

    taken = set()
    dev_fams = take_from_pool(pool_primary, target_dev, taken)
    test_fams = take_from_pool(pool_primary, target_test, taken)

    # fallback if strict not enough
    if len(dev_fams) < target_dev:
        dev_fams += take_from_pool(pool_fallback, target_dev - len(dev_fams), taken)
    if len(test_fams) < target_test:
        test_fams += take_from_pool(pool_fallback, target_test - len(test_fams), taken)

    relaxed_used = []
    if args.allow_relax:
        relax_pool = list(near_seed_noop) + list(near_seed_sem)
        rng.shuffle(relax_pool)
        if len(dev_fams) < target_dev:
            add = take_from_pool(relax_pool, target_dev - len(dev_fams), taken)
            dev_fams += add
            relaxed_used += add
        if len(test_fams) < target_test:
            add = take_from_pool(relax_pool, target_test - len(test_fams), taken)
            test_fams += add
            relaxed_used += add

    dev_set = set(dev_fams)
    test_set = set(test_fams)
    train_set = set(families) - dev_set - test_set

    assert train_set.isdisjoint(dev_set)
    assert train_set.isdisjoint(test_set)
    assert dev_set.isdisjoint(test_set)

    # 5) write splits (max utilization: train gets ALL remaining families)
    train_items = collect(fam2samples, train_set)
    dev_items = collect(fam2samples, dev_set)
    test_items = collect(fam2samples, test_set)

    write_jsonl(args.out_train, train_items)
    write_jsonl(args.out_dev, dev_items)
    write_jsonl(args.out_test, test_items)

    # 6) write family id lists for traceability
    train_list = sorted(list(train_set))
    dev_list = sorted(list(dev_set))
    test_list = sorted(list(test_set))
    write_list(f"{args.out_dir_lists}/train_families.txt", train_list)
    write_list(f"{args.out_dir_lists}/dev_families.txt", dev_list)
    write_list(f"{args.out_dir_lists}/test_families.txt", test_list)

    # 7) report json
    report = {
        "timestamp": ts,
        "input": {
            "path": args.input_sft,
            "families": n_families,
            "samples": n_samples,
            "family_size_distribution": dict(sorted(fam_sizes.items()))
        },
        "family_buckets": {
            "strict_families": len(strict_fams),
            "complete_families": len(complete_fams),
            "near_seed_noop": len(near_seed_noop),
            "near_seed_sem": len(near_seed_sem),
            "others": len(others)
        },
        "split_policy": {
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
            "min_eval_families": args.min_eval_families,
            "prefer_strict": args.prefer_strict,
            "allow_relax": args.allow_relax
        },
        "split_result": {
            "train_families": len(train_set),
            "dev_families": len(dev_set),
            "test_families": len(test_set),
            "train_samples": len(train_items),
            "dev_samples": len(dev_items),
            "test_samples": len(test_items),
            "relaxed_used_families": len(set(relaxed_used))
        },
        "outputs": {
            "train": args.out_train,
            "dev": args.out_dev,
            "test": args.out_test,
            "report": args.out_report,
            "log": log_path,
            "train_families": f"{args.out_dir_lists}/train_families.txt",
            "dev_families": f"{args.out_dir_lists}/dev_families.txt",
            "test_families": f"{args.out_dir_lists}/test_families.txt"
        }
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("==== SMART SPLIT DONE ====")
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Outputs written. Report={args.out_report}, Log={log_path}")


if __name__ == "__main__":
    main()