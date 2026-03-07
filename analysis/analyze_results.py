import argparse
import json
import os
from collections import defaultdict

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def read_labels(path):
    # If label file exists (GSM-IC format), read it
    # Expects list of dicts with 'new_question' and labels
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item['new_question']: item for item in data}
    except Exception as e:
        print(f"Warning: Could not read label file {path}: {e}")
        return {}

def analyze(results, label_map=None):
    stats = {
        "total": len(results),
        "correct": sum(1 for r in results if r.get("correct")),
        "accuracy": 0.0,
        "by_label": defaultdict(lambda: {"total": 0, "correct": 0})
    }
    stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    if label_map:
        for r in results:
            q = r.get("question")
            is_correct = r.get("correct")
            
            # Find matching label
            # Exact match might fail due to whitespace, try simple normalization
            label_info = label_map.get(q)
            if not label_info:
                # Try stripped
                label_info = label_map.get(q.strip())
            
            if label_info:
                for key in ['role_label', 'number_label', 'sentence_label', 'n_steps']:
                    if key in label_info:
                        val = label_info[key]
                        group_key = f"{key}:{val}"
                        stats["by_label"][group_key]["total"] += 1
                        if is_correct:
                            stats["by_label"][group_key]["correct"] += 1

    return stats

def generate_report(stats, output_path):
    lines = []
    lines.append("# Analysis Report")
    lines.append(f"## Overall")
    lines.append(f"- Total: {stats['total']}")
    lines.append(f"- Correct: {stats['correct']}")
    lines.append(f"- Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
    
    if stats["by_label"]:
        lines.append("\n## Detailed Breakdown")
        
        # Group by label type
        groups = defaultdict(list)
        for k, v in stats["by_label"].items():
            type_name, label_val = k.split(":", 1)
            acc = v["correct"] / v["total"] if v["total"] > 0 else 0
            groups[type_name].append((label_val, v["total"], v["correct"], acc))
            
        for type_name, items in groups.items():
            lines.append(f"\n### {type_name}")
            lines.append("| Label | Total | Correct | Accuracy |")
            lines.append("|---|---|---|---|")
            # Sort by label value
            for item in sorted(items, key=lambda x: x[0]):
                lines.append(f"| {item[0]} | {item[1]} | {item[2]} | {item[3]:.4f} |")

    report = "\n".join(lines)
    print(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True, help="Path to results.jsonl from eval")
    parser.add_argument("--label_file", help="Path to GSM-IC labels json file (optional)")
    parser.add_argument("--output_report", help="Path to save markdown report")
    args = parser.parse_args()

    results = read_jsonl(args.results_file)
    label_map = None
    if args.label_file:
        label_map = read_labels(args.label_file)
        
    stats = analyze(results, label_map)
    generate_report(stats, args.output_report)

if __name__ == "__main__":
    main()
