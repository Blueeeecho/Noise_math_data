import argparse
import re
from pathlib import Path


ACCURACY_PATTERN = re.compile(r"Current average accuracy across 13 datasets:\s*([0-9.]+)")
METRIC_PATTERNS = {
    "r_acc_mean1": re.compile(r"val-aux/math_noise/r_acc/mean@1['\"]?:\s*([0-9.\-]+)"),
    "step_good_count_mean1": re.compile(r"val-aux/math_noise/step_good_count/mean@1['\"]?:\s*([0-9.\-]+)"),
    "step_count_mean1": re.compile(r"val-aux/math_noise/step_count/mean@1['\"]?:\s*([0-9.\-]+)"),
    "global_format_pass_mean1": re.compile(r"val-aux/math_noise/global_format_pass/mean@1['\"]?:\s*([0-9.\-]+)"),
    "final_answer_parsed_mean1": re.compile(r"val-aux/math_noise/final_answer_parsed/mean@1['\"]?:\s*([0-9.\-]+)"),
}


def extract_first_match(text: str, pattern: re.Pattern[str]) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def extract_accuracy_series(text: str) -> list[float]:
    return [float(match.group(1)) for match in ACCURACY_PATTERN.finditer(text)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_log(path: Path, top_n: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    accuracies = extract_accuracy_series(text)
    if not accuracies:
        raise ValueError(f"No validation accuracy series found in {path}")

    top_values = accuracies[:top_n]
    first_window = top_values[: min(3, len(top_values))]
    last_window = top_values[-min(3, len(top_values)) :]

    metrics = {
        metric_name: extract_first_match(text, pattern)
        for metric_name, pattern in METRIC_PATTERNS.items()
    }

    return {
        "path": str(path),
        "count_found": len(accuracies),
        "count_used": len(top_values),
        "values": top_values,
        "mean": mean(top_values),
        "best": max(top_values),
        "worst": min(top_values),
        "start": top_values[0],
        "end": top_values[-1],
        "delta": top_values[-1] - top_values[0],
        "range": max(top_values) - min(top_values),
        "first3_mean": mean(first_window),
        "last3_mean": mean(last_window),
        "metrics": metrics,
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def print_summary(summary: dict, top_n: int) -> None:
    print(f"Log: {summary['path']}")
    print(f"Top-{top_n} values: {', '.join(f'{value:.4f}' for value in summary['values'])}")
    print(f"Used validations: {summary['count_used']} / Found validations: {summary['count_found']}")
    print(f"Mean: {summary['mean']:.4f}")
    print(f"Best: {summary['best']:.4f}")
    print(f"Worst: {summary['worst']:.4f}")
    print(f"Start -> End: {summary['start']:.4f} -> {summary['end']:.4f}")
    print(f"Delta: {summary['delta']:.4f}")
    print(f"Range: {summary['range']:.4f}")
    print(f"First3 Mean: {summary['first3_mean']:.4f}")
    print(f"Last3 Mean: {summary['last3_mean']:.4f}")
    print("Early structure metrics:")
    print(f"  r_acc/mean@1: {fmt(summary['metrics']['r_acc_mean1'])}")
    print(f"  step_good_count/mean@1: {fmt(summary['metrics']['step_good_count_mean1'])}")
    print(f"  step_count/mean@1: {fmt(summary['metrics']['step_count_mean1'])}")
    print(f"  global_format_pass/mean@1: {fmt(summary['metrics']['global_format_pass_mean1'])}")
    print(f"  final_answer_parsed/mean@1: {fmt(summary['metrics']['final_answer_parsed_mean1'])}")
    print()


def print_comparison(left: dict, right: dict) -> None:
    mean_winner = left if left["mean"] >= right["mean"] else right
    stability_winner = left if abs(left["delta"]) <= abs(right["delta"]) else right
    print("Comparison")
    print(f"  Higher short-budget mean: {mean_winner['path']} ({mean_winner['mean']:.4f})")
    print(f"  More stable early curve: {stability_winner['path']} (|delta|={abs(stability_winner['delta']):.4f})")
    print(
        "  Mean gap: "
        f"{abs(left['mean'] - right['mean']):.4f} "
        f"({Path(left['path']).name} vs {Path(right['path']).name})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_a")
    parser.add_argument("log_b")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_a = summarize_log(Path(args.log_a), args.top_n)
    summary_b = summarize_log(Path(args.log_b), args.top_n)
    print_summary(summary_a, args.top_n)
    print_summary(summary_b, args.top_n)
    print_comparison(summary_a, summary_b)


if __name__ == "__main__":
    main()
