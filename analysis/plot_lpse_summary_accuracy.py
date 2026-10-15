#!/usr/bin/env python3
"""Plot average accuracy over training steps from LPSE summary JSONL."""

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_points(summary_path: Path) -> tuple[list[int], list[float], int]:
    by_step: dict[int, float] = {}
    skipped = 0

    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            record_text = line.strip()
            if not record_text:
                continue

            try:
                record = json.loads(record_text)
            except json.JSONDecodeError:
                skipped += 1
                continue

            step = record.get("step")
            avg_accuracy = record.get("avg_accuracy")
            if isinstance(step, (int, float)) and isinstance(
                avg_accuracy, (int, float)
            ):
                by_step[int(step)] = float(avg_accuracy)
            else:
                skipped += 1

    steps = sorted(by_step)
    accuracies = [by_step[s] for s in steps]
    return steps, accuracies, skipped


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)

    out: list[float] = []
    q: deque[float] = deque()
    running_sum = 0.0

    for value in values:
        q.append(value)
        running_sum += value
        if len(q) > window:
            running_sum -= q.popleft()
        out.append(running_sum / len(q))

    return out


def _plot(
    steps: list[int], accuracies: list[float], output_path: Path, smooth_window: int
) -> None:
    smoothed = _moving_average(accuracies, smooth_window)

    plt.figure(figsize=(11, 4.8))
    plt.plot(steps, smoothed, linewidth=1.4, color="#1f77b4")
    plt.title("LPSE Average Accuracy vs Training Step")
    plt.xlabel("Training Step")
    plt.ylabel("Average Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average accuracy over training steps from lpse_summary.txt"
    )
    parser.add_argument(
        "--summary",
        default="lpse_summary.txt",
        help="Path to summary txt file written by train_lpse.py",
    )
    parser.add_argument(
        "--output",
        default="plots/lpse_avg_accuracy.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window over accuracy points (>=1)",
    )
    args = parser.parse_args()

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")

    summary_path = Path(args.summary).expanduser()
    output_path = Path(args.output).expanduser()

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    steps, accuracies, skipped = _read_points(summary_path)
    if not steps:
        raise RuntimeError(
            f"No valid step/avg_accuracy records found in {summary_path}"
        )

    _plot(steps, accuracies, output_path, args.smooth_window)

    print(f"Read {len(steps)} points from {summary_path}")
    if skipped:
        print(f"Skipped {skipped} malformed or incomplete lines")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
