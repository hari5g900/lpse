#!/usr/bin/env python3
"""Compare LPSE summary JSONL files across three training runs.

The script reads the shared metrics from each summary file, aligns records by
training step, plots accuracy and loss trends, and prints a compact comparison
table with final/best metrics for each run.

Usage:
  python compare_lpse_summaries.py \
    --summary1 lpse_summary.txt \
    --summary2 lpse_summary2.txt \
    --summary3 lpse_summary3.txt
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SummarySeries:
    label: str
    steps: list[int]
    losses: list[float]
    accuracies: list[float]
    skipped: int


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_series(summary_path: Path, label: str) -> SummarySeries:
    by_step_loss: dict[int, float] = {}
    by_step_accuracy: dict[int, float] = {}
    skipped = 0

    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record_text = raw_line.strip()
            if not record_text:
                continue

            try:
                record = json.loads(record_text)
            except json.JSONDecodeError:
                skipped += 1
                continue

            step = record.get("step")
            avg_loss = record.get("avg_loss")
            avg_accuracy = record.get("avg_accuracy")
            if _is_number(step) and _is_number(avg_loss) and _is_number(avg_accuracy):
                step_key = int(step)
                by_step_loss[step_key] = float(avg_loss)
                by_step_accuracy[step_key] = float(avg_accuracy)
            else:
                skipped += 1

    steps = sorted(set(by_step_loss) & set(by_step_accuracy))
    losses = [by_step_loss[step] for step in steps]
    accuracies = [by_step_accuracy[step] for step in steps]
    return SummarySeries(
        label=label, steps=steps, losses=losses, accuracies=accuracies, skipped=skipped
    )


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)

    smoothed: list[float] = []
    queue: deque[float] = deque()
    running_sum = 0.0

    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.popleft()
        smoothed.append(running_sum / len(queue))

    return smoothed


def _last_value(values: Sequence[float]) -> float | None:
    return None if not values else float(values[-1])


def _best_value(values: Sequence[float], maximize: bool) -> tuple[int, float] | None:
    if not values:
        return None

    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if (maximize and value > best_value) or (not maximize and value < best_value):
            best_index = index
            best_value = value
    return best_index, float(best_value)


def _plot(
    series_list: Sequence[SummarySeries], output_path: Path, smooth_window: int
) -> None:
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for index, series in enumerate(series_list):
        color = colors[index % len(colors)]
        smoothed_accuracy = _moving_average(series.accuracies, smooth_window)
        smoothed_loss = _moving_average(series.losses, smooth_window)

        ax_acc.plot(
            series.steps,
            smoothed_accuracy,
            label=series.label,
            linewidth=1.5,
            color=color,
        )
        ax_loss.plot(
            series.steps, smoothed_loss, label=series.label, linewidth=1.5, color=color
        )

    ax_acc.set_title("LPSE Summary Comparison")
    ax_acc.set_ylabel("Average Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best")

    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Average Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _print_report(series_list: Sequence[SummarySeries]) -> None:
    print("Summary comparison")
    print("label | points | final_acc | best_acc | final_loss | best_loss | skipped")
    print("-" * 76)

    for series in series_list:
        final_acc = _last_value(series.accuracies)
        final_loss = _last_value(series.losses)
        best_acc = _best_value(series.accuracies, maximize=True)
        best_loss = _best_value(series.losses, maximize=False)
        print(
            f"{series.label} | {len(series.steps)} | {_format_float(final_acc)} | "
            f"{_format_float(None if best_acc is None else best_acc[1])} | "
            f"{_format_float(final_loss)} | {_format_float(None if best_loss is None else best_loss[1])} | {series.skipped}"
        )

    common_steps = set(series_list[0].steps)
    for series in series_list[1:]:
        common_steps &= set(series.steps)

    if common_steps:
        print()
        print(f"Common aligned steps: {len(common_steps)}")
        print(f"Step range: {min(common_steps)}..{max(common_steps)}")
    else:
        print()
        print("No shared steps across all three summary files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three LPSE summary JSONL files"
    )
    parser.add_argument(
        "--summary1", default="lpse_summary.txt", help="First summary JSONL file"
    )
    parser.add_argument(
        "--summary2", default="lpse_summary2.txt", help="Second summary JSONL file"
    )
    parser.add_argument(
        "--summary3", default="lpse_summary3.txt", help="Third summary JSONL file"
    )
    parser.add_argument(
        "--output",
        default="plots/lpse_summary_comparison.png",
        help="Output PNG path for the comparison plot",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Moving-average window for plotted curves (>=1)",
    )
    args = parser.parse_args()

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")

    summary_paths = [
        Path(args.summary1).expanduser(),
        Path(args.summary2).expanduser(),
        Path(args.summary3).expanduser(),
    ]
    labels = ["summary1", "summary2", "summary3"]

    for summary_path in summary_paths:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

    series_list = [
        _read_series(path, label) for path, label in zip(summary_paths, labels)
    ]

    if not any(series.steps for series in series_list):
        raise RuntimeError(
            "No valid step/avg_loss/avg_accuracy records found in the provided summaries"
        )

    output_path = Path(args.output).expanduser()
    _plot(series_list, output_path, args.smooth_window)
    _print_report(series_list)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
