#!/usr/bin/env python3
"""Analyze training CSV for accuracy and loss plots.

Auto-discovers the most-recent ``lpse_training_log*.csv`` if ``--csv`` is
not given. Tolerant of old and new CSV schemas: shows whichever loss
columns are present (qa_loss, ssl_loss/jepa_loss, total_loss, var_loss,
cov_loss, latent_mean_std).
"""

import argparse
import csv
import glob
import math
import os

import matplotlib.pyplot as plt

DEFAULT_DENOMS = [8, 16, 32, 64, 128, 256]
MAX_POINTS = 2000
SMOOTH_WINDOW = 101
CLIP_PCT = 1.0


LOSS_PANELS = [
    (("qa_loss",), "qa_loss", True),
    (("ssl_loss", "jepa_loss"), "ssl_loss", False),
    (("total_loss",), "total_loss", True),
    (("var_loss",), "var_loss", False),
    (("cov_loss",), "cov_loss", False),
    (("latent_mean_std",), "latent_mean_std", False),
]


def _find_latest_csv(pattern: str = "lpse_training_log*.csv") -> str | None:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _read_csv(path: str) -> dict[str, list[float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                val = row.get(name, "")
                try:
                    columns[name].append(float(val))
                except ValueError:
                    columns[name].append(float("nan"))
    return columns


def _resolve_panel(
    columns: dict[str, list[float]], candidates: tuple[str, ...]
) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _infer_batch_size(acc_values: list[float], denoms: list[int]) -> int:
    denom_hits = {d: 0 for d in denoms}
    for acc in acc_values:
        if not (0.0 <= acc <= 1.0):
            continue
        if acc in (0.0, 1.0):
            continue
        for d in denoms:
            scaled = acc * d
            if abs(scaled - round(scaled)) < 1e-6:
                denom_hits[d] += 1
    best = max(denom_hits, key=lambda d: denom_hits[d])
    return best if denom_hits[best] > 0 else 32


def _compute_counts(acc_values: list[float], batch_size: int) -> tuple[int, int, float]:
    total_correct = 0
    total_samples = 0
    for acc in acc_values:
        if acc != acc or acc < 0.0 or acc > 1.0:
            continue
        correct = int(round(acc * batch_size))
        correct = min(batch_size, max(0, correct))
        total_correct += correct
        total_samples += batch_size
    total_incorrect = total_samples - total_correct
    overall_acc = (total_correct / total_samples) if total_samples else 0.0
    return total_correct, total_incorrect, overall_acc


def _plot_accuracy_counts(out_path: str, correct: int, incorrect: int) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Correct", "Incorrect"], [correct, incorrect], color=["#2ca02c", "#d62728"]
    )
    plt.title("Correct vs Incorrect Predictions")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _log10_safe(values: list[float], eps: float = 1e-12) -> list[float]:
    out: list[float] = []
    for v in values:
        if v != v:
            out.append(float("nan"))
            continue
        v = max(float(v), eps)
        out.append(math.log10(v))
    return out


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        chunk = [v for v in values[start:end] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def _downsample(
    steps: list[float], values: list[float], max_points: int
) -> tuple[list[float], list[float]]:
    if len(steps) <= max_points:
        return steps, values
    stride = max(1, len(steps) // max_points)
    return steps[::stride], values[::stride]


def _percentile_clip(values: list[float], pct: float) -> list[float]:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return values
    lo_idx = int(len(clean) * (pct / 100.0))
    hi_idx = int(len(clean) * (1.0 - pct / 100.0)) - 1
    lo = clean[max(0, min(lo_idx, len(clean) - 1))]
    hi = clean[max(0, min(hi_idx, len(clean) - 1))]
    return [min(hi, max(lo, v)) if v == v else v for v in values]


def _plot_loss_subplots(
    out_path: str, steps: list[float], panels: list[tuple[str, str, list[float], bool]]
) -> None:
    """panels: list of (column-name, panel-title, values, log10?)."""
    n = len(panels)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), sharex=True)
    if rows == 1:
        axes_iter = list(axes) if cols > 1 else [axes]
    else:
        axes_iter = [ax for row in axes for ax in row]
    for ax, (name, title, values, log_scale) in zip(axes_iter, panels):
        clipped = _percentile_clip(values, CLIP_PCT)
        smoothed = _moving_average(clipped, SMOOTH_WINDOW)
        s_steps, s_values = _downsample(steps, smoothed, MAX_POINTS)
        y_values = _log10_safe(s_values) if log_scale else s_values
        ax.plot(s_steps, y_values, color="#1f77b4", linewidth=1.2)
        ax.set_title(f"log10({title})" if log_scale else title)
        ax.set_xlabel("Step")
        ax.set_ylabel("log10(value)" if log_scale else "value")
        ax.grid(alpha=0.3)

    for ax in axes_iter[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy and plot loss from training CSV."
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Path to training CSV. Default: latest lpse_training_log*.csv by mtime.",
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Directory to save plots."
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="Batch size used to compute top1_acc."
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path:
        csv_path = _find_latest_csv()
        if not csv_path:
            raise SystemExit(
                "No CSV found. Pass --csv explicitly or run from a directory with lpse_training_log*.csv files."
            )
        print(f"Auto-selected latest CSV: {csv_path}")

    columns = _read_csv(csv_path)
    if "top1_acc" not in columns:
        raise KeyError("CSV is missing 'top1_acc' column.")
    if "step" not in columns:
        raise KeyError("CSV is missing 'step' column.")

    panels: list[tuple[str, str, list[float], bool]] = []
    for candidates, title, log_scale in LOSS_PANELS:
        col = _resolve_panel(columns, candidates)
        if col is not None:
            panels.append((col, title, columns[col], log_scale))

    if not panels:
        raise KeyError(
            "CSV has no recognised loss columns (qa_loss / ssl_loss / total_loss / var_loss / cov_loss / latent_mean_std)."
        )

    print(f"Loss panels found: {[p[1] for p in panels]}")

    acc_values = columns["top1_acc"]
    steps = columns["step"]

    batch_size = (
        args.batch_size
        if args.batch_size > 0
        else _infer_batch_size(acc_values, DEFAULT_DENOMS)
    )
    correct, incorrect, overall_acc = _compute_counts(acc_values, batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    counts_path = os.path.join(args.output_dir, "accuracy_counts.png")
    loss_path = os.path.join(args.output_dir, "loss_curves_log10.png")

    _plot_accuracy_counts(counts_path, correct, incorrect)
    _plot_loss_subplots(loss_path, steps, panels)

    print(f"Batch size used: {batch_size}")
    print(f"Total samples: {correct + incorrect}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Overall accuracy: {overall_acc * 100:.2f}%")
    print(f"Saved: {counts_path}")
    print(f"Saved: {loss_path}")


if __name__ == "__main__":
    main()
