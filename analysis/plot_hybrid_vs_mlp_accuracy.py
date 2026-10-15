#!/usr/bin/env python3
"""Plot training and validation accuracy for hybrid LPSE, MLP baseline, and
XGBoost baseline.

Reads per-step CSV logs (columns: step, top1_acc, val_top1) for each model and
overlays the curves on a single plot. Validation rows where val_top1 is
NaN/blank are skipped so the validation curve only shows actual eval points.

Note on x-axis: hybrid and MLP "step" = SGD optimizer step; XGB "step" =
boosting round. They are plotted on the same training-step axis so the reader
can see XGB converges in its own rounds budget while SGD models continue to
improve over many more steps.
"""

import argparse
import csv
import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_float(value: str) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _read_log(csv_path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    """Read per-step rows, splitting on step-decrease (restart) and keeping the
    longest contiguous run. Concatenated CSVs from multiple training restarts
    otherwise produce duplicate step values that show as horizontal artifacts."""

    runs: list[list[tuple[int, float, float]]] = [[]]
    last_step = -1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "top1_acc", "val_top1"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue

            if step <= last_step:
                runs.append([])
            last_step = step

            top1 = _parse_float(row.get("top1_acc", ""))
            val = _parse_float(row.get("val_top1", ""))
            runs[-1].append((step, top1, val))

    chosen = max(runs, key=len)

    train_steps: list[int] = []
    train_acc: list[float] = []
    val_steps: list[int] = []
    val_acc: list[float] = []
    for step, top1, val in chosen:
        if not math.isnan(top1):
            train_steps.append(step)
            train_acc.append(top1)
        if not math.isnan(val):
            val_steps.append(step)
            val_acc.append(val)

    return train_steps, train_acc, val_steps, val_acc


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


def _plot(
    hybrid_train: tuple[list[int], list[float]],
    hybrid_val: tuple[list[int], list[float]],
    mlp_train: tuple[list[int], list[float]],
    mlp_val: tuple[list[int], list[float]],
    xgb_train: tuple[list[int], list[float]],
    xgb_val: tuple[list[int], list[float]],
    output_path: Path,
    train_smooth: int,
    val_smooth: int,
    xgb_smooth: int,
) -> None:
    plt.figure(figsize=(11, 5.2))

    hybrid_color = "#1f77b4"
    mlp_color = "#d62728"
    xgb_color = "#2ca02c"

    if hybrid_train[0]:
        plt.plot(
            hybrid_train[0],
            _moving_average(hybrid_train[1], train_smooth),
            color=hybrid_color,
            linewidth=1.4,
            label="Hybrid train",
        )
    if hybrid_val[0]:
        plt.plot(
            hybrid_val[0],
            _moving_average(hybrid_val[1], val_smooth),
            color=hybrid_color,
            linewidth=1.6,
            linestyle="--",
            label="Hybrid val",
        )
    if mlp_train[0]:
        plt.plot(
            mlp_train[0],
            _moving_average(mlp_train[1], train_smooth),
            color=mlp_color,
            linewidth=1.4,
            label="MLP train",
        )
    if mlp_val[0]:
        plt.plot(
            mlp_val[0],
            _moving_average(mlp_val[1], val_smooth),
            color=mlp_color,
            linewidth=1.6,
            linestyle="--",
            label="MLP val",
        )
    if xgb_train[0]:
        plt.plot(
            xgb_train[0],
            _moving_average(xgb_train[1], xgb_smooth),
            color=xgb_color,
            linewidth=1.6,
            label=f"XGB train (final {xgb_train[1][-1]:.2f})",
        )

        plt.axhline(xgb_train[1][-1], color=xgb_color, linewidth=1.0, alpha=0.35)
    if xgb_val[0]:
        plt.plot(
            xgb_val[0],
            _moving_average(xgb_val[1], xgb_smooth),
            color=xgb_color,
            linewidth=1.6,
            linestyle="--",
            label=f"XGB val (final {xgb_val[1][-1]:.2f})",
        )
        plt.axhline(
            xgb_val[1][-1], color=xgb_color, linewidth=1.0, alpha=0.35, linestyle="--"
        )

    plt.title("Hybrid LPSE vs MLP vs XGBoost: Training and Validation Accuracy")
    plt.xlabel("Training Step  (XGB step = boosting round)")
    plt.ylabel("Top-1 Accuracy")
    plt.xlim(0, 30000)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=3, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _report(
    name: str, train: tuple[list[int], list[float]], val: tuple[list[int], list[float]]
) -> None:
    train_n = len(train[0])
    val_n = len(val[0])
    final_train = train[1][-1] if train_n else float("nan")
    final_val = val[1][-1] if val_n else float("nan")
    best_train = max(train[1]) if train_n else float("nan")
    best_val = max(val[1]) if val_n else float("nan")
    print(
        f"{name}: train_points={train_n} val_points={val_n} "
        f"final_train={final_train:.4f} best_train={best_train:.4f} "
        f"final_val={final_val:.4f} best_val={best_val:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare training and validation accuracy of hybrid LPSE and MLP baseline"
    )
    parser.add_argument(
        "--hybrid-log",
        default="lpse_training_loghybrid.csv",
        help="Per-step CSV log for the hybrid LPSE run",
    )
    parser.add_argument(
        "--mlp-log",
        default="baseline_mlp_v2_training_log.csv",
        help="Per-step CSV log for the MLP baseline run (v2 = streaming offline)",
    )
    parser.add_argument(
        "--xgb-log",
        default="baseline_xgb_v2_training_log.csv",
        help="Per-step CSV log for the XGBoost baseline run (step = boosting round)",
    )
    parser.add_argument(
        "--output",
        default="plots/hybrid_vs_mlp_xgb_accuracy.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--train-smooth-window",
        type=int,
        default=500,
        help="Moving-average window for noisy per-step training accuracy (>=1)",
    )
    parser.add_argument(
        "--val-smooth-window",
        type=int,
        default=500,
        help="Moving-average window for validation accuracy points (>=1)",
    )
    parser.add_argument(
        "--xgb-smooth-window",
        type=int,
        default=1,
        help="Moving-average window for the sparser XGB curve (>=1)",
    )
    args = parser.parse_args()

    if (
        args.train_smooth_window < 1
        or args.val_smooth_window < 1
        or args.xgb_smooth_window < 1
    ):
        raise ValueError("smoothing windows must be >= 1")

    hybrid_path = Path(args.hybrid_log).expanduser()
    mlp_path = Path(args.mlp_log).expanduser()
    xgb_path = Path(args.xgb_log).expanduser()
    output_path = Path(args.output).expanduser()

    for path in (hybrid_path, mlp_path):
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

    h_train_s, h_train_a, h_val_s, h_val_a = _read_log(hybrid_path)
    m_train_s, m_train_a, m_val_s, m_val_a = _read_log(mlp_path)
    if xgb_path.exists():
        x_train_s, x_train_a, x_val_s, x_val_a = _read_log(xgb_path)
    else:
        print(f"XGB log {xgb_path} not found — plotting without XGB.")
        x_train_s, x_train_a, x_val_s, x_val_a = [], [], [], []

    if not (h_train_s or h_val_s or m_train_s or m_val_s):
        raise RuntimeError("No usable accuracy points found in either log")

    _plot(
        (h_train_s, h_train_a),
        (h_val_s, h_val_a),
        (m_train_s, m_train_a),
        (m_val_s, m_val_a),
        (x_train_s, x_train_a),
        (x_val_s, x_val_a),
        output_path,
        args.train_smooth_window,
        args.val_smooth_window,
        args.xgb_smooth_window,
    )

    _report("Hybrid", (h_train_s, h_train_a), (h_val_s, h_val_a))
    _report("MLP   ", (m_train_s, m_train_a), (m_val_s, m_val_a))
    _report("XGB   ", (x_train_s, x_train_a), (x_val_s, x_val_a))
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
