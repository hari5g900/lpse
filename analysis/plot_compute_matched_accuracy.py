#!/usr/bin/env python3
"""Plot the accuracy vs compute tradeoff for LPSE, MLP, and XGBoost.

Motivation: CSV "step" inflates XGBoost's apparent efficiency because each
XGB row covers ~100 accumulated scrapes plus 50 fresh boosting trees fit on
the full live buffer (~20k samples), while each LPSE / MLP row is a
single 32-sample SGD update. Comparing on step makes XGB look "free" — it
is anything but. The honest comparison is per sample-view of compute.

Sample-view accounting (matches train_*.py defaults):
  - SGD models: BATCH_SIZE (=32) per logged row.
  - XGB:        XGB_N_ESTIMATORS_PER_REFIT (=50)
                * min(buffer_size, MAX_TRAIN_SAMPLES=20000) per logged row,
                since each boosting round scans the whole training set.

Visualization: side-by-side horizontal bars sharing model rows.
  - Left:  validation top-1 accuracy (linear 0–1).
  - Right: cumulative compute in sample-views (log scale).
A summary annotation calls out the headline tradeoff between the top SGD
model and XGBoost (Δaccuracy vs ×compute multiplier).
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "Times",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.titlesize": "medium",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


SGD_BATCH_SIZE = 32
XGB_TREES_PER_REFIT = 50
XGB_MAX_TRAIN_SAMPLES = 20000


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


def _read_log(csv_path: Path) -> list[tuple[float, int, float, float, int]]:
    """Read rows, splitting on step-decrease (restart) and keeping the longest
    contiguous run. Returns (timestamp, step, top1_acc, val_top1, buffer_size)."""

    runs: list[list[tuple[float, int, float, float, int]]] = [[]]
    last_step = -1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "step", "top1_acc", "val_top1", "buffer_size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue
            ts = _parse_float(row.get("timestamp", ""))
            if math.isnan(ts):
                continue

            if step <= last_step:
                runs.append([])
            last_step = step

            top1 = _parse_float(row.get("top1_acc", ""))
            val = _parse_float(row.get("val_top1", ""))
            try:
                buf = int(row.get("buffer_size", "") or 0)
            except (TypeError, ValueError):
                buf = 0

            runs[-1].append((ts, step, top1, val, buf))

    return max(runs, key=len) if runs and runs[0] else []


def _cumulative_samples(rows, kind: str) -> list[float]:
    cum: list[float] = []
    total = 0.0
    for _ts, _step, _top1, _val, buf in rows:
        if kind == "sgd":
            total += SGD_BATCH_SIZE
        elif kind == "xgb":
            n = min(buf if buf > 0 else XGB_MAX_TRAIN_SAMPLES, XGB_MAX_TRAIN_SAMPLES)
            total += XGB_TREES_PER_REFIT * n
        else:
            raise ValueError(kind)
        cum.append(total)
    return cum


def _final_val(rows, smooth_radius: int) -> float:
    if not rows:
        return float("nan")
    n = len(rows)
    lo = max(0, n - 1 - smooth_radius)
    vals = [rows[i][3] for i in range(lo, n) if not math.isnan(rows[i][3])]
    return float(np.mean(vals)) if vals else float("nan")


def _format_compute(c: float) -> str:
    if c <= 0 or math.isnan(c):
        return "n/a"
    exp = int(math.floor(math.log10(c)))
    mantissa = c / (10**exp)
    return f"{mantissa:.1f}×10$^{{{exp}}}$"


_MODELS = ["hybrid", "mlp", "xgb"]
_LABELS = {"hybrid": "LPSE", "mlp": "MLP", "xgb": "XGBoost"}

_PALETTE = {
    "hybrid": ("#d0cee2", "#56517e"),
    "mlp": ("#fff2cc", "#d6b656"),
    "xgb": ("#d5e8d4", "#82b366"),
}
_FILL_ALPHA = 0.55


_FIGSIZE = (3.5, 1.9)
_FS_TICK = 9
_FS_LABEL = 9
_FS_VALUE = 9.5


def _setup_axis(ax) -> None:
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(axis="x", colors="#666666", labelsize=_FS_TICK)
    ax.tick_params(axis="y", colors="#666666", labelsize=_FS_TICK)


def _draw_yticks(ax, y_pos) -> None:
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_LABELS[m] for m in _MODELS], fontsize=_FS_TICK)
    for tick_label, m in zip(ax.get_yticklabels(), _MODELS):
        tick_label.set_color(_PALETTE[m][1])
        tick_label.set_fontweight("bold")


def _plot_accuracy(finals: dict[str, tuple[float, float]], output_path: Path) -> None:
    accs = [finals[m][0] for m in _MODELS]
    fills = [_PALETTE[m][0] for m in _MODELS]
    edges = [_PALETTE[m][1] for m in _MODELS]
    y_pos = np.arange(len(_MODELS))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.barh(
        y_pos,
        accs,
        color=fills,
        edgecolor=edges,
        linewidth=1.0,
        height=0.62,
        alpha=_FILL_ALPHA,
    )
    for y, m, v in zip(y_pos, _MODELS, accs):
        ax.text(
            v + 0.012,
            y,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=_FS_VALUE,
            fontweight="bold",
            color=_PALETTE[m][1],
        )

    _draw_yticks(ax, y_pos)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Validation top-1 accuracy", color="#444444", fontsize=_FS_LABEL)
    ax.invert_yaxis()
    _setup_axis(ax)

    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_compute(finals: dict[str, tuple[float, float]], output_path: Path) -> None:
    comps = [finals[m][1] for m in _MODELS]
    fills = [_PALETTE[m][0] for m in _MODELS]
    edges = [_PALETTE[m][1] for m in _MODELS]
    y_pos = np.arange(len(_MODELS))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.barh(
        y_pos,
        comps,
        color=fills,
        edgecolor=edges,
        linewidth=1.0,
        height=0.62,
        alpha=_FILL_ALPHA,
    )
    for y, m, c in zip(y_pos, _MODELS, comps):
        ax.text(
            c * 1.25,
            y,
            _format_compute(c),
            va="center",
            ha="left",
            fontsize=_FS_VALUE,
            fontweight="bold",
            color=_PALETTE[m][1],
        )

    ax.set_xscale("log")
    max_c = max(comps) if comps else 1.0
    ax.set_xlim(1e5, max_c * 8)
    _draw_yticks(ax, y_pos)
    ax.set_xlabel("Cumulative sample-views (log)", color="#444444", fontsize=_FS_LABEL)
    ax.invert_yaxis()
    _setup_axis(ax)

    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute-vs-accuracy tradeoff bar chart."
    )
    parser.add_argument("--hybrid-log", default="lpse_training_loghybrid.csv")
    parser.add_argument("--mlp-log", default="baseline_mlp_v2_training_log.csv")
    parser.add_argument("--xgb-log", default="baseline_xgb_v2_training_log.csv")
    parser.add_argument(
        "--output-accuracy",
        default="plots/compute_matched_accuracy_a.pdf",
        help="Output PDF for panel (a): accuracy bars (single-column width).",
    )
    parser.add_argument(
        "--output-compute",
        default="plots/compute_matched_accuracy_b.pdf",
        help="Output PDF for panel (b): compute bars (single-column width).",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=200,
        help="Rows averaged for the SGD final accuracy. SGD logs are per-batch "
        "and noisy; XGB logs are per-refit so a window of 2 is plenty.",
    )
    args = parser.parse_args()

    paths = {
        "hybrid": Path(args.hybrid_log).expanduser(),
        "mlp": Path(args.mlp_log).expanduser(),
        "xgb": Path(args.xgb_log).expanduser(),
    }
    kinds = {"hybrid": "sgd", "mlp": "sgd", "xgb": "xgb"}

    finals: dict[str, tuple[float, float]] = {}
    for m, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")
        rows = _read_log(p)
        cum = _cumulative_samples(rows, kinds[m])
        sr = 2 if kinds[m] == "xgb" else max(1, args.smooth_radius)
        finals[m] = (_final_val(rows, sr), cum[-1] if cum else 0.0)

    out_a = Path(args.output_accuracy).expanduser()
    out_b = Path(args.output_compute).expanduser()
    _plot_accuracy(finals, out_a)
    _plot_compute(finals, out_b)

    print("Final state per model:")
    for m, (v, c) in finals.items():
        print(f"  {m:6s}  val_top1={v:.4f}  compute={c:.3e}")
    sgd_best = "hybrid" if finals["hybrid"][0] >= finals["mlp"][0] else "mlp"
    print(
        f"Headline: XGBoost +{(finals['xgb'][0] - finals[sgd_best][0]) * 100:.1f} pp "
        f"vs {sgd_best}, at {finals['xgb'][1] / finals[sgd_best][1]:.0f}× the compute."
    )
    print(f"Saved (a) accuracy: {out_a}")
    print(f"Saved (b) compute:  {out_b}")


if __name__ == "__main__":
    main()
