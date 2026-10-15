#!/usr/bin/env python3
"""Sample-efficiency curve for LPSE, MLP, and XGBoost.

Shows validation top-1 accuracy as a function of cumulative training
compute, on a common axis: sample-views = (sample, feature-vector) touches
performed by the trainer. SGD models contribute BATCH_SIZE=32 per logged
step; XGBoost contributes XGB_N_ESTIMATORS_PER_REFIT × min(buffer,
XGB_MAX_TRAIN_SAMPLES) per logged refit, since each boosting round scans
the full training set. Output is a single line plot, vector PDF, sized for
single-column IEEE width with serif IEEE-style fonts.
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SGD_BATCH_SIZE = 32
XGB_TREES_PER_REFIT = 50
XGB_MAX_TRAIN_SAMPLES = 20000

PALETTE = {
    "hybrid": ("#d0cee2", "#56517e"),
    "mlp": ("#fff2cc", "#d6b656"),
    "xgb": ("#d5e8d4", "#82b366"),
}
LABELS = {"hybrid": "LPSE", "mlp": "MLP", "xgb": "XGBoost"}


def _parse_float(v: str) -> float:
    if v is None:
        return math.nan
    s = v.strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _read(csv_path: Path):
    runs: list[list[tuple[float, int, float, int]]] = [[]]
    last_step = -1
    with csv_path.open(newline="") as h:
        r = csv.DictReader(h)
        for row in r:
            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue
            if step <= last_step:
                runs.append([])
            last_step = step
            ts = _parse_float(row.get("timestamp", ""))
            val = _parse_float(row.get("val_top1", ""))
            try:
                buf = int(row.get("buffer_size", "") or 0)
            except (TypeError, ValueError):
                buf = 0
            runs[-1].append((ts, step, val, buf))
    return max(runs, key=len) if runs and runs[0] else []


def _cum_samples(rows, kind: str) -> list[float]:
    out: list[float] = []
    total = 0.0
    for _ts, _step, _val, buf in rows:
        if kind == "sgd":
            total += SGD_BATCH_SIZE
        else:
            n = min(buf if buf > 0 else XGB_MAX_TRAIN_SAMPLES, XGB_MAX_TRAIN_SAMPLES)
            total += XGB_TREES_PER_REFIT * n
        out.append(total)
    return out


def _moving_avg(xs: Sequence[float], w: int) -> list[float]:
    if w <= 1:
        return list(xs)
    out: list[float] = []
    q: deque[float] = deque()
    s = 0.0
    for x in xs:
        q.append(x)
        s += x
        if len(q) > w:
            s -= q.popleft()
        out.append(s / len(q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-log", default="lpse_training_loghybrid.csv")
    ap.add_argument("--mlp-log", default="baseline_mlp_v2_training_log.csv")
    ap.add_argument("--xgb-log", default="baseline_xgb_v2_training_log.csv")
    ap.add_argument("--output", default="plots/sample_efficiency.pdf")
    ap.add_argument(
        "--smooth",
        type=int,
        default=400,
        help="Moving-average window for noisy per-step val curves.",
    )
    args = ap.parse_args()

    paths = {
        "hybrid": (Path(args.hybrid_log), "sgd"),
        "mlp": (Path(args.mlp_log), "sgd"),
        "xgb": (Path(args.xgb_log), "xgb"),
    }

    NUM_ANSWER_TEMPLATES = 94
    CHANCE_ACC = 1.0 / NUM_ANSWER_TEMPLATES
    series = {}
    for k, (p, kind) in paths.items():
        rows = _read(p)
        cum = _cum_samples(rows, kind)
        xs, ys = [], []
        for (_ts, _step, val, _buf), c in zip(rows, cum):
            if not math.isnan(val):
                xs.append(c)
                ys.append(val)

        ys_s = _moving_avg(ys, args.smooth) if kind == "sgd" else ys

        anchor_x = 1.0
        xs = [anchor_x] + list(xs)
        ys_s = [CHANCE_ACC] + list(ys_s)
        series[k] = (xs, ys_s)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.patch.set_facecolor("white")
    for k in ("hybrid", "mlp", "xgb"):
        xs, ys = series[k]
        if not xs:
            continue
        ax.plot(xs, ys, color=PALETTE[k][1], linewidth=1.6, label=LABELS[k])
        ax.fill_between(xs, 0, ys, color=PALETTE[k][0], alpha=0.30)

    ax.set_xscale("log")
    ax.set_xlim(1, 2e8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cumulative training sample-views (log)", fontsize=9, color="#444444")
    ax.set_ylabel("Validation top-1 accuracy", fontsize=9, color="#444444")
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(axis="both", colors="#666666", labelsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout(pad=0.4)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    for k in ("hybrid", "mlp", "xgb"):
        xs, ys = series[k]
        if xs:
            print(
                f"{LABELS[k]:8s}  end_compute={xs[-1]:.2e}  final_val={ys[-1]:.4f}  n_points={len(xs)}"
            )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
