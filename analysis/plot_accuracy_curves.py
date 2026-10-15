#!/usr/bin/env python3
"""Validation top-1 accuracy vs training step for LPSE, MLP, XGBoost.
Step axis to ~30000. Style matches sample_efficiency.pdf and
compute_matched_accuracy_*.pdf: LPSE purple, MLP yellow, XGB green,
IEEE serif fonts, no title."""

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

PALETTE = {
    "hybrid": ("#d0cee2", "#56517e"),
    "mlp": ("#fff2cc", "#d6b656"),
    "xgb": ("#d5e8d4", "#82b366"),
}
LABELS = {"hybrid": "LPSE", "mlp": "MLP", "xgb": "XGBoost"}


def _f(v):
    if v is None:
        return math.nan
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _read(csv_path: Path) -> list[tuple[int, float]]:
    runs = [[]]
    last_step = -1
    with csv_path.open(newline="") as h:
        for row in csv.DictReader(h):
            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue
            if step <= last_step:
                runs.append([])
            last_step = step
            runs[-1].append((step, _f(row.get("val_top1"))))
    return max(runs, key=len) if runs and runs[0] else []


def _moving_avg(xs: Sequence[float], w: int) -> list[float]:
    if w <= 1:
        return list(xs)
    out: list[float] = []
    q: deque[float] = deque()
    s = 0.0
    n = 0
    for x in xs:
        if not math.isnan(x):
            q.append(x)
            s += x
            n += 1
            if len(q) > w:
                s -= q.popleft()
                n -= 1
            out.append(s / max(1, n))
        else:
            out.append(math.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-log", default="lpse_training_loghybrid.csv")
    ap.add_argument("--mlp-log", default="baseline_mlp_v2_training_log.csv")
    ap.add_argument("--xgb-log", default="baseline_xgb_v2_training_log.csv")
    ap.add_argument("--output", default="figures/accuracy_curves.pdf")
    ap.add_argument("--max-step", type=int, default=30000)
    ap.add_argument("--smooth-sgd", type=int, default=400)
    args = ap.parse_args()

    paths = {
        "hybrid": (Path(args.hybrid_log), "sgd"),
        "mlp": (Path(args.mlp_log), "sgd"),
        "xgb": (Path(args.xgb_log), "xgb"),
    }
    series = {}
    for k, (p, kind) in paths.items():
        rows = _read(p)
        steps = [s for s, v in rows if s <= args.max_step]
        vals = [v for s, v in rows if s <= args.max_step]
        if kind == "sgd":
            vals_s = _moving_avg(vals, args.smooth_sgd)
        else:
            vals_s = vals

        first = next((i for i, v in enumerate(vals_s) if not math.isnan(v)), None)
        if first is None:
            series[k] = ([], [])
            continue
        steps = steps[first:]
        vals_s = vals_s[first:]
        series[k] = (steps, vals_s)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.patch.set_facecolor("white")
    for k in ("hybrid", "mlp", "xgb"):
        xs, ys = series[k]
        if not xs:
            continue
        fill, line = PALETTE[k]
        ax.plot(xs, ys, color=line, linewidth=1.6, label=LABELS[k])
        ax.fill_between(xs, 0, ys, color=fill, alpha=0.30)

    ax.set_xlim(0, args.max_step)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training step", fontsize=9, color="#444444")
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
                f"{LABELS[k]:8s}  end_step={xs[-1]}  final_val={ys[-1]:.4f}  n={len(xs)}"
            )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
