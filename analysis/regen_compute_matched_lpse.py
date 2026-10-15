#!/usr/bin/env python3
"""Regenerate compute_matched_accuracy_{a,b}.pdf with the legend relabelled
NL-JEPA -> LPSE, changing *nothing else* about the figures.

Unlike the original version of this script (which drove the plotting code with
final values hand-recovered from the committed PDFs), this version derives the
final accuracy / cumulative sample-view numbers from the training logs using
the exact same accounting as plot_compute_matched_accuracy.py. That keeps the
figures reproducible and self-consistent with the accuracy/compute figures,
and fails loudly instead of silently plotting stale numbers if a log is gone.

To keep everything except the legend pixel-stable, the axes are laid out with
the original (wider) "NL-JEPA" tick label so tight_layout reserves the same
left margin, then the y-tick labels are swapped to "LPSE" without re-running
layout. The result overlays the original except for the relabelled tick.
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "plot_compute_matched_accuracy",
    Path(__file__).with_name("plot_compute_matched_accuracy.py"),
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(
        "could not import plot_compute_matched_accuracy.py next to this script"
    )
cma = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cma)

M = cma._MODELS
PAL = cma._PALETTE
ALPHA = cma._FILL_ALPHA
FIG = cma._FIGSIZE
FT, FL, FV = cma._FS_TICK, cma._FS_LABEL, cma._FS_VALUE


LAYOUT_LABELS = {"hybrid": "NL-JEPA", "mlp": "MLP", "xgb": "XGBoost"}
DISPLAY_LABELS = {"hybrid": "LPSE", "mlp": "MLP", "xgb": "XGBoost"}

OUT_A = Path("figures/compute_matched_accuracy_a.pdf")
OUT_B = Path("figures/compute_matched_accuracy_b.pdf")
OUT_A.parent.mkdir(parents=True, exist_ok=True)


def _finals_from_logs(hybrid_log: str, mlp_log: str, xgb_log: str, smooth_radius: int):
    paths = {
        "hybrid": Path(hybrid_log).expanduser(),
        "mlp": Path(mlp_log).expanduser(),
        "xgb": Path(xgb_log).expanduser(),
    }
    kinds = {"hybrid": "sgd", "mlp": "sgd", "xgb": "xgb"}
    finals = {}
    for m, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")
        rows = cma._read_log(p)
        if not rows:
            raise ValueError(f"No rows parsed from {p}")
        cum = cma._cumulative_samples(rows, kinds[m])
        sr = 2 if kinds[m] == "xgb" else max(1, smooth_radius)
        finals[m] = (cma._final_val(rows, sr), cum[-1] if cum else 0.0)
    return finals


def _yticks(ax, y, labels):
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in M], fontsize=FT)
    for tick_label, m in zip(ax.get_yticklabels(), M):
        tick_label.set_color(PAL[m][1])
        tick_label.set_fontweight("bold")


def _finish_and_relabel(fig, ax, y):

    fig.tight_layout(pad=0.4)
    _yticks(ax, y, DISPLAY_LABELS)


def plot_accuracy(out_path, finals):
    y = np.arange(len(M))
    accs = [finals[m][0] for m in M]
    fills = [PAL[m][0] for m in M]
    edges = [PAL[m][1] for m in M]

    fig, ax = plt.subplots(figsize=FIG)
    fig.patch.set_facecolor("white")
    ax.barh(
        y, accs, color=fills, edgecolor=edges, linewidth=1.0, height=0.62, alpha=ALPHA
    )
    for yy, m, v in zip(y, M, accs):
        ax.text(
            v + 0.012,
            yy,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=FV,
            fontweight="bold",
            color=PAL[m][1],
        )
    _yticks(ax, y, LAYOUT_LABELS)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Validation top-1 accuracy", color="#444444", fontsize=FL)
    ax.invert_yaxis()
    cma._setup_axis(ax)
    _finish_and_relabel(fig, ax, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_compute(out_path, finals):
    y = np.arange(len(M))
    comps = [finals[m][1] for m in M]
    fills = [PAL[m][0] for m in M]
    edges = [PAL[m][1] for m in M]

    fig, ax = plt.subplots(figsize=FIG)
    fig.patch.set_facecolor("white")
    ax.barh(
        y, comps, color=fills, edgecolor=edges, linewidth=1.0, height=0.62, alpha=ALPHA
    )
    for yy, m, c in zip(y, M, comps):
        ax.text(
            c * 1.25,
            yy,
            cma._format_compute(c),
            va="center",
            ha="left",
            fontsize=FV,
            fontweight="bold",
            color=PAL[m][1],
        )
    ax.set_xscale("log")
    ax.set_xlim(1e5, max(comps) * 8)
    _yticks(ax, y, LAYOUT_LABELS)
    ax.set_xlabel("Cumulative sample-views (log)", color="#444444", fontsize=FL)
    ax.invert_yaxis()
    cma._setup_axis(ax)
    _finish_and_relabel(fig, ax, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate compute_matched_accuracy_*.pdf with NL-JEPA -> LPSE labels, "
        "deriving final values from the training logs."
    )
    parser.add_argument("--hybrid-log", default="lpse_training_loghybrid.csv")
    parser.add_argument("--mlp-log", default="baseline_mlp_v2_training_log.csv")
    parser.add_argument("--xgb-log", default="baseline_xgb_v2_training_log.csv")
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=200,
        help="Rows averaged for the SGD final accuracy (XGB uses 2).",
    )
    args = parser.parse_args()

    finals = _finals_from_logs(
        args.hybrid_log, args.mlp_log, args.xgb_log, args.smooth_radius
    )

    print("Final state per model (derived from logs):")
    for m, (v, c) in finals.items():
        print(f"  {m:6s}  val_top1={v:.4f}  compute={c:.3e}")

    plot_accuracy(OUT_A, finals)
    plot_compute(OUT_B, finals)
    print(f"Saved (a) accuracy: {OUT_A}")
    print(f"Saved (b) compute:  {OUT_B}")


if __name__ == "__main__":
    main()
