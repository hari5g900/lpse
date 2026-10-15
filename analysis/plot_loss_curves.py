#!/usr/bin/env python3
"""Loss-curve figure for LPSE vs. the MLP baseline. A single panel
overlaying total_loss across training steps for both models.
Style matches sample_efficiency.pdf / compute_matched_accuracy_*.pdf:
LPSE purple palette, MLP cream palette, IEEE serif fonts, log-y axis,
smoothed line with a faint raw trace underneath. No titles.

Both curves are drawn from the same canonical runs used everywhere else
in the paper: lpse_training_loghybrid.csv and baseline_mlp_v2_training_log.csv
(MLP checkpoint baseline_mlp_v2.pt, the default in every eval/plot script)."""

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

JEPA_FILL = "#d0cee2"
JEPA_LINE = "#56517e"
MLP_FILL = "#fff2cc"
MLP_LINE = "#d6b656"


def _f(v: str) -> float:
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
            runs[-1].append(
                (
                    step,
                    _f(row.get("qa_loss", "")),
                    _f(row.get("ssl_loss", "")),
                    _f(row.get("var_loss", "")),
                    _f(row.get("cov_loss", "")),
                    _f(row.get("total_loss", "")),
                )
            )
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


def _style(ax, ylabel: str, *, show_xlabel: bool = True):
    ax.set_yscale("log")
    if show_xlabel:
        ax.set_xlabel("Training step", fontsize=9, color="#444444")
    ax.set_ylabel(ylabel, fontsize=9, color="#444444")
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(axis="both", colors="#666666", labelsize=8)


def _total_series(
    csv_path: Path, max_step: int, smooth: int
) -> tuple[list[int], list[float], list[float]]:
    """Return (steps, raw total_loss, smoothed total_loss) capped at max_step."""
    rows = _read(csv_path)
    if not rows:
        raise SystemExit(f"no rows parsed from {csv_path}")
    steps = [r[0] for r in rows if r[0] <= max_step]
    tot = [r[5] for r in rows if r[0] <= max_step]
    return steps, tot, _moving_avg(tot, smooth)


def _normalise(
    raw: Sequence[float], smoothed: Sequence[float], window: int
) -> tuple[list[float], list[float], float]:
    """Divide both raw and smoothed by the run's initial loss so the curve
    starts at 1.0 and expresses the *fraction* of the initial loss remaining.
    The baseline is the mean of the first `window` valid raw points rather than
    a single noisy step-0 sample; the spike-prone tail never enters it."""
    head = [v for v in raw[:window] if not math.isnan(v)]
    base = sum(head) / len(head) if head else math.nan
    if not base or math.isnan(base):
        return list(raw), list(smoothed), math.nan

    def norm(xs):
        return [v / base if not math.isnan(v) else math.nan for v in xs]

    return norm(raw), norm(smoothed), base


def _report(name: str, steps: Sequence[int], tot: Sequence[float]) -> None:
    print(f"{name}: steps {steps[0]}..{steps[-1]}  (n={len(steps)})")
    clean = [v for v in tot if not math.isnan(v)]
    if clean:
        print(
            f"  total: min={min(clean):.4g}  max={max(clean):.4g}  "
            f"last10_mean={sum(clean[-10:]) / 10:.4g}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="lpse_training_loghybrid.csv")
    ap.add_argument("--mlp-log", default="baseline_mlp_v2_training_log.csv")
    ap.add_argument("--output", default="figures/loss_curves.pdf")
    ap.add_argument("--smooth", type=int, default=100)
    ap.add_argument(
        "--norm-window",
        type=int,
        default=50,
        help="steps averaged for each run's initial-loss baseline",
    )
    args = ap.parse_args()

    MAX_STEP = 5000
    j_steps, j_tot, j_tot_s = _total_series(Path(args.log), MAX_STEP, args.smooth)
    m_steps, m_tot, m_tot_s = _total_series(Path(args.mlp_log), MAX_STEP, args.smooth)

    j_tot, j_tot_s, j_base = _normalise(j_tot, j_tot_s, args.norm_window)
    m_tot, m_tot_s, m_base = _normalise(m_tot, m_tot_s, args.norm_window)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.patch.set_facecolor("white")

    ax.plot(m_steps, m_tot, color=MLP_FILL, linewidth=0.6, alpha=0.9)
    ax.plot(j_steps, j_tot, color=JEPA_FILL, linewidth=0.6, alpha=0.9)
    ax.plot(m_steps, m_tot_s, color=MLP_LINE, linewidth=1.4, label="MLP")
    ax.plot(j_steps, j_tot_s, color=JEPA_LINE, linewidth=1.4, label="LPSE")

    _style(ax, r"$\mathcal{L}_{\mathrm{total}}\,/\,\mathcal{L}_0$", show_xlabel=True)
    ax.set_xlim(0, MAX_STEP)

    leg = ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.2,
        labelcolor="#444444",
    )
    for line in leg.get_lines():
        line.set_linewidth(1.6)

    fig.tight_layout(pad=0.4)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"baselines (initial loss): LPSE={j_base:.4g}  MLP={m_base:.4g}")
    _report("LPSE (normalised)", j_steps, j_tot)
    _report("MLP  (normalised)", m_steps, m_tot)


if __name__ == "__main__":
    main()
