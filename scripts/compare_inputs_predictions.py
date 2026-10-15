#!/usr/bin/env python3
"""Compare llm_inputs.txt vs llm_predictions.txt.

Assumptions based on `test_llm.py`:
- `llm_inputs.txt` is NDJSON; each line is a dict: ip -> list[list[float]] (time-major history).
- `llm_predictions.txt` is NDJSON; each line ideally is ip -> {"t+1": [...], "t+2": [...], "t+3": [...]}
  but may contain malformed JSON, empty objects, error objects, nulls, or alternate schemas.

This script:
- Streams both files (does not load into memory).
- Aligns records by line index (first prediction line with first input line, etc.).
- Uses input[i+1..i+3] last-timestep values as ground-truth for horizons t+1..t+3.
- Skips prediction records with structural issues (configurable strictness).

Usage:
  python compare_inputs_predictions.py --inputs llm_inputs.txt --predictions llm_predictions.txt

"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

HORIZONS: tuple[int, int, int] = (1, 2, 3)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _reduce_vec(vec: list[float], core_index: int | None) -> float | None:
    """Reduce a per-core vector to a single scalar for plotting.

    If core_index is provided, pick that core; otherwise return mean.
    Returns None if core_index is out of range.
    """
    if core_index is None:
        return _mean(vec)
    if core_index < 0 or core_index >= len(vec):
        return None
    return float(vec[core_index])


def _is_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def _iter_nonempty_lines(path: str) -> Iterator[tuple[int, str]]:
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            yield lineno, line


def _parse_json(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def parse_input_record(obj: Any) -> dict[str, list[list[float]]] | None:
    """Validate and normalize one input record."""
    if not isinstance(obj, dict) or not obj:
        return None

    out: dict[str, list[list[float]]] = {}
    for ip, series in obj.items():
        if not isinstance(ip, str):
            return None
        if not isinstance(series, list) or not series:
            return None

        norm_series: list[list[float]] = []
        for row in series:
            if not isinstance(row, list) or not row:
                return None
            if any(not _is_number(v) for v in row):
                return None
            norm_series.append([float(v) for v in row])
        out[ip] = norm_series

    return out


def _parse_pred_vector(vec: Any) -> list[float] | None:
    if not isinstance(vec, list) or not vec:
        return None
    if any(not _is_number(v) for v in vec):
        return None
    return [float(v) for v in vec]


def parse_prediction_record(obj: Any) -> dict[str, dict[int, list[float]]] | None:
    """Normalize one prediction record.

    Returns: ip -> {1: vec, 2: vec, 3: vec}

    Accepted schemas per ip:
    - {"t+1": [...], "t+2": [...], "t+3": [...]}  (preferred)
    - [[...], [...], [...]]  (index 0..2 correspond to horizons 1..3)

    Rejects records like: {}, {"error": ...}, or any record that isn't a dict.
    """
    if not isinstance(obj, dict) or not obj:
        return None
    if "error" in obj:
        return None

    out: dict[str, dict[int, list[float]]] = {}
    for ip, payload in obj.items():
        if not isinstance(ip, str):
            continue

        if isinstance(payload, dict):
            t1 = _parse_pred_vector(payload.get("t+1"))
            t2 = _parse_pred_vector(payload.get("t+2"))
            t3 = _parse_pred_vector(payload.get("t+3"))
            if t1 and t2 and t3:
                out[ip] = {1: t1, 2: t2, 3: t3}
            continue

        if isinstance(payload, list) and len(payload) >= 3:
            v1 = _parse_pred_vector(payload[0])
            v2 = _parse_pred_vector(payload[1])
            v3 = _parse_pred_vector(payload[2])
            if v1 and v2 and v3:
                out[ip] = {1: v1, 2: v2, 3: v3}
            continue

    return out or None


@dataclass
class Agg:
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    n: int = 0

    def add_vecs(self, pred: list[float], truth: list[float]) -> None:
        m = min(len(pred), len(truth))
        for i in range(m):
            e = pred[i] - truth[i]
            self.abs_sum += abs(e)
            self.sq_sum += e * e
            self.n += 1

    def mae(self) -> float | None:
        return None if self.n == 0 else self.abs_sum / self.n

    def rmse(self) -> float | None:
        return None if self.n == 0 else math.sqrt(self.sq_sum / self.n)


@dataclass
class PlotSeries:
    actual: dict[str, list[float | None]]
    pred_by_h: dict[int, dict[str, list[float | None]]]

    def ensure_len(self, ip: str, length: int) -> None:
        a = self.actual.setdefault(ip, [])
        if len(a) < length:
            a.extend([None] * (length - len(a)))
        for h in HORIZONS:
            p = self.pred_by_h.setdefault(h, {}).setdefault(ip, [])
            if len(p) < length:
                p.extend([None] * (length - len(p)))


def _write_plots(
    series: PlotSeries,
    out_dir: str,
    core_index: int | None,
    only_ips: list[str] | None,
) -> None:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ips = sorted(series.actual.keys())
    if only_ips:
        only = set(only_ips)
        ips = [ip for ip in ips if ip in only]

    suffix = "mean" if core_index is None else f"core{core_index}"

    for ip in ips:
        actual = series.actual.get(ip, [])
        if not actual:
            continue

        max_len = len(actual)
        for h in HORIZONS:
            max_len = max(max_len, len(series.pred_by_h.get(h, {}).get(ip, [])))

        def _xy(vals: list[float | None]):
            x_out = []
            y_out = []
            for x, y in enumerate(vals[:max_len]):
                if y is None:
                    continue
                x_out.append(x)
                y_out.append(y)
            return x_out, y_out

        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(f"{ip} actual vs predictions ({suffix})")
        ax.set_xlabel("time index (input line)")
        ax.set_ylabel("cpu utilization")

        xa, ya = _xy(actual)
        if xa:
            ax.plot(xa, ya, label="actual", linewidth=1.6)

        for h in HORIZONS:
            preds = series.pred_by_h.get(h, {}).get(ip, [])
            xp, yp = _xy(preds)
            if xp:
                ax.plot(xp, yp, label=f"pred t+{h}", linewidth=1.2, linestyle="--")

        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")

        out_path = os.path.join(out_dir, f"compare_{ip}_{suffix}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def _rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values
    out: list[float | None] = [None] * len(values)
    buf: list[float] = []
    buf_sum = 0.0
    for i, v in enumerate(values):
        if v is not None:
            buf.append(v)
            buf_sum += v

        if len(buf) > window:
            buf_sum -= buf.pop(0)
        if buf:
            out[i] = buf_sum / len(buf)
    return out


def _write_avg_diff_plot(
    diffs_by_h: dict[int, list[float | None]],
    out_path: str,
    rolling: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title("Average absolute difference (MAE) vs time")
    ax.set_xlabel("time index (prediction made at i)")
    ax.set_ylabel("mean absolute error")
    ax.grid(True, alpha=0.2)

    for h in HORIZONS:
        vals = diffs_by_h.get(h, [])
        vals = _rolling_mean(vals, rolling)

        xs = []
        ys = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            xs.append(i)
            ys.append(v)
        if xs:
            label = f"t+{h}"
            if rolling and rolling > 1:
                label += f" (roll={rolling})"
            ax.plot(xs, ys, label=label, linewidth=1.3)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare inputs vs predictions (skip malformed outputs)."
    )
    ap.add_argument(
        "--inputs", default="llm_inputs.txt", help="Path to llm_inputs.txt (NDJSON)"
    )
    ap.add_argument(
        "--predictions",
        default="llm_predictions.txt",
        help="Path to llm_predictions.txt (NDJSON)",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Limit number of aligned input/prediction lines to scan (0=all)",
    )
    ap.add_argument(
        "--require-all-nodes",
        action="store_true",
        help="If set, skip a prediction record unless it has valid predictions for every node in the input.",
    )
    ap.add_argument(
        "--require-all-horizons",
        action="store_true",
        help="If set, skip a prediction record unless it has at least one node with all 3 horizons valid.",
    )
    ap.add_argument(
        "--plot-dir",
        default="",
        help="If set, write PNG plots into this directory (e.g. plots/)",
    )
    ap.add_argument(
        "--plot-core-index",
        type=int,
        default=-1,
        help="If >=0, plot that single core index instead of mean across cores.",
    )
    ap.add_argument(
        "--plot-ip",
        action="append",
        default=None,
        help="Limit plots to a specific IP (repeatable). If omitted, plots all IPs.",
    )
    ap.add_argument(
        "--diff-plot",
        action="store_true",
        help="If set, write a PNG showing average absolute difference (MAE) over time.",
    )
    ap.add_argument(
        "--diff-plot-path",
        default="",
        help="Output path for diff plot PNG. If empty and --plot-dir is set, writes <plot-dir>/avg_diff.png",
    )
    ap.add_argument(
        "--diff-rolling",
        type=int,
        default=1,
        help="Rolling mean window (in samples) for diff plot smoothing.",
    )

    args = ap.parse_args()

    core_index: int | None = None if args.plot_core_index < 0 else args.plot_core_index
    want_plots = bool(args.plot_dir)
    want_diff_plot = bool(args.diff_plot)

    input_iter = _iter_nonempty_lines(args.inputs)
    pred_iter = _iter_nonempty_lines(args.predictions)

    window: list[dict[str, list[list[float]]] | None] = []
    input_line_numbers: list[int] = []

    bad_input = 0

    def read_next_input() -> bool:
        nonlocal bad_input
        try:
            lineno, line = next(input_iter)
        except StopIteration:
            return False
        obj = _parse_json(line)
        rec = parse_input_record(obj) if obj is not None else None
        if rec is None:
            bad_input += 1
        window.append(rec)
        input_line_numbers.append(lineno)
        return True

    while len(window) < 4 and read_next_input():
        pass

    total_inputs_seen = len(window)
    total_preds_seen = 0

    skipped_pred_json = 0
    skipped_pred_structure = 0
    skipped_no_future_truth = 0
    aligned = 0
    processed = 0

    agg_by_h = {h: Agg() for h in HORIZONS}
    plot_series = PlotSeries(actual={}, pred_by_h={})
    diffs_by_h: dict[int, list[float | None]] = {h: [] for h in HORIZONS}

    while len(window) == 4:
        if args.max_records and aligned >= args.max_records:
            break
        aligned += 1

        try:
            _, pred_line = next(pred_iter)
            total_preds_seen += 1
        except StopIteration:
            break

        in0, in1, in2, in3 = window
        if in0 is None or in1 is None or in2 is None or in3 is None:
            skipped_no_future_truth += 1
        else:
            if want_plots:
                idx = aligned - 1
                for ip, series0 in in0.items():
                    v = _reduce_vec(series0[-1], core_index)
                    plot_series.ensure_len(ip, idx + 1)
                    if v is not None:
                        plot_series.actual[ip][idx] = v

            pred_obj = _parse_json(pred_line)
            if pred_obj is None:
                skipped_pred_json += 1
            else:
                pred_norm = parse_prediction_record(pred_obj)
                if pred_norm is None:
                    skipped_pred_structure += 1
                else:
                    if args.require_all_horizons:
                        ok_any = any(
                            all(h in per_h for h in HORIZONS)
                            for per_h in pred_norm.values()
                        )
                        if not ok_any:
                            skipped_pred_structure += 1
                            pred_norm = None

                    if pred_norm is not None and args.require_all_nodes:
                        needed = set(in0.keys())
                        ok = True
                        for ip in needed:
                            per_h = pred_norm.get(ip)
                            if per_h is None or any(h not in per_h for h in HORIZONS):
                                ok = False
                                break
                        if not ok:
                            skipped_pred_structure += 1
                            pred_norm = None

                if pred_norm is not None:
                    truth_by_h = {
                        1: {ip: in1[ip][-1] for ip in in1.keys()},
                        2: {ip: in2[ip][-1] for ip in in2.keys()},
                        3: {ip: in3[ip][-1] for ip in in3.keys()},
                    }

                    if want_plots:
                        idx = aligned - 1
                        for ip, per_h in pred_norm.items():
                            for h in HORIZONS:
                                pred_vec = per_h.get(h)
                                if pred_vec is None:
                                    continue
                                pv = _reduce_vec(pred_vec, core_index)
                                if pv is None:
                                    continue
                                target = idx + h
                                plot_series.ensure_len(ip, target + 1)
                                plot_series.pred_by_h[h][ip][target] = pv

                    for ip, per_h in pred_norm.items():
                        for h in HORIZONS:
                            pred_vec = per_h.get(h)
                            truth_vec = truth_by_h[h].get(ip)
                            if pred_vec is None or truth_vec is None:
                                continue

                            agg_by_h[h].add_vecs(pred_vec, truth_vec)

                    if want_diff_plot:
                        for h in HORIZONS:
                            err_sum = 0.0
                            err_n = 0
                            for ip, per_h in pred_norm.items():
                                pred_vec = per_h.get(h)
                                truth_vec = truth_by_h[h].get(ip)
                                if pred_vec is None or truth_vec is None:
                                    continue
                                m = min(len(pred_vec), len(truth_vec))
                                for i in range(m):
                                    err_sum += abs(pred_vec[i] - truth_vec[i])
                                    err_n += 1
                            diffs_by_h[h].append(
                                None if err_n == 0 else (err_sum / err_n)
                            )
                    else:
                        for h in HORIZONS:
                            pass

                    processed += 1
                else:
                    if want_diff_plot:
                        for h in HORIZONS:
                            diffs_by_h[h].append(None)
            if want_diff_plot and pred_obj is None:
                for h in HORIZONS:
                    diffs_by_h[h].append(None)
        if want_diff_plot and (
            in0 is None or in1 is None or in2 is None or in3 is None
        ):
            for h in HORIZONS:
                diffs_by_h[h].append(None)

        window.pop(0)
        input_line_numbers.pop(0)
        if read_next_input():
            total_inputs_seen += 1
        else:
            break

    print("=== inputs vs predictions comparison ===")
    print(f"inputs_path={args.inputs}")
    print(f"predictions_path={args.predictions}")
    print(f"inputs_seen={total_inputs_seen} (bad_inputs={bad_input})")
    print(f"predictions_seen={total_preds_seen}")
    print(f"aligned_pairs={aligned}")
    print(f"processed_pairs={processed}")
    print(f"skipped_pred_json={skipped_pred_json}")
    print(f"skipped_pred_structure={skipped_pred_structure}")
    print(f"skipped_no_future_truth={skipped_no_future_truth}")

    for h in HORIZONS:
        mae = agg_by_h[h].mae()
        rmse = agg_by_h[h].rmse()
        mae_s = "n/a" if mae is None else f"{mae:.6f}"
        rmse_s = "n/a" if rmse is None else f"{rmse:.6f}"
        print(f"t+{h}: n={agg_by_h[h].n} mae={mae_s} rmse={rmse_s}")

    if want_plots:
        _write_plots(
            plot_series, args.plot_dir, core_index=core_index, only_ips=args.plot_ip
        )
        print(f"plots_written_dir={args.plot_dir}")

    if want_diff_plot:
        if args.diff_plot_path:
            diff_out = args.diff_plot_path
        elif args.plot_dir:
            diff_out = os.path.join(args.plot_dir, "avg_diff.png")
        else:
            diff_out = "avg_diff.png"

        _write_avg_diff_plot(
            diffs_by_h, diff_out, rolling=max(1, int(args.diff_rolling))
        )
        print(f"diff_plot_written={diff_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
