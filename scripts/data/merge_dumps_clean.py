#!/usr/bin/env python3
"""Clean + chronologically-interleave multiple replay dumps.

Fixes found in jepa_v2 that poisoned the naive merge:
  1. Feature-distribution mismatch: jepa_v2 carries an abnormal
     `node_procs_blocked` regime (values up to 4000 vs ~69 max in
     jepa_run) plus a `node_disk_io_time` shift and a rare -90k
     network-counter glitch.  Merging raw lets the metric encoder /
     codebook waste capacity reconciling two incompatible regimes.
     Fix: clip every context value into per-feature robust bounds
     [p_low, p_high] estimated from the reference dump (jepa_run).
  2. Temporal discontinuity: the naive merge concatenated run-then-v2,
     so the timeline jumped ~16h backwards mid-epoch.  Fix: interleave
     snapshots from all dumps in global chronological (timestamp) order.

Usage:
  python scripts/data/merge_dumps_clean.py \
      --ref artifacts/dumps/jepa_run \
      --out artifacts/dumps/jepa_merged_clean \
      artifacts/dumps/jepa_run artifacts/dumps/jepa_v2
"""

import argparse
import glob
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("merge_dumps_clean")


def iter_snapshots(dump_dir: str):
    """Yield (context, is_val, node_ips, node_cores, timestamp, qa_pairs)."""
    shards = sorted(glob.glob(os.path.join(dump_dir, "shard_*.pt")))
    if not shards:
        logger.warning("No shards under %s", dump_dir)
        return
    for shard_path in shards:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        contexts = payload["contexts"]
        is_val = payload["is_val"]
        node_ips = payload["node_ips"]
        node_cores = payload["node_cores"]
        ts = payload["timestamps"]
        qa = payload["qa"]
        per_snapshot: dict[int, list[tuple[dict, int]]] = {}
        for local_idx, q_dict, ans_idx in qa:
            per_snapshot.setdefault(int(local_idx), []).append((q_dict, int(ans_idx)))
        for local_idx in range(contexts.shape[0]):
            yield (
                contexts[local_idx],
                bool(is_val[local_idx].item()),
                node_ips[local_idx],
                node_cores[local_idx],
                float(ts[local_idx].item()),
                per_snapshot.get(local_idx, []),
            )


def estimate_bounds(
    ref_dir: str, low_q: float = 0.001, high_q: float = 0.999
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature robust bounds across the reference dump."""
    logger.info(
        "Estimating per-feature bounds from %s (q=%.3f..%.3f)", ref_dir, low_q, high_q
    )
    samples: list[torch.Tensor] = []
    n = 0
    for p in sorted(glob.glob(os.path.join(ref_dir, "shard_*.pt")))[::4]:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        samples.append(payload["contexts"])
        n += payload["contexts"].shape[0]
    X = torch.cat(samples, dim=0)
    flat = X.reshape(-1, X.shape[-1])
    flat = flat[::8]
    lo = torch.quantile(flat, low_q, dim=0)
    hi = torch.quantile(flat, high_q, dim=0)
    logger.info(
        "Bounds: %d samples, clipped range example f=1127 [%.2f, %.2f]",
        n,
        lo[1127].item(),
        hi[1127].item(),
    )
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="input dump dirs")
    ap.add_argument("--ref", required=True, help="reference dump for feature bounds")
    ap.add_argument("--out", required=True, help="output dump dir")
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument(
        "--no-clip",
        action="store_true",
        help="disable feature clipping (chrono interleave only)",
    )
    args = ap.parse_args()

    lo, hi = None, None
    if not args.no_clip:
        lo, hi = estimate_bounds(args.ref)

    os.makedirs(args.out, exist_ok=True)
    out_shard_idx = 0
    snapshots_written = 0
    qa_written = 0
    contexts: list[torch.Tensor] = []
    is_val: list[bool] = []
    qa: list[tuple[int, dict, int]] = []
    node_ips: list[list[str]] = []
    node_cores: list[list[int]] = []
    timestamps: list[float] = []

    def flush():
        nonlocal out_shard_idx, snapshots_written, qa_written
        if not contexts:
            return
        path = os.path.join(args.out, f"shard_{out_shard_idx:06d}.pt")
        payload = {
            "contexts": torch.stack(contexts, dim=0),
            "is_val": torch.tensor(is_val, dtype=torch.bool),
            "qa": qa,
            "node_ips": node_ips,
            "node_cores": node_cores,
            "timestamps": torch.tensor(timestamps, dtype=torch.float64),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        snapshots_written += len(contexts)
        qa_written += len(qa)
        logger.info(
            "Wrote %s (snapshots=%d qa=%d)",
            os.path.basename(path),
            len(contexts),
            len(qa),
        )
        out_shard_idx += 1
        contexts.clear()
        is_val.clear()
        qa.clear()
        node_ips.clear()
        node_cores.clear()
        timestamps.clear()

    def emit(snap):
        nonlocal snapshots_written
        ctx, isv, ips, cores, ts, pairs = snap
        if lo is not None:
            ctx = torch.clamp(ctx, lo, hi)
        local_idx = len(contexts)
        contexts.append(ctx)
        is_val.append(isv)
        node_ips.append(list(ips))
        node_cores.append(list(cores))
        timestamps.append(ts)
        for q_dict, ans_idx in pairs:
            qa.append((local_idx, q_dict, ans_idx))
        if len(contexts) >= args.shard_size:
            flush()

    streams = []
    for dump in args.dumps:
        it = iter_snapshots(dump)
        streams.append(it)

    pending = []
    for s_idx, it in enumerate(streams):
        try:
            nxt = next(it)
            pending.append((nxt[4], s_idx, nxt))
        except StopIteration:
            pass

    node_universe = set()
    n_total = 0
    while pending:
        pending.sort(key=lambda x: x[0])
        _ts, s_idx, snap = pending.pop(0)
        node_universe.update(snap[2])
        emit(snap)
        n_total += 1
        try:
            nxt = next(streams[s_idx])
            pending.append((nxt[4], s_idx, nxt))
        except StopIteration:
            pass
        if n_total % 20000 == 0:
            logger.info("Interleaved %d snapshots...", n_total)
    flush()

    meta = {
        "snapshots_written": snapshots_written,
        "qa_pairs_written": qa_written,
        "shards_written": out_shard_idx,
        "shard_size": args.shard_size,
        "node_universe": sorted(node_universe),
        "source_dumps": args.dumps,
        "ref_dump": args.ref,
        "clipped": not args.no_clip,
        "bounds_q": None if args.no_clip else [0.001, 0.999],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Clean merged dataset: %d snapshots, %d QA pairs, %d shards, %d nodes",
        snapshots_written,
        qa_written,
        out_shard_idx,
        len(node_universe),
    )


if __name__ == "__main__":
    main()
