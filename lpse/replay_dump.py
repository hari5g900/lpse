"""Sharded on-disk dump of (context, QA, metadata) triples produced by the JEPA trainer and consumed by the baseline trainers.

Layout under DUMP_DIR/:
    shard_000000.pt, shard_000001.pt, ...   Atomic via .pt.tmp -> rename.
    meta.json                               Cumulative counts (rewritten on flush).

Each shard is a torch.save of:
    contexts:    Tensor[N, T, F]
    is_val:      Tensor[N], bool
    qa:          List[(snapshot_idx_in_shard, q_dict, ans_idx)]
    node_ips:    List[List[str]]
    node_cores:  List[List[int]]
    timestamps:  Tensor[N]

Contexts are deduped within a shard (many QA pairs share one context).
Partial shards flush via atexit on process exit.
"""

from __future__ import annotations

import atexit
import glob
import json
import logging
import os
import time
from collections.abc import Iterator

import torch

logger = logging.getLogger("replay_dump")


class DumpWriter:
    """Accumulates snapshots into shards and atomically flushes them."""

    def __init__(self, dump_dir: str, shard_size: int = 100):
        self.dump_dir = os.path.abspath(dump_dir)
        os.makedirs(self.dump_dir, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self._next_shard_idx = self._discover_next_shard_idx()

        self._contexts: list[torch.Tensor] = []
        self._is_val: list[bool] = []
        self._qa: list[tuple[int, dict, int]] = []
        self._node_ips: list[list[str]] = []
        self._node_cores: list[list[int]] = []
        self._timestamps: list[float] = []

        self._snapshots_written = 0
        self._qa_pairs_written = 0

        atexit.register(self._flush_safely)
        logger.info(
            "DumpWriter ready dir=%s shard_size=%d next_shard_idx=%d",
            self.dump_dir,
            self.shard_size,
            self._next_shard_idx,
        )

    def _discover_next_shard_idx(self) -> int:
        existing = sorted(glob.glob(os.path.join(self.dump_dir, "shard_*.pt")))
        if not existing:
            return 0
        last = os.path.basename(existing[-1])
        try:
            return int(last[len("shard_") : -len(".pt")]) + 1
        except ValueError:
            return len(existing)

    def add_snapshot(
        self,
        context_seq: torch.Tensor,
        qa_pairs: list[tuple[dict, int]],
        is_val: bool,
        node_ips: list[str],
        node_cores: list[int],
        timestamp: float | None = None,
    ):
        """Stage one snapshot's context tensor and the QA pairs derived from it."""
        local_idx = len(self._contexts)
        self._contexts.append(context_seq.detach().cpu())
        self._is_val.append(bool(is_val))
        self._node_ips.append(list(node_ips))
        self._node_cores.append(list(node_cores))
        self._timestamps.append(
            float(timestamp if timestamp is not None else time.time())
        )
        for q_dict, ans_idx in qa_pairs:
            self._qa.append((local_idx, q_dict, int(ans_idx)))

        if len(self._contexts) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self._contexts:
            return
        shard_idx = self._next_shard_idx
        path = os.path.join(self.dump_dir, f"shard_{shard_idx:06d}.pt")
        tmp = path + ".tmp"
        try:
            payload = {
                "contexts": torch.stack(self._contexts, dim=0),
                "is_val": torch.tensor(self._is_val, dtype=torch.bool),
                "qa": self._qa,
                "node_ips": self._node_ips,
                "node_cores": self._node_cores,
                "timestamps": torch.tensor(self._timestamps, dtype=torch.float64),
            }
            torch.save(payload, tmp)
            os.replace(tmp, path)
            self._snapshots_written += len(self._contexts)
            self._qa_pairs_written += len(self._qa)
            logger.info(
                "Flushed shard %s (snapshots=%d qa=%d cumulative_snapshots=%d cumulative_qa=%d)",
                os.path.basename(path),
                len(self._contexts),
                len(self._qa),
                self._snapshots_written,
                self._qa_pairs_written,
            )
            self._next_shard_idx += 1
            self._contexts.clear()
            self._is_val.clear()
            self._qa.clear()
            self._node_ips.clear()
            self._node_cores.clear()
            self._timestamps.clear()
            self._write_meta()
        except Exception as e:
            logger.error("Failed to flush shard %s: %s", path, e)
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _write_meta(self):
        meta = {
            "snapshots_written": self._snapshots_written,
            "qa_pairs_written": self._qa_pairs_written,
            "shards_written": self._next_shard_idx,
            "shard_size": self.shard_size,
            "last_flush_ts": time.time(),
        }
        try:
            with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write meta.json: %s", e)

    def _flush_safely(self):
        try:
            self._flush()
        except Exception as e:
            logger.error("atexit flush failed: %s", e)


def iter_dump(
    dump_dir: str,
) -> Iterator[tuple[torch.Tensor, dict, int, bool, list[str], list[int], float]]:
    """
    Stream every (context_seq, question_dict, answer_idx, is_val, node_ips,
    node_cores, timestamp) tuple from the dump in shard order.
    """
    dump_dir = os.path.abspath(dump_dir)
    shards = sorted(glob.glob(os.path.join(dump_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {dump_dir}")
    for shard_path in shards:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        contexts = payload["contexts"]
        is_val = payload["is_val"]
        node_ips = payload["node_ips"]
        node_cores = payload["node_cores"]
        ts = payload["timestamps"]
        qa = payload["qa"]
        for local_idx, q_dict, ans_idx in qa:
            yield (
                contexts[local_idx],
                q_dict,
                int(ans_idx),
                bool(is_val[local_idx].item()),
                node_ips[local_idx],
                node_cores[local_idx],
                float(ts[local_idx].item()),
            )


def iter_dump_by_snapshot(
    dump_dir: str,
) -> Iterator[
    tuple[torch.Tensor, list[tuple[dict, int]], bool, list[str], list[int], float]
]:
    """Yield one (context, qa_pairs, is_val, node_ips, node_cores, timestamp)
    per snapshot in chronological shard order. Groups the per-QA stream from
    iter_dump() back into snapshots so callers can mimic the live trainer's
    one-step-per-scrape cadence."""
    dump_dir = os.path.abspath(dump_dir)
    shards = sorted(glob.glob(os.path.join(dump_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {dump_dir}")
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
            per_snapshot.setdefault(local_idx, []).append((q_dict, int(ans_idx)))
        for local_idx in range(contexts.shape[0]):
            yield (
                contexts[local_idx],
                per_snapshot.get(local_idx, []),
                bool(is_val[local_idx].item()),
                node_ips[local_idx],
                node_cores[local_idx],
                float(ts[local_idx].item()),
            )


def load_dump_summary(dump_dir: str) -> dict:
    """Return cumulative counts from meta.json plus a shard listing."""
    dump_dir = os.path.abspath(dump_dir)
    meta_path = os.path.join(dump_dir, "meta.json")
    summary: dict = {"shards": sorted(glob.glob(os.path.join(dump_dir, "shard_*.pt")))}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                summary.update(json.load(f))
        except Exception:
            pass
    return summary
