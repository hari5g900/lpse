#!/usr/bin/env python3
"""Evaluate the LPSE, MLP, and XGBoost models on the JEPA dump
(jepa_run), splitting accuracy by question type to fill Table 2 of the
paper.

The dump is streamed in chronological order; held-out val snapshots are
kept in a FIFO buffer (matching the trainer's val_replay capacity). At
end of stream the surviving buffer is the same window the trainer
evaluated against to log val_top1, so per-type accuracy here is directly
comparable to the trainer's reported numbers.

Bucketing follows check_lpse_accuracy.py: questions are split by the
'metric' field of the parsed question dict (cpu/memory/disk/load/process/
health/overall). Contexts in the dump are already normalised, so no
RunningNormalizer is needed at eval time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from infer_lpse import load_model as load_jepa_model
from lpse.prometheus_ingest import CLUSTER_TOKEN_DIM
from lpse.question_encoder import QUESTION_DIM, build_question_encoder, encode_questions
from lpse.replay_dump import iter_dump_by_snapshot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAPER_TYPES = ["cpu", "memory", "disk", "load", "process", "health", "overall"]


@dataclass
class Stat:
    total: int = 0
    correct: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.correct += 1

    def acc(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total


class FlatMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_answers: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_answers))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mlp(path: str) -> tuple[nn.Module, nn.Module, int]:
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ck["model_config"]
    mlp = FlatMLP(
        in_dim=cfg["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["hidden_layers"],
        num_answers=cfg["num_answers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    mlp.load_state_dict(ck["model"])
    mlp.eval()
    backend = ck.get("q_encoder_backend", "hybrid")
    qe = build_question_encoder(backend=backend, out_dim=QUESTION_DIM).to(DEVICE)
    qe.load_state_dict(ck.get("question_encoder_state", {}), strict=False)
    qe.eval()
    seq_len = int(ck.get("seq_len", 16))
    return mlp, qe, seq_len


def load_xgb(model_path: str, encoder_path: str) -> tuple[xgb.Booster, nn.Module]:
    booster = xgb.Booster()
    booster.load_model(model_path)
    blob = torch.load(encoder_path, map_location=DEVICE, weights_only=False)
    backend = blob.get("q_encoder_backend", "hybrid")
    qe = build_question_encoder(backend=backend, out_dim=QUESTION_DIM).to(DEVICE)
    qe.load_state_dict(blob.get("question_encoder_state", {}), strict=False)
    qe.eval()
    return booster, qe


def categorise(metric: str) -> str | None:
    """Map a parsed-question metric to one of the paper's row labels.
    Anything outside the bucket list is returned as None and only counted
    in the "All types" aggregate."""
    m = (metric or "").lower()
    if m in PAPER_TYPES:
        return m
    return None


def run_eval(
    dump_dir: str,
    jepa_ckpt: str,
    mlp_ckpt: str,
    xgb_model: str,
    xgb_encoder: str,
    val_qa_capacity: int,
    val_context_capacity: int,
    max_per_type: int,
    log_every: int,
) -> dict:
    print(f"Loading LPSE from {jepa_ckpt}")
    jepa, qe_jepa, _norm, _ips, _cores, _ans = load_jepa_model(jepa_ckpt)
    jepa.eval()
    qe_jepa.eval()
    print(f"Loading MLP from {mlp_ckpt}")
    mlp, qe_mlp, seq_len = load_mlp(mlp_ckpt)
    print(f"Loading XGBoost from {xgb_model} / {xgb_encoder}")
    booster, qe_xgb = load_xgb(xgb_model, xgb_encoder)

    flat_dim = seq_len * CLUSTER_TOKEN_DIM
    print(f"seq_len={seq_len}  flat_dim={flat_dim}  q_dim={QUESTION_DIM}")

    from collections import OrderedDict, deque

    val_ctx: OrderedDict[int, tuple[torch.Tensor, list[str]]] = OrderedDict()
    val_qa: deque[tuple[int, dict, int]] = deque(maxlen=val_qa_capacity)
    next_id = 0

    streamed = 0
    val_snaps = 0
    t0 = time.time()
    for ctx, qa_pairs, is_val, ips, _cores, _ts in iter_dump_by_snapshot(dump_dir):
        streamed += 1
        if not is_val:
            continue
        val_snaps += 1
        cid = next_id
        next_id += 1
        val_ctx[cid] = (ctx, list(ips) if ips else [])
        while len(val_ctx) > val_context_capacity:
            val_ctx.popitem(last=False)
        for q, a in qa_pairs:
            val_qa.append((cid, q, int(a)))
        if log_every and streamed % log_every == 0:
            print(
                f"streamed snapshots={streamed} val_snaps={val_snaps} live_val_qa={len(val_qa)}"
            )

    live = [(cid, q, a) for cid, q, a in val_qa if cid in val_ctx]
    print(
        f"\nFinished streaming. snapshots={streamed} val_snaps={val_snaps} "
        f"live_val_qa={len(live)} unique_contexts={len(val_ctx)} "
        f"stream_elapsed={time.time() - t0:.1f}s"
    )

    by_cat: dict[str, list[tuple[int, dict, int]]] = {t: [] for t in PAPER_TYPES}
    extra_metrics: dict[str, int] = defaultdict(int)
    for cid, q, a in live:
        cat = categorise(q.get("metric") or "")
        if cat is None:
            extra_metrics[str(q.get("metric")).lower()] += 1
            continue
        if max_per_type > 0 and len(by_cat[cat]) >= max_per_type:
            continue
        by_cat[cat].append((cid, q, a))
    for t in PAPER_TYPES:
        print(f"  {t}: {len(by_cat[t])} samples available for eval")

    jepa_stats: dict[str, Stat] = {t: Stat() for t in PAPER_TYPES}
    mlp_stats: dict[str, Stat] = {t: Stat() for t in PAPER_TYPES}
    xgb_stats: dict[str, Stat] = {t: Stat() for t in PAPER_TYPES}
    jepa_all = Stat()
    mlp_all = Stat()
    xgb_all = Stat()

    qa_pairs_seen = 0
    BATCH = 64
    t1 = time.time()
    for cat in PAPER_TYPES:
        items = by_cat[cat]
        if not items:
            continue
        for start in range(0, len(items), BATCH):
            chunk = items[start : start + BATCH]
            cids = [c[0] for c in chunk]
            qs = [c[1] for c in chunk]
            ans = np.array([c[2] for c in chunk], dtype=np.int64)

            ctxs = torch.stack([val_ctx[cid][0] for cid in cids], dim=0).to(DEVICE)

            last_ips = val_ctx[cids[-1]][1]

            with torch.no_grad():
                ctx_flat = ctxs.flatten(1)

                q_emb_jepa = encode_questions(qe_jepa, qs, last_ips).to(DEVICE)
                _seq_latent, ctx_latent = jepa.metric_encoder(ctxs)
                pred_latent = jepa.predict_answer_latent(ctx_latent, q_emb_jepa)
                jepa_logits = jepa.codebook.logits_from_latent(pred_latent)
                jepa_pred = jepa_logits.argmax(-1).detach().cpu().numpy()

                q_emb_mlp = encode_questions(qe_mlp, qs, last_ips).to(DEVICE)
                x_mlp = torch.cat([ctx_flat, q_emb_mlp], dim=-1)
                mlp_pred = mlp(x_mlp).argmax(-1).detach().cpu().numpy()

                q_emb_xgb = encode_questions(qe_xgb, qs, last_ips).to(DEVICE)
                x_xgb = (
                    torch.cat([ctx_flat, q_emb_xgb], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                xgb_probs = booster.predict(xgb.DMatrix(x_xgb))
                xgb_pred = xgb_probs.argmax(axis=1)

            for i in range(len(chunk)):
                ok_jepa = bool(jepa_pred[i] == ans[i])
                ok_mlp = bool(mlp_pred[i] == ans[i])
                ok_xgb = bool(xgb_pred[i] == ans[i])
                jepa_all.add(ok_jepa)
                mlp_all.add(ok_mlp)
                xgb_all.add(ok_xgb)
                jepa_stats[cat].add(ok_jepa)
                mlp_stats[cat].add(ok_mlp)
                xgb_stats[cat].add(ok_xgb)
                qa_pairs_seen += 1

    elapsed = time.time() - t0
    print(
        f"\nScored {qa_pairs_seen} QA pairs in {time.time() - t1:.1f}s "
        f"(total elapsed {elapsed:.1f}s)"
    )

    def _row(stats: dict[str, Stat], all_stat: Stat) -> dict:
        row = {}
        for t in PAPER_TYPES:
            s = stats[t]
            row[t] = {
                "total": s.total,
                "correct": s.correct,
                "acc": (None if (a := s.acc()) is None else float(a)),
            }
        row["all_types"] = {
            "total": all_stat.total,
            "correct": all_stat.correct,
            "acc": (None if (a := all_stat.acc()) is None else float(a)),
        }
        return row

    summary = {
        "dump_dir": dump_dir,
        "jepa_checkpoint": jepa_ckpt,
        "mlp_checkpoint": mlp_ckpt,
        "xgb_model": xgb_model,
        "xgb_encoder": xgb_encoder,
        "val_qa_capacity": val_qa_capacity,
        "val_context_capacity": val_context_capacity,
        "max_per_type": max_per_type,
        "stream_snapshots": streamed,
        "val_snapshots_streamed": val_snaps,
        "live_val_qa_at_end": len(live),
        "qa_pairs_evaluated": qa_pairs_seen,
        "elapsed_seconds": elapsed,
        "jepa": _row(jepa_stats, jepa_all),
        "mlp": _row(mlp_stats, mlp_all),
        "xgb": _row(xgb_stats, xgb_all),
        "uncategorised_metric_counts": dict(extra_metrics),
    }

    def _print(label: str, stats: dict[str, Stat], all_stat: Stat) -> None:
        print(f"\n=== {label} per-type accuracy ===")
        for t in PAPER_TYPES:
            s = stats[t]
            a = s.acc()
            print(
                f"  {t:8s} {s.correct:5d}/{s.total:5d}  acc={'n/a' if a is None else f'{a * 100:.2f}%'}"
            )
        a = all_stat.acc()
        print(
            f"  {'All':8s} {all_stat.correct:5d}/{all_stat.total:5d}  acc={'n/a' if a is None else f'{a * 100:.2f}%'}"
        )

    _print("LPSE", jepa_stats, jepa_all)
    _print("MLP", mlp_stats, mlp_all)
    _print("XGBoost", xgb_stats, xgb_all)

    if extra_metrics:
        print("\nUncategorised metric counts (only counted in All types):")
        for k, v in sorted(extra_metrics.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")

    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", default="artifacts/dumps/jepa_run")
    p.add_argument("--jepa-checkpoint", default="checkpoints/nljepahybrid.pt")
    p.add_argument("--mlp-checkpoint", default="checkpoints/baseline_mlp_v2.pt")
    p.add_argument("--xgb-model", default="checkpoints/baseline_xgb_v2.json")
    p.add_argument("--xgb-encoder", default="checkpoints/baseline_xgb_v2_encoder.pt")
    p.add_argument(
        "--val-qa-capacity",
        type=int,
        default=4096,
        help="FIFO val_qa buffer capacity (mirror of trainer's VAL_QA_CAPACITY).",
    )
    p.add_argument(
        "--val-context-capacity",
        type=int,
        default=512,
        help="FIFO val_context buffer capacity (mirror of trainer's VAL_CONTEXT_CAPACITY).",
    )
    p.add_argument(
        "--max-per-type",
        type=int,
        default=100,
        help="Stop sampling a category once this many examples have been scored. 0 = no cap.",
    )
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--output-json", default="baselines_per_type_summary.json")
    args = p.parse_args()

    summary = run_eval(
        dump_dir=args.dump_dir,
        jepa_ckpt=args.jepa_checkpoint,
        mlp_ckpt=args.mlp_checkpoint,
        xgb_model=args.xgb_model,
        xgb_encoder=args.xgb_encoder,
        val_qa_capacity=args.val_qa_capacity,
        val_context_capacity=args.val_context_capacity,
        max_per_type=args.max_per_type,
        log_every=args.log_every,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
