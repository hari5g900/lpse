#!/usr/bin/env python3
"""Evaluate LPSE accuracy by question type using live cluster telemetry.

This script mirrors the training data path:
- scrape live metrics
- build a temporal context window
- generate QA pairs from the snapshot
- evaluate answer-JEPA predictions

By default it reports accuracy split by metric+intent sub-types.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import torch

from lpse import seed_everything
from lpse.apps.infer_lpse import load_model
from lpse.collectors import get_node_ips, get_num_cores_from_node_exporter
from lpse.prometheus_ingest import (
    cluster_snapshot_to_token,
    collect_full_cluster_snapshot,
    compute_semantic_features,
)
from lpse.qa_generator import generate_qa_pairs
from lpse.question_encoder import encode_question_tokens, encode_questions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed_everything(int(os.getenv("SEED", "1337")))

DEFAULT_TYPES = [
    "cpu",
    "memory",
    "disk",
    "load",
    "process",
    "health",
    "overall",
]


@dataclass
class AccuracyStats:
    total: int = 0
    correct: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1

    def acc(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total


def _discover_live_topology() -> tuple[list[str], list[int]]:
    node_ips = get_node_ips()
    node_cores: list[int] = []
    for ip in node_ips:
        cores = get_num_cores_from_node_exporter(ip)
        if cores <= 0:
            cores = 10
        node_cores.append(cores)
    return node_ips, node_cores


def _types_from_arg(arg: str) -> list[str]:
    if not arg:
        return []
    return [t.strip().lower() for t in arg.split(",") if t.strip()]


def _all_complete(stats: dict[str, AccuracyStats], target: int) -> bool:
    if not stats:
        return False
    return all(s.total >= target for s in stats.values())


def _print_progress(stats: dict[str, AccuracyStats], target: int) -> None:
    if not stats:
        print("progress: (no types yet)")
        return
    parts = []
    for key in sorted(stats.keys()):
        s = stats[key]
        parts.append(f"{key}:{s.total}/{target}")
    print("progress: " + " | ".join(parts))


def _type_key(q_struct: dict, split_by: str) -> str:
    metric = str(q_struct.get("metric", "unknown")).lower()
    intent = str(q_struct.get("intent", "unknown")).lower()
    scope = str(q_struct.get("scope", "unknown")).lower()
    if split_by == "metric":
        return metric
    if split_by == "intent":
        return intent
    if split_by == "scope":
        return scope
    if split_by == "metric_intent":
        return f"{metric}:{intent}"
    if split_by == "metric_intent_scope":
        return f"{metric}:{intent}:{scope}"
    return metric


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate LPSE accuracy split by question type."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("CHECKPOINT_PATH", "checkpoints/lpse3.pt"),
        help="Path to checkpoint (default: checkpoints/lpse3.pt).",
    )
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", "16")))
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=float(os.getenv("SCRAPE_INTERVAL", "1.0")),
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=100,
        help="Number of questions to evaluate per type.",
    )
    parser.add_argument(
        "--split-by",
        choices=["metric", "intent", "scope", "metric_intent", "metric_intent_scope"],
        default="metric_intent",
        help="How to bucket question types.",
    )
    parser.add_argument(
        "--question-types",
        default="",
        help="Comma-separated list of question type keys for the selected split.",
    )
    parser.add_argument(
        "--max-pairs-per-step",
        type=int,
        default=40,
        help="Max QA pairs to generate per snapshot.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Safety cap on snapshot iterations.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Optional time cap in seconds (0 disables).",
    )
    parser.add_argument(
        "--update-normalizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Update running normalizer during evaluation.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N snapshots.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON summary.",
    )

    args = parser.parse_args()

    question_types = _types_from_arg(args.question_types)
    allow_new_types = False
    if question_types:
        stats = {q_type: AccuracyStats() for q_type in question_types}
    elif args.split_by == "metric":
        question_types = list(DEFAULT_TYPES)
        stats = {q_type: AccuracyStats() for q_type in question_types}
    else:
        stats = {}
        allow_new_types = True
    overall = AccuracyStats()

    model, question_encoder, normalizer, _ips, _cores, _answers = load_model(
        args.checkpoint
    )
    model.eval()
    question_encoder.eval()

    buffer = deque(maxlen=args.seq_len)
    start = time.monotonic()

    step = 0
    while step < args.max_steps and not _all_complete(stats, args.per_type):
        if args.max_seconds and (time.monotonic() - start) >= args.max_seconds:
            print("Reached max-seconds; stopping early.")
            break

        live_node_ips, live_node_cores = _discover_live_topology()
        if not live_node_ips:
            time.sleep(args.sample_interval_s)
            continue

        node_tensors, raw_parsed = collect_full_cluster_snapshot(live_node_ips)
        cluster_token = cluster_snapshot_to_token(node_tensors, live_node_ips)
        if args.update_normalizer:
            normalizer.update(cluster_token)
        token_norm = normalizer.normalize(cluster_token)
        buffer.append(token_norm)

        step += 1
        if len(buffer) < args.seq_len:
            time.sleep(args.sample_interval_s)
            continue

        semantic = compute_semantic_features(raw_parsed, live_node_ips)
        qa_pairs = generate_qa_pairs(
            semantic,
            live_node_ips,
            live_node_cores,
            max_pairs=args.max_pairs_per_step,
        )

        candidates: list[tuple[dict, int, str]] = []
        per_type_added: dict[str, int] = defaultdict(int)
        for q_struct, ans_idx in qa_pairs:
            q_type = _type_key(q_struct, args.split_by)
            if q_type not in stats:
                if allow_new_types:
                    stats[q_type] = AccuracyStats()
                else:
                    continue
            if stats[q_type].total + per_type_added[q_type] >= args.per_type:
                continue
            candidates.append((q_struct, ans_idx, q_type))
            per_type_added[q_type] += 1

        if candidates:
            ctx = torch.stack(list(buffer), dim=0).unsqueeze(0).to(DEVICE)
            ctx_batch = ctx.expand(len(candidates), -1, -1).contiguous()
            q_list = [c[0] for c in candidates]
            ans_list = [c[1] for c in candidates]
            q_emb = encode_questions(question_encoder, q_list, live_node_ips).to(DEVICE)

            with torch.no_grad():
                q_tok = (
                    encode_question_tokens(question_encoder, q_list).to(DEVICE)
                    if getattr(model, "use_token_sequence", False)
                    else None
                )
                _seq_latent, ctx_latent = model.metric_encoder(ctx_batch)
                pred_latent = model.predict_answer_latent(ctx_latent, q_emb, q_tok)
                logits = model.codebook.logits_from_latent(pred_latent)
                preds = logits.argmax(dim=-1).tolist()

            for pred, ans_idx, item in zip(preds, ans_list, candidates):
                q_type = item[2]
                is_correct = int(pred) == int(ans_idx)
                stats[q_type].add(is_correct)
                overall.add(is_correct)

        if args.log_every and step % args.log_every == 0:
            _print_progress(stats, args.per_type)

        time.sleep(args.sample_interval_s)

    print(f"=== LPSE accuracy by question type ({args.split_by}) ===")
    ordered_types = question_types if question_types else sorted(stats.keys())
    for q_type in ordered_types:
        s = stats[q_type]
        acc = s.acc()
        acc_pct = "n/a" if acc is None else f"{acc * 100:.2f}%"
        print(f"{q_type:12s} {s.correct:4d}/{s.total:4d}  acc={acc_pct}")

    overall_acc = overall.acc()
    overall_pct = "n/a" if overall_acc is None else f"{overall_acc * 100:.2f}%"
    print(f"overall      {overall.correct:4d}/{overall.total:4d}  acc={overall_pct}")

    if not _all_complete(stats, args.per_type):
        print("WARNING: Not all question types reached the target count.")

    if args.output_json:
        payload = {
            "checkpoint": args.checkpoint,
            "per_type_target": args.per_type,
            "split_by": args.split_by,
            "question_types": ordered_types,
            "totals": {k: stats[k].total for k in ordered_types},
            "correct": {k: stats[k].correct for k in ordered_types},
            "accuracy": {
                k: None if (a := stats[k].acc()) is None else float(a)
                for k in ordered_types
            },
            "overall": {
                "total": overall.total,
                "correct": overall.correct,
                "accuracy": None if overall_acc is None else float(overall_acc),
            },
        }
        _write_json(args.output_json, payload)
        print(f"Wrote summary JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
