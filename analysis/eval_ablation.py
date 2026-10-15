"""Evaluate the main and no-future checkpoints on the saved replay-dump validation snapshots.

Reports top-1 / top-5 accuracy on every QA pair whose context snapshot was
flagged as held-out validation during training. Uses the same snapshots, same
QA pairs, same question encoder, same codebook for both checkpoints, so the
gap is attributable to the missing future-prediction loss term.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

from lpse.lpse import (
    ANSWER_DIM,
    FUSION_DIM,
    LATENT_DIM,
    LPSE,
)
from lpse.prometheus_ingest import CLUSTER_TOKEN_DIM, RunningNormalizer
from lpse.qa_generator import ANSWER_TEMPLATES
from lpse.question_encoder import (
    QUESTION_DIM,
    build_question_encoder,
    encode_questions,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablation_eval")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(path: str, q_encoder_backend: str = "sentence_transformer"):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("model_config") or {}
    answer_templates = ckpt.get("answer_templates", ANSWER_TEMPLATES)

    qenc = build_question_encoder(
        backend=q_encoder_backend,
        out_dim=QUESTION_DIM,
    ).to(DEVICE)

    model = LPSE(
        input_dim=int(cfg.get("input_dim", CLUSTER_TOKEN_DIM)),
        num_answers=len(answer_templates),
        question_encoder=qenc,
        latent_dim=int(cfg.get("latent_dim", LATENT_DIM)),
        fusion_dim=int(cfg.get("fusion_dim", FUSION_DIM)),
        answer_dim=int(cfg.get("answer_dim", ANSWER_DIM)),
        encoder_heads=int(cfg.get("encoder_heads", 16)),
        encoder_layers=int(cfg.get("encoder_layers", 12)),
        fusion_heads=int(cfg.get("fusion_heads", 16)),
        fusion_layers=int(cfg.get("fusion_layers", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        codebook_temperature=float(ckpt.get("codebook_temperature", 0.07)),
    ).to(DEVICE)

    state = model.state_dict()
    src = ckpt.get("model", {})
    compat = {k: v for k, v in src.items() if k in state and state[k].shape == v.shape}
    skipped = [k for k in src if k not in compat]
    state.update(compat)
    model.load_state_dict(state)
    if skipped:
        logger.info(f"  skipped {len(skipped)} incompatible keys (e.g. {skipped[:3]})")
    model.eval()

    normalizer = RunningNormalizer(int(cfg.get("input_dim", CLUSTER_TOKEN_DIM)))
    if "normalizer" in ckpt:
        try:
            normalizer.load_state_dict(ckpt["normalizer"])
        except Exception as e:
            logger.info(f"  normalizer load failed: {e}")

    node_ips = ckpt.get("node_ips", [])
    return model, qenc, normalizer, node_ips, ckpt.get("step", -1)


def gather_val_pairs(
    dump_path: str, max_shards: int | None = None
) -> list[tuple[torch.Tensor, dict, int]]:
    """Return list of (context_tensor[T, F], parsed_question, answer_idx) for val snapshots."""
    if not os.path.isdir(dump_path):
        raise FileNotFoundError(
            f"Dump directory not found: {dump_path}. Point --dump at the replay dump "
            f"(e.g. artifacts/dumps/jepa_run)."
        )
    pairs: list[tuple[torch.Tensor, dict, int]] = []
    shards = sorted(f for f in os.listdir(dump_path) if f.startswith("shard_"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files under {dump_path}")
    if max_shards is not None:
        shards = shards[:max_shards]
    for sf in shards:
        d = torch.load(
            os.path.join(dump_path, sf), map_location="cpu", weights_only=False
        )
        is_val = d["is_val"].tolist()
        contexts = d["contexts"]
        for snap_idx, parsed_q, ans_idx in d["qa"]:
            if is_val[snap_idx]:
                pairs.append((contexts[snap_idx], parsed_q, int(ans_idx)))
    return pairs


@torch.no_grad()
def evaluate(
    model, qenc, normalizer, node_ips, pairs, batch_size: int = 64
) -> tuple[float, float, int]:
    correct1 = 0
    correct5 = 0
    total = 0
    n = len(pairs)
    for start in range(0, n, batch_size):
        batch = pairs[start : start + batch_size]
        ctx = torch.stack([p[0] for p in batch], dim=0).to(DEVICE)

        parsed = [p[1] for p in batch]
        targets = torch.tensor([p[2] for p in batch], dtype=torch.long, device=DEVICE)
        q_emb = encode_questions(qenc, parsed, node_ips)
        logits = model(ctx, q_emb)
        top5 = logits.topk(5, dim=-1).indices
        top1 = top5[:, 0]
        correct1 += (top1 == targets).sum().item()
        correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
        if (start // batch_size) % 50 == 0:
            logger.info(
                f"  {total}/{n} top1={correct1 / total:.4f} top5={correct5 / total:.4f}"
            )
    return correct1 / total, correct5 / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="checkpoints/nljepahybrid.pt")
    ap.add_argument(
        "--ablation", default="ablation_runs/no_future/checkpoints/lpse_no_future.pt"
    )
    ap.add_argument("--dump", default="artifacts/dumps/jepa_run")
    ap.add_argument(
        "--max-shards", type=int, default=None, help="cap shards loaded (default: all)"
    )
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading val pairs from {args.dump} (max_shards={args.max_shards})...")
    pairs = gather_val_pairs(args.dump, args.max_shards)
    logger.info(f"  -> {len(pairs)} val QA pairs")

    results: dict[str, tuple[float, float, int, int]] = {}
    for label, path in [
        ("main_full_jepa", args.main),
        ("no_future_ablation", args.ablation),
    ]:
        if not os.path.isfile(path):
            logger.info(f"\nSKIP {label}: {path} not found")
            continue
        logger.info(f"\n=== {label}: {path} ===")
        model, qenc, normalizer, node_ips, step = load_ckpt(path)
        logger.info(f"  step={step} num_node_ips={len(node_ips)}")
        top1, top5, total = evaluate(
            model, qenc, normalizer, node_ips, pairs, batch_size=args.batch
        )
        results[label] = (top1, top5, total, step)
        logger.info(
            f"  FINAL: top1={top1:.4f} top5={top5:.4f} (n={total}, ckpt step {step})"
        )

    print("\n=== SUMMARY ===")
    for label, (t1, t5, n, step) in results.items():
        print(f"{label:30s}  top1={t1:.4f}  top5={t5:.4f}  n={n}  step={step}")


if __name__ == "__main__":
    sys.exit(main())
