#!/usr/bin/env python3
"""Test-time Monte-Carlo voting over perturbed question embeddings.

The model was trained (lever A) to be locally constant in question-embedding
space. At inference we average the answer-codebook logits over N noisy
question embeddings — the perturbed votes concentrate on the stable answer,
reducing boundary flips caused by paraphrase text drift. This uses the
trained model as-is; NO paraphrase text and NO retraining.
"""

import argparse
import csv
import os
import random
import sys

import torch

sys_path_setup = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path_setup not in sys.path:
    sys.path.insert(0, sys_path_setup)

from benchmark_llm_forecast import _paraphrase_question
from infer_lpse import load_model
from lpse.question_encoder import encode_questions
from lpse.replay_dump import iter_dump_by_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="artifacts/dumps/jepa_run")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--paraphrase-questions", action="store_true")
    ap.add_argument("--snapshots", type=int, default=40)
    ap.add_argument("--tail-fraction", type=float, default=0.2)
    ap.add_argument("--questions-per-snapshot", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--results-csv", default="/tmp/mc_vote.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, question_encoder, _norm, node_ips, node_cores, answer_templates = load_model(
        args.checkpoint
    )
    model.eval()
    question_encoder.eval()
    device = next(model.parameters()).device

    snapshots = []
    for ctx, qa_pairs, is_val, ips, _c, _ts in iter_dump_by_snapshot(args.dump):
        snapshots.append((ctx, qa_pairs, ips))
    n_total = len(snapshots)
    tail = snapshots[int(n_total * (1 - args.tail_fraction)) :]
    step = max(1, len(tail) // args.snapshots)
    chosen = tail[::step][: args.snapshots]
    print(
        f"trace snapshots={n_total} tail={len(tail)} sampled={len(chosen)}", flush=True
    )

    rows = []
    n = 0
    correct_single = 0
    correct_mc = 0
    for ctx, qa_pairs, ips in chosen:
        ctx_t = ctx.unsqueeze(0).to(device)
        if not qa_pairs:
            continue
        random.shuffle(qa_pairs)
        for q_dict, truth_idx in qa_pairs[: args.questions_per_snapshot]:
            if n >= args.max_questions:
                break
            if args.paraphrase_questions and isinstance(q_dict, dict):
                q_dict = dict(q_dict)
                q_dict["raw"] = _paraphrase_question(q_dict.get("raw", ""))
            q_text = q_dict.get("raw", "") if isinstance(q_dict, dict) else str(q_dict)

            with torch.no_grad():
                _, state_latent = model.metric_encoder(ctx_t)

            with torch.no_grad():
                q_emb = encode_questions(question_encoder, [q_dict], ips)
                q_tok = model._tokens(
                    encode_question_tokens_opt(question_encoder, model, [q_dict])
                )
                fused = model.fusion(
                    state_latent.unsqueeze(1)
                    if state_latent.dim() == 2
                    else state_latent,
                    q_emb,
                    q_tok,
                )
                logits_single = model.codebook.logits_from_latent(
                    model.codebook.project(fused)
                )[0]
                idx_single = int(logits_single.argmax().item())

            logits_sum = torch.zeros_like(logits_single)
            with torch.no_grad():
                for _ in range(args.mc_samples):
                    q_emb_n = encode_questions(question_encoder, [q_dict], ips)
                    q_emb_n = q_emb_n + torch.randn_like(q_emb_n) * (
                        q_emb_n.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                        * args.noise_std
                    )
                    fused_n = model.fusion(
                        state_latent.unsqueeze(1)
                        if state_latent.dim() == 2
                        else state_latent,
                        q_emb_n,
                        q_tok,
                    )
                    logits_sum += model.codebook.logits_from_latent(
                        model.codebook.project(fused_n)
                    )[0]
            logits_mc = logits_sum / args.mc_samples
            idx_mc = int(logits_mc.argmax().item())

            model.train()
            d_sum = torch.zeros_like(logits_single)
            with torch.no_grad():
                for _ in range(args.mc_samples):
                    fused_d = model.fusion(
                        state_latent.unsqueeze(1)
                        if state_latent.dim() == 2
                        else state_latent,
                        q_emb,
                        q_tok,
                    )
                    d_sum += model.codebook.logits_from_latent(
                        model.codebook.project(fused_d)
                    )[0]
            model.eval()
            logits_mcd = d_sum / args.mc_samples
            idx_mcd = int(logits_mcd.argmax().item())

            c1 = int(idx_single == int(truth_idx))
            cm = int(idx_mc == int(truth_idx))
            cmd = int(idx_mcd == int(truth_idx))
            correct_single += c1
            correct_mc += cm
            rows.append([q_text, truth_idx, idx_single, idx_mc, idx_mcd, c1, cm, cmd])
            n += 1
        if n >= args.max_questions:
            break

    print(f"single-forward: {correct_single}/{n} = {correct_single / n * 100:.1f}%")
    print(
        f"MC-vote (n={args.mc_samples}, std={args.noise_std}): {correct_mc}/{n} = {correct_mc / n * 100:.1f}%"
    )
    print(
        f"MC-dropout (n={args.mc_samples}): {sum(r[7] for r in rows)}/{n} = {sum(r[7] for r in rows) / n * 100:.1f}%"
    )
    with open(args.results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["question", "truth", "single", "mc", "mcd", "single_ok", "mc_ok", "mcd_ok"]
        )
        w.writerows(rows)


def encode_question_tokens_opt(encoder, model, qs):
    if not getattr(model, "use_token_sequence", False):
        return None
    from lpse.question_encoder import encode_question_tokens

    return encode_question_tokens(encoder, qs)


if __name__ == "__main__":
    main()
