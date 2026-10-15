#!/usr/bin/env python3
"""Offline benchmark: evaluate LPSE, MLP, XGBoost, and an LLM on the SAME
replay trace (training-time snapshots) so all systems see the identical
(normalized) context distribution the models were trained on.

Reads the JEPA replay dump (artifacts/dumps/jepa_run), samples snapshots in
chronological order, and for each sampled snapshot evaluates every system on
the same generated QA pairs. Mirrors benchmark_llm_forecast.py's protocol
(open-set 94-way answer decision, live question generation, optional
paraphrase rewrite) but substitutes the dump's stored contexts for live
cluster scraping, removing the normalizer/telemetry drift that depresses
live numbers months after training.

Usage:
  python scripts/eval/benchmark_offline_trace.py \
      --dump artifacts/dumps/jepa_run \
      --checkpoint checkpoints/nljepahybrid.pt \
      --mlp-checkpoint checkpoints/baseline_mlp_v2.pt \
      --xgb-model checkpoints/baseline_xgb_v2.json \
      --xgb-norm checkpoints/baseline_xgb_v2_norm.pkl \
      --xgb-encoder-state checkpoints/baseline_xgb_v2_encoder.pt \
      --snapshots 30 --questions-per-snapshot 5 --max-questions 100 \
      --vllm-chat-url http://localhost:8001/v1/chat/completions \
      --vllm-model "Qwen/Qwen3-4B-Instruct-2507-FP8" \
      [--paraphrase-questions] [--llm-options open|category]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests
import torch
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lpse.apps.benchmark_llm_forecast import (
    AccuracyStats,
    _build_llm_messages,
    _category_option_indices,
    _ensure_chat_url,
    _extract_json_candidates,
    _paraphrase_question,
    _parse_llm_answer_index,
)
from lpse.apps.infer_lpse import load_model
from lpse.qa_generator import ANSWER_TEMPLATES, NUM_ANSWERS
from lpse.question_encoder import QUESTION_DIM, encode_question_tokens, encode_questions
from lpse.replay_dump import iter_dump_by_snapshot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _FlatMLPClassifier(torch.nn.Module):
    in_dim: int
    hidden_dim: int
    num_layers: int
    num_answers: int
    dropout: float

    def __post_init__(self):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = self.in_dim
        for _ in range(self.num_layers):
            layers.append(torch.nn.Linear(prev, self.hidden_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(self.dropout))
            prev = self.hidden_dim
        layers.append(torch.nn.Linear(prev, self.num_answers))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _LSTMBaseline(torch.nn.Module):
    in_dim: int
    hidden_dim: int
    num_layers: int
    num_answers: int
    dropout: float
    head_hidden: int
    question_dim: int

    def __post_init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + self.question_dim, self.head_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.head_hidden, self.num_answers),
        )

    def forward(self, seq: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        return self.head(torch.cat([last, q_emb], dim=-1))


@dataclass
class _TransformerBaseline(torch.nn.Module):
    in_dim: int
    d_model: int
    num_layers: int
    num_heads: int
    num_answers: int
    dropout: float
    head_hidden: int
    question_dim: int

    def __post_init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(self.in_dim, self.d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            enc_layer, num_layers=self.num_layers
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d_model + self.question_dim, self.head_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.head_hidden, self.num_answers),
        )

    def forward(self, seq: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.in_proj(seq))
        pooled = h.mean(dim=1)
        return self.head(torch.cat([pooled, q_emb], dim=-1))


def _count_snapshots(dump_dir: str) -> int:
    """Count snapshots in the dump from its meta.json, falling back to
    scanning shard files if the meta is missing."""
    try:
        with open(os.path.join(dump_dir, "meta.json")) as f:
            meta = json.load(f)
        n = int(meta.get("snapshots_written", 0))
        if n > 0:
            return n
    except Exception:
        pass
    import glob

    total = 0
    for shard in sorted(glob.glob(os.path.join(dump_dir, "shard_*.pt"))):
        p = torch.load(shard, map_location="cpu", weights_only=False)
        total += int(p["contexts"].shape[0])
    return total


def _load_mlp(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("model_config") or {}
    model = _FlatMLPClassifier(
        in_dim=int(cfg["in_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 1024)),
        num_layers=int(cfg.get("hidden_layers", 4)),
        num_answers=int(cfg.get("num_answers", NUM_ANSWERS)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("question_encoder_state")


def _load_lstm(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("model_config") or {}
    model = _LSTMBaseline(
        in_dim=int(cfg["in_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_answers=int(cfg.get("num_answers", NUM_ANSWERS)),
        dropout=float(cfg.get("dropout", 0.1)),
        head_hidden=int(cfg.get("head_hidden", 512)),
        question_dim=int(cfg.get("question_dim", QUESTION_DIM)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("question_encoder_state")


def _load_transformer(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("model_config") or {}
    model = _TransformerBaseline(
        in_dim=int(cfg["in_dim"]),
        d_model=int(cfg.get("d_model", 256)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_answers=int(cfg.get("num_answers", NUM_ANSWERS)),
        dropout=float(cfg.get("dropout", 0.1)),
        head_hidden=int(cfg.get("head_hidden", 512)),
        question_dim=int(cfg.get("question_dim", QUESTION_DIM)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("question_encoder_state")


def _build_encoder(persisted_state):
    from lpse.question_encoder import build_question_encoder

    if persisted_state is None:
        return None

    proj_key = "proj.0.weight"
    if proj_key not in persisted_state:
        proj_key = "proj.weight"
    if proj_key in persisted_state:
        proj_in = int(persisted_state[proj_key].shape[1])
        if proj_in == 384:
            enc = build_question_encoder(
                backend="sentence_transformer", out_dim=QUESTION_DIM
            ).to(DEVICE)
        elif proj_in == 416:
            enc = build_question_encoder(
                backend="hybrid", out_dim=QUESTION_DIM, use_structured=False
            ).to(DEVICE)
        else:
            enc = build_question_encoder(
                backend="hybrid", out_dim=QUESTION_DIM, use_structured=True
            ).to(DEVICE)
    else:
        enc = build_question_encoder(
            backend="hybrid", out_dim=QUESTION_DIM, use_structured=True
        ).to(DEVICE)
    enc.load_state_dict(persisted_state, strict=False)
    enc.eval()
    return enc


def _call_llm(
    session,
    chat_url,
    model,
    api_key,
    question_text,
    semantic_features,
    answer_templates,
    node_ips,
    node_cores,
    option_indices,
    timeout_s,
    temperature,
    max_tokens,
):
    messages = _build_llm_messages(
        question_text=question_text,
        semantic_features=semantic_features,
        answer_templates=answer_templates,
        node_ips=node_ips,
        node_cores=node_cores,
        option_indices=option_indices,
    )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    try:
        r = session.post(chat_url, json=payload, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})

        idx = None
        for cand in _extract_json_candidates(content):
            idx = _parse_llm_answer_index(cand, answer_templates)
            if idx is not None:
                break
        return idx, latency_ms, None, content, usage
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return None, latency_ms, str(e), "", {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline trace benchmark (all systems, same replay data)"
    )
    ap.add_argument("--dump", required=True, help="Replay dump directory")
    ap.add_argument("--checkpoint", default="checkpoints/nljepahybrid.pt")
    ap.add_argument("--mlp-checkpoint", default="checkpoints/baseline_mlp_v2.pt")
    ap.add_argument("--lstm-checkpoint", default="")
    ap.add_argument("--transformer-checkpoint", default="")
    ap.add_argument("--xgb-model", default="checkpoints/baseline_xgb_v2.json")
    ap.add_argument("--xgb-norm", default="checkpoints/baseline_xgb_v2_norm.pkl")
    ap.add_argument(
        "--xgb-encoder-state", default="checkpoints/baseline_xgb_v2_encoder.pt"
    )
    ap.add_argument("--llm-options", choices=("open", "category"), default="open")
    ap.add_argument("--paraphrase-questions", action="store_true")
    ap.add_argument("--snapshots", type=int, default=30)
    ap.add_argument(
        "--tail-fraction",
        type=float,
        default=None,
        help="Sample only the last fraction (0..1) of the trace, "
        "e.g. 0.2 = last 20% (matches evaluating the final "
        "trained model on end-of-training distribution).",
    )
    ap.add_argument("--questions-per-snapshot", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--vllm-chat-url", default="")
    ap.add_argument("--vllm-model", default="")
    ap.add_argument("--vllm-api-key", default="")
    ap.add_argument("--vllm-timeout-s", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--results-csv", default="offline_trace_benchmark.csv")
    ap.add_argument("--summary-json", default="offline_trace_summary.json")
    ap.add_argument("--llm-raw-out", default="")
    args = ap.parse_args()

    random.seed(args.seed)

    model, question_encoder, _norm, node_ips, node_cores, answer_templates = load_model(
        args.checkpoint
    )
    model.eval()
    question_encoder.eval()
    device = next(model.parameters()).device
    if not answer_templates:
        answer_templates = ANSWER_TEMPLATES

    mlp_model = None
    mlp_encoder = None
    if args.mlp_checkpoint and os.path.isfile(args.mlp_checkpoint):
        mlp_model, mlp_enc_state = _load_mlp(args.mlp_checkpoint)
        mlp_encoder = _build_encoder(mlp_enc_state)
        print(f"MLP loaded: {args.mlp_checkpoint}")

    lstm_model = None
    lstm_encoder = None
    if args.lstm_checkpoint and os.path.isfile(args.lstm_checkpoint):
        lstm_model, lstm_enc_state = _load_lstm(args.lstm_checkpoint)
        lstm_encoder = _build_encoder(lstm_enc_state)
        print(f"LSTM loaded: {args.lstm_checkpoint}")

    transformer_model = None
    transformer_encoder = None
    if args.transformer_checkpoint and os.path.isfile(args.transformer_checkpoint):
        transformer_model, tf_enc_state = _load_transformer(args.transformer_checkpoint)
        transformer_encoder = _build_encoder(tf_enc_state)
        print(f"Transformer loaded: {args.transformer_checkpoint}")

    xgb_booster = None
    xgb_encoder = None
    if args.xgb_model and os.path.isfile(args.xgb_model):
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(args.xgb_model)
        if os.path.isfile(args.xgb_encoder_state):
            enc_blob = torch.load(
                args.xgb_encoder_state, map_location="cpu", weights_only=False
            )
            xgb_encoder = _build_encoder(enc_blob.get("question_encoder_state"))
        print(f"XGB loaded: {args.xgb_model} rounds={xgb_booster.num_boosted_rounds()}")

    llm_enabled = bool(args.vllm_chat_url and args.vllm_model) and not args.skip_llm
    chat_url = _ensure_chat_url(args.vllm_chat_url) if llm_enabled else ""

    jepa_acc = AccuracyStats()
    mlp_acc = AccuracyStats()
    xgb_acc = AccuracyStats()
    lstm_acc = AccuracyStats()
    transformer_acc = AccuracyStats()
    llm_overall = AccuracyStats()
    llm_valid = AccuracyStats()

    csv_fields = [
        "snapshot",
        "question",
        "truth_idx",
        "jepa_idx",
        "jepa_correct",
        "mlp_idx",
        "mlp_correct",
        "lstm_idx",
        "lstm_correct",
        "transformer_idx",
        "transformer_correct",
        "xgb_idx",
        "xgb_correct",
        "llm_idx",
        "llm_correct",
        "llm_valid",
    ]
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

    total_snapshots = _count_snapshots(args.dump)
    if args.tail_fraction is not None:
        start_idx = int(
            total_snapshots * (1.0 - max(0.0, min(1.0, args.tail_fraction)))
        )
        step_size = max(1, int((total_snapshots - start_idx) // max(1, args.snapshots)))
        selected = (
            set(range(start_idx, total_snapshots, step_size))
            if args.snapshots > 0
            else None
        )
    else:
        step_size = max(1, total_snapshots // max(1, args.snapshots))
        selected = (
            set(range(0, total_snapshots, step_size)) if args.snapshots > 0 else None
        )

    with open(args.results_csv, "w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()
        raw_fh = (
            open(args.llm_raw_out, "w", encoding="utf-8") if args.llm_raw_out else None
        )

        n_questions = 0
        with requests.Session() as session:
            for snap_idx, (ctx, qa_pairs, is_val, ips_n, cores_n, ts) in enumerate(
                iter_dump_by_snapshot(args.dump)
            ):
                if n_questions >= args.max_questions:
                    break
                if snap_idx >= total_snapshots:
                    break
                if selected is not None and snap_idx not in selected:
                    continue
                ctx_t = ctx.unsqueeze(0).to(device)

                if not qa_pairs:
                    continue
                random.shuffle(qa_pairs)
                chosen = qa_pairs[: args.questions_per_snapshot]

                for q_dict, truth_idx in chosen:
                    if n_questions >= args.max_questions:
                        break
                    q_text = (
                        q_dict.get("raw", "")
                        if isinstance(q_dict, dict)
                        else str(q_dict)
                    )
                    if args.paraphrase_questions and isinstance(q_dict, dict):
                        q_dict = dict(q_dict)
                        q_dict["raw"] = _paraphrase_question(q_text)
                        q_text = q_dict["raw"]

                    option_indices = (
                        _category_option_indices(int(truth_idx))
                        if args.llm_options == "category"
                        else None
                    )

                    q_emb = encode_questions(question_encoder, [q_dict], ips_n)
                    q_tok = (
                        encode_question_tokens(question_encoder, [q_dict])
                        if getattr(model, "use_token_sequence", False)
                        else None
                    )
                    with torch.no_grad():
                        jepa_idx, _, _ = model.infer(ctx_t, q_emb, q_tok)
                    jepa_correct = int(int(jepa_idx) == int(truth_idx))
                    jepa_acc.add(bool(jepa_correct))

                    mlp_idx = None
                    mlp_correct = 0
                    if mlp_model is not None:
                        mlp_q = (
                            q_emb
                            if mlp_encoder is None
                            else encode_questions(mlp_encoder, [q_dict], ips_n)
                        )
                        with torch.no_grad():
                            x = torch.cat([ctx_t.flatten(1), mlp_q], dim=-1)
                            mlp_idx = int(mlp_model(x).argmax(dim=-1).item())
                        mlp_correct = int(mlp_idx == int(truth_idx))
                        mlp_acc.add(bool(mlp_correct))

                    lstm_idx = None
                    lstm_correct = 0
                    if lstm_model is not None:
                        lstm_q = (
                            q_emb
                            if lstm_encoder is None
                            else encode_questions(lstm_encoder, [q_dict], ips_n)
                        )
                        with torch.no_grad():
                            lstm_idx = int(
                                lstm_model(ctx_t, lstm_q).argmax(dim=-1).item()
                            )
                        lstm_correct = int(lstm_idx == int(truth_idx))
                        lstm_acc.add(bool(lstm_correct))

                    transformer_idx = None
                    transformer_correct = 0
                    if transformer_model is not None:
                        tf_q = (
                            q_emb
                            if transformer_encoder is None
                            else encode_questions(transformer_encoder, [q_dict], ips_n)
                        )
                        with torch.no_grad():
                            transformer_idx = int(
                                transformer_model(ctx_t, tf_q).argmax(dim=-1).item()
                            )
                        transformer_correct = int(transformer_idx == int(truth_idx))
                        transformer_acc.add(bool(transformer_correct))

                    xgb_idx = None
                    xgb_correct = 0
                    if xgb_booster is not None:
                        xgb_q = (
                            q_emb
                            if xgb_encoder is None
                            else encode_questions(xgb_encoder, [q_dict], ips_n)
                        )
                        flat_np = (
                            ctx_t.flatten(1).detach().cpu().numpy().astype(np.float32)
                        )
                        q_np = xgb_q.detach().cpu().numpy().astype(np.float32)
                        x = np.concatenate([flat_np, q_np], axis=1)
                        probs = xgb_booster.predict(xgb.DMatrix(x))
                        xgb_idx = int(probs.argmax(axis=1)[0])
                        xgb_correct = int(xgb_idx == int(truth_idx))
                        xgb_acc.add(bool(xgb_correct))

                    llm_idx = None
                    llm_correct = 0
                    llm_ok = False
                    if llm_enabled:
                        idx, _lat, err, raw, usage = _call_llm(
                            session=session,
                            chat_url=chat_url,
                            model=args.vllm_model,
                            api_key=args.vllm_api_key,
                            question_text=q_text,
                            semantic_features={},
                            answer_templates=answer_templates,
                            node_ips=ips_n,
                            node_cores=cores_n,
                            option_indices=option_indices,
                            timeout_s=args.vllm_timeout_s,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        if idx is not None:
                            llm_ok = True
                            llm_idx = idx
                            llm_correct = int(idx == int(truth_idx))
                            llm_valid.add(bool(llm_correct))
                        llm_overall.add(bool(llm_correct))
                        if raw_fh is not None:
                            raw_fh.write(
                                json.dumps(
                                    {
                                        "snapshot": snap_idx,
                                        "question": q_text,
                                        "truth_idx": int(truth_idx),
                                        "llm_idx": llm_idx,
                                        "llm_correct": llm_correct,
                                        "llm_error": err,
                                        "llm_raw_content": raw,
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )

                    writer.writerow(
                        {
                            "snapshot": snap_idx,
                            "question": q_text,
                            "truth_idx": int(truth_idx),
                            "jepa_idx": jepa_idx,
                            "jepa_correct": jepa_correct,
                            "mlp_idx": ("" if mlp_idx is None else mlp_idx),
                            "mlp_correct": mlp_correct,
                            "lstm_idx": ("" if lstm_idx is None else lstm_idx),
                            "lstm_correct": lstm_correct,
                            "transformer_idx": (
                                "" if transformer_idx is None else transformer_idx
                            ),
                            "transformer_correct": transformer_correct,
                            "xgb_idx": ("" if xgb_idx is None else xgb_idx),
                            "xgb_correct": xgb_correct,
                            "llm_idx": ("" if llm_idx is None else llm_idx),
                            "llm_correct": llm_correct,
                            "llm_valid": int(llm_ok),
                        }
                    )
                    n_questions += 1

        if raw_fh is not None:
            raw_fh.close()

    def _fmt(d):
        return d if d is not None else float("nan")

    summary = {
        "generated_at": time.time(),
        "config": {
            "dump": args.dump,
            "checkpoint": args.checkpoint,
            "mlp": args.mlp_checkpoint,
            "xgb": args.xgb_model,
            "llm_options": args.llm_options,
            "paraphrase": bool(args.paraphrase_questions),
            "snapshots": args.snapshots,
            "questions_per_snapshot": args.questions_per_snapshot,
            "max_questions": args.max_questions,
            "seed": args.seed,
            "llm": args.vllm_model if llm_enabled else None,
        },
        "jepa": {
            "n": jepa_acc.total,
            "correct": jepa_acc.correct,
            "accuracy": _fmt(jepa_acc.acc()),
        },
        "mlp": {
            "n": mlp_acc.total,
            "correct": mlp_acc.correct,
            "accuracy": _fmt(mlp_acc.acc()),
        },
        "lstm": {
            "n": lstm_acc.total,
            "correct": lstm_acc.correct,
            "accuracy": _fmt(lstm_acc.acc()),
        },
        "transformer": {
            "n": transformer_acc.total,
            "correct": transformer_acc.correct,
            "accuracy": _fmt(transformer_acc.acc()),
        },
        "xgb": {
            "n": xgb_acc.total,
            "correct": xgb_acc.correct,
            "accuracy": _fmt(xgb_acc.acc()),
        },
        "llm": {
            "n": llm_overall.total,
            "correct": llm_overall.correct,
            "accuracy_overall": _fmt(llm_overall.acc()),
            "valid": llm_valid.total,
            "accuracy_valid_only": _fmt(llm_valid.acc()),
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== offline trace benchmark ===")
    print(
        f"LPSE        accuracy={_fmt(jepa_acc.acc()) * 100 if jepa_acc.acc() else 0:.2f}% (n={jepa_acc.total})"
    )
    print(
        f"MLP         accuracy={_fmt(mlp_acc.acc()) * 100 if mlp_acc.acc() else 0:.2f}% (n={mlp_acc.total})"
    )
    print(
        f"LSTM        accuracy={_fmt(lstm_acc.acc()) * 100 if lstm_acc.acc() else 0:.2f}% (n={lstm_acc.total})"
    )
    print(
        f"Transformer accuracy={_fmt(transformer_acc.acc()) * 100 if transformer_acc.acc() else 0:.2f}% (n={transformer_acc.total})"
    )
    print(
        f"XGBoost     accuracy={_fmt(xgb_acc.acc()) * 100 if xgb_acc.acc() else 0:.2f}% (n={xgb_acc.total})"
    )
    if llm_enabled:
        print(
            f"LLM     accuracy_overall={_fmt(llm_overall.acc()) * 100 if llm_overall.acc() else 0:.2f}% "
            f"valid_only={_fmt(llm_valid.acc()) * 100 if llm_valid.acc() else 0:.2f}% "
            f"(n={llm_overall.total}, valid={llm_valid.total})"
        )
    print(f"results_csv={args.results_csv}")
    print(f"summary_json={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
