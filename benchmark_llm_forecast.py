#!/usr/bin/env python3
"""JEPA vs LLM benchmark on live cluster telemetry. Compares latency and top-1 accuracy on a shared QA stream.

LLM answer protocol: by default the LLM answers open-set over the full
answer-template inventory — the same 94-way decision LPSE/MLP/XGBoost make.
`--llm-options category` optionally restricts the prompt's offered options to
the truth's whole answer category (truth + all same-category distractors).

Earlier revisions of this script passed the LLM a `qa_pair_options` dict built
from a singleton candidate list containing only the ground-truth index, and
additionally rejected any parsed answer outside that list as invalid rather
than scoring it wrong. Under that protocol the LLM rows measured instruction-
following, not accuracy (`accuracy_valid_only` was 1.0 by construction). LLM
results produced before this fix are not comparable and must be re-run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re as _re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import psutil
import requests
import torch
import torch.nn as nn
import xgboost as xgb

from infer_lpse import load_model
from lpse.collectors import get_node_ips, get_num_cores_from_node_exporter
from lpse.prometheus_ingest import (
    CLUSTER_TOKEN_DIM,
    RunningNormalizer,
    cluster_snapshot_to_token,
    collect_full_cluster_snapshot,
    compute_semantic_features,
)
from lpse.qa_generator import (
    _CATEGORY_RANGES,
    ANSWER_TEMPLATES,
    NUM_ANSWERS,
    generate_qa_pairs,
)
from lpse.question_encoder import QUESTION_DIM, encode_question_tokens, encode_questions


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_PARAPHRASE_PATTERNS: list[tuple[Any, Any]] = [
    (
        _re.compile(r"^Is node (\S+) CPU-bound\?$"),
        lambda m: f"Has the processor on box {m.group(1)} hit its ceiling?",
    ),
    (
        _re.compile(r"^Is node (\S+) memory-bound\?$"),
        lambda m: f"Is RAM the bottleneck for {m.group(1)} right now?",
    ),
    (
        _re.compile(r"^How much disk space is available on node (\S+)\?$"),
        lambda m: f"How much room is left on {m.group(1)}'s filesystem?",
    ),
    (
        _re.compile(r"^What is the load average on node (\S+)\?$"),
        lambda m: f"What load is {m.group(1)} carrying?",
    ),
    (
        _re.compile(r"^Is load increasing or decreasing on node (\S+)\?$"),
        lambda m: f"Which way is {m.group(1)}'s load trending?",
    ),
    (
        _re.compile(r"^Are any processes blocked on node (\S+)\?$"),
        lambda m: f"Has anything stalled in {m.group(1)}'s run queue?",
    ),
    (
        _re.compile(r"^What is the temperature on node (\S+)\?$"),
        lambda m: f"How hot is {m.group(1)} running?",
    ),
    (
        _re.compile(
            r"^(?:What is the CPU usage|How much CPU is being used|Is CPU utilisation high|What is the CPU load on this node|Tell me about CPU usage|How busy is the processor|What's the compute usage|Give me the CPU stats|How are the cores doing) on node ([\d\.]+)\??$"
        ),
        lambda m: f"How hard is {m.group(1)}'s processor working at the moment?",
    ),
    (
        _re.compile(
            r"^(?:What is the memory usage|How much memory is being used|Is memory under pressure|How much RAM is available|Tell me about memory usage|What's the memory utilisation|How much memory is free|Are we running low on memory) on node ([\d\.]+)\??$"
        ),
        lambda m: f"What's the RAM picture for {m.group(1)} right now?",
    ),
    (
        _re.compile(
            r"^(?:Is the cluster healthy|Is everything running normally|Are all nodes OK|Give me a cluster overview|Summarize the cluster status|What's the overall system health|Is there anything wrong)\??$"
        ),
        lambda m: "From a 30,000-foot view, is anything off across the fleet?",
    ),
    (
        _re.compile(
            r"^(?:Give me a status summary|What's the overall cluster state|How is the system doing|System health report|Any issues to worry about)\??$"
        ),
        lambda m: "Wrap up the cluster's vibe in one go.",
    ),
]


def _paraphrase_question(text: str) -> str:
    """Return a semantically equivalent rephrasing of `text`. Falls back to
    a generic prefix if no pattern matches so we never silently leave a
    template-shaped question unrewritten."""
    s = (text or "").strip()
    for pat, fn in _PARAPHRASE_PATTERNS:
        m = pat.match(s)
        if m:
            return fn(m)

    return f"Quick question: {s.rstrip('?').lower()}, can you tell?"


def _ensure_chat_url(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    if s.endswith("/v1/chat/completions") or "/chat/completions" in s:
        return s
    return s.rstrip("/") + "/v1/chat/completions"


def _mean(values: list[float]) -> float | None:
    return None if not values else (sum(values) / len(values))


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _parse_json(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _strip_thinking_blocks(content: str) -> str:
    """Remove think tags/blocks often emitted by reasoning-enabled models.

    Handles the real reasoning formats used by vLLM / OpenAI-style models:
    <|thinking|>...</thinking|>, <thinking>...</thinking>, {thinking}...{/thinking},
    and  thinking... response wrappers. Anything between a pair of matching
    think open/close tags is removed so a JSON object inside the reasoning
    trace is not parsed as the answer.
    """
    s = content or ""

    for open_pat, close_pat in [
        (r"<\|thinking\|>", r"<\|/?thinking\|>"),
        (r"<think(?:ing)?>", r"</?think(?:ing)?>"),
        (r"\{thinking\}", r"\{/?thinking\}"),
    ]:
        while True:
            m = _re.search(open_pat, s, _re.IGNORECASE)
            if not m:
                break
            start = m.start()
            end_m = _re.search(close_pat, s[start + 1 :], _re.IGNORECASE)
            if not end_m:
                s = s[:start] + s[m.end() :]
                break
            end = start + 1 + end_m.end()
            s = s[:start] + s[end:]

    for pat in (r"<\|/?thinking\|>", r"</?think(?:ing)?>", r"\{/?thinking\}"):
        s = _re.sub(pat, " ", s, flags=_re.IGNORECASE)
    return s.strip()


def _extract_json_candidates(content: str) -> list[Any]:
    """Extract all parseable JSON candidates from free-form model text."""
    s = _strip_thinking_blocks(content)
    if not s:
        return []

    candidates: list[Any] = []

    direct = _parse_json(s)
    if direct is not None:
        candidates.append(direct)

    if s.startswith("```"):
        s2 = s.strip("`")
        if s2.startswith("json"):
            s2 = s2[4:].strip()
        parsed = _parse_json(s2)
        if parsed is not None:
            candidates.append(parsed)

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
        except Exception:
            continue
        candidates.append(obj)

    out: list[Any] = []
    seen = set()
    for obj in candidates:
        try:
            key = json.dumps(obj, sort_keys=True)
        except Exception:
            key = repr(obj)
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out


def _normalize_text(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


def _parse_llm_answer_index(
    obj: Any,
    answer_templates: list[str],
) -> int | None:

    max_idx = len(answer_templates) - 1

    def _valid_idx(x: Any) -> int | None:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            i = int(x)
            if 0 <= i <= max_idx:
                return i
        return None

    idx = _valid_idx(obj)
    if idx is not None:
        return idx

    if not isinstance(obj, dict):
        return None

    for k in ("answer_index", "index", "template_index"):
        idx = _valid_idx(obj.get(k))
        if idx is not None:
            return idx

    template_text = None
    for k in ("answer_template", "template", "answer"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            template_text = v
            break
    if template_text is None:
        return None

    norm_map = {_normalize_text(t): i for i, t in enumerate(answer_templates)}
    return norm_map.get(_normalize_text(template_text))


def _category_option_indices(truth_idx: int) -> list[int]:
    """All template indices in the answer category containing `truth_idx`.

    Used by `--llm-options category`: the LLM is offered the truth plus every
    same-category distractor, never a narrower set. Falls back to the full
    inventory if the index is not covered by a category range.
    """
    for start, end in _CATEGORY_RANGES.values():
        if start <= truth_idx < end:
            return list(range(start, end))
    return list(range(NUM_ANSWERS))


def _build_llm_messages(
    question_text: str,
    semantic_features: dict[str, dict[str, float | None]],
    answer_templates: list[str],
    node_ips: list[str],
    node_cores: list[int],
    option_indices: list[int] | None = None,
) -> list[dict[str, str]]:
    indexed_templates = {str(i): t for i, t in enumerate(answer_templates)}
    qa_pair_options = None
    if option_indices:
        qa_pair_options = {
            str(i): answer_templates[i]
            for i in option_indices
            if 0 <= i < len(answer_templates)
        }

    payload = {
        "question": question_text,
        "node_ips": node_ips,
        "node_cores": node_cores,
        "semantic_features": semantic_features,
        "answer_templates": indexed_templates,
    }
    if qa_pair_options:
        payload["qa_pair_options"] = qa_pair_options

    system = (
        "You are selecting one answer template index for network-state QA. "
        "Use only the provided semantic_features and question. "
        "Never output <think> tags or explanations. "
        'Return only JSON: {"answer_index": <int>}'
    )
    option_rule = (
        "1) answer_index must be an integer key from qa_pair_options.\n"
        if qa_pair_options
        else "1) answer_index must be an integer key from answer_templates.\n"
    )
    user = (
        "Select the best answer template index for the given question.\n"
        "Constraints:\n" + option_rule + "2) Do not include explanation.\n"
        "3) Do not include think tags.\n"
        "4) Output JSON only.\n\n"
        f"data={json.dumps(payload, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm_for_answer_index(
    session: requests.Session,
    chat_url: str,
    model: str,
    api_key: str | None,
    question_text: str,
    semantic_features: dict[str, dict[str, float | None]],
    answer_templates: list[str],
    node_ips: list[str],
    node_cores: list[int],
    option_indices: list[int] | None,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
) -> tuple[int | None, float, str, str, dict[str, int | None]]:
    """Return (answer_index, latency_ms, error, raw_content, usage_tokens)."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": _build_llm_messages(
            question_text=question_text,
            semantic_features=semantic_features,
            answer_templates=answer_templates,
            node_ips=node_ips,
            node_cores=node_cores,
            option_indices=option_indices,
        ),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    t0 = time.perf_counter()
    try:
        resp = session.post(chat_url, json=body, headers=headers, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
    except Exception as e:
        return (
            None,
            (time.perf_counter() - t0) * 1000.0,
            f"request_failed: {e}",
            "",
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        )

    try:
        data = resp.json()
    except Exception as e:
        return (
            None,
            latency_ms,
            f"invalid_json_response: {e}",
            "",
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        )

    usage = data.get("usage") or {}
    usage_tokens = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning") or ""
    if not content:
        return None, latency_ms, "empty_content", "", usage_tokens

    cleaned_content = _strip_thinking_blocks(content)
    candidates = _extract_json_candidates(cleaned_content)
    if not candidates:
        return (
            None,
            latency_ms,
            "no_json_object_in_content",
            cleaned_content,
            usage_tokens,
        )

    for cand in reversed(candidates):
        idx = _parse_llm_answer_index(cand, answer_templates)
        if idx is not None:
            return idx, latency_ms, "", cleaned_content, usage_tokens

    return None, latency_ms, "no_valid_answer_index", cleaned_content, usage_tokens


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


class _FlatMLPClassifier(nn.Module):
    """Mirror of train_baseline_mlp.py:153 — kept here so the benchmark can
    rebuild the architecture from the checkpoint's model_config without
    importing the trainer module (which side-effects on Kubernetes APIs)."""

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
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_answers))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_mlp(
    ckpt_path: str, device: torch.device
) -> tuple[
    _FlatMLPClassifier,
    RunningNormalizer,
    dict[str, Any],
    dict[str, torch.Tensor] | None,
]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"MLP checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config") or {}
    raw_in_dim = cfg.get("in_dim")
    if raw_in_dim is None:
        raise ValueError(f"MLP checkpoint missing in_dim: {ckpt_path}")
    in_dim = int(raw_in_dim)
    if in_dim <= 0:
        raise ValueError(f"MLP checkpoint missing in_dim: {ckpt_path}")
    model = _FlatMLPClassifier(
        in_dim=in_dim,
        hidden_dim=int(cfg.get("hidden_dim", 1024)),
        num_layers=int(cfg.get("hidden_layers", 4)),
        num_answers=int(cfg.get("num_answers", NUM_ANSWERS)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = RunningNormalizer(int(ckpt.get("input_dim", CLUSTER_TOKEN_DIM)))
    if "normalizer" in ckpt:
        try:
            norm.load_state_dict(ckpt["normalizer"])
        except Exception:
            pass
    encoder_state = ckpt.get("question_encoder_state")
    return model, norm, cfg, encoder_state


def _load_xgb(
    model_path: str, norm_path: str, encoder_state_path: str = ""
) -> tuple[xgb.Booster, RunningNormalizer, dict[str, torch.Tensor] | None]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"XGB model not found: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    norm = RunningNormalizer(CLUSTER_TOKEN_DIM)
    if os.path.isfile(norm_path):
        try:
            with open(norm_path, "rb") as f:
                state = pickle.load(f)
            norm.load_state_dict(state)
        except Exception:
            pass
    encoder_state: dict[str, torch.Tensor] | None = None
    if encoder_state_path and os.path.isfile(encoder_state_path):
        try:
            blob = torch.load(
                encoder_state_path, map_location="cpu", weights_only=False
            )
            encoder_state = blob.get("question_encoder_state")
        except Exception:
            encoder_state = None
    return booster, norm, encoder_state


def _gpu_alloc_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _measure_peak_mb(fn, n_warmup: int = 2, n_runs: int = 5) -> dict[str, float]:
    """Run fn() a few times, capture peak GPU + delta RSS during the call.

    Returns {"gpu_peak_mb": ..., "rss_peak_mb_delta": ...}.
    For CPU-only models the GPU figure will be ~0; the RSS delta is the
    relevant number there."""
    for _ in range(n_warmup):
        fn()
    rss_before = _process_rss_mb()
    rss_peak = rss_before
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(n_runs):
        fn()
        rss_now = _process_rss_mb()
        if rss_now > rss_peak:
            rss_peak = rss_now
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        gpu_peak = 0.0
    return {
        "gpu_peak_mb": float(gpu_peak),
        "rss_peak_mb_delta": float(max(0.0, rss_peak - rss_before)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live JEPA vs LLM benchmark using fresh cluster telemetry."
    )
    ap.add_argument(
        "--checkpoint", default=os.getenv("CHECKPOINT_PATH", "checkpoints/lpse2.pt")
    )
    ap.add_argument(
        "--mlp-checkpoint",
        default=os.getenv("MLP_CHECKPOINT_PATH", "checkpoints/baseline_mlp_v2.pt"),
        help="Path to MLP baseline checkpoint (set empty to skip).",
    )
    ap.add_argument(
        "--xgb-model",
        default=os.getenv("XGB_MODEL_PATH", "checkpoints/baseline_xgb_v2.json"),
        help="Path to XGBoost booster JSON (set empty to skip).",
    )
    ap.add_argument(
        "--xgb-norm",
        default=os.getenv(
            "XGB_NORMALIZER_PATH", "checkpoints/baseline_xgb_v2_norm.pkl"
        ),
        help="Path to XGBoost normalizer pickle.",
    )
    ap.add_argument(
        "--xgb-encoder-state",
        default=os.getenv(
            "XGB_ENCODER_STATE_PATH", "checkpoints/baseline_xgb_v2_encoder.pt"
        ),
        help="Path to XGBoost question_encoder state file (saved at training).",
    )
    ap.add_argument(
        "--llm-options",
        choices=("open", "category"),
        default=os.getenv("LLM_OPTIONS", "open"),
        help="LLM answer protocol: 'open' = choose from the full "
        "answer-template inventory (default, matches the "
        "94-way decision the local models make); 'category' = "
        "prompt offers the truth's whole answer category "
        "(truth + all same-category distractors).",
    )
    ap.add_argument(
        "--paraphrase-questions",
        action="store_true",
        help="Rewrite each generated question's `raw` text with a "
        "semantically equivalent but lexically distinct phrasing "
        "not present in the training question templates. The "
        "answer index is unchanged.",
    )
    ap.add_argument(
        "--exclude-node-ip",
        default="",
        help="If set, exclude this node IP from live discovery. The "
        "cluster token, semantic features, and QA generation all "
        "operate on the remaining nodes only --- simulates a "
        "node going dark for the duration of the benchmark.",
    )
    ap.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", "16")))
    ap.add_argument(
        "--sample-interval-s",
        type=float,
        default=float(os.getenv("SCRAPE_INTERVAL", "1.0")),
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of live benchmark steps after warmup",
    )
    ap.add_argument("--questions-per-step", type=int, default=6)
    ap.add_argument(
        "--max-questions",
        type=int,
        default=100,
        help="Safety cap across all steps (0=unlimited)",
    )
    ap.add_argument(
        "--normalizer-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Update running normalizer with live data before normalization.",
    )
    ap.add_argument("--seed", type=int, default=1337)

    default_chat_url = os.getenv("VLLM_CHAT_URL", "")
    if not default_chat_url:
        base = os.getenv("VLLM_BASE_URL", "")
        if base:
            default_chat_url = base

    ap.add_argument("--vllm-chat-url", default=default_chat_url)
    ap.add_argument("--vllm-model", default=os.getenv("VLLM_MODEL", ""))
    ap.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY", ""))
    ap.add_argument(
        "--vllm-timeout-s", type=float, default=float(os.getenv("VLLM_TIMEOUT_S", "60"))
    )
    ap.add_argument(
        "--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.0"))
    )
    ap.add_argument(
        "--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "1024"))
    )

    ap.add_argument("--results-csv", default="jepa_llm_live_comparison.csv")
    ap.add_argument("--summary-json", default="jepa_llm_live_summary.json")
    ap.add_argument("--llm-raw-out", default="jepa_llm_live_raw_outputs.ndjson")

    args = ap.parse_args()

    random.seed(args.seed)

    chat_url = _ensure_chat_url(args.vllm_chat_url)
    if not chat_url or not args.vllm_model:
        raise SystemExit(
            "LLM endpoint config is required for this benchmark. Provide --vllm-chat-url and --vllm-model."
        )

    print("=== live jepa vs llm benchmark ===")
    print(f"checkpoint={args.checkpoint}")
    print(f"seq_len={args.seq_len} sample_interval_s={args.sample_interval_s}")
    print(
        f"steps={args.steps} questions_per_step={args.questions_per_step} max_questions={args.max_questions}"
    )
    print(f"vllm_chat_url={chat_url}")
    print(f"vllm_model={args.vllm_model}")
    print(f"results_csv={args.results_csv}")
    print(f"summary_json={args.summary_json}")

    model, question_encoder, normalizer, node_ips, node_cores, answer_templates = (
        load_model(args.checkpoint)
    )
    model.eval()
    question_encoder.eval()
    device = next(model.parameters()).device

    if not answer_templates:
        answer_templates = ANSWER_TEMPLATES

    from lpse.question_encoder import build_question_encoder

    def _build_encoder_for_baseline(
        persisted_state: dict[str, torch.Tensor] | None,
    ) -> torch.nn.Module:
        if persisted_state is None:
            print(
                "  [warn] no persisted baseline encoder state; falling back to shared LPSE encoder"
            )
            return question_encoder

        proj_key = "proj.0.weight"
        use_structured = True
        if proj_key in persisted_state:
            proj_in = int(persisted_state[proj_key].shape[1])
            if proj_in == 416:
                use_structured = False
        enc = build_question_encoder(
            backend="hybrid", out_dim=QUESTION_DIM, use_structured=use_structured
        ).to(device)
        try:
            missing, unexpected = enc.load_state_dict(persisted_state, strict=False)

            trainable_missing = [k for k in missing if not k.startswith("st_model.")]
            if trainable_missing:
                print(
                    f"  [warn] encoder load missing trainable keys={len(trainable_missing)} "
                    f"(unexpected={len(unexpected)}); proj may be partially initialized"
                )
            else:
                print(
                    f"  encoder load: missing={len(missing)} unexpected={len(unexpected)} "
                    f"(use_structured={use_structured})"
                )
        except Exception as e:
            print(f"  [warn] encoder load failed ({e}); using LPSE encoder")
            return question_encoder
        enc.eval()
        return enc

    mlp_model: _FlatMLPClassifier | None = None
    mlp_norm: RunningNormalizer | None = None
    mlp_encoder: torch.nn.Module = question_encoder
    mlp_in_dim: int | None = None
    if args.mlp_checkpoint:
        mlp_model, mlp_norm, mlp_cfg, mlp_enc_state = _load_mlp(
            args.mlp_checkpoint, device
        )
        mlp_in_dim = int(mlp_cfg.get("in_dim", -1))
        mlp_encoder = _build_encoder_for_baseline(mlp_enc_state)
        used = "persisted" if mlp_enc_state is not None else "shared LPSE"
        print(
            f"loaded MLP: {args.mlp_checkpoint}  params={sum(p.numel() for p in mlp_model.parameters())}  in_dim={mlp_in_dim}  encoder={used}"
        )
    else:
        print("MLP skipped (no checkpoint)")

    xgb_booster: xgb.Booster | None = None
    xgb_norm: RunningNormalizer | None = None
    xgb_encoder: torch.nn.Module = question_encoder
    if args.xgb_model:
        xgb_booster, xgb_norm, xgb_enc_state = _load_xgb(
            args.xgb_model, args.xgb_norm, args.xgb_encoder_state
        )
        xgb_encoder = _build_encoder_for_baseline(xgb_enc_state)
        used = "persisted" if xgb_enc_state is not None else "shared LPSE"
        print(
            f"loaded XGB: {args.xgb_model}  trees={xgb_booster.num_boosted_rounds()}  encoder={used}"
        )
    else:
        print("XGB skipped (no model)")

    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be > 0")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.questions_per_step <= 0:
        raise SystemExit("--questions-per-step must be > 0")

    print(f"checkpoint_nodes={len(node_ips)}")
    for i, ip in enumerate(node_ips):
        cores = node_cores[i] if i < len(node_cores) else -1
        print(f"  node[{i}] ip={ip} cores={cores}")

    ctx_buffer_raw: deque[torch.Tensor] = deque(maxlen=args.seq_len)
    print(f"warming_up_context: need {args.seq_len} fresh snapshots")

    latest_semantic: dict[str, dict[str, float | None]] = {}

    excluded_ip_warmup = (args.exclude_node_ip or "").strip()
    for i in range(args.seq_len):
        live_node_ips = get_node_ips() or node_ips
        if excluded_ip_warmup:
            live_node_ips = [ip for ip in live_node_ips if ip != excluded_ip_warmup]
        live_node_cores = []
        for ip in live_node_ips:
            cores = get_num_cores_from_node_exporter(ip)
            live_node_cores.append(cores if cores > 0 else 10)

        node_tensors, raw_parsed = collect_full_cluster_snapshot(live_node_ips)
        flat = cluster_snapshot_to_token(node_tensors, live_node_ips)
        if args.normalizer_update:
            normalizer.update(flat)
        ctx_buffer_raw.append(flat.detach().cpu())
        latest_semantic = compute_semantic_features(raw_parsed, live_node_ips)

        print(f"  warmup {i + 1}/{args.seq_len}")
        if i < args.seq_len - 1:
            time.sleep(max(0.0, float(args.sample_interval_s)))

    csv_fields = [
        "timestamp",
        "step",
        "question_id",
        "question",
        "truth_idx",
        "truth_template",
        "jepa_idx",
        "jepa_template",
        "jepa_confidence",
        "jepa_correct",
        "jepa_latency_ms",
        "mlp_idx",
        "mlp_template",
        "mlp_correct",
        "mlp_latency_ms",
        "xgb_idx",
        "xgb_template",
        "xgb_correct",
        "xgb_latency_ms",
        "llm_idx",
        "llm_template",
        "llm_correct",
        "llm_latency_ms",
        "llm_error",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
    ]

    jepa_acc = AccuracyStats()
    mlp_acc = AccuracyStats()
    xgb_acc = AccuracyStats()
    llm_acc_overall = AccuracyStats()
    llm_acc_valid = AccuracyStats()
    agreement = AccuracyStats()

    jepa_latencies: list[float] = []
    mlp_latencies: list[float] = []
    xgb_latencies: list[float] = []
    llm_latencies: list[float] = []
    llm_failures = 0
    llm_valid = 0

    question_id = 0

    raw_window_probe = torch.stack(list(ctx_buffer_raw), dim=0)
    probe_q = [
        {"raw": "what is the cpu usage", "intent": "metric_value", "scope": "cluster"}
    ]
    probe_q_emb = encode_questions(question_encoder, probe_q, node_ips)
    jepa_window_probe = (
        torch.stack([normalizer.normalize(r) for r in raw_window_probe], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    def _params_mb(m: nn.Module) -> float:
        return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 * 1024)

    def _working_set_mb(fn) -> float:
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return max(0.0, (peak - before) / (1024 * 1024))

    print("\n=== memory footprint ===")
    jepa_static_mb = _params_mb(model) + _params_mb(question_encoder)
    jepa_ws_mb = _working_set_mb(lambda: model.infer(jepa_window_probe, probe_q_emb))
    print(f"LPSE: params_mb={jepa_static_mb:.1f}  working_set_mb={jepa_ws_mb:.1f}")
    mem_jepa = {
        "params_mb": jepa_static_mb,
        "working_set_mb": jepa_ws_mb,
        "total_mb": jepa_static_mb + jepa_ws_mb,
    }

    if mlp_model is not None:
        _mlp_norm_for_probe = (
            normalizer if (mlp_norm is None or mlp_norm.count == 0) else mlp_norm
        )
        mlp_window_probe = (
            torch.stack(
                [_mlp_norm_for_probe.normalize(r) for r in raw_window_probe], dim=0
            )
            .unsqueeze(0)
            .to(device)
        )
        mlp_static_mb = _params_mb(mlp_model)

        def _do_mlp():
            x = torch.cat([mlp_window_probe.flatten(1), probe_q_emb], dim=-1)
            with torch.no_grad():
                mlp_model(x)

        mlp_ws_mb = _working_set_mb(_do_mlp)
        print(f"MLP    : params_mb={mlp_static_mb:.1f}  working_set_mb={mlp_ws_mb:.1f}")
        mem_mlp = {
            "params_mb": mlp_static_mb,
            "working_set_mb": mlp_ws_mb,
            "total_mb": mlp_static_mb + mlp_ws_mb,
        }
    else:
        mem_mlp = {"params_mb": 0.0, "working_set_mb": 0.0, "total_mb": 0.0}

    if xgb_booster is not None:
        _xgb_norm_for_probe = (
            normalizer if (xgb_norm is None or xgb_norm.count == 0) else xgb_norm
        )
        xgb_window_probe = (
            torch.stack(
                [_xgb_norm_for_probe.normalize(r) for r in raw_window_probe], dim=0
            )
            .unsqueeze(0)
            .to(device)
        )
        xgb_disk_mb = os.path.getsize(args.xgb_model) / (1024 * 1024)

        def _do_xgb():
            flat = xgb_window_probe.flatten(1).detach().cpu().numpy().astype(np.float32)
            q_np = probe_q_emb.detach().cpu().numpy().astype(np.float32)
            x = np.concatenate([flat, q_np], axis=1)
            xgb_booster.predict(xgb.DMatrix(x))

        xgb_ws_mb = _working_set_mb(_do_xgb)
        print(
            f"XGBoost: model_disk_mb={xgb_disk_mb:.1f}  working_set_mb={xgb_ws_mb:.1f}"
        )
        mem_xgb = {
            "params_mb": xgb_disk_mb,
            "working_set_mb": xgb_ws_mb,
            "total_mb": xgb_disk_mb + xgb_ws_mb,
        }
    else:
        mem_xgb = {"params_mb": 0.0, "working_set_mb": 0.0, "total_mb": 0.0}

    raw_out_fh = (
        open(args.llm_raw_out, "w", encoding="utf-8") if args.llm_raw_out else None
    )

    with open(args.results_csv, "w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()

        excluded_ip = (args.exclude_node_ip or "").strip()
        if excluded_ip:
            print(f"PERTURB: excluding node IP {excluded_ip} from live discovery")
        with requests.Session() as session:
            for step in range(1, args.steps + 1):
                live_node_ips = get_node_ips() or node_ips
                if excluded_ip:
                    live_node_ips = [ip for ip in live_node_ips if ip != excluded_ip]
                live_node_cores = []
                for ip in live_node_ips:
                    cores = get_num_cores_from_node_exporter(ip)
                    live_node_cores.append(cores if cores > 0 else 10)

                node_tensors, raw_parsed = collect_full_cluster_snapshot(live_node_ips)
                flat = cluster_snapshot_to_token(node_tensors, live_node_ips)
                if args.normalizer_update:
                    normalizer.update(flat)
                ctx_buffer_raw.append(flat.detach().cpu())
                latest_semantic = compute_semantic_features(raw_parsed, live_node_ips)

                raw_window = torch.stack(list(ctx_buffer_raw), dim=0)
                jepa_window = torch.stack(
                    [normalizer.normalize(r) for r in raw_window], dim=0
                )
                context = jepa_window.unsqueeze(0).to(device)

                def _norm_for(baseline_norm):
                    return (
                        normalizer
                        if baseline_norm is None or baseline_norm.count == 0
                        else baseline_norm
                    )

                if mlp_model is not None:
                    eff_norm = _norm_for(mlp_norm)
                    if eff_norm is normalizer:
                        mlp_context = context
                    else:
                        mlp_window = torch.stack(
                            [eff_norm.normalize(r) for r in raw_window], dim=0
                        )
                        mlp_context = mlp_window.unsqueeze(0).to(device)
                else:
                    mlp_context = None
                if xgb_booster is not None:
                    eff_norm = _norm_for(xgb_norm)
                    if eff_norm is normalizer:
                        xgb_context = context
                    else:
                        xgb_window = torch.stack(
                            [eff_norm.normalize(r) for r in raw_window], dim=0
                        )
                        xgb_context = xgb_window.unsqueeze(0).to(device)
                else:
                    xgb_context = None

                qa_pairs = generate_qa_pairs(
                    semantic_features=latest_semantic,
                    node_ips=live_node_ips,
                    node_cores=live_node_cores,
                    max_pairs=args.questions_per_step,
                )
                if args.paraphrase_questions and qa_pairs:
                    new_pairs: list[tuple[dict, int]] = []
                    for q_struct, ans_idx in qa_pairs:
                        if isinstance(q_struct, dict):
                            new_q = dict(q_struct)
                            new_q["raw"] = _paraphrase_question(new_q.get("raw", ""))
                        else:
                            new_q = q_struct
                        new_pairs.append((new_q, ans_idx))
                    qa_pairs = new_pairs
                if not qa_pairs:
                    print(f"step={step}: no QA pairs generated, skipping")
                    time.sleep(max(0.0, float(args.sample_interval_s)))
                    continue

                random.shuffle(qa_pairs)

                print(f"step={step}: evaluating {len(qa_pairs)} questions")

                for q_struct, truth_idx in qa_pairs:
                    if args.max_questions > 0 and question_id >= args.max_questions:
                        break

                    question_id += 1
                    question_text = (
                        q_struct.get("raw", "") if isinstance(q_struct, dict) else ""
                    )

                    if args.llm_options == "category":
                        llm_option_indices: list[int] | None = _category_option_indices(
                            int(truth_idx)
                        )
                    else:
                        llm_option_indices = None

                    t0 = time.perf_counter()
                    q_emb = encode_questions(
                        question_encoder, [q_struct], live_node_ips
                    )
                    q_tok = (
                        encode_question_tokens(question_encoder, [q_struct])
                        if getattr(model, "use_token_sequence", False)
                        else None
                    )
                    with torch.no_grad():
                        jepa_idx, jepa_conf, _ = model.infer(context, q_emb, q_tok)
                    jepa_latency_ms = (time.perf_counter() - t0) * 1000.0

                    jepa_is_correct = int(jepa_idx == int(truth_idx))
                    jepa_acc.add(bool(jepa_is_correct))
                    jepa_latencies.append(jepa_latency_ms)

                    mlp_idx: int | None = None
                    mlp_latency_ms = float("nan")
                    if mlp_model is not None and mlp_context is not None:
                        t1 = time.perf_counter()
                        with torch.no_grad():
                            mlp_q_emb = (
                                q_emb
                                if mlp_encoder is question_encoder
                                else encode_questions(
                                    mlp_encoder, [q_struct], live_node_ips
                                )
                            )
                            x_mlp = torch.cat(
                                [mlp_context.flatten(1), mlp_q_emb], dim=-1
                            )
                            logits_mlp = mlp_model(x_mlp)
                            mlp_idx_t = int(logits_mlp.argmax(dim=-1).item())
                        mlp_latency_ms = (time.perf_counter() - t1) * 1000.0
                        mlp_idx = mlp_idx_t
                        mlp_is_correct = int(mlp_idx == int(truth_idx))
                        mlp_acc.add(bool(mlp_is_correct))
                        mlp_latencies.append(mlp_latency_ms)
                    else:
                        mlp_is_correct = 0

                    xgb_idx: int | None = None
                    xgb_latency_ms = float("nan")
                    if xgb_booster is not None and xgb_context is not None:
                        t2 = time.perf_counter()
                        with torch.no_grad():
                            xgb_q_emb = (
                                q_emb
                                if xgb_encoder is question_encoder
                                else encode_questions(
                                    xgb_encoder, [q_struct], live_node_ips
                                )
                            )
                        flat_np = (
                            xgb_context.flatten(1)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        q_np = xgb_q_emb.detach().cpu().numpy().astype(np.float32)
                        x_xgb = np.concatenate([flat_np, q_np], axis=1)
                        d = xgb.DMatrix(x_xgb)
                        probs = xgb_booster.predict(d)
                        xgb_idx_t = int(probs.argmax(axis=1)[0])
                        xgb_latency_ms = (time.perf_counter() - t2) * 1000.0
                        xgb_idx = xgb_idx_t
                        xgb_is_correct = int(xgb_idx == int(truth_idx))
                        xgb_acc.add(bool(xgb_is_correct))
                        xgb_latencies.append(xgb_latency_ms)
                    else:
                        xgb_is_correct = 0

                    llm_idx, llm_latency_ms, llm_error, llm_raw_content, usage = (
                        call_llm_for_answer_index(
                            session=session,
                            chat_url=chat_url,
                            model=args.vllm_model,
                            api_key=(args.vllm_api_key or None),
                            question_text=question_text,
                            semantic_features=latest_semantic,
                            answer_templates=answer_templates,
                            node_ips=live_node_ips,
                            node_cores=live_node_cores,
                            option_indices=llm_option_indices,
                            timeout_s=float(args.vllm_timeout_s),
                            temperature=float(args.temperature),
                            max_tokens=int(args.max_tokens),
                        )
                    )

                    if llm_idx is None:
                        llm_failures += 1
                    else:
                        llm_valid += 1
                        llm_latencies.append(llm_latency_ms)

                    llm_is_correct = (
                        int(llm_idx == int(truth_idx)) if llm_idx is not None else 0
                    )
                    llm_acc_overall.add(bool(llm_is_correct))
                    if llm_idx is not None:
                        llm_acc_valid.add(bool(llm_is_correct))
                        agreement.add(bool(llm_idx == jepa_idx))

                    def _tpl(idx):
                        return (
                            answer_templates[int(idx)]
                            if idx is not None and 0 <= int(idx) < len(answer_templates)
                            else ""
                        )

                    truth_template = _tpl(truth_idx)
                    jepa_template = _tpl(jepa_idx)
                    mlp_template = _tpl(mlp_idx)
                    xgb_template = _tpl(xgb_idx)
                    llm_template = _tpl(llm_idx)

                    writer.writerow(
                        {
                            "timestamp": _utc_now_iso(),
                            "step": step,
                            "question_id": question_id,
                            "question": question_text,
                            "truth_idx": int(truth_idx),
                            "truth_template": truth_template,
                            "jepa_idx": int(jepa_idx),
                            "jepa_template": jepa_template,
                            "jepa_confidence": float(jepa_conf),
                            "jepa_correct": jepa_is_correct,
                            "jepa_latency_ms": jepa_latency_ms,
                            "mlp_idx": ("" if mlp_idx is None else int(mlp_idx)),
                            "mlp_template": mlp_template,
                            "mlp_correct": mlp_is_correct,
                            "mlp_latency_ms": (
                                "" if mlp_idx is None else mlp_latency_ms
                            ),
                            "xgb_idx": ("" if xgb_idx is None else int(xgb_idx)),
                            "xgb_template": xgb_template,
                            "xgb_correct": xgb_is_correct,
                            "xgb_latency_ms": (
                                "" if xgb_idx is None else xgb_latency_ms
                            ),
                            "llm_idx": ("" if llm_idx is None else int(llm_idx)),
                            "llm_template": llm_template,
                            "llm_correct": llm_is_correct,
                            "llm_latency_ms": (
                                "" if llm_idx is None else llm_latency_ms
                            ),
                            "llm_error": llm_error,
                            "llm_prompt_tokens": usage.get("prompt_tokens"),
                            "llm_completion_tokens": usage.get("completion_tokens"),
                            "llm_total_tokens": usage.get("total_tokens"),
                        }
                    )

                    if raw_out_fh is not None:
                        raw_out_fh.write(
                            json.dumps(
                                {
                                    "timestamp": _utc_now_iso(),
                                    "step": step,
                                    "question_id": question_id,
                                    "question": question_text,
                                    "truth_idx": int(truth_idx),
                                    "llm_options_mode": args.llm_options,
                                    "llm_option_indices": llm_option_indices,
                                    "llm_idx": llm_idx,
                                    "llm_error": llm_error,
                                    "llm_raw_content": llm_raw_content,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                if args.max_questions > 0 and question_id >= args.max_questions:
                    break

                if step < args.steps:
                    time.sleep(max(0.0, float(args.sample_interval_s)))

    if raw_out_fh is not None:
        raw_out_fh.close()

    def _stats(
        acc: AccuracyStats,
        lats: list[float],
        mem: dict[str, float],
        num_params: int | None = None,
    ) -> dict[str, Any]:
        return {
            "questions": acc.total,
            "correct": acc.correct,
            "accuracy": acc.acc(),
            "latency_ms": {
                "mean": _mean(lats),
                "p50": _percentile(lats, 50),
                "p95": _percentile(lats, 95),
            },
            "memory": {
                "params_mb": mem.get("params_mb"),
                "working_set_mb": mem.get("working_set_mb"),
                "total_mb": mem.get("total_mb"),
            },
            "num_params": num_params,
        }

    jepa_summary = _stats(
        jepa_acc,
        jepa_latencies,
        mem_jepa,
        num_params=sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in question_encoder.parameters()),
    )
    mlp_summary = _stats(
        mlp_acc,
        mlp_latencies,
        mem_mlp,
        num_params=(
            sum(p.numel() for p in mlp_model.parameters())
            if mlp_model is not None
            else None
        ),
    )
    xgb_summary = _stats(
        xgb_acc,
        xgb_latencies,
        mem_xgb,
        num_params=(
            xgb_booster.num_boosted_rounds() if xgb_booster is not None else None
        ),
    )

    llm_summary = {
        "questions": llm_acc_overall.total,
        "correct_overall": llm_acc_overall.correct,
        "accuracy_overall": llm_acc_overall.acc(),
        "valid_predictions": llm_valid,
        "parse_failures": llm_failures,
        "accuracy_valid_only": llm_acc_valid.acc(),
        "latency_ms_valid_only": {
            "mean": _mean(llm_latencies),
            "p50": _percentile(llm_latencies, 50),
            "p95": _percentile(llm_latencies, 95),
        },
    }

    summary = {
        "generated_at": _utc_now_iso(),
        "config": {
            "checkpoint": args.checkpoint,
            "seq_len": args.seq_len,
            "sample_interval_s": args.sample_interval_s,
            "steps": args.steps,
            "questions_per_step": args.questions_per_step,
            "max_questions": args.max_questions,
            "vllm_chat_url": chat_url,
            "vllm_model": args.vllm_model,
            "llm_options": args.llm_options,
            "normalizer_update": bool(args.normalizer_update),
            "node_ips": node_ips,
            "node_cores": node_cores,
            "num_answer_templates": len(answer_templates),
        },
        "jepa": jepa_summary,
        "mlp": mlp_summary,
        "xgb": xgb_summary,
        "llm": llm_summary,
        "agreement": {
            "compared": agreement.total,
            "same_prediction": agreement.correct,
            "rate": agreement.acc(),
        },
        "deltas": {
            "accuracy_overall_llm_minus_jepa": (
                None
                if llm_summary["accuracy_overall"] is None
                or jepa_summary["accuracy"] is None
                else llm_summary["accuracy_overall"] - jepa_summary["accuracy"]
            ),
            "latency_mean_ratio_llm_over_jepa": (
                None
                if llm_summary["latency_ms_valid_only"]["mean"] is None
                or jepa_summary["latency_ms"]["mean"] is None
                or jepa_summary["latency_ms"]["mean"] == 0
                else llm_summary["latency_ms_valid_only"]["mean"]
                / jepa_summary["latency_ms"]["mean"]
            ),
        },
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def _print_model(name: str, s: dict[str, Any]) -> None:
        if s["accuracy"] is None:
            print(f"{name}: skipped")
            return
        print(
            f"{name} accuracy={s['accuracy'] * 100:.2f}% (n={s['questions']})  "
            f"latency_ms mean={_fmt(s['latency_ms']['mean'])} "
            f"p50={_fmt(s['latency_ms']['p50'])} p95={_fmt(s['latency_ms']['p95'])}  "
            f"mem_total_mb={_fmt(s['memory']['total_mb'], 1)} "
            f"(params={_fmt(s['memory']['params_mb'], 1)} ws={_fmt(s['memory']['working_set_mb'], 1)})"
        )

    print("\n=== benchmark summary ===")
    _print_model("LPSE", jepa_summary)
    _print_model("MLP    ", mlp_summary)
    _print_model("XGBoost", xgb_summary)

    print(
        f"LLM accuracy_overall={_fmt(llm_summary['accuracy_overall'] * 100 if llm_summary['accuracy_overall'] is not None else None, 2)}% "
        f"(n={llm_summary['questions']}, valid={llm_summary['valid_predictions']}, failures={llm_summary['parse_failures']})"
    )
    print(
        f"LLM accuracy_valid_only={_fmt(llm_summary['accuracy_valid_only'] * 100 if llm_summary['accuracy_valid_only'] is not None else None, 2)}%"
    )
    print(
        "LLM latency_ms(valid) "
        f"mean={_fmt(llm_summary['latency_ms_valid_only']['mean'])} "
        f"p50={_fmt(llm_summary['latency_ms_valid_only']['p50'])} "
        f"p95={_fmt(llm_summary['latency_ms_valid_only']['p95'])}"
    )

    print(
        f"agreement(JEPA==LLM)={_fmt(summary['agreement']['rate'] * 100 if summary['agreement']['rate'] is not None else None, 2)}% "
        f"(n={summary['agreement']['compared']})"
    )

    print(
        f"accuracy_delta(llm-jepa)={_fmt(summary['deltas']['accuracy_overall_llm_minus_jepa'], 6)}"
    )
    print(
        f"latency_ratio_mean(llm/jepa)={_fmt(summary['deltas']['latency_mean_ratio_llm_over_jepa'], 3)}"
    )

    print(f"\nresults_csv={args.results_csv}")
    print(f"summary_json={args.summary_json}")
    if args.llm_raw_out:
        print(f"llm_raw_out={args.llm_raw_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
