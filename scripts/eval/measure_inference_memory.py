#!/usr/bin/env python3
"""Standalone inference-memory probe for the four local systems used in
Table 1. Reports static parameter MB and one-call working-set MB for each
model loaded in isolation, then a sanity check across the combined load.

This exists because the in-process benchmark accounting in
benchmark_llm_forecast.py double-counted LPSE's question encoder (it is
a submodule of the JEPA model, so summing both was redundant) and
under-counted MLP/XGBoost (their checkpoints don't bundle the encoder, but
they still need it at inference). Run this directly to print the corrected
numbers; the accuracy/latency numbers in the JSON summary are unaffected.
"""

import os
import sys

import torch
import torch.nn as nn
import xgboost as xgb

os.environ.setdefault("CHECKPOINT_PATH", "checkpoints/lpsehybrid.pt")

from lpse.apps.infer_lpse import load_model
from lpse.prometheus_ingest import CLUSTER_TOKEN_DIM
from lpse.question_encoder import QUESTION_DIM, build_question_encoder, encode_questions

SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _params_mb(m: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 * 1024)


def _buffers_mb(m: nn.Module) -> float:
    return sum(b.numel() * b.element_size() for b in m.buffers()) / (1024 * 1024)


def _ws_mb(fn) -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0.0, (peak - before) / (1024 * 1024))


class _FlatMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_answers, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_answers))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> int:

    probe_q = [
        {"raw": "what is the cpu usage", "intent": "metric_value", "scope": "cluster"}
    ]
    ips = os.environ.get(
        "INFERENCE_NODES",
        "10.68.184.21,10.68.184.22,10.68.184.23,"
        "10.68.184.24,10.68.184.25,10.68.184.26,10.68.184.122",
    ).split(",")

    jepa, qe_jepa, _, _, _, _ = load_model("checkpoints/lpsehybrid.pt")
    jepa.eval()
    qe_jepa.eval()

    jepa_params = _params_mb(jepa) + _buffers_mb(jepa)
    raw = torch.zeros(1, SEQ_LEN, CLUSTER_TOKEN_DIM, device=DEVICE)
    q_emb = encode_questions(qe_jepa, probe_q, ips)
    jepa_ws = _ws_mb(lambda: jepa.infer(raw, q_emb))
    print(
        f"LPSE  params_mb={jepa_params:7.1f}  working_set_mb={jepa_ws:5.1f}  total_mb={jepa_params + jepa_ws:7.1f}"
    )

    mlp_ck = torch.load(
        "checkpoints/baseline_mlp_v2.pt", map_location=DEVICE, weights_only=False
    )
    cfg = mlp_ck["model_config"]
    mlp = _FlatMLP(
        cfg["in_dim"],
        cfg["hidden_dim"],
        cfg["hidden_layers"],
        cfg["num_answers"],
        cfg["dropout"],
    ).to(DEVICE)
    mlp.load_state_dict(mlp_ck["model"])
    mlp.eval()
    qe_mlp = build_question_encoder(backend="hybrid", out_dim=QUESTION_DIM).to(DEVICE)
    qe_mlp.load_state_dict(mlp_ck.get("question_encoder_state", {}), strict=False)
    qe_mlp.eval()

    mlp_params = (
        _params_mb(mlp) + _params_mb(qe_mlp) + _buffers_mb(mlp) + _buffers_mb(qe_mlp)
    )
    q_emb_mlp = encode_questions(qe_mlp, probe_q, ips)

    def _do_mlp():
        with torch.no_grad():
            x = torch.cat([raw.flatten(1), q_emb_mlp], dim=-1)
            mlp(x)

    mlp_ws = _ws_mb(_do_mlp)
    print(
        f"MLP      params_mb={mlp_params:7.1f}  working_set_mb={mlp_ws:5.1f}  total_mb={mlp_params + mlp_ws:7.1f}"
    )

    booster = xgb.Booster()
    booster.load_model("checkpoints/baseline_xgb_v2.json")
    xgb_disk = os.path.getsize("checkpoints/baseline_xgb_v2.json") / (1024 * 1024)
    qe_xgb = build_question_encoder(backend="hybrid", out_dim=QUESTION_DIM).to(DEVICE)
    enc_blob = torch.load(
        "checkpoints/baseline_xgb_v2_encoder.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    qe_xgb.load_state_dict(enc_blob.get("question_encoder_state", {}), strict=False)
    qe_xgb.eval()

    xgb_params = xgb_disk + _params_mb(qe_xgb) + _buffers_mb(qe_xgb)
    q_emb_xgb = encode_questions(qe_xgb, probe_q, ips)

    def _do_xgb():
        flat = raw.flatten(1).detach().cpu().numpy()
        q_np = q_emb_xgb.detach().cpu().numpy()
        import numpy as np

        x = np.concatenate([flat, q_np], axis=1).astype("float32")
        booster.predict(xgb.DMatrix(x))

    xgb_ws = _ws_mb(_do_xgb)
    print(
        f"XGBoost  params_mb={xgb_params:7.1f}  working_set_mb={xgb_ws:5.1f}  total_mb={xgb_params + xgb_ws:7.1f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
