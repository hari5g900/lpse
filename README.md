# LPSE

LPSE (Low-latency Predictive State Estimator): a low-latency semantic state
estimator for dynamic network monitoring and orchestration, built on
I-JEPA-style latent predictive learning over streaming Kubernetes telemetry.

## What it does

LPSE continuously scrapes per-node node-exporter metrics from a running
Kubernetes cluster, packs each snapshot into a fixed-size cluster token, and
trains with a JEPA-style self-supervised loss (EMA future-latent prediction
plus a VICReg anti-collapse term) alongside a supervised QA loss over a fixed
answer-codebook. At inference it answers natural-language questions about
cluster state ("Is any node CPU-bound?") by scoring the answer codebook, in a
single forward pass — no LLM call on the serving path. The same replay data
pipeline trains MLP, LSTM/Transformer, and XGBoost baselines so all systems
are compared on identical (context, question, answer) data.

## Architecture

The `lpse/` package contains the model and data pipeline shared by every
trainer:

| Module | Role |
| --- | --- |
| `lpse/collectors.py` | Kubernetes/node-exporter topology discovery (node IPs, core counts); loads the kubeconfig lazily on first API use |
| `lpse/metrics_shared.py` | Shared node-exporter fetch/parse helpers and capacity features |
| `lpse/prometheus_ingest.py` | Scrapes `/metrics`, summarizes per metric family, packs a fixed per-node feature tensor into a cluster token; running normalizer and semantic features |
| `lpse/question_encoder.py` | Question → fixed-size embedding; backends `keyword`, `sentence_transformer`, `hybrid` |
| `lpse/qa_generator.py` | Generates (question, answer) pairs from a snapshot; answers are indices into a fixed template codebook |
| `lpse/lpse.py` | The LPSE model: temporal metric encoder, fusion transformer, and answer-codebook scoring (plus EMA target encoder and JEPA predictor) |
| `lpse/replay_buffer.py` | Replay buffer shared by the LPSE, MLP, and XGBoost trainers; contexts interned so each snapshot is stored once |
| `lpse/replay_dump.py` | Sharded on-disk dump of (context, QA, metadata) triples, written by the live trainer and replayed offline by the baselines |

## Install

Requires Python >= 3.10.

```bash
pip install -e .
```

Torch is installed with the package; install a CUDA build for GPU training.
Note: pin `transformers` below 5.x — 5.x breaks the sentence-transformers
import path.

## Cluster access

The live trainers and `lpse/apps/infer_lpse.py` use the Kubernetes Python client. Point
it at your cluster:

```bash
export KUBECONFIG_PATH=/path/to/your/kubeconfig
```

Or place the file at `./config` (the default the loader looks for). Offline
replay training does not need cluster access.

## Quickstart — training

All entry points run from the repository root.

**1. LPSE (live cluster, JEPA SSL + QA losses).** Writes a sharded replay
dump by default (set `DUMP_DIR=""` to disable):

```bash
Q_ENCODER_BACKEND=hybrid LPSE_RUN_SUFFIX=hybrid python -m lpse.apps.train_lpse
```

Writes `artifacts/checkpoints/lpsehybrid.pt`,
`artifacts/logs/lpse_training_loghybrid.csv`,
`artifacts/logs/lpse_summaryhybrid.txt`, and replay shards plus
`meta.json` under `artifacts/dumps/jepa_live_lap/`. Shards are atomic and
flush via `atexit` on SIGTERM / Ctrl+C.

**2. Baselines, offline replay of the dump above** (no cluster access):

```bash
LOAD_REPLAY_PATH=artifacts/dumps/jepa_live_lap Q_ENCODER_BACKEND=hybrid \
OFFLINE_EPOCHS=10 python -m lpse.apps.train_baseline_mlp

LOAD_REPLAY_PATH=artifacts/dumps/jepa_live_lap Q_ENCODER_BACKEND=hybrid \
XGB_DEVICE=cuda OFFLINE_NUM_ROUNDS=500 OFFLINE_EVAL_EVERY=10 \
python -m lpse.apps.train_baseline_xgb
```

Also available: `lpse/apps/train_baseline_lstm.py` and
`lpse/apps/train_baseline_transformer.py` (temporal baselines over the full 16-step
window, same env vars as the MLP baseline), `lpse/apps/train_lpse_offline.py`
(trains LPSE itself from a replay dump), and `lpse/apps/train_lpse_no_future.py`
(single-variable ablation with `JEPA_SSL_WEIGHT=0`).

All trainers emit the same CSV schema:

```bash
python analysis/analyze_training_metrics.py artifacts/logs/lpse_training_loghybrid.csv
```

### Environment variables

| Variable | Default | Used by | Effect |
| --- | --- | --- | --- |
| `Q_ENCODER_BACKEND` | `sentence_transformer` | all trainers | `keyword`, `sentence_transformer`, or `hybrid`; use the same value for LPSE and its baselines |
| `KUBECONFIG_PATH` | `./config` | live trainers, inference | kubeconfig file |
| `DUMP_DIR` | `artifacts/dumps/jepa_live_lap` | `lpse/apps/train_lpse.py` | Replay dump output directory; set to `""` to disable dumping |
| `DUMP_SHARD_SIZE` | 100 | `lpse/apps/train_lpse.py` | Snapshots per shard |
| `LOAD_REPLAY_PATH` | (unset) | offline trainers | Replay a dump instead of scraping live |
| `OFFLINE_EPOCHS` | 10 | MLP/LSTM/Transformer baselines | Passes over the dump |
| `OFFLINE_NUM_ROUNDS` | 500 | XGBoost baseline | Boosting rounds total |
| `OFFLINE_EVAL_EVERY` | 10 | XGBoost baseline | Eval cadence |
| `XGB_DEVICE` | `cuda` if available | XGBoost baseline | `cuda` or `cpu` |
| `LPSE_RUN_SUFFIX` | `2` | LPSE trainers | Disambiguates checkpoint/log names |
| `BATCH_SIZE` | 32 | all trainers | SGD batch size |
| `SEQ_LEN` | 16 | all trainers | Temporal context window |
| `NODE_DROPOUT_PROB` | 0.20 | live trainers | Topology augmentation rate |

See the top of each trainer file for the full list.

### GPU sizing

Tested on a single RTX 6000 Ada (48 GB VRAM): LPSE training 10–12 GB,
MLP baseline ~2 GB, XGBoost (GPU hist) 4–6 GB peak,
SentenceTransformer encoder ~100 MB per process. All trainers can run
concurrently on this card.

## Inference

```bash
CHECKPOINT_PATH=artifacts/checkpoints/lpsehybrid.pt python -m lpse.apps.infer_lpse              # REPL
python -m lpse.apps.infer_lpse --query "Is any node CPU-bound?"                                 # one-shot
python -m lpse.apps.infer_lpse --serve --port 8080                                              # HTTP
```

The inference loader reads `q_encoder_backend` from the checkpoint metadata,
so you do not need to set it unless overriding.

## Evaluation

**Offline trace benchmark** — evaluates LPSE, MLP, XGBoost, and an LLM on the
same replay trace, so all systems see the identical (normalized) context
distribution they were trained on:

```bash
python scripts/eval/benchmark_offline_trace.py \
    --dump artifacts/dumps/jepa_live_lap \
    --checkpoint artifacts/checkpoints/lpsehybrid.pt \
    --snapshots 30 --questions-per-snapshot 5 --max-questions 100 \
    --vllm-chat-url http://localhost:8001/v1/chat/completions \
    --vllm-model "Qwen/Qwen3-4B-Instruct-2507-FP8"
```

Pass `--mlp-checkpoint` / `--xgb-model` (plus `--xgb-norm`,
`--xgb-encoder-state`) to include the baselines, and
`--paraphrase-questions` for the paraphrase condition.

**Live LLM benchmark** — `lpse/apps/benchmark_llm_forecast.py` compares LPSE against a
vLLM endpoint on a live QA stream. The endpoint is configured via env vars
(`VLLM_CHAT_URL` or `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`) or the
equivalent `--vllm-*` flags. `scripts/eval/run_llm_benchmark_matrix.sh` runs the
full condition matrix (clean / drop-node / paraphrased, open and category
answer modes) against the endpoints configured in that script; it needs
trained checkpoints, a kubeconfig, and a running cluster.

## Results

From the camera-ready evaluation, against a deployable 4B LLM endpoint:

| Metric | Result |
| --- | --- |
| Semantic prediction accuracy under changing node configurations | 82.42% |
| Mean inference latency | ~41× lower |
| Memory footprint | 15× smaller |

## Repository layout

```
lpse/                Library: model, encoders, ingestion, replay
lpse/apps/           Entry-point applications:
  train_lpse.py        LPSE trainer (live scrape + JEPA SSL + QA losses)
  train_lpse_offline.py        LPSE trainer on a replay dump
  train_lpse_no_future.py      LPSE ablation (future objective removed)
  train_baseline_mlp.py        MLP baseline (live or offline replay)
  train_baseline_lstm.py       LSTM temporal baseline
  train_baseline_transformer.py Transformer temporal baseline
  train_baseline_xgb.py        XGBoost baseline (live or offline replay)
  infer_lpse.py        Inference: CLI REPL, --query, or HTTP server
  benchmark_llm_forecast.py    LPSE-vs-LLM live benchmark
scripts/train/       Experiment launchers and training watchers
scripts/eval/        Benchmark, accuracy, and memory utilities
scripts/data/        Dump cleaning and cluster stress tooling
scripts/infra/       Cluster infrastructure helpers
analysis/            Plot and ablation/per-type analysis scripts
tests/               Unit tests and live-cluster check scripts
deploy/              Cluster manifests
artifacts/           Run outputs, checkpoints, dumps (gitignored)
```

## Tests

```bash
pytest tests/
```

`tests/test_core.py` is self-contained and covers the core math and data
pipeline. `tests/test_llm.py` targets a live vLLM endpoint (`VLLM_CHAT_URL` /
`VLLM_MODEL`); under pytest it skips automatically when the endpoint is
unreachable, and when run directly it defaults to deterministic baseline
buckets unless `USE_LLM=1`.

## Citation

If you use LPSE, please cite:

```bibtex
@inproceedings{lpse2026,
  title     = {A Low-Latency Semantic State Estimator using Latent Predictive
               Learning for Dynamic Network Monitoring and Orchestration},
  booktitle = {IEEE GLOBECOM},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
