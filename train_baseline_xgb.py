"""XGBoost baseline for the QA task. Periodic warm-started refits over the replay buffer; ssl_loss is always 0."""

import csv
import json
import logging
import os
import pickle
import random
import time
import traceback
from collections import OrderedDict, deque
from typing import Any

import numpy as np
import torch
import xgboost as xgb

from lpse import seed_everything
from lpse.collectors import get_node_ips, get_num_cores_from_node_exporter
from lpse.prometheus_ingest import (
    CLUSTER_TOKEN_DIM,
    RunningNormalizer,
    cluster_snapshot_to_token,
    collect_full_cluster_snapshot,
    compute_semantic_features,
)
from lpse.qa_generator import (
    NUM_ANSWERS,
    generate_qa_pairs,
)
from lpse.question_encoder import (
    QUESTION_DIM,
    build_question_encoder,
    encode_questions,
)
from lpse.replay_buffer import QAReplayBuffer
from lpse.replay_dump import iter_dump_by_snapshot, load_dump_summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = int(os.getenv("SEED", "1337"))
seed_everything(SEED)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_baseline_xgb")


SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))
SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "0.5"))


REPLAY_QA_CAPACITY = int(os.getenv("REPLAY_QA_CAPACITY", "32768"))
REPLAY_CONTEXT_CAPACITY = int(os.getenv("REPLAY_CONTEXT_CAPACITY", "4096"))
QA_PAIRS_PER_STEP = int(os.getenv("QA_PAIRS_PER_STEP", "10"))
NORMALIZER_BURN_IN = int(os.getenv("NORMALIZER_BURN_IN", "2000"))

NODE_DROPOUT_PROB = float(os.getenv("NODE_DROPOUT_PROB", "0.20"))
MAX_NODES_TO_DROP = int(os.getenv("MAX_NODES_TO_DROP", "2"))
MIN_NODES_AFTER_DROPOUT = int(os.getenv("MIN_NODES_AFTER_DROPOUT", "3"))


VAL_FRACTION_DENOM = int(os.getenv("VAL_FRACTION_DENOM", "10"))
VAL_QA_CAPACITY = int(os.getenv("VAL_QA_CAPACITY", "4096"))
VAL_CONTEXT_CAPACITY = int(os.getenv("VAL_CONTEXT_CAPACITY", "512"))
VAL_BATCH_SIZE = int(os.getenv("VAL_BATCH_SIZE", "1024"))


REFIT_INTERVAL_SCRAPES = int(os.getenv("XGB_REFIT_INTERVAL_SCRAPES", "100"))
MIN_REFIT_SAMPLES = int(os.getenv("XGB_MIN_REFIT_SAMPLES", "1024"))
MAX_TRAIN_SAMPLES = int(os.getenv("XGB_MAX_TRAIN_SAMPLES", "20000"))


XGB_N_ESTIMATORS_PER_REFIT = int(os.getenv("XGB_N_ESTIMATORS_PER_REFIT", "50"))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.5"))
XGB_TREE_METHOD = os.getenv("XGB_TREE_METHOD", "hist")
_requested_xgb_device = os.getenv(
    "XGB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)


def _resolve_xgb_device(requested: str) -> str:
    """Validate the requested XGBoost device against the installed build and
    fall back to CPU. Previously a GPU build mismatch made every refit raise,
    which was swallowed by `except: continue`, silently training nothing."""
    requested = requested.strip().lower()
    if requested == "cuda":
        try:
            import xgboost as _xgb

            _xgb.DMatrix(np.zeros((2, 2), dtype=np.float32))
            params = {"tree_method": "hist", "device": "cuda", "verbosity": 0}
            try:
                _xgb.train(
                    params,
                    _xgb.DMatrix(
                        np.zeros((4, 2), dtype=np.float32),
                        label=np.zeros(4, dtype=np.int64),
                    ),
                    num_boost_round=1,
                )
                return "cuda"
            except Exception:
                pass
        except Exception:
            pass
        logger.warning(
            "XGB_DEVICE=cuda requested but the installed XGBoost build does not "
            "support GPU training (or the CUDA plugin failed); falling back to CPU."
        )
        return "cpu"
    return requested


XGB_DEVICE = _resolve_xgb_device(_requested_xgb_device)


XGB_MAX_TOTAL_ROUNDS = int(os.getenv("XGB_MAX_TOTAL_ROUNDS", "1000"))

XGB_MAX_DELTA_STEP = float(os.getenv("XGB_MAX_DELTA_STEP", "1.0"))
XGB_REG_LAMBDA = float(os.getenv("XGB_REG_LAMBDA", "1.0"))
XGB_REG_ALPHA = float(os.getenv("XGB_REG_ALPHA", "0.0"))
XGB_MIN_CHILD_WEIGHT = float(os.getenv("XGB_MIN_CHILD_WEIGHT", "5.0"))

Q_ENCODER_BACKEND = os.getenv("Q_ENCODER_BACKEND", "sentence_transformer")


LOAD_REPLAY_PATH = os.getenv("LOAD_REPLAY_PATH", "")
OFFLINE_NUM_ROUNDS = int(os.getenv("OFFLINE_NUM_ROUNDS", "500"))
OFFLINE_EVAL_EVERY = int(os.getenv("OFFLINE_EVAL_EVERY", "10"))


RUN_SUFFIX = os.getenv("BASELINE_RUN_SUFFIX", "xgb")
MODEL_PATH = os.path.abspath(
    os.getenv("XGB_MODEL_PATH", f"artifacts/checkpoints/baseline_{RUN_SUFFIX}.json")
)
NORMALIZER_PATH = os.path.abspath(
    os.getenv(
        "XGB_NORMALIZER_PATH", f"artifacts/checkpoints/baseline_{RUN_SUFFIX}_norm.pkl"
    )
)
ENCODER_STATE_PATH = os.path.abspath(
    os.getenv(
        "XGB_ENCODER_STATE_PATH",
        f"artifacts/checkpoints/baseline_{RUN_SUFFIX}_encoder.pt",
    )
)
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "1"))
SUMMARY_PATH = os.path.abspath(
    os.getenv("SUMMARY_PATH", f"artifacts/logs/baseline_{RUN_SUFFIX}_summary.txt")
)
LOG_CSV_PATH = os.path.abspath(
    os.getenv("LOG_CSV_PATH", f"artifacts/logs/baseline_{RUN_SUFFIX}_training_log.csv")
)


XGB_FRESH_START = os.getenv("XGB_FRESH", "0").strip() not in ("", "0", "false", "False")


node_ips: list[str] = []
node_cores: list[int] = []
if not LOAD_REPLAY_PATH:
    logger.info("Discovering cluster topology...")
    node_ips = get_node_ips()
    if not node_ips:
        raise RuntimeError("No Kubernetes nodes found via API")
    for ip in node_ips:
        cores = get_num_cores_from_node_exporter(ip)
        if cores <= 0:
            logger.warning("Could not detect cores for %s, falling back to 10", ip)
            cores = 10
        node_cores.append(cores)

INPUT_DIM = CLUSTER_TOKEN_DIM
FLAT_CONTEXT_DIM = SEQ_LEN * INPUT_DIM
XGB_INPUT_DIM = FLAT_CONTEXT_DIM + QUESTION_DIM
logger.info(
    "Cluster: %d nodes, input_dim=%d, flat_context=%d, xgb_input=%d, answers=%d",
    len(node_ips),
    INPUT_DIM,
    FLAT_CONTEXT_DIM,
    XGB_INPUT_DIM,
    NUM_ANSWERS,
)


question_encoder = build_question_encoder(
    backend=Q_ENCODER_BACKEND,
    out_dim=QUESTION_DIM,
).to(DEVICE)
question_encoder.eval()

normalizer = RunningNormalizer(INPUT_DIM)
if XGB_FRESH_START:
    logger.info("XGB_FRESH=1: skipping normalizer load")
elif os.path.isfile(NORMALIZER_PATH):
    try:
        with open(NORMALIZER_PATH, "rb") as f:
            state = pickle.load(f)
        normalizer.load_state_dict(state)
        logger.info(
            "Loaded normalizer state from %s (count=%d)",
            NORMALIZER_PATH,
            normalizer.count,
        )
    except Exception as e:
        logger.warning("Failed to load normalizer: %s", e)


if not XGB_FRESH_START and os.path.isfile(ENCODER_STATE_PATH):
    try:
        enc_blob = torch.load(
            ENCODER_STATE_PATH, map_location=DEVICE, weights_only=False
        )
        qe_state = enc_blob.get("question_encoder_state", {})
        missing, unexpected = question_encoder.load_state_dict(qe_state, strict=False)
        logger.info(
            "Loaded encoder state from %s (missing=%d unexpected=%d)",
            ENCODER_STATE_PATH,
            len(missing),
            len(unexpected),
        )
    except Exception as e:
        logger.warning("Failed to load encoder state: %s", e)


def _materialize(
    entries, live_node_ips: list[str], cap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Turn replay tuples into a dense (X, y) numpy pair, batched through the
    sentence encoder for question embedding."""
    if cap is not None and len(entries) > cap:
        entries = random.sample(entries, cap)
    if not entries:
        return np.zeros((0, XGB_INPUT_DIM), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )

    all_q_emb_chunks: list[np.ndarray] = []
    all_ctx_chunks: list[np.ndarray] = []
    all_y: list[int] = []
    chunk = 256
    with torch.no_grad():
        for i in range(0, len(entries), chunk):
            batch = entries[i : i + chunk]
            seqs = torch.stack([b[0] for b in batch], dim=0)
            qs = [b[1] for b in batch]
            ys = [b[2] for b in batch]
            q_emb = encode_questions(question_encoder, qs, live_node_ips)
            all_ctx_chunks.append(seqs.flatten(1).numpy().astype(np.float32))
            all_q_emb_chunks.append(q_emb.detach().cpu().numpy().astype(np.float32))
            all_y.extend(ys)
    X = np.concatenate(
        [
            np.concatenate([c, q], axis=1)
            for c, q in zip(all_ctx_chunks, all_q_emb_chunks)
        ],
        axis=0,
    )
    y = np.asarray(all_y, dtype=np.int64)
    return X, y


def _load_booster() -> xgb.Booster | None:
    if XGB_FRESH_START:
        logger.info("XGB_FRESH=1: skipping booster load (training from scratch)")
        return None
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        b = xgb.Booster()
        b.load_model(MODEL_PATH)
        logger.info("Loaded existing booster from %s", MODEL_PATH)
        return b
    except Exception as e:
        logger.warning("Failed to load booster: %s", e)
        return None


def _open_csv_log() -> tuple[Any, Any, list[str]]:
    """Open the CSV log. With XGB_FRESH, truncate; otherwise append."""
    csv_header = [
        "timestamp",
        "step",
        "qa_loss",
        "ssl_loss",
        "total_loss",
        "top1_acc",
        "top5_acc",
        "val_top1",
        "val_top5",
        "buffer_size",
        "val_buffer_size",
    ]
    os.makedirs(os.path.dirname(LOG_CSV_PATH) or ".", exist_ok=True)
    if XGB_FRESH_START:
        mode = "w"
        write_header = True
    else:
        mode = "a"
        write_header = not os.path.isfile(LOG_CSV_PATH)
    handle = open(LOG_CSV_PATH, mode=mode, newline="")
    writer = csv.writer(handle)
    if write_header:
        writer.writerow(csv_header)
    return writer, handle, csv_header


def _truncate_summary_if_fresh():
    if XGB_FRESH_START and os.path.isfile(SUMMARY_PATH):
        os.makedirs(os.path.dirname(SUMMARY_PATH) or ".", exist_ok=True)
        open(SUMMARY_PATH, "w").close()
        logger.info("XGB_FRESH=1: truncated summary file %s", SUMMARY_PATH)


def _save_booster(b: xgb.Booster):
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    b.save_model(MODEL_PATH)
    try:
        with open(NORMALIZER_PATH, "wb") as f:
            pickle.dump(normalizer.state_dict(), f)
    except Exception as e:
        logger.warning("Failed to save normalizer: %s", e)

    try:
        qe_state = {
            k: v.detach().cpu()
            for k, v in question_encoder.state_dict().items()
            if not k.startswith("st_model.")
        }
        torch.save(
            {
                "question_encoder_state": qe_state,
                "q_encoder_backend": Q_ENCODER_BACKEND,
                "use_structured": getattr(question_encoder, "use_structured", None),
            },
            ENCODER_STATE_PATH,
        )
    except Exception as e:
        logger.warning("Failed to save encoder state: %s", e)


XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": NUM_ANSWERS,
    "max_depth": XGB_MAX_DEPTH,
    "eta": XGB_LEARNING_RATE,
    "subsample": XGB_SUBSAMPLE,
    "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    "tree_method": XGB_TREE_METHOD,
    "device": XGB_DEVICE,
    "max_delta_step": XGB_MAX_DELTA_STEP,
    "reg_lambda": XGB_REG_LAMBDA,
    "reg_alpha": XGB_REG_ALPHA,
    "min_child_weight": XGB_MIN_CHILD_WEIGHT,
    "eval_metric": ["mlogloss", "merror"],
    "verbosity": 0,
}


def _refit(
    booster: xgb.Booster | None,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> tuple[xgb.Booster, float, float, float, float, float]:
    """Add `XGB_N_ESTIMATORS_PER_REFIT` boosting rounds to the booster.
    Returns (booster, train_mlogloss, train_top1, train_top5, val_top1, val_top5).

    Does not pass `evals=` to xgb.train: per-round evaluation on multi:softprob
    with 94 classes adds ~3s per round per dataset (~120s overhead at 20 rounds
    × 2 datasets). We compute the final metrics from explicit predict() calls
    below instead."""
    dtrain = xgb.DMatrix(X, label=y)
    dval = (
        xgb.DMatrix(X_val, label=y_val)
        if X_val is not None and len(X_val) > 0
        else None
    )

    existing_rounds = booster.num_boosted_rounds() if booster is not None else 0
    if booster is not None and existing_rounds >= XGB_MAX_TOTAL_ROUNDS:
        logger.warning(
            "Booster already at %d rounds (cap %d); skipping this refit.",
            existing_rounds,
            XGB_MAX_TOTAL_ROUNDS,
        )
        train_probs = booster.predict(dtrain)
        train_top1 = float((train_probs.argmax(axis=1) == y).mean())
        top5_idx = np.argpartition(-train_probs, 5, axis=1)[:, :5]
        train_top5 = float((top5_idx == y[:, None]).any(axis=1).mean())
        train_mlogloss = float(_safe_mlogloss(train_probs, y))
        val_top1 = float("nan")
        val_top5 = float("nan")
        if dval is not None and y_val is not None:
            val_probs = booster.predict(dval)
            val_top1 = float((val_probs.argmax(axis=1) == y_val).mean())
            v_top5_idx = np.argpartition(-val_probs, 5, axis=1)[:, :5]
            val_top5 = float((v_top5_idx == y_val[:, None]).any(axis=1).mean())
        return booster, train_mlogloss, train_top1, train_top5, val_top1, val_top5
    max_rounds_this_fit = min(
        XGB_N_ESTIMATORS_PER_REFIT, XGB_MAX_TOTAL_ROUNDS - existing_rounds
    )

    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=max_rounds_this_fit,
        xgb_model=booster,
        verbose_eval=False,
    )

    train_probs = booster.predict(dtrain)
    train_top1 = float((train_probs.argmax(axis=1) == y).mean())
    top5_idx = np.argpartition(-train_probs, 5, axis=1)[:, :5]
    train_top5 = float((top5_idx == y[:, None]).any(axis=1).mean())
    train_mlogloss = float(_safe_mlogloss(train_probs, y))

    val_top1 = float("nan")
    val_top5 = float("nan")
    if dval is not None and y_val is not None:
        val_probs = booster.predict(dval)
        val_top1 = float((val_probs.argmax(axis=1) == y_val).mean())
        v_top5_idx = np.argpartition(-val_probs, 5, axis=1)[:, :5]
        val_top5 = float((v_top5_idx == y_val[:, None]).any(axis=1).mean())

    return booster, train_mlogloss, train_top1, train_top5, val_top1, val_top5


def _safe_mlogloss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0 - eps)
    return -float(np.log(p).mean())


def _write_summary(
    step: int,
    semantic: dict,
    train_top1: float,
    val_top1: float,
    val_top5: float,
    mlogloss: float,
):
    try:
        summary = {
            "step": step,
            "timestamp": time.time(),
            "model": "baseline_xgb_v1",
            "train_top1": train_top1,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "mlogloss": mlogloss,
            "num_answers": NUM_ANSWERS,
            "nodes": {
                ip: {k: v for k, v in feats.items() if v is not None}
                for ip, feats in semantic.items()
            },
        }
        os.makedirs(os.path.dirname(SUMMARY_PATH) or ".", exist_ok=True)
        with open(SUMMARY_PATH, "a") as f:
            f.write(json.dumps(summary, separators=(",", ":")) + "\n")
        logger.info("Wrote summary → %s", SUMMARY_PATH)
    except Exception as e:
        logger.warning("Failed to write summary: %s", e)


def main():
    booster = _load_booster()

    metric_buffer: deque[torch.Tensor] = deque(maxlen=SEQ_LEN)
    val_metric_buffer: deque[torch.Tensor] = deque(maxlen=SEQ_LEN)
    replay = QAReplayBuffer(REPLAY_QA_CAPACITY, REPLAY_CONTEXT_CAPACITY)
    val_replay = QAReplayBuffer(VAL_QA_CAPACITY, VAL_CONTEXT_CAPACITY)
    scrape_counter = 0
    refit_step = 0
    scrapes_since_last_refit = 0

    _truncate_summary_if_fresh()
    csv_writer, log_file, _ = _open_csv_log()

    logger.info(
        "Starting XGBoost baseline training\n"
        "  Refit every %d scrapes (≥ %d samples), %d boosting rounds per refit\n"
        "  Tree depth=%d, lr=%.2f, subsample=%.2f, colsample=%.2f, device=%s\n"
        "  Question encoder: %s",
        REFIT_INTERVAL_SCRAPES,
        MIN_REFIT_SAMPLES,
        XGB_N_ESTIMATORS_PER_REFIT,
        XGB_MAX_DEPTH,
        XGB_LEARNING_RATE,
        XGB_SUBSAMPLE,
        XGB_COLSAMPLE_BYTREE,
        XGB_DEVICE,
        Q_ENCODER_BACKEND,
    )

    def _discover_live_topology() -> tuple[list[str], list[int]]:
        live_ips = get_node_ips()
        live_cores: list[int] = []
        for ip in live_ips:
            cores = get_num_cores_from_node_exporter(ip)
            if cores <= 0:
                cores = 10
            live_cores.append(cores)
        return live_ips, live_cores

    def _maybe_dropout_nodes(node_ips_, node_cores_, node_tensors, raw_parsed):
        n = len(node_ips_)
        if NODE_DROPOUT_PROB <= 0.0 or n <= MIN_NODES_AFTER_DROPOUT:
            return node_ips_, node_cores_, node_tensors, raw_parsed
        if random.random() >= NODE_DROPOUT_PROB:
            return node_ips_, node_cores_, node_tensors, raw_parsed
        max_drop = min(MAX_NODES_TO_DROP, n - MIN_NODES_AFTER_DROPOUT)
        if max_drop < 1:
            return node_ips_, node_cores_, node_tensors, raw_parsed
        k = random.randint(1, max_drop)
        drop_idx = set(random.sample(range(n), k))
        new_ips = [ip for i, ip in enumerate(node_ips_) if i not in drop_idx]
        new_cores = [c for i, c in enumerate(node_cores_) if i not in drop_idx]
        keep_idx = torch.tensor(
            [i for i in range(n) if i not in drop_idx], dtype=torch.long
        )
        new_tensors = node_tensors.index_select(0, keep_idx)
        new_parsed = {ip: raw_parsed.get(ip, {}) for ip in new_ips}
        return new_ips, new_cores, new_tensors, new_parsed

    last_semantic: dict = {}

    while True:
        try:
            live_node_ips, live_node_cores = _discover_live_topology()
            if not live_node_ips:
                logger.warning("No live nodes discovered; waiting")
                time.sleep(SAMPLE_INTERVAL)
                continue
            node_tensors, raw_parsed = collect_full_cluster_snapshot(live_node_ips)
            live_node_ips, live_node_cores, node_tensors, raw_parsed = (
                _maybe_dropout_nodes(
                    live_node_ips, live_node_cores, node_tensors, raw_parsed
                )
            )
            cluster_token = cluster_snapshot_to_token(node_tensors, live_node_ips)
            if normalizer.count < NORMALIZER_BURN_IN:
                normalizer.update(cluster_token)
            token_norm = normalizer.normalize(cluster_token)
        except Exception as e:
            logger.error("Metric collection failed: %s\n%s", e, traceback.format_exc())
            time.sleep(SAMPLE_INTERVAL)
            continue

        time.sleep(SAMPLE_INTERVAL)

        scrape_counter += 1
        scrapes_since_last_refit += 1

        is_val_snapshot = (
            VAL_FRACTION_DENOM > 0 and scrape_counter % VAL_FRACTION_DENOM == 0
        )
        target_metric_buffer = val_metric_buffer if is_val_snapshot else metric_buffer
        target_metric_buffer.append(token_norm)

        if len(target_metric_buffer) < SEQ_LEN:
            continue

        context_seq = torch.stack(list(target_metric_buffer), dim=0)
        semantic = compute_semantic_features(raw_parsed, live_node_ips)
        last_semantic = semantic
        qa_pairs = generate_qa_pairs(
            semantic, live_node_ips, live_node_cores, max_pairs=QA_PAIRS_PER_STEP
        )

        target_buffer = val_replay if is_val_snapshot else replay
        ctx_id = target_buffer.add_context(context_seq)
        for q_struct, ans_idx in qa_pairs:
            target_buffer.add_qa(ctx_id, q_struct, ans_idx)

        ready = (
            scrapes_since_last_refit >= REFIT_INTERVAL_SCRAPES
            and len(replay) >= MIN_REFIT_SAMPLES
        )
        if not ready:
            continue

        scrapes_since_last_refit = 0
        refit_step += 1

        try:
            train_entries = replay.all_live()
            val_entries = val_replay.all_live()
            X_train, y_train = _materialize(
                train_entries, live_node_ips, cap=MAX_TRAIN_SAMPLES
            )
            X_val, y_val = _materialize(val_entries, live_node_ips, cap=VAL_BATCH_SIZE)
            t0 = time.time()
            booster, mll, train_top1, train_top5, val_top1, val_top5 = _refit(
                booster,
                X_train,
                y_train,
                X_val if len(X_val) else None,
                y_val if len(y_val) else None,
            )
            fit_secs = time.time() - t0
            _save_booster(booster)
        except Exception as e:
            logger.error("Refit failed: %s\n%s", e, traceback.format_exc())
            continue

        csv_writer.writerow(
            [
                time.time(),
                refit_step,
                mll,
                0.0,
                mll,
                train_top1,
                train_top5,
                val_top1,
                val_top5,
                len(replay),
                len(val_replay),
            ]
        )
        log_file.flush()

        logger.info(
            "refit=%d  fit_s=%.1f  train_mlogloss=%.4f train_top1=%.2f%% train_top5=%.2f%%  "
            "val_top1=%.2f%% val_top5=%.2f%%  buf=%d val_buf=%d  X=%dx%d",
            refit_step,
            fit_secs,
            mll,
            train_top1 * 100,
            train_top5 * 100,
            val_top1 * 100 if val_top1 == val_top1 else float("nan"),
            val_top5 * 100 if val_top5 == val_top5 else float("nan"),
            len(replay),
            len(val_replay),
            X_train.shape[0],
            X_train.shape[1],
        )

        if SUMMARY_INTERVAL > 0 and refit_step % SUMMARY_INTERVAL == 0:
            _write_summary(
                refit_step, last_semantic, train_top1, val_top1, val_top5, mll
            )


def main_offline():
    """Stream the JEPA dump chronologically and refit XGBoost periodically,
    mirroring the live trainer's per-scrape cadence so the step axis lines up
    with train_lpse.py's logged step (one scrape = one JEPA SGD step).

    Each refit fits XGB_N_ESTIMATORS_PER_REFIT additional boosting rounds onto
    a warm-started booster using the current replay buffer. Snapshots flagged
    is_val in the dump go to a held-out val buffer used for evaluation only.
    """
    summary = load_dump_summary(LOAD_REPLAY_PATH)
    logger.info(
        "Streaming offline mode: dump=%s (snapshots=%s qa_pairs=%s shards=%d)",
        LOAD_REPLAY_PATH,
        summary.get("snapshots_written", "?"),
        summary.get("qa_pairs_written", "?"),
        len(summary.get("shards", [])),
    )

    booster: xgb.Booster | None = _load_booster()

    train_contexts: OrderedDict[int, np.ndarray] = OrderedDict()
    train_entries: deque[tuple[int, np.ndarray, int]] = deque(maxlen=REPLAY_QA_CAPACITY)
    val_contexts: OrderedDict[int, np.ndarray] = OrderedDict()
    val_entries: deque[tuple[int, np.ndarray, int]] = deque(maxlen=VAL_QA_CAPACITY)
    next_ctx_id = 0

    def _add(ctx_dict, qa_dq, cap, ctx_flat, q_embs_np, ans_idxs):
        nonlocal next_ctx_id
        cid = next_ctx_id
        next_ctx_id += 1
        ctx_dict[cid] = ctx_flat
        while len(ctx_dict) > cap:
            ctx_dict.popitem(last=False)
        for qe, a in zip(q_embs_np, ans_idxs):
            qa_dq.append((cid, qe, int(a)))

    def _live_count(ctx_dict, qa_dq):
        return sum(1 for cid, _, _ in qa_dq if cid in ctx_dict)

    def _materialize_buffer(ctx_dict, qa_dq, cap):
        live = [(cid, qe, a) for cid, qe, a in qa_dq if cid in ctx_dict]
        if cap is not None and len(live) > cap:
            live = random.sample(live, cap)
        if not live:
            return np.zeros((0, XGB_INPUT_DIM), dtype=np.float32), np.zeros(
                (0,), dtype=np.int64
            )
        n = len(live)
        X = np.empty((n, XGB_INPUT_DIM), dtype=np.float32)
        y = np.empty((n,), dtype=np.int64)
        for i, (cid, qe, a) in enumerate(live):
            X[i, :FLAT_CONTEXT_DIM] = ctx_dict[cid]
            X[i, FLAT_CONTEXT_DIM:] = qe
            y[i] = a
        return X, y

    _truncate_summary_if_fresh()
    csv_writer, log_file, _ = _open_csv_log()

    refit_interval = max(1, REFIT_INTERVAL_SCRAPES)
    n_rounds_per_refit = max(1, XGB_N_ESTIMATORS_PER_REFIT)

    logger.info(
        "Streaming XGBoost: refit every %d scrapes (≥%d train samples), "
        "%d rounds per refit, depth=%d, lr=%.2f, max_delta_step=%.1f, device=%s "
        "(q_emb cached on insertion)",
        refit_interval,
        MIN_REFIT_SAMPLES,
        n_rounds_per_refit,
        XGB_MAX_DEPTH,
        XGB_LEARNING_RATE,
        XGB_MAX_DELTA_STEP,
        XGB_DEVICE,
    )

    scrape_counter = 0
    scrapes_since_last_refit = 0
    refit_step = 0
    last_ips: list[str] = []
    t0 = time.time()

    for ctx, qa_pairs, is_val, ips, _cores, _ts in iter_dump_by_snapshot(
        LOAD_REPLAY_PATH
    ):
        scrape_counter += 1
        scrapes_since_last_refit += 1
        if ips:
            last_ips = ips
        if not qa_pairs:
            continue

        qs = [q for q, _ in qa_pairs]
        ans_idxs = [a for _, a in qa_pairs]
        with torch.no_grad():
            q_embs_t = encode_questions(question_encoder, qs, last_ips)
        q_embs_np = q_embs_t.detach().cpu().numpy().astype(np.float32)
        ctx_flat = ctx.flatten().numpy().astype(np.float32)

        if is_val:
            _add(
                val_contexts,
                val_entries,
                VAL_CONTEXT_CAPACITY,
                ctx_flat,
                q_embs_np,
                ans_idxs,
            )
        else:
            _add(
                train_contexts,
                train_entries,
                REPLAY_CONTEXT_CAPACITY,
                ctx_flat,
                q_embs_np,
                ans_idxs,
            )

        train_live = _live_count(train_contexts, train_entries)
        if scrapes_since_last_refit < refit_interval or train_live < MIN_REFIT_SAMPLES:
            continue

        scrapes_since_last_refit = 0
        refit_step += 1

        try:
            X_train, y_train = _materialize_buffer(
                train_contexts, train_entries, MAX_TRAIN_SAMPLES
            )
            X_val, y_val = _materialize_buffer(
                val_contexts, val_entries, VAL_BATCH_SIZE
            )
            t_refit = time.time()
            booster, mll, train_top1, train_top5, val_top1, val_top5 = _refit(
                booster,
                X_train,
                y_train,
                X_val if len(X_val) else None,
                y_val if len(y_val) else None,
            )
            fit_secs = time.time() - t_refit
        except Exception as e:
            logger.error(
                "Streaming refit failed at scrape=%d: %s\n%s",
                scrape_counter,
                e,
                traceback.format_exc(),
            )
            continue

        csv_writer.writerow(
            [
                time.time(),
                scrape_counter,
                mll,
                0.0,
                mll,
                train_top1,
                train_top5,
                val_top1,
                val_top5,
                train_live,
                _live_count(val_contexts, val_entries),
            ]
        )
        log_file.flush()
        _save_booster(booster)

        elapsed = time.time() - t0
        logger.info(
            "scrape=%d  refit=%d  elapsed=%.0fs  fit_s=%.1f  "
            "train_mlogloss=%.4f train_top1=%.2f%% train_top5=%.2f%%  "
            "val_top1=%.2f%% val_top5=%.2f%%  buf=%d val_buf=%d",
            scrape_counter,
            refit_step,
            elapsed,
            fit_secs,
            mll,
            train_top1 * 100,
            train_top5 * 100,
            val_top1 * 100 if val_top1 == val_top1 else float("nan"),
            val_top5 * 100 if val_top5 == val_top5 else float("nan"),
            train_live,
            _live_count(val_contexts, val_entries),
        )

    logger.info(
        "Streaming offline XGBoost done. Total scrapes=%d, refits=%d. Booster saved to %s",
        scrape_counter,
        refit_step,
        MODEL_PATH,
    )


if __name__ == "__main__":
    if LOAD_REPLAY_PATH:
        main_offline()
    else:
        main()
