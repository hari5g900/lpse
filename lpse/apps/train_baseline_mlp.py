"""MLP baseline for the QA task. Shares the data pipeline with train_lpse.py; ssl_loss is always 0."""

import csv
import json
import logging
import os
import random
import time
import traceback
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    build_answer_soft_targets,
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
logger = logging.getLogger("train_baseline_mlp")


SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))
SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "0.5"))


MODEL_VARIANT = "baseline_mlp_v1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
REPLAY_QA_CAPACITY = int(os.getenv("REPLAY_QA_CAPACITY", "32768"))
REPLAY_CONTEXT_CAPACITY = int(os.getenv("REPLAY_CONTEXT_CAPACITY", "4096"))
QA_PAIRS_PER_STEP = int(os.getenv("QA_PAIRS_PER_STEP", "10"))
NORMALIZER_BURN_IN = int(os.getenv("NORMALIZER_BURN_IN", "2000"))

SOFT_LABEL_MASS = float(os.getenv("SOFT_LABEL_MASS", "0.30"))
SOFT_LABEL_TAU = float(os.getenv("SOFT_LABEL_TAU", "1.0"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))

NODE_DROPOUT_PROB = float(os.getenv("NODE_DROPOUT_PROB", "0.20"))
MAX_NODES_TO_DROP = int(os.getenv("MAX_NODES_TO_DROP", "2"))
MIN_NODES_AFTER_DROPOUT = int(os.getenv("MIN_NODES_AFTER_DROPOUT", "3"))


VAL_FRACTION_DENOM = int(os.getenv("VAL_FRACTION_DENOM", "10"))
VAL_QA_CAPACITY = int(os.getenv("VAL_QA_CAPACITY", "4096"))
VAL_CONTEXT_CAPACITY = int(os.getenv("VAL_CONTEXT_CAPACITY", "512"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
VAL_BATCH_SIZE = int(os.getenv("VAL_BATCH_SIZE", "256"))


HIDDEN_DIM = int(os.getenv("MLP_HIDDEN_DIM", "1024"))
HIDDEN_LAYERS = int(os.getenv("MLP_HIDDEN_LAYERS", "4"))
DROPOUT = float(os.getenv("MLP_DROPOUT", "0.1"))

Q_ENCODER_BACKEND = os.getenv("Q_ENCODER_BACKEND", "sentence_transformer")


LOAD_REPLAY_PATH = os.getenv("LOAD_REPLAY_PATH", "")
OFFLINE_EPOCHS = int(os.getenv("OFFLINE_EPOCHS", "10"))


RUN_SUFFIX = os.getenv("BASELINE_RUN_SUFFIX", "mlp")
CHECKPOINT_PATH = os.path.abspath(
    os.getenv("CHECKPOINT_PATH", f"artifacts/checkpoints/baseline_{RUN_SUFFIX}.pt")
)
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "200"))
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "50"))
SUMMARY_PATH = os.path.abspath(
    os.getenv("SUMMARY_PATH", f"artifacts/logs/baseline_{RUN_SUFFIX}_summary.txt")
)
LOG_CSV_PATH = os.path.abspath(
    os.getenv("LOG_CSV_PATH", f"artifacts/logs/baseline_{RUN_SUFFIX}_training_log.csv")
)


MLP_FRESH_START = os.getenv("MLP_FRESH", "0").strip() not in ("", "0", "false", "False")


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
MLP_INPUT_DIM = FLAT_CONTEXT_DIM + QUESTION_DIM
logger.info(
    "Cluster: %d nodes, input_dim=%d, flat_context=%d, mlp_input=%d, answers=%d",
    len(node_ips),
    INPUT_DIM,
    FLAT_CONTEXT_DIM,
    MLP_INPUT_DIM,
    NUM_ANSWERS,
)


class FlatMLPClassifier(nn.Module):
    """Vanilla MLP over [flatten(context), question_embedding] → answer logits."""

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


question_encoder = build_question_encoder(
    backend=Q_ENCODER_BACKEND,
    out_dim=QUESTION_DIM,
).to(DEVICE)

model = FlatMLPClassifier(
    in_dim=MLP_INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=HIDDEN_LAYERS,
    num_answers=NUM_ANSWERS,
    dropout=DROPOUT,
).to(DEVICE)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


logger.info(
    "Model params: total=%d mlp=%d question_encoder=%d (backend=%s)",
    _count_params(model) + _count_params(question_encoder),
    _count_params(model),
    _count_params(question_encoder),
    Q_ENCODER_BACKEND,
)

normalizer = RunningNormalizer(INPUT_DIM)

trainable_params = list(model.parameters()) + [
    p for p in question_encoder.parameters() if p.requires_grad
]
optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


_soft_target_rows = build_answer_soft_targets(
    num_answers=NUM_ANSWERS,
    soft_mass=SOFT_LABEL_MASS,
    tau=SOFT_LABEL_TAU,
    uniform_smoothing=LABEL_SMOOTHING,
)
SOFT_TARGET_MATRIX = torch.tensor(_soft_target_rows, dtype=torch.float32, device=DEVICE)


def soft_cross_entropy(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    soft_targets = SOFT_TARGET_MATRIX.index_select(0, target_idx)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def save_checkpoint(
    step: int, current_node_ips: list[str], current_node_cores: list[int]
):
    ckpt_dir = os.path.dirname(CHECKPOINT_PATH)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    qe_state = {
        k: v.detach().cpu()
        for k, v in question_encoder.state_dict().items()
        if not k.startswith("st_model.")
    }
    torch.save(
        {
            "model": model.state_dict(),
            "question_encoder_state": qe_state,
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "step": step,
            "node_ips": current_node_ips,
            "node_cores": current_node_cores,
            "input_dim": INPUT_DIM,
            "seq_len": SEQ_LEN,
            "model_variant": MODEL_VARIANT,
            "q_encoder_backend": Q_ENCODER_BACKEND,
            "model_config": {
                "in_dim": MLP_INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "dropout": DROPOUT,
                "num_answers": NUM_ANSWERS,
                "model_variant": MODEL_VARIANT,
                "q_encoder_backend": Q_ENCODER_BACKEND,
                "use_structured": getattr(question_encoder, "use_structured", None),
            },
        },
        CHECKPOINT_PATH,
    )
    logger.info("Checkpoint saved at step %d → %s", step, CHECKPOINT_PATH)


def load_checkpoint() -> int:
    if MLP_FRESH_START:
        logger.info("MLP_FRESH=1: skipping checkpoint load (training from scratch)")
        return 0
    if not os.path.isfile(CHECKPOINT_PATH):
        return 0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if ckpt.get("model_variant") != MODEL_VARIANT:
        logger.warning("Checkpoint variant mismatch; starting fresh.")
        return 0
    if (
        int(ckpt.get("input_dim", -1)) != INPUT_DIM
        or int(ckpt.get("seq_len", -1)) != SEQ_LEN
    ):
        logger.warning("Checkpoint shape mismatch; starting fresh.")
        return 0
    try:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "normalizer" in ckpt:
            try:
                normalizer.load_state_dict(ckpt["normalizer"])
            except Exception:
                pass
        if "question_encoder_state" in ckpt:
            try:
                qe_state = ckpt["question_encoder_state"]
                missing, unexpected = question_encoder.load_state_dict(
                    qe_state, strict=False
                )
                logger.info(
                    "Loaded question_encoder_state (missing=%d unexpected=%d)",
                    len(missing),
                    len(unexpected),
                )
            except Exception as e:
                logger.warning("Failed to load persisted encoder state: %s", e)
        return int(ckpt.get("step", 0))
    except Exception as e:
        logger.warning("Checkpoint load failed (%s); starting fresh.", e)
        return 0


def _evaluate_validation(
    val_replay: "QAReplayBuffer", live_node_ips: list[str]
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            seq_batch, q_batch, ans_batch = val_replay.sample(
                min(VAL_BATCH_SIZE, len(val_replay))
            )
            seq_batch = seq_batch.to(DEVICE)
            ans_batch = ans_batch.to(DEVICE)
            q_emb = encode_questions(question_encoder, q_batch, live_node_ips)
            x = torch.cat([seq_batch.flatten(1), q_emb], dim=-1)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            top1 = (preds == ans_batch).float().mean().item()
            _, top5_idx = logits.topk(5, dim=-1)
            top5 = (
                (top5_idx == ans_batch.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )
            return top1, top5
    except Exception as e:
        logger.warning("Validation eval failed: %s", e)
        return float("nan"), float("nan")
    finally:
        if was_training:
            model.train()


def _write_summary(
    step, semantic, loss_window, acc_window, val_top1=None, val_top5=None
):
    try:
        summary = {
            "step": step,
            "timestamp": time.time(),
            "model": MODEL_VARIANT,
            "avg_loss": sum(loss_window) / max(1, len(loss_window)),
            "avg_accuracy": sum(acc_window) / max(1, len(acc_window)),
            "val_top1": val_top1,
            "val_top5": val_top5,
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
    MIN_REPLAY_SIZE = BATCH_SIZE * 10
    global_step = load_checkpoint()

    metric_buffer: deque[torch.Tensor] = deque(maxlen=SEQ_LEN)
    val_metric_buffer: deque[torch.Tensor] = deque(maxlen=SEQ_LEN)
    replay = QAReplayBuffer(REPLAY_QA_CAPACITY, REPLAY_CONTEXT_CAPACITY)
    val_replay = QAReplayBuffer(VAL_QA_CAPACITY, VAL_CONTEXT_CAPACITY)
    scrape_counter = 0
    model.train()

    os.makedirs(os.path.dirname(LOG_CSV_PATH) or ".", exist_ok=True)
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
    needs_header = True
    log_mode = "a"
    if os.path.isfile(LOG_CSV_PATH):
        try:
            with open(LOG_CSV_PATH) as f:
                existing_header = f.readline().rstrip("\n").split(",")
            if existing_header == csv_header:
                needs_header = False
            else:
                logger.warning(
                    "CSV %s has incompatible header (%d cols vs expected %d); truncating.",
                    LOG_CSV_PATH,
                    len(existing_header),
                    len(csv_header),
                )
                log_mode = "w"
        except Exception:
            pass
    log_file = open(LOG_CSV_PATH, mode=log_mode, newline="")
    csv_writer = csv.writer(log_file)
    if needs_header:
        csv_writer.writerow(csv_header)

    logger.info(
        "Starting MLP baseline training\n"
        "  Device: %s | SEQ_LEN: %d | BATCH_SIZE: %d | LR: %s\n"
        "  Hidden: %d × %d, dropout=%.2f\n"
        "  Question encoder: %s",
        DEVICE,
        SEQ_LEN,
        BATCH_SIZE,
        LEARNING_RATE,
        HIDDEN_DIM,
        HIDDEN_LAYERS,
        DROPOUT,
        Q_ENCODER_BACKEND,
    )

    loss_window: deque[float] = deque(maxlen=100)
    acc_window: deque[float] = deque(maxlen=100)
    val_top1_last = float("nan")
    val_top5_last = float("nan")

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

        is_val_snapshot = (
            VAL_FRACTION_DENOM > 0 and scrape_counter % VAL_FRACTION_DENOM == 0
        )
        target_metric_buffer = val_metric_buffer if is_val_snapshot else metric_buffer
        target_metric_buffer.append(token_norm)

        if len(target_metric_buffer) < SEQ_LEN:
            logger.info(
                "Filling %s buffer: %d/%d",
                "val" if is_val_snapshot else "temporal",
                len(target_metric_buffer),
                SEQ_LEN,
            )
            continue

        context_seq = torch.stack(list(target_metric_buffer), dim=0)

        semantic = compute_semantic_features(raw_parsed, live_node_ips)
        qa_pairs = generate_qa_pairs(
            semantic, live_node_ips, live_node_cores, max_pairs=QA_PAIRS_PER_STEP
        )

        target_buffer = val_replay if is_val_snapshot else replay
        ctx_id = target_buffer.add_context(context_seq)
        for q_struct, ans_idx in qa_pairs:
            target_buffer.add_qa(ctx_id, q_struct, ans_idx)

        if len(replay) < MIN_REPLAY_SIZE:
            logger.info("Filling replay buffer: %d/%d", len(replay), MIN_REPLAY_SIZE)
            continue

        try:
            seq_batch, q_batch, ans_batch = replay.sample(BATCH_SIZE)
        except Exception:
            continue

        seq_batch = seq_batch.to(DEVICE)
        ans_batch = ans_batch.to(DEVICE)

        q_emb = encode_questions(question_encoder, q_batch, live_node_ips)
        x = torch.cat([seq_batch.flatten(1), q_emb], dim=-1)
        logits = model(x)
        loss = soft_cross_entropy(logits, ans_batch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            top1 = (preds == ans_batch).float().mean().item()
            _, top5_idx = logits.topk(5, dim=-1)
            top5 = (
                (top5_idx == ans_batch.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )

        loss_window.append(loss.item())
        acc_window.append(top1)
        global_step += 1

        if (
            VAL_INTERVAL > 0
            and global_step % VAL_INTERVAL == 0
            and len(val_replay) >= BATCH_SIZE
        ):
            val_top1_last, val_top5_last = _evaluate_validation(
                val_replay, live_node_ips
            )

        csv_writer.writerow(
            [
                time.time(),
                global_step,
                loss.item(),
                0.0,
                loss.item(),
                top1,
                top5,
                val_top1_last,
                val_top5_last,
                len(replay),
                len(val_replay),
            ]
        )
        log_file.flush()

        if global_step % 10 == 0:
            avg_loss = sum(loss_window) / len(loss_window)
            avg_acc = sum(acc_window) / len(acc_window)
            logger.info(
                "step=%d  loss=%.4f  top1=%.2f%% top5=%.2f%%  "
                "avg_loss=%.4f avg_acc=%.2f%%  "
                "val_top1=%.2f%% val_top5=%.2f%%  grad=%.3f buf=%d val_buf=%d",
                global_step,
                loss.item(),
                top1 * 100,
                top5 * 100,
                avg_loss,
                avg_acc * 100,
                val_top1_last * 100 if val_top1_last == val_top1_last else float("nan"),
                val_top5_last * 100 if val_top5_last == val_top5_last else float("nan"),
                float(grad_norm),
                len(replay),
                len(val_replay),
            )

        if CHECKPOINT_INTERVAL > 0 and global_step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(global_step, live_node_ips, live_node_cores)

        if SUMMARY_INTERVAL > 0 and global_step % SUMMARY_INTERVAL == 0:
            _write_summary(
                global_step,
                semantic,
                loss_window,
                acc_window,
                val_top1=val_top1_last if val_top1_last == val_top1_last else None,
                val_top5=val_top5_last if val_top5_last == val_top5_last else None,
            )


def main_offline():
    """Stream the JEPA dump chronologically through the MLP, mirroring the
    live trainer's per-scrape cadence: each snapshot adds its QA pairs to a
    replay buffer (or held-out val buffer if the dump flagged it), then one
    SGD step is taken per scrape sampling BATCH_SIZE from the replay buffer.
    Step numbers in the CSV equal the scrape counter, matching JEPA's logged
    step axis."""
    summary = load_dump_summary(LOAD_REPLAY_PATH)
    logger.info(
        "Streaming offline mode: dump=%s (snapshots=%s qa_pairs=%s shards=%d)",
        LOAD_REPLAY_PATH,
        summary.get("snapshots_written", "?"),
        summary.get("qa_pairs_written", "?"),
        len(summary.get("shards", [])),
    )

    global_step = load_checkpoint()
    replay = QAReplayBuffer(REPLAY_QA_CAPACITY, REPLAY_CONTEXT_CAPACITY)
    val_replay = QAReplayBuffer(VAL_QA_CAPACITY, VAL_CONTEXT_CAPACITY)

    os.makedirs(os.path.dirname(LOG_CSV_PATH) or ".", exist_ok=True)
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
    if MLP_FRESH_START:
        log_mode = "w"
        write_header = True
        if os.path.isfile(SUMMARY_PATH):
            open(SUMMARY_PATH, "w").close()
            logger.info(
                "MLP_FRESH=1: truncated %s and %s",
                os.path.basename(LOG_CSV_PATH),
                os.path.basename(SUMMARY_PATH),
            )
    else:
        log_mode = "a"
        write_header = not os.path.isfile(LOG_CSV_PATH)
    log_file = open(LOG_CSV_PATH, mode=log_mode, newline="")
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow(csv_header)

    MIN_REPLAY_SIZE = BATCH_SIZE * 10
    model.train()

    val_top1_last = float("nan")
    val_top5_last = float("nan")
    loss_window: deque[float] = deque(maxlen=100)
    acc_window: deque[float] = deque(maxlen=100)

    def _eval_val(live_ips: list[str]) -> tuple[float, float]:
        if len(val_replay) < BATCH_SIZE:
            return float("nan"), float("nan")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                seq_batch, q_batch, ans_batch = val_replay.sample(
                    min(VAL_BATCH_SIZE, len(val_replay))
                )
                seq_batch = seq_batch.to(DEVICE)
                ans_batch = ans_batch.to(DEVICE)
                q_emb = encode_questions(question_encoder, q_batch, live_ips)
                x = torch.cat([seq_batch.flatten(1), q_emb], dim=-1)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                top1 = (preds == ans_batch).float().mean().item()
                _, top5_idx = logits.topk(5, dim=-1)
                top5 = (
                    (top5_idx == ans_batch.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
                return top1, top5
        except Exception as e:
            logger.warning("Val eval failed: %s", e)
            return float("nan"), float("nan")
        finally:
            if was_training:
                model.train()

    logger.info(
        "Starting streaming MLP offline training (1 SGD step per scrape, "
        "batch_size=%d, replay_capacity=%d, val_capacity=%d)",
        BATCH_SIZE,
        REPLAY_QA_CAPACITY,
        VAL_QA_CAPACITY,
    )

    scrape_counter = 0
    last_ips: list[str] = []
    t0 = time.time()

    for _epoch in range(max(1, OFFLINE_EPOCHS)):
        epoch_t0 = time.time()
        for ctx, qa_pairs, is_val, ips, _cores, _ts in iter_dump_by_snapshot(
            LOAD_REPLAY_PATH
        ):
            target = val_replay if is_val else replay
            ctx_id = target.add_context(ctx)
            for q_struct, ans_idx in qa_pairs:
                target.add_qa(ctx_id, q_struct, ans_idx)
            scrape_counter += 1
            if ips:
                last_ips = ips

            if len(replay) < MIN_REPLAY_SIZE:
                continue

            try:
                seq_batch, q_batch, ans_batch = replay.sample(BATCH_SIZE)
            except Exception:
                continue

            seq_batch = seq_batch.to(DEVICE)
            ans_batch = ans_batch.to(DEVICE)

            q_emb = encode_questions(question_encoder, q_batch, last_ips)
            x = torch.cat([seq_batch.flatten(1), q_emb], dim=-1)
            logits = model(x)
            loss = soft_cross_entropy(logits, ans_batch)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                top1 = (preds == ans_batch).float().mean().item()
                _, top5_idx = logits.topk(5, dim=-1)
                top5 = (
                    (top5_idx == ans_batch.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )

            loss_window.append(loss.item())
            acc_window.append(top1)
            global_step += 1

            if VAL_INTERVAL > 0 and global_step % VAL_INTERVAL == 0:
                val_top1_last, val_top5_last = _eval_val(last_ips)

            csv_writer.writerow(
                [
                    time.time(),
                    scrape_counter,
                    loss.item(),
                    0.0,
                    loss.item(),
                    top1,
                    top5,
                    val_top1_last,
                    val_top5_last,
                    len(replay),
                    len(val_replay),
                ]
            )
            log_file.flush()

            if global_step % 100 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "epoch=%d scrape=%d step=%d elapsed=%.0fs  loss=%.4f top1=%.2f%%  "
                    "avg_loss=%.4f avg_acc=%.2f%%  val_top1=%.2f%% val_top5=%.2f%%  "
                    "grad=%.3f buf=%d val_buf=%d",
                    _epoch + 1,
                    scrape_counter,
                    global_step,
                    elapsed,
                    loss.item(),
                    top1 * 100,
                    sum(loss_window) / len(loss_window),
                    sum(acc_window) / len(acc_window) * 100,
                    val_top1_last * 100
                    if val_top1_last == val_top1_last
                    else float("nan"),
                    val_top5_last * 100
                    if val_top5_last == val_top5_last
                    else float("nan"),
                    float(grad_norm),
                    len(replay),
                    len(val_replay),
                )

            if CHECKPOINT_INTERVAL > 0 and global_step % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(global_step, last_ips, [])

        logger.info(
            "Epoch %d/%d complete in %.0fs",
            _epoch + 1,
            OFFLINE_EPOCHS,
            time.time() - epoch_t0,
        )

    val_top1_last, val_top5_last = _eval_val(last_ips)
    logger.info(
        "Streaming offline MLP done. scrapes=%d sgd_steps=%d epochs=%d. "
        "Final val_top1=%.2f%% val_top5=%.2f%%",
        scrape_counter,
        global_step,
        OFFLINE_EPOCHS,
        val_top1_last * 100 if val_top1_last == val_top1_last else float("nan"),
        val_top5_last * 100 if val_top5_last == val_top5_last else float("nan"),
    )
    save_checkpoint(global_step, last_ips, [])
    _write_summary(
        global_step,
        {},
        loss_window,
        acc_window,
        val_top1=val_top1_last if val_top1_last == val_top1_last else None,
        val_top5=val_top5_last if val_top5_last == val_top5_last else None,
    )


if __name__ == "__main__":
    if LOAD_REPLAY_PATH:
        main_offline()
    else:
        main()
