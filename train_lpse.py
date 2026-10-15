"""Streaming training loop for LPSE against a live Kubernetes cluster."""

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
from lpse.lpse import (
    ANSWER_DIM,
    FUSION_DIM,
    LATENT_DIM,
    LPSE,
    JEPAPredictor,
    MomentumMetricEncoder,
)
from lpse.prometheus_ingest import (
    CLUSTER_TOKEN_DIM,
    MAX_NODE_SLOTS,
    RunningNormalizer,
    cluster_snapshot_to_token,
    collect_full_cluster_snapshot,
    compute_semantic_features,
)
from lpse.qa_generator import (
    ANSWER_TEMPLATES,
    NUM_ANSWERS,
    build_answer_soft_targets,
    fill_answer_template,
    generate_qa_pairs,
)
from lpse.question_encoder import (
    QUESTION_DIM,
    build_question_encoder,
    encode_questions,
)
from lpse.replay_buffer import QAReplayBuffer
from lpse.replay_dump import DumpWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = int(os.getenv("SEED", "1337"))
seed_everything(SEED)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_lpse")


SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))
SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "0.5"))


MODEL_VARIANT = "answer_jepa_v1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
REPLAY_QA_CAPACITY = int(
    os.getenv("REPLAY_QA_CAPACITY", os.getenv("REPLAY_CAPACITY", "32768"))
)
REPLAY_CONTEXT_CAPACITY = int(os.getenv("REPLAY_CONTEXT_CAPACITY", "4096"))
QA_PAIRS_PER_STEP = int(os.getenv("QA_PAIRS_PER_STEP", "10"))
NORMALIZER_BURN_IN = int(os.getenv("NORMALIZER_BURN_IN", "2000"))


SOFT_LABEL_MASS = float(os.getenv("SOFT_LABEL_MASS", "0.30"))
SOFT_LABEL_TAU = float(os.getenv("SOFT_LABEL_TAU", "1.0"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))


JEPA_SSL_WEIGHT = float(os.getenv("JEPA_SSL_WEIGHT", "0.5"))
JEPA_EMA_MOMENTUM = float(os.getenv("JEPA_EMA_MOMENTUM", "0.996"))
JEPA_PRED_HORIZON = int(os.getenv("JEPA_PRED_HORIZON", "1"))


VAR_WEIGHT = 1.0
VAR_GAMMA = 1.0
COV_WEIGHT = 0.005


NODE_DROPOUT_PROB = float(os.getenv("NODE_DROPOUT_PROB", "0.20"))
MAX_NODES_TO_DROP = int(os.getenv("MAX_NODES_TO_DROP", "2"))
MIN_NODES_AFTER_DROPOUT = int(os.getenv("MIN_NODES_AFTER_DROPOUT", "3"))


VAL_FRACTION_DENOM = int(os.getenv("VAL_FRACTION_DENOM", "10"))
VAL_QA_CAPACITY = int(os.getenv("VAL_QA_CAPACITY", "4096"))
VAL_CONTEXT_CAPACITY = int(os.getenv("VAL_CONTEXT_CAPACITY", "512"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
VAL_BATCH_SIZE = int(os.getenv("VAL_BATCH_SIZE", "256"))


RUN_SUFFIX = os.getenv("LPSE_RUN_SUFFIX", "2")
CHECKPOINT_PATH = os.path.abspath(
    os.getenv("CHECKPOINT_PATH", f"artifacts/checkpoints/lpse{RUN_SUFFIX}.pt")
)
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "200"))
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "50"))
SUMMARY_PATH = os.path.abspath(
    os.getenv("SUMMARY_PATH", f"artifacts/logs/lpse_summary{RUN_SUFFIX}.txt")
)
LOG_CSV_PATH = os.path.abspath(
    os.getenv("LOG_CSV_PATH", f"artifacts/logs/lpse_training_log{RUN_SUFFIX}.csv")
)


ENCODER_HEADS = int(os.getenv("ENCODER_HEADS", "16"))
ENCODER_LAYERS = int(os.getenv("ENCODER_LAYERS", "12"))
FUSION_HEADS_N = int(os.getenv("FUSION_HEADS", "16"))
FUSION_LAYERS_N = int(os.getenv("FUSION_LAYERS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
CODEBOOK_TAU = float(os.getenv("CODEBOOK_TAU", "0.07"))


Q_ENCODER_BACKEND = os.getenv("Q_ENCODER_BACKEND", "sentence_transformer")


STRUCTURED_QUESTION = os.getenv("STRUCTURED_QUESTION", "1").strip() not in (
    "",
    "0",
    "false",
    "False",
)
TOKEN_QUESTION = os.getenv("TOKEN_QUESTION", "0").strip() not in (
    "",
    "0",
    "false",
    "False",
)


LAP_EPS = float(os.getenv("LAP_EPS", "0.0"))
LAP_INNER_STEPS = int(os.getenv("LAP_INNER_STEPS", "2"))
LAP_LR = float(os.getenv("LAP_LR", "3e-4"))


DUMP_DIR = os.getenv("DUMP_DIR", "artifacts/dumps/jepa_live_lap")
DUMP_SHARD_SIZE = int(os.getenv("DUMP_SHARD_SIZE", "100"))


logger.info("Discovering cluster topology...")
node_ips = get_node_ips()
if not node_ips:
    raise RuntimeError("No Kubernetes nodes found via API")

NUM_NODES = len(node_ips)
node_cores = []
for ip in node_ips:
    cores = get_num_cores_from_node_exporter(ip)
    if cores <= 0:
        logger.warning("Could not detect cores for %s, falling back to 10", ip)
        cores = 10
    node_cores.append(cores)

INPUT_DIM = CLUSTER_TOKEN_DIM
logger.info(
    "Cluster: %d nodes, %d total cores, input_dim=%d, answer_codebook=%d",
    NUM_NODES,
    sum(node_cores),
    INPUT_DIM,
    NUM_ANSWERS,
)


_enc_kwargs = {}
if Q_ENCODER_BACKEND == "hybrid":
    _enc_kwargs = {
        "use_structured": STRUCTURED_QUESTION,
    }
question_encoder = build_question_encoder(
    backend=Q_ENCODER_BACKEND,
    out_dim=QUESTION_DIM,
    **_enc_kwargs,
).to(DEVICE)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class LatentParaphraser(nn.Module):
    """Learns a bounded drift direction in question-embedding space.

    Maps a question embedding to an unnormalized direction; the training
    loop normalizes and scales it to an epsilon ball (fraction of ||q_emb||).
    Trained adversarially (maximize QA loss) in a dual loop with the model,
    so it discovers the actual latent directions paraphrases drift along
    instead of hammering the single strongest PGD gradient.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.out_linear: nn.Linear = self.net[-1]
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def forward(self, q_emb: torch.Tensor, eps: float) -> torch.Tensor:

        d = self.net(q_emb)
        d = F.normalize(d, dim=-1)
        scale = q_emb.norm(dim=-1, keepdim=True).clamp_min(1e-4) * eps
        return d * scale


def _load_matching_state_dict(
    module: nn.Module, state_dict: dict[str, torch.Tensor]
) -> list[str]:
    current_state = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    current_state.update(compatible)
    module.load_state_dict(current_state)
    return skipped


model = LPSE(
    input_dim=INPUT_DIM,
    num_answers=NUM_ANSWERS,
    question_encoder=question_encoder,
    latent_dim=LATENT_DIM,
    fusion_dim=FUSION_DIM,
    answer_dim=ANSWER_DIM,
    encoder_heads=ENCODER_HEADS,
    encoder_layers=ENCODER_LAYERS,
    fusion_heads=FUSION_HEADS_N,
    fusion_layers=FUSION_LAYERS_N,
    dropout=DROPOUT,
    codebook_temperature=CODEBOOK_TAU,
).to(DEVICE)

logger.info(
    "Model params: total=%d metric_encoder=%d fusion=%d codebook=%d question_encoder=%d",
    _count_params(model),
    _count_params(model.metric_encoder),
    _count_params(model.fusion),
    _count_params(model.codebook),
    _count_params(model.question_encoder),
)


ema_encoder = MomentumMetricEncoder(
    model.metric_encoder, momentum=JEPA_EMA_MOMENTUM
).to(DEVICE)
ema_encoder.eval()

jepa_predictor = JEPAPredictor(latent_dim=LATENT_DIM).to(DEVICE)
logger.info(
    "JEPA predictor params: %d (SSL weight=%.2f)",
    _count_params(jepa_predictor),
    JEPA_SSL_WEIGHT,
)


normalizer = RunningNormalizer(INPUT_DIM)


lap_perturber = LatentParaphraser(dim=QUESTION_DIM).to(DEVICE)
lap_optimizer = optim.AdamW(lap_perturber.parameters(), lr=LAP_LR)


trainable_params = list(model.parameters()) + list(jepa_predictor.parameters())
optimizer = optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


_soft_target_rows = build_answer_soft_targets(
    num_answers=NUM_ANSWERS,
    soft_mass=SOFT_LABEL_MASS,
    tau=SOFT_LABEL_TAU,
    uniform_smoothing=LABEL_SMOOTHING,
)
SOFT_TARGET_MATRIX = torch.tensor(_soft_target_rows, dtype=torch.float32, device=DEVICE)


def soft_cross_entropy(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """KL-style cross-entropy against soft targets indexed by `target_idx`."""
    soft_targets = SOFT_TARGET_MATRIX.index_select(0, target_idx)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def vicreg_losses(
    z: torch.Tensor, gamma: float = VAR_GAMMA, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VICReg variance and covariance terms for a batch of latents.

    Variance: hinge penalty when per-dim std falls below `gamma`. Forces
        each latent dimension to be informative across the batch.
    Covariance: squared off-diagonal penalty on the empirical covariance
        matrix. Forces dimensions to encode different information.

    Returns: (var_loss, cov_loss). z: [B, D].
    """
    n, d = z.shape
    if n < 2:
        zero = z.new_zeros(())
        return zero, zero
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (n - 1)
    off_diag_sq = (cov - torch.diag(torch.diag(cov))) ** 2
    cov_loss = off_diag_sq.sum() / d
    return var_loss, cov_loss


def save_checkpoint(
    step: int,
    current_node_ips: list[str] | None = None,
    current_node_cores: list[int] | None = None,
):
    ckpt_dir = os.path.dirname(CHECKPOINT_PATH)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    save_node_ips = current_node_ips if current_node_ips is not None else node_ips
    save_node_cores = (
        current_node_cores if current_node_cores is not None else node_cores
    )
    torch.save(
        {
            "model": model.state_dict(),
            "ema_encoder": ema_encoder.state_dict(),
            "jepa_predictor": jepa_predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "lap_perturber": lap_perturber.state_dict()
            if lap_perturber is not None
            else None,
            "lap_optimizer": lap_optimizer.state_dict()
            if lap_optimizer is not None
            else None,
            "step": step,
            "node_ips": save_node_ips,
            "node_cores": save_node_cores,
            "answer_templates": ANSWER_TEMPLATES,
            "input_dim": INPUT_DIM,
            "max_node_slots": MAX_NODE_SLOTS,
            "model_variant": MODEL_VARIANT,
            "q_encoder_backend": Q_ENCODER_BACKEND,
            "codebook_temperature": CODEBOOK_TAU,
            "model_config": {
                "input_dim": INPUT_DIM,
                "latent_dim": LATENT_DIM,
                "fusion_dim": FUSION_DIM,
                "answer_dim": ANSWER_DIM,
                "encoder_heads": ENCODER_HEADS,
                "encoder_layers": ENCODER_LAYERS,
                "fusion_heads": FUSION_HEADS_N,
                "fusion_layers": FUSION_LAYERS_N,
                "dropout": DROPOUT,
                "codebook_temperature": CODEBOOK_TAU,
                "model_variant": MODEL_VARIANT,
                "q_encoder_backend": Q_ENCODER_BACKEND,
                "max_node_slots": MAX_NODE_SLOTS,
                "use_structured": getattr(question_encoder, "use_structured", None),
                "use_token_sequence": model.use_token_sequence,
            },
        },
        CHECKPOINT_PATH,
    )
    logger.info("Checkpoint saved at step %d → %s", step, CHECKPOINT_PATH)


def load_checkpoint() -> int:
    if not os.path.isfile(CHECKPOINT_PATH):
        return 0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    ckpt_config = ckpt.get("model_config", {})
    ckpt_variant = ckpt.get("model_variant", ckpt_config.get("model_variant"))
    if ckpt_variant != MODEL_VARIANT:
        logger.warning(
            "Checkpoint model_variant=%s does not match current model_variant=%s; starting fresh.",
            ckpt_variant,
            MODEL_VARIANT,
        )
        return 0
    ckpt_input_dim = ckpt.get("input_dim", ckpt_config.get("input_dim"))
    if ckpt_input_dim is None or int(ckpt_input_dim) != INPUT_DIM:
        logger.warning(
            "Checkpoint input_dim=%s does not match current input_dim=%d; starting fresh.",
            ckpt_input_dim,
            INPUT_DIM,
        )
        return 0
    try:
        skipped = _load_matching_state_dict(model, ckpt["model"])
        if skipped:
            logger.info("Skipping incompatible model keys: %s", skipped[:6])
    except Exception as e:
        logger.warning("Partial model load: %s", e)
    if "ema_encoder" in ckpt:
        try:
            _load_matching_state_dict(ema_encoder, ckpt["ema_encoder"])
        except Exception as e:
            logger.warning("Skipping EMA encoder state: %s", e)
    else:
        ema_encoder.encoder.load_state_dict(model.metric_encoder.state_dict())
    if "jepa_predictor" in ckpt:
        try:
            _load_matching_state_dict(jepa_predictor, ckpt["jepa_predictor"])
        except Exception as e:
            logger.warning("Skipping JEPA predictor state: %s", e)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        logger.info(
            "Optimizer LR overridden to %s after checkpoint load", LEARNING_RATE
        )
    except Exception as e:
        logger.warning("Skipping optimizer state: %s", e)
    if "normalizer" in ckpt:
        try:
            normalizer.load_state_dict(ckpt["normalizer"])
        except Exception:
            pass
    if (
        lap_perturber is not None
        and "lap_perturber" in ckpt
        and ckpt.get("lap_perturber") is not None
    ):
        try:
            lap_perturber.load_state_dict(ckpt["lap_perturber"])
            if (
                lap_optimizer is not None
                and "lap_optimizer" in ckpt
                and ckpt.get("lap_optimizer") is not None
            ):
                lap_optimizer.load_state_dict(ckpt["lap_optimizer"])
                for g in lap_optimizer.param_groups:
                    g["lr"] = LAP_LR
            logger.info("LAP perturber state loaded")
        except Exception as e:
            logger.warning("Partial LAP state load: %s", e)
    step = int(ckpt.get("step", 0))
    logger.info("Loaded checkpoint from %s at step %d", CHECKPOINT_PATH, step)
    return step


def main():
    MIN_REPLAY_SIZE = BATCH_SIZE * 10
    global_step = load_checkpoint()

    metric_buffer = deque(maxlen=SEQ_LEN)
    val_metric_buffer = deque(maxlen=SEQ_LEN)
    replay = QAReplayBuffer(REPLAY_QA_CAPACITY, REPLAY_CONTEXT_CAPACITY)
    val_replay = QAReplayBuffer(VAL_QA_CAPACITY, VAL_CONTEXT_CAPACITY)
    scrape_counter = 0
    model.train()

    dump_writer = DumpWriter(DUMP_DIR, shard_size=DUMP_SHARD_SIZE) if DUMP_DIR else None
    if dump_writer is not None:
        logger.info(
            "Replay dump enabled → %s (shard_size=%d)", DUMP_DIR, DUMP_SHARD_SIZE
        )

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
        "var_loss",
        "cov_loss",
        "latent_mean_std",
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
        "Starting LPSE training on live cluster metrics\n"
        "  Device: %s | SEQ_LEN: %d | BATCH_SIZE: %d | LR: %s\n"
        "  Answer codebook: %d templates | QA loss: soft-CE (mass=%.2f, smooth=%.2f)\n"
        "  JEPA SSL weight: %.2f | EMA momentum: %.4f\n"
        "  Replay: qa=%d ctx=%d | Val: 1/%d snapshots, qa=%d ctx=%d, eval every %d steps\n"
        "  Question encoder: %s",
        DEVICE,
        SEQ_LEN,
        BATCH_SIZE,
        LEARNING_RATE,
        NUM_ANSWERS,
        SOFT_LABEL_MASS,
        LABEL_SMOOTHING,
        JEPA_SSL_WEIGHT,
        JEPA_EMA_MOMENTUM,
        REPLAY_QA_CAPACITY,
        REPLAY_CONTEXT_CAPACITY,
        VAL_FRACTION_DENOM,
        VAL_QA_CAPACITY,
        VAL_CONTEXT_CAPACITY,
        VAL_INTERVAL,
        Q_ENCODER_BACKEND,
    )

    loss_window = deque(maxlen=100)
    acc_window = deque(maxlen=100)
    ssl_window = deque(maxlen=100)
    val_top1_last = float("nan")
    val_top5_last = float("nan")

    def _discover_live_topology() -> tuple[list[str], list[int]]:
        live_node_ips = get_node_ips()
        live_node_cores: list[int] = []
        for ip in live_node_ips:
            cores = get_num_cores_from_node_exporter(ip)
            if cores <= 0:
                cores = 10
            live_node_cores.append(cores)
        return live_node_ips, live_node_cores

    def _maybe_dropout_nodes(
        node_ips: list[str],
        node_cores: list[int],
        node_tensors: torch.Tensor,
        raw_parsed: dict[str, dict[str, list[float]]],
    ) -> tuple[list[str], list[int], torch.Tensor, dict[str, dict[str, list[float]]]]:
        """
        Randomly drop up to MAX_NODES_TO_DROP nodes from this snapshot.
        With probability (1 - NODE_DROPOUT_PROB) the full snapshot is kept;
        otherwise k ~ Uniform{1, ..., max_drop} nodes are dropped, where
        max_drop respects MIN_NODES_AFTER_DROPOUT.

        All four parallel structures are filtered together so the cluster token,
        semantic features, and the QA pairs generated from this snapshot agree
        on which nodes exist this step.
        """
        n = len(node_ips)
        if NODE_DROPOUT_PROB <= 0.0 or n <= MIN_NODES_AFTER_DROPOUT:
            return node_ips, node_cores, node_tensors, raw_parsed
        if random.random() >= NODE_DROPOUT_PROB:
            return node_ips, node_cores, node_tensors, raw_parsed

        max_drop = min(MAX_NODES_TO_DROP, n - MIN_NODES_AFTER_DROPOUT)
        if max_drop < 1:
            return node_ips, node_cores, node_tensors, raw_parsed

        k = random.randint(1, max_drop)
        drop_idx = set(random.sample(range(n), k))

        new_ips = [ip for i, ip in enumerate(node_ips) if i not in drop_idx]
        new_cores = [c for i, c in enumerate(node_cores) if i not in drop_idx]
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
                logger.warning("No live nodes discovered; waiting for topology refresh")
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
            split_name = "val" if is_val_snapshot else "temporal"
            logger.info(
                "Filling %s buffer: %d/%d",
                split_name,
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

        if dump_writer is not None:
            dump_writer.add_snapshot(
                context_seq=context_seq,
                qa_pairs=qa_pairs,
                is_val=is_val_snapshot,
                node_ips=live_node_ips,
                node_cores=live_node_cores,
            )

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

        _online_seq_latent, online_ctx = model.metric_encoder(seq_batch)

        if LAP_EPS > 0.0 and model.training:
            clean_q = q_emb.detach()
            for _ in range(max(1, LAP_INNER_STEPS)):
                lap_optimizer.zero_grad()
                delta = lap_perturber(clean_q, LAP_EPS)
                adv_q = clean_q + delta
                _adv_latent = model.predict_answer_latent(online_ctx, adv_q)
                _adv_logits = model.codebook.logits_from_latent(_adv_latent)
                adv_loss = soft_cross_entropy(_adv_logits, ans_batch)
                grads = torch.autograd.grad(adv_loss, list(lap_perturber.parameters()))
                for p, g in zip(lap_perturber.parameters(), grads):
                    p.grad = -g
                lap_optimizer.step()
            with torch.no_grad():
                delta = lap_perturber(q_emb.detach(), LAP_EPS)
            q_emb = q_emb + delta

        pred_answer_latent = model.predict_answer_latent(online_ctx, q_emb)
        logits = model.codebook.logits_from_latent(pred_answer_latent)
        qa_loss = soft_cross_entropy(logits, ans_batch)

        if JEPA_SSL_WEIGHT > 0.0 and seq_batch.size(1) > JEPA_PRED_HORIZON:
            horizon = max(1, JEPA_PRED_HORIZON)
            short_seq = seq_batch[:, :-horizon, :]
            online_short_seq, online_short_ctx = model.metric_encoder(short_seq)
            with torch.no_grad():
                ema_full_seq, _ = ema_encoder(seq_batch)
                target_latents = ema_full_seq[:, -horizon:, :]
                target_latents = F.normalize(target_latents, dim=-1)
            predicted_latent = jepa_predictor(online_short_ctx)
            predicted_latent = F.normalize(predicted_latent, dim=-1)
            ssl_per_step = 1.0 - (predicted_latent.unsqueeze(1) * target_latents).sum(
                dim=-1
            )
            ssl_loss = ssl_per_step.mean()
        else:
            ssl_loss = torch.zeros((), device=DEVICE)

        var_loss, cov_loss = vicreg_losses(pred_answer_latent)

        total_loss = (
            qa_loss
            + JEPA_SSL_WEIGHT * ssl_loss
            + VAR_WEIGHT * var_loss
            + COV_WEIGHT * cov_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        ema_encoder.update(model.metric_encoder)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            top1_acc = (preds == ans_batch).float().mean().item()
            _, top5_idx = logits.topk(5, dim=-1)
            top5_acc = (
                (top5_idx == ans_batch.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )
            latent_mean_std = float(
                torch.sqrt(pred_answer_latent.var(dim=0, unbiased=False) + 1e-4)
                .mean()
                .item()
            )

        loss_window.append(total_loss.item())
        acc_window.append(top1_acc)
        ssl_window.append(float(ssl_loss.item()))

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
                qa_loss.item(),
                float(ssl_loss.item()),
                total_loss.item(),
                top1_acc,
                top5_acc,
                val_top1_last,
                val_top5_last,
                len(replay),
                len(val_replay),
                float(var_loss.item()),
                float(cov_loss.item()),
                latent_mean_std,
            ]
        )
        log_file.flush()

        if global_step % 10 == 0:
            avg_loss = sum(loss_window) / len(loss_window)
            avg_acc = sum(acc_window) / len(acc_window)
            avg_ssl = sum(ssl_window) / max(1, len(ssl_window))
            sample_idx = 0
            pred_ans = int(preds[sample_idx].item())
            true_ans = int(ans_batch[sample_idx].item())
            logger.info(
                "step=%d  loss=%.4f (qa=%.4f ssl=%.4f var=%.4f cov=%.4f std=%.3f)  "
                "top1=%.2f%% top5=%.2f%%  "
                "avg_loss=%.4f avg_acc=%.2f%% avg_ssl=%.4f  "
                "val_top1=%.2f%% val_top5=%.2f%%  grad=%.3f lr=%.2e  buf=%d val_buf=%d",
                global_step,
                total_loss.item(),
                qa_loss.item(),
                float(ssl_loss.item()),
                float(var_loss.item()),
                float(cov_loss.item()),
                latent_mean_std,
                top1_acc * 100,
                top5_acc * 100,
                avg_loss,
                avg_acc * 100,
                avg_ssl,
                val_top1_last * 100 if val_top1_last == val_top1_last else float("nan"),
                val_top5_last * 100 if val_top5_last == val_top5_last else float("nan"),
                float(grad_norm),
                optimizer.param_groups[0]["lr"],
                len(replay),
                len(val_replay),
            )
            if pred_ans < NUM_ANSWERS and true_ans < NUM_ANSWERS:
                rendered_pred = fill_answer_template(pred_ans, live_node_ips, semantic)
                rendered_true = fill_answer_template(true_ans, live_node_ips, semantic)
                logger.info(
                    "  sample Q: %s\n  predicted: %s\n  true:      %s",
                    q_batch[sample_idx].get("raw", "?"),
                    rendered_pred,
                    rendered_true,
                )

        if CHECKPOINT_INTERVAL > 0 and global_step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(global_step, live_node_ips, live_node_cores)

        if SUMMARY_INTERVAL > 0 and global_step % SUMMARY_INTERVAL == 0:
            _write_summary(
                global_step,
                semantic,
                loss_window,
                acc_window,
                ssl_window=ssl_window,
                val_top1=val_top1_last if val_top1_last == val_top1_last else None,
                val_top5=val_top5_last if val_top5_last == val_top5_last else None,
            )


def _evaluate_validation(
    val_replay: "QAReplayBuffer", live_node_ips: list[str]
) -> tuple[float, float]:
    """Run model.eval() over the held-out validation buffer; return (top1, top5)."""
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
            _seq_latent, ctx = model.metric_encoder(seq_batch)
            pred_latent = model.predict_answer_latent(ctx, q_emb)
            logits = model.codebook.logits_from_latent(pred_latent)
            preds = logits.argmax(dim=-1)
            top1 = (preds == ans_batch).float().mean().item()
            _, top5_idx = logits.topk(5, dim=-1)
            top5 = (
                (top5_idx == ans_batch.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )
            return top1, top5
    except Exception as e:
        logger.error("Validation eval failed: %s\n%s", e, traceback.format_exc())
        return float("nan"), float("nan")
    finally:
        if was_training:
            model.train()


def _write_summary(
    step,
    semantic,
    loss_window,
    acc_window,
    ssl_window=None,
    val_top1=None,
    val_top5=None,
):
    try:
        summary = {
            "step": step,
            "timestamp": time.time(),
            "avg_loss": sum(loss_window) / max(1, len(loss_window)),
            "avg_accuracy": sum(acc_window) / max(1, len(acc_window)),
            "avg_ssl_loss": (sum(ssl_window) / max(1, len(ssl_window)))
            if ssl_window
            else None,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "num_answers": NUM_ANSWERS,
            "nodes": {},
        }
        for ip, feats in semantic.items():
            summary["nodes"][ip] = {k: v for k, v in feats.items() if v is not None}

        os.makedirs(os.path.dirname(SUMMARY_PATH) or ".", exist_ok=True)
        with open(SUMMARY_PATH, "a") as f:
            f.write(json.dumps(summary, separators=(",", ":")) + "\n")
        logger.info("Wrote summary → %s", SUMMARY_PATH)
    except Exception as e:
        logger.warning("Failed to write summary: %s", e)


if __name__ == "__main__":
    main()
