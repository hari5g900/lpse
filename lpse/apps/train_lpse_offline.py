#!/usr/bin/env python3
"""Offline LPSE trainer: stream a replay dump (as the baseline trainers do) and
train LPSE with the exact losses of train_lpse.py — soft-CE QA + JEPA SSL
(EMA future-latent prediction) + VICReg — taking one SGD step per snapshot
sampled from a replay buffer.

This is the fast path for architecture experiments: the dump contexts are
already normalised (stored as the live trainer produced them), so no live
scraping and no normalizer update are needed. The QA distribution is identical
to live training; only the accumulation source differs.

Usage:
  LOAD_REPLAY_PATH=artifacts/dumps/jepa_run Q_ENCODER_BACKEND=hybrid \
      python -m lpse.apps.train_lpse_offline
"""

import csv
import logging
import os
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lpse import seed_everything
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
)
from lpse.qa_generator import (
    ANSWER_TEMPLATES,
    NUM_ANSWERS,
    build_answer_soft_targets,
)
from lpse.question_encoder import (
    QUESTION_DIM,
    build_question_encoder,
    encode_question_tokens,
    encode_questions,
)
from lpse.replay_buffer import QAReplayBuffer
from lpse.replay_dump import iter_dump_by_snapshot, load_dump_summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_lpse_offline")

LOAD_REPLAY_PATH = os.getenv("LOAD_REPLAY_PATH", "")
OFFLINE_EPOCHS = int(os.getenv("OFFLINE_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
REPLAY_QA_CAPACITY = int(os.getenv("REPLAY_QA_CAPACITY", "32768"))
REPLAY_CONTEXT_CAPACITY = int(os.getenv("REPLAY_CONTEXT_CAPACITY", "4096"))
VAL_QA_CAPACITY = int(os.getenv("VAL_QA_CAPACITY", "4096"))
VAL_CONTEXT_CAPACITY = int(os.getenv("VAL_CONTEXT_CAPACITY", "512"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
VAL_BATCH_SIZE = int(os.getenv("VAL_BATCH_SIZE", "256"))

SOFT_LABEL_MASS = float(os.getenv("SOFT_LABEL_MASS", "0.30"))
SOFT_LABEL_TAU = float(os.getenv("SOFT_LABEL_TAU", "1.0"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))

JEPA_SSL_WEIGHT = float(os.getenv("JEPA_SSL_WEIGHT", "0.5"))
JEPA_EMA_MOMENTUM = float(os.getenv("JEPA_EMA_MOMENTUM", "0.996"))
JEPA_PRED_HORIZON = int(os.getenv("JEPA_PRED_HORIZON", "1"))
VAR_WEIGHT = 1.0
COV_WEIGHT = 0.005


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


if TOKEN_QUESTION and Q_ENCODER_BACKEND != "hybrid":
    raise SystemExit(
        f"TOKEN_QUESTION=1 requires Q_ENCODER_BACKEND=hybrid (got {Q_ENCODER_BACKEND!r}). "
        "The token-sequence path needs HybridQuestionEncoder.token_states."
    )


Q_EMB_NOISE_STD = float(os.getenv("Q_EMB_NOISE_STD", "0.0"))
Q_EMB_DROPOUT = float(os.getenv("Q_EMB_DROPOUT", "0.0"))


TEXT_NOISE_STD = float(os.getenv("TEXT_NOISE_STD", "0.0"))
TEXT_DROPOUT = float(os.getenv("TEXT_DROPOUT", "0.0"))


ADV_EPS = float(os.getenv("ADV_EPS", "0.0"))
ADV_STEPS = int(os.getenv("ADV_STEPS", "1"))


LAP_EPS = float(os.getenv("LAP_EPS", "0.0"))
LAP_INNER_STEPS = int(os.getenv("LAP_INNER_STEPS", "2"))
LAP_LR = float(os.getenv("LAP_LR", "3e-4"))

ENCODER_HEADS = int(os.getenv("ENCODER_HEADS", "16"))
ENCODER_LAYERS = int(os.getenv("ENCODER_LAYERS", "12"))
FUSION_HEADS_N = int(os.getenv("FUSION_HEADS", "16"))
FUSION_LAYERS_N = int(os.getenv("FUSION_LAYERS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
CODEBOOK_TAU = float(os.getenv("CODEBOOK_TAU", "0.07"))


MODEL_VARIANT = "answer_jepa_offline_v1"

RUN_SUFFIX = os.getenv("LPSE_RUN_SUFFIX", "offline")
CHECKPOINT_PATH = os.path.abspath(
    os.getenv("CHECKPOINT_PATH", f"artifacts/checkpoints/lpse{RUN_SUFFIX}.pt")
)
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "500"))


EPOCH_CKPT = int(os.getenv("EPOCH_CKPT", "0"))
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "500"))
LOG_CSV_PATH = os.path.abspath(
    os.getenv("LOG_CSV_PATH", f"artifacts/logs/lpse_offline{RUN_SUFFIX}.csv")
)


SEED = int(os.getenv("SEED", "1337"))
seed_everything(SEED)


def soft_cross_entropy(
    logits: torch.Tensor, target_idx: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
    st = soft_targets.index_select(0, target_idx)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(st * log_probs).sum(dim=-1).mean()


def _perturb_question_embedding(q_emb: torch.Tensor, training: bool) -> torch.Tensor:
    """Apply lever A: Gaussian noise + embedding dropout to question embeddings.

    Training-only regularizer. Perturbs the question embedding so the
    fusion/readout is locally constant in question space, i.e. robust to the
    small text-embedding drift that paraphrases introduce. Structured fields
    are untouched by this noise, mirroring the benchmark where paraphrase
    changes only the raw text.
    """
    if not training:
        return q_emb
    if Q_EMB_NOISE_STD > 0.0:
        norms = q_emb.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        q_emb = q_emb + torch.randn_like(q_emb) * (norms * Q_EMB_NOISE_STD)
    if Q_EMB_DROPOUT > 0.0:
        mask = torch.bernoulli(torch.full_like(q_emb, 1.0 - Q_EMB_DROPOUT))
        q_emb = q_emb * mask / (1.0 - Q_EMB_DROPOUT)
    return q_emb


def vicreg_losses(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    n, d = z.shape
    if n < 2:
        return z.new_zeros(()), z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    off = (cov - torch.diag(torch.diag(cov))) ** 2
    return var_loss, off.sum() / d


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


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _save_ckpt(
    step: int,
    last_ips: list[str],
    path: str,
    model,
    ema_encoder,
    jepa_predictor,
    optimizer,
    lap_perturber,
    lap_optimizer,
) -> None:
    """Atomically save a full training checkpoint (model + EMA + LAP state)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema_encoder": ema_encoder.state_dict(),
        "jepa_predictor": jepa_predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lap_perturber": lap_perturber.state_dict(),
        "lap_optimizer": lap_optimizer.state_dict(),
        "step": step,
        "node_ips": last_ips,
        "node_cores": [],
        "answer_templates": ANSWER_TEMPLATES,
        "input_dim": CLUSTER_TOKEN_DIM,
        "max_node_slots": MAX_NODE_SLOTS,
        "model_variant": MODEL_VARIANT,
        "q_encoder_backend": Q_ENCODER_BACKEND,
        "codebook_temperature": CODEBOOK_TAU,
        "model_config": {
            "input_dim": CLUSTER_TOKEN_DIM,
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
            "use_structured": STRUCTURED_QUESTION,
            "use_token_sequence": TOKEN_QUESTION,
        },
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info("Checkpoint saved at step %d -> %s", step, path)


def _load_checkpoint(
    model,
    ema_encoder,
    jepa_predictor,
    optimizer,
    lap_perturber=None,
    lap_optimizer=None,
) -> int:
    """Load an existing checkpoint (if any) and return the resume step."""
    if not os.path.isfile(CHECKPOINT_PATH):
        return 0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if ckpt.get("model_variant") != MODEL_VARIANT:
        logger.warning("Checkpoint variant mismatch; starting fresh.")
        return 0
    ckpt_config = ckpt.get("model_config", {})
    ckpt_input_dim = ckpt.get("input_dim", ckpt_config.get("input_dim"))
    if ckpt_input_dim is None or int(ckpt_input_dim) != CLUSTER_TOKEN_DIM:
        logger.warning("Checkpoint input_dim mismatch; starting fresh.")
        return 0
    current = model.state_dict()
    compatible = {
        k: v
        for k, v in ckpt["model"].items()
        if k in current and current[k].shape == v.shape
    }
    current.update(compatible)
    model.load_state_dict(current)
    try:
        ema_encoder.load_state_dict(ckpt["ema_encoder"])
        jepa_predictor.load_state_dict(ckpt["jepa_predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE
    except Exception as e:
        logger.warning("Partial resume state load: %s", e)
    if lap_perturber is not None and "lap_perturber" in ckpt:
        try:
            lap_perturber.load_state_dict(ckpt["lap_perturber"])
            if lap_optimizer is not None and "lap_optimizer" in ckpt:
                lap_optimizer.load_state_dict(ckpt["lap_optimizer"])
                for g in lap_optimizer.param_groups:
                    g["lr"] = LAP_LR
        except Exception as e:
            logger.warning("Partial LAP state load: %s", e)
    step = int(ckpt.get("step", 0))
    logger.info("Resumed from %s at step %d", CHECKPOINT_PATH, step)
    return step


def main():
    if not LOAD_REPLAY_PATH:
        raise SystemExit("LOAD_REPLAY_PATH is required (a replay dump directory).")
    summary = load_dump_summary(LOAD_REPLAY_PATH)
    logger.info(
        "Dump: %s (snapshots=%s qa=%s shards=%d)",
        LOAD_REPLAY_PATH,
        summary.get("snapshots_written", "?"),
        summary.get("qa_pairs_written", "?"),
        len(summary.get("shards", [])),
    )

    question_encoder = build_question_encoder(
        backend=Q_ENCODER_BACKEND,
        out_dim=QUESTION_DIM,
        use_structured=STRUCTURED_QUESTION,
        text_noise_std=TEXT_NOISE_STD,
        text_dropout=TEXT_DROPOUT,
    ).to(DEVICE)
    model = LPSE(
        input_dim=CLUSTER_TOKEN_DIM,
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
        use_token_sequence=TOKEN_QUESTION,
    ).to(DEVICE)
    ema_encoder = MomentumMetricEncoder(
        model.metric_encoder, momentum=JEPA_EMA_MOMENTUM
    ).to(DEVICE)
    ema_encoder.eval()
    jepa_predictor = JEPAPredictor(latent_dim=LATENT_DIM).to(DEVICE)
    logger.info(
        "Model params: total=%d (metric=%d fusion=%d codebook=%d)",
        _count_params(model),
        _count_params(model.metric_encoder),
        _count_params(model.fusion),
        _count_params(model.codebook),
    )

    trainable = list(model.parameters()) + list(jepa_predictor.parameters())
    optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    lap_perturber = LatentParaphraser(dim=QUESTION_DIM).to(DEVICE)
    lap_optimizer = optim.AdamW(lap_perturber.parameters(), lr=LAP_LR)

    soft_targets = torch.tensor(
        build_answer_soft_targets(
            num_answers=NUM_ANSWERS,
            soft_mass=SOFT_LABEL_MASS,
            tau=SOFT_LABEL_TAU,
            uniform_smoothing=LABEL_SMOOTHING,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    replay = QAReplayBuffer(REPLAY_QA_CAPACITY, REPLAY_CONTEXT_CAPACITY)
    val_replay = QAReplayBuffer(VAL_QA_CAPACITY, VAL_CONTEXT_CAPACITY)
    MIN_REPLAY_SIZE = BATCH_SIZE * 10

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
    resuming = os.path.isfile(CHECKPOINT_PATH)
    log_mode = "a"
    write_header = True
    if resuming and os.path.isfile(LOG_CSV_PATH):
        try:
            with open(LOG_CSV_PATH) as f:
                existing_header = f.readline().rstrip("\n").split(",")
            write_header = existing_header != csv_header
            if write_header:
                logger.warning(
                    "CSV %s has incompatible header; truncating.", LOG_CSV_PATH
                )
                log_mode = "w"
        except Exception:
            write_header = True
    log_file = open(LOG_CSV_PATH, mode=log_mode, newline="")
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow(csv_header)

    model.train()
    global_step = _load_checkpoint(
        model, ema_encoder, jepa_predictor, optimizer, lap_perturber, lap_optimizer
    )
    last_ips: list[str] = []
    loss_window: deque[float] = deque(maxlen=100)
    acc_window: deque[float] = deque(maxlen=100)
    t0 = time.time()

    def _eval_val():
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
                q_emb = encode_questions(question_encoder, q_batch, last_ips)
                q_tok = (
                    encode_question_tokens(question_encoder, q_batch)
                    if model.use_token_sequence
                    else None
                )
                _seq, ctx = model.metric_encoder(seq_batch)
                pred = model.predict_answer_latent(ctx, q_emb, q_tok)
                logits = model.codebook.logits_from_latent(pred)
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
        finally:
            if was_training:
                model.train()

    for _epoch in range(max(1, OFFLINE_EPOCHS)):
        epoch_t0 = time.time()
        for ctx, qa_pairs, is_val, ips, _cores, _ts in iter_dump_by_snapshot(
            LOAD_REPLAY_PATH
        ):
            if ips:
                last_ips = ips
            target = val_replay if is_val else replay
            ctx_id = target.add_context(ctx)
            for q_struct, ans_idx in qa_pairs:
                target.add_qa(ctx_id, q_struct, ans_idx)
            if len(replay) < MIN_REPLAY_SIZE:
                continue

            seq_batch, q_batch, ans_batch = replay.sample(BATCH_SIZE)
            seq_batch = seq_batch.to(DEVICE)
            ans_batch = ans_batch.to(DEVICE)
            q_emb = encode_questions(question_encoder, q_batch, last_ips)
            q_emb = _perturb_question_embedding(q_emb, model.training)
            q_tok = (
                encode_question_tokens(question_encoder, q_batch)
                if model.use_token_sequence
                else None
            )

            _seq, online_ctx = model.metric_encoder(seq_batch)

            if LAP_EPS > 0.0 and model.training:
                clean_q = q_emb.detach()
                for _ in range(max(1, LAP_INNER_STEPS)):
                    lap_optimizer.zero_grad()
                    delta = lap_perturber(clean_q, LAP_EPS)
                    adv_q = clean_q + delta
                    _adv_latent = model.predict_answer_latent(
                        online_ctx.detach(), adv_q, q_tok
                    )
                    _adv_logits = model.codebook.logits_from_latent(_adv_latent)
                    adv_loss = soft_cross_entropy(_adv_logits, ans_batch, soft_targets)
                    grads = torch.autograd.grad(
                        adv_loss, list(lap_perturber.parameters())
                    )
                    for p, g in zip(lap_perturber.parameters(), grads):
                        p.grad = -g
                    lap_optimizer.step()

                with torch.no_grad():
                    delta = lap_perturber(q_emb.detach(), LAP_EPS)
                q_emb = q_emb + delta

            if ADV_EPS > 0.0 and model.training and LAP_EPS <= 0.0:
                q_base = q_emb.detach().clone().requires_grad_(True)
                for _ in range(max(1, ADV_STEPS)):
                    q_p = q_base
                    _adv_latent = model.predict_answer_latent(online_ctx, q_p, q_tok)
                    _adv_logits = model.codebook.logits_from_latent(_adv_latent)
                    adv_loss = soft_cross_entropy(_adv_logits, ans_batch, soft_targets)
                    (grad,) = torch.autograd.grad(adv_loss, q_p)
                    step = (
                        grad.sign()
                        * q_p.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                        * ADV_EPS
                    )
                    q_base.data.add_(step)

                    delta = q_base - q_emb.detach()
                    max_norm = (
                        q_emb.detach().norm(dim=-1, keepdim=True).clamp_min(1e-4)
                        * ADV_EPS
                    )
                    scale = (delta.norm(dim=-1, keepdim=True) / max_norm).clamp_min(1.0)
                    q_base.data = q_emb.detach() + delta / scale
                q_emb = q_emb + (q_base - q_emb.detach())

            pred_answer_latent = model.predict_answer_latent(online_ctx, q_emb, q_tok)
            logits = model.codebook.logits_from_latent(pred_answer_latent)
            qa_loss = soft_cross_entropy(logits, ans_batch, soft_targets)

            if JEPA_SSL_WEIGHT > 0.0 and seq_batch.size(1) > JEPA_PRED_HORIZON:
                horizon = max(1, JEPA_PRED_HORIZON)
                short_seq = seq_batch[:, :-horizon, :]
                _s, online_short_ctx = model.metric_encoder(short_seq)
                with torch.no_grad():
                    ema_full, _ = ema_encoder(seq_batch)
                    targets = F.normalize(ema_full[:, -horizon:, :], dim=-1)
                predicted = F.normalize(jepa_predictor(online_short_ctx), dim=-1)
                ssl_loss = (1.0 - (predicted.unsqueeze(1) * targets).sum(dim=-1)).mean()
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
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            ema_encoder.update(model.metric_encoder)

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

            loss_window.append(total_loss.item())
            acc_window.append(top1)
            global_step += 1

            val_top1 = val_top5 = float("nan")
            if VAL_INTERVAL > 0 and global_step % VAL_INTERVAL == 0:
                val_top1, val_top5 = _eval_val()

            csv_writer.writerow(
                [
                    time.time(),
                    global_step,
                    qa_loss.item(),
                    float(ssl_loss.item()),
                    total_loss.item(),
                    top1,
                    top5,
                    val_top1,
                    val_top5,
                    len(replay),
                    len(val_replay),
                ]
            )
            log_file.flush()

            if global_step % 100 == 0:
                logger.info(
                    "step=%d  loss=%.4f (qa=%.4f ssl=%.4f)  top1=%.2f%% top5=%.2f%%  "
                    "avg_loss=%.4f avg_acc=%.2f%%  val_top1=%.2f%% val_top5=%.2f%%  buf=%d  %.0fs",
                    global_step,
                    total_loss.item(),
                    qa_loss.item(),
                    float(ssl_loss.item()),
                    top1 * 100,
                    top5 * 100,
                    sum(loss_window) / max(1, len(loss_window)),
                    sum(acc_window) / max(1, len(acc_window)) * 100,
                    val_top1 * 100 if val_top1 == val_top1 else float("nan"),
                    val_top5 * 100 if val_top5 == val_top5 else float("nan"),
                    len(replay),
                    time.time() - t0,
                )

            if CHECKPOINT_INTERVAL > 0 and global_step % CHECKPOINT_INTERVAL == 0:
                _save_ckpt(
                    global_step,
                    last_ips,
                    CHECKPOINT_PATH,
                    model,
                    ema_encoder,
                    jepa_predictor,
                    optimizer,
                    lap_perturber,
                    lap_optimizer,
                )
        if EPOCH_CKPT > 0:
            _save_ckpt(
                global_step,
                last_ips,
                f"{CHECKPOINT_PATH}.epoch{_epoch + 1}.step{global_step}.pt",
                model,
                ema_encoder,
                jepa_predictor,
                optimizer,
                lap_perturber,
                lap_optimizer,
            )
        logger.info(
            "Epoch %d/%d complete in %.0fs",
            _epoch + 1,
            OFFLINE_EPOCHS,
            time.time() - epoch_t0,
        )
    logger.info(
        "Offline training done at step %d (%d epochs)", global_step, OFFLINE_EPOCHS
    )


if __name__ == "__main__":
    main()
