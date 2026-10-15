"""LPSE: temporal metric encoder, fusion transformer, and answer-codebook scoring."""

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .question_encoder import QUESTION_DIM

logger = logging.getLogger("lpse")


LATENT_DIM = 1024
FUSION_DIM = 1024
ANSWER_DIM = 512
HIDDEN_DIM = 4096
METRIC_MODEL_DIM = 1024
FUSION_HEADS = 16
FUSION_LAYERS = 4


class SinusoidalPE(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


class MetricEncoder(nn.Module):
    """
    Consumes a sequence of metric snapshots [B, T, F_in] and produces
    a latent representation [B, T, LATENT_DIM] (full sequence) plus a
    context summary [B, LATENT_DIM] (last-token pooling).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = LATENT_DIM,
        model_dim: int = METRIC_MODEL_DIM,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = SinusoidalPE(model_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(model_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, F_in]
        Returns:
            seq_latent: [B, T, latent_dim]  -- full temporal latent
            ctx:        [B, latent_dim]      -- context summary (last timestep)
        """
        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h)

        seq_latent = self.to_latent(h)
        ctx = seq_latent[:, -1, :]
        return seq_latent, ctx


class FusionTransformer(nn.Module):
    """
    Cross-attends question embedding over the metric temporal sequence.

    Input:
        metric_seq: [B, T, LATENT_DIM]   — from MetricEncoder
        question:   [B, QUESTION_DIM]     — from QuestionEncoder

    Output:
        fused: [B, FUSION_DIM]
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        question_dim: int = QUESTION_DIM,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = FUSION_HEADS,
        num_layers: int = FUSION_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.q_proj = nn.Linear(question_dim, latent_dim)

        self.cross_attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ff_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim * 4, latent_dim),
                    nn.Dropout(dropout),
                )
            )
            self.norms1.append(nn.LayerNorm(latent_dim))
            self.norms2.append(nn.LayerNorm(latent_dim))

        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        metric_seq: torch.Tensor,
        question: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q_orig = self.q_proj(question).unsqueeze(1)
        q = q_orig

        kv = metric_seq
        if question_tokens is not None:
            kv = torch.cat([metric_seq, question_tokens], dim=1)

        for ca, ff, n1, n2 in zip(
            self.cross_attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            attn_out, _ = ca(query=q, key=kv, value=kv)
            q = n1(q + attn_out)
            q = n2(q + ff(q))

        q = q + q_orig

        return self.out_proj(q.squeeze(1))


class AnswerCodebook(nn.Module):
    """
    A learnable set of answer embeddings. Each code corresponds to a
    template answer string.

    During training, the model predicts an answer latent from the JEPA state
    latent plus question, then scores that latent against all answer codes.

    During inference, we do nearest-neighbour lookup.
    """

    def __init__(
        self,
        num_answers: int,
        answer_dim: int = ANSWER_DIM,
        fusion_dim: int = FUSION_DIM,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_answers = num_answers
        self.answer_dim = answer_dim
        self.temperature = max(1e-4, float(temperature))

        self.embeddings = nn.Embedding(num_answers, answer_dim)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        self.proj = nn.Linear(fusion_dim, answer_dim)

    def project(self, fused: torch.Tensor) -> torch.Tensor:
        """Project fused state-question features into normalized answer space."""
        return F.normalize(self.proj(fused), dim=-1)

    def logits_from_latent(self, answer_latent: torch.Tensor) -> torch.Tensor:
        """Score predicted answer latents against the full answer codebook."""
        q_emb = F.normalize(answer_latent, dim=-1)
        a_emb = F.normalize(self.embeddings.weight, dim=-1)
        logits = torch.matmul(q_emb, a_emb.T)
        return logits / self.temperature

    @torch.no_grad()
    def max_cosine(self, answer_latent: torch.Tensor) -> torch.Tensor:
        """Max UNSCALED cosine similarity between predicted latents and codes.

        Unlike `logits_from_latent`, this does not divide by temperature, so
        the returned value is a true cosine in [-1, 1] and can be compared
        against an absolute OOD threshold.
        """
        q_emb = F.normalize(answer_latent, dim=-1)
        a_emb = F.normalize(self.embeddings.weight, dim=-1)
        return torch.matmul(q_emb, a_emb.T).max(dim=-1).values

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        fused: [B, FUSION_DIM]
        Returns: [B, num_answers] — similarity scores (logits)
        """
        return self.logits_from_latent(self.project(fused))

    def decode(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (best_indices, similarity_scores) for greedy decoding.
        """
        logits = self.forward(fused)
        scores, indices = logits.max(dim=-1)
        return indices, scores


class LPSE(nn.Module):
    """
    Full model: Metric Encoder + question-conditioned Answer-JEPA + Codebook.

    Training:
        state = encoder(metrics).ctx
        answer_latent = answer_jepa(state, q_emb)
        loss = cross_entropy(sim(answer_latent, answer_codebook), answer_idx)

    Inference:
        answer_idx, score = model.infer(metric_seq, question_text)
        answer_text = answer_templates[answer_idx]
    """

    def __init__(
        self,
        input_dim: int,
        num_answers: int,
        question_encoder: nn.Module,
        latent_dim: int = LATENT_DIM,
        fusion_dim: int = FUSION_DIM,
        answer_dim: int = ANSWER_DIM,
        encoder_heads: int = 8,
        encoder_layers: int = 4,
        fusion_heads: int = FUSION_HEADS,
        fusion_layers: int = FUSION_LAYERS,
        dropout: float = 0.1,
        codebook_temperature: float = 1.0,
        use_token_sequence: bool = False,
    ):
        super().__init__()
        self.use_token_sequence = bool(use_token_sequence)

        self.metric_encoder = MetricEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            model_dim=METRIC_MODEL_DIM,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        self.fusion = FusionTransformer(
            latent_dim=latent_dim,
            question_dim=QUESTION_DIM,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            dropout=dropout,
        )

        self.codebook = AnswerCodebook(
            num_answers=num_answers,
            answer_dim=answer_dim,
            fusion_dim=fusion_dim,
            temperature=codebook_temperature,
        )

        self.question_encoder = question_encoder
        if self.use_token_sequence:
            self.token_proj = nn.Linear(384, latent_dim)

    def _tokens(self, question_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_token_sequence or question_tokens is None:
            return None
        return self.token_proj(question_tokens)

    def forward(
        self,
        metric_seq: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns answer logits [B, num_answers]."""
        _seq_latent, state_latent = self.metric_encoder(metric_seq)
        return self.answer_logits_from_state(
            state_latent, question_emb, question_tokens
        )

    def predict_answer_latent(
        self,
        state_latent: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict an answer latent from a JEPA/network-state latent and question.

        state_latent can be [B, LATENT_DIM] or [B, T, LATENT_DIM]. The QA path
        intentionally consumes this state representation, not raw metrics.
        """
        if state_latent.dim() == 2:
            state_seq = state_latent.unsqueeze(1)
        elif state_latent.dim() == 3:
            state_seq = state_latent
        else:
            raise ValueError(
                f"Expected state_latent with rank 2 or 3, got {state_latent.dim()}"
            )
        fused = self.fusion(state_seq, question_emb, self._tokens(question_tokens))
        return self.codebook.project(fused)

    def answer_logits_from_state(
        self,
        state_latent: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return answer-codebook logits from a JEPA/network-state latent."""
        answer_latent = self.predict_answer_latent(
            state_latent, question_emb, question_tokens
        )
        return self.codebook.logits_from_latent(answer_latent)

    @torch.no_grad()
    def infer(
        self,
        metric_seq: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> tuple[int, float, torch.Tensor]:
        """
        Single-sample inference.

        Returns:
            answer_idx: int
            confidence: float (cosine similarity)
            all_logits: [num_answers]
        """
        self.eval()
        logits = self.forward(metric_seq, question_emb, question_tokens)
        scores = logits[0]
        idx = int(scores.argmax().item())
        conf = float(scores[idx].item())
        return idx, conf, scores

    @torch.no_grad()
    def infer_topk(
        self,
        metric_seq: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return top-k (answer_idx, score) pairs."""
        self.eval()
        logits = self.forward(metric_seq, question_emb, question_tokens)[0]
        topk = torch.topk(logits, min(k, logits.size(0)))
        return list(zip(topk.indices.tolist(), topk.values.tolist()))

    def get_ood_score(
        self,
        metric_seq: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> float:
        """
        OOD detection: return the max UNSCALED cosine similarity between the
        predicted answer latent and any answer code.

        Low score (< ~0.3) means the question+state doesn't match any learned
        answer — the model "doesn't know". This deliberately ignores the
        codebook temperature: the scaled logits in `infer` are inflated by
        1/tau (tau ~ 0.07), so a threshold meant for raw cosine must be
        evaluated on the raw cosine, not on the scaled logit.
        """
        return self.ood_cosine(metric_seq, question_emb, question_tokens)

    @torch.no_grad()
    def ood_cosine(
        self,
        metric_seq: torch.Tensor,
        question_emb: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
    ) -> float:
        """Max unscaled cosine similarity to any answer code (not temperature-scaled)."""
        self.eval()
        state_latent = self.metric_encoder(metric_seq)[1]
        answer_latent = self.predict_answer_latent(
            state_latent, question_emb, question_tokens
        )
        return float(self.codebook.max_cosine(answer_latent))


class MomentumMetricEncoder(nn.Module):
    """
    EMA copy of MetricEncoder used as prediction target (JEPA objective).
    The student encoder predicts the EMA encoder's latent for future timesteps.
    """

    def __init__(self, encoder: MetricEncoder, momentum: float = 0.996):
        super().__init__()

        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, online_encoder: MetricEncoder):
        for p_ema, p_online in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            p_ema.data.mul_(self.momentum).add_(p_online.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """
    Predicts the latent of a future timestep given the online encoder's
    summary of the past. Trained against the EMA target encoder.
    """

    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim or latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.GELU(),
            nn.Linear(h, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
