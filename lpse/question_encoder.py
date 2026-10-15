"""Question -> fixed-size embedding. Backends: keyword, sentence_transformer, hybrid."""

import logging
import re
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("question_encoder")

QUESTION_DIM = 256


INTENTS = [
    "query_value",
    "compare",
    "find_extreme",
    "predict",
    "is_condition",
    "summarize",
    "count",
    "list",
    "trend",
    "threshold",
]
INTENT_INDEX = {name: i for i, name in enumerate(INTENTS)}


METRICS = [
    "cpu",
    "memory",
    "network",
    "disk",
    "filesystem",
    "load",
    "temperature",
    "process",
    "tcp",
    "overall",
    "health",
]
METRIC_INDEX = {name: i for i, name in enumerate(METRICS)}


SCOPES = ["single_node", "all_nodes", "cluster"]
SCOPE_INDEX = {name: i for i, name in enumerate(SCOPES)}


HORIZONS = ["current", "recent", "trend", "prediction"]
HORIZON_INDEX = {name: i for i, name in enumerate(HORIZONS)}


_INTENT_PATTERNS = [
    (r"\b(predict|forecast|will|future|next)\b", "predict"),
    (
        r"\b(trend|trending|increasing|decreasing|going up|going down|changing)\b",
        "trend",
    ),
    (r"\b(compare|comparison|versus|vs|differ)\b", "compare"),
    (
        r"\b(highest|lowest|most|least|maximum|minimum|max|min|top|bottom|worst|best)\b",
        "find_extreme",
    ),
    (
        r"\b(is|are|does|do|has|have|any|check)\b.*\b(bound|overloaded|idle|healthy|unhealthy|critical|saturated|bottleneck)\b",
        "is_condition",
    ),
    (r"\b(how many|count|number of)\b", "count"),
    (r"\b(list|show|display|enumerate)\b", "list"),
    (r"\b(summary|summarize|overview|status|report)\b", "summarize"),
    (r"\b(alert|warn|threshold|exceed|above|below|over|under)\b", "threshold"),
    (r"\b(what|how much|how|tell|get|query|value)\b", "query_value"),
]

_METRIC_PATTERNS = [
    (r"\b(cpu|processor|core|cores|compute|utilization|utilisation)\b", "cpu"),
    (r"\b(mem|memory|ram|swap|oom)\b", "memory"),
    (
        r"\b(net|network|bandwidth|throughput|rx|tx|packet|latency|tcp|socket|connection)\b",
        "network",
    ),
    (r"\b(disk|io|iops|storage|read|write|ssd|hdd|nvme)\b", "disk"),
    (r"\b(filesystem|fs|mount|partition|volume|space|capacity)\b", "filesystem"),
    (r"\b(load|load average|load1|load5|load15|runqueue)\b", "load"),
    (r"\b(temp|temperature|thermal|heat|celsius|fahrenheit)\b", "temperature"),
    (r"\b(process|proc|pid|fork|context switch|blocked|running)\b", "process"),
    (r"\b(tcp|established|retransmit|connection|socket)\b", "tcp"),
    (r"\b(health|healthy|status|ok|fine|good)\b", "health"),
    (r"\b(overall|everything|all|general|cluster|total)\b", "overall"),
]

_SCOPE_PATTERNS = [
    (r"\b(node\s+\d+\.\d+\.\d+\.\d+|node\s*#?\s*\d+|specific node)\b", "single_node"),
    (r"\b(all nodes|every node|each node|across nodes)\b", "all_nodes"),
    (r"\b(cluster|total|aggregate|overall)\b", "cluster"),
]

_HORIZON_PATTERNS = [
    (r"\b(predict|forecast|will|next|future|upcoming)\b", "prediction"),
    (r"\b(trend|trending|over time|history|historical|past)\b", "trend"),
    (r"\b(recent|last|previous|ago)\b", "recent"),
]

_IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")
_NODE_NUM_RE = re.compile(r"\bnode\s*#?\s*(\d+)\b", re.IGNORECASE)


def parse_question(text: str, node_ips: list[str] | None = None) -> dict:
    """
    Parse a natural language question into a structured representation.

    Returns:
        {
            "intent": str,
            "metric": str,
            "scope": str,
            "horizon": str,
            "node_ref": Optional[str],   # IP address or node index
            "raw": str,
        }
    """
    lower = text.lower().strip()

    intent = "query_value"
    for pattern, label in _INTENT_PATTERNS:
        if re.search(pattern, lower):
            intent = label
            break

    metric = "overall"
    for pattern, label in _METRIC_PATTERNS:
        if re.search(pattern, lower):
            metric = label
            break

    scope = "cluster"
    for pattern, label in _SCOPE_PATTERNS:
        if re.search(pattern, lower):
            scope = label
            break

    node_ref = None
    ip_match = _IP_RE.search(text)
    if ip_match:
        node_ref = ip_match.group(1)
        scope = "single_node"
    else:
        num_match = _NODE_NUM_RE.search(text)
        if num_match and node_ips:
            idx = int(num_match.group(1))
            if 0 <= idx < len(node_ips):
                node_ref = node_ips[idx]
                scope = "single_node"

    horizon = "current"
    for pattern, label in _HORIZON_PATTERNS:
        if re.search(pattern, lower):
            horizon = label
            break

    return {
        "intent": intent,
        "metric": metric,
        "scope": scope,
        "horizon": horizon,
        "node_ref": node_ref,
        "raw": text,
    }


class KeywordQuestionEncoder(nn.Module):
    """
    Encodes a parsed question struct into a dense embedding.

    This is deterministic + learnable: the categorical features are embedded
    and passed through a small MLP.  No tokeniser, no pretrained LM.
    """

    def __init__(self, out_dim: int = QUESTION_DIM, max_nodes: int = 64):
        super().__init__()
        self.intent_emb = nn.Embedding(len(INTENTS), 32)
        self.metric_emb = nn.Embedding(len(METRICS), 32)
        self.scope_emb = nn.Embedding(len(SCOPES), 16)
        self.horizon_emb = nn.Embedding(len(HORIZONS), 16)
        self.node_ip_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.no_node_emb = nn.Parameter(torch.zeros(16))

        concat_dim = 32 + 32 + 16 + 16 + 16

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def _encode_ip(self, ip: str) -> torch.Tensor | None:
        parts = ip.split(".")
        if len(parts) != 4:
            return None
        octets = []
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                return None
            if value < 0 or value > 255:
                return None
            octets.append(float(value) / 255.0)
        device = next(self.parameters()).device
        return torch.tensor(octets, dtype=torch.float32, device=device)

    def _encode_struct(
        self, q: dict, node_ips: list[str] | None = None
    ) -> torch.Tensor:
        """Build raw feature vector from parsed question."""
        intent_idx = INTENT_INDEX.get(q["intent"], 0)
        metric_idx = METRIC_INDEX.get(q["metric"], 0)
        scope_idx = SCOPE_INDEX.get(q["scope"], 0)
        horizon_idx = HORIZON_INDEX.get(q["horizon"], 0)

        device = next(self.parameters()).device
        ie = self.intent_emb(torch.tensor(intent_idx, device=device))
        me = self.metric_emb(torch.tensor(metric_idx, device=device))
        se = self.scope_emb(torch.tensor(scope_idx, device=device))
        he = self.horizon_emb(torch.tensor(horizon_idx, device=device))

        node_ref = q.get("node_ref")
        if node_ref:
            ip_vec = self._encode_ip(node_ref)
            if ip_vec is not None:
                ne = self.node_ip_proj(ip_vec)
            else:
                ne = self.no_node_emb.to(device)
        else:
            ne = self.no_node_emb.to(device)

        return torch.cat([ie, me, se, he, ne], dim=-1)

    def forward(
        self,
        questions: list[dict],
        node_ips: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Encode a batch of parsed question dicts.

        Returns: [B, QUESTION_DIM]
        """
        feats = torch.stack(
            [self._encode_struct(q, node_ips) for q in questions],
            dim=0,
        )
        return self.mlp(feats)

    def encode_text(
        self,
        text: str,
        node_ips: list[str] | None = None,
    ) -> torch.Tensor:
        """Convenience: parse + encode one question string → [1, QUESTION_DIM]."""
        q = parse_question(text, node_ips)
        return self.forward([q], node_ips)


class SentenceTransformerQuestionEncoder(nn.Module):
    """
    Uses a pretrained sentence-transformers model to encode questions,
    then projects to QUESTION_DIM with a learned linear layer.

    Requires: `pip install sentence-transformers`
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        out_dim: int = QUESTION_DIM,
        freeze_backbone: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_device = device

        self.st_model = SentenceTransformer(model_name, device=self.st_device)
        st_dim = self.st_model.get_embedding_dimension()
        if st_dim is None:
            raise RuntimeError(
                f"SentenceTransformer model '{model_name}' returned no embedding dimension"
            )

        if freeze_backbone:
            for p in self.st_model.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(st_dim, out_dim)
        logger.info(
            "SentenceTransformerQuestionEncoder initialized model=%s device=%s freeze_backbone=%s",
            model_name,
            self.st_device,
            freeze_backbone,
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of question strings → [B, QUESTION_DIM]."""
        with torch.no_grad():
            embs = self.st_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.st_device,
            )

        device = next(self.proj.parameters()).device
        embs = embs.detach().to(device).clone()
        return self.proj(embs)


class HybridQuestionEncoder(nn.Module):
    """
    Combines SentenceTransformer (paraphrase-robust intent semantics) with
    explicit IP-octet and scope features (precise node grounding).

    The text path captures *what* the user is asking about — this is where
    "is the box roasting" maps near "is CPU high".

    The structured path captures *which specific node* via four normalised
    IP octets, plus a small scope embedding (single_node / all_nodes /
    cluster). These are signals that get buried in BERT subword tokens
    when ST encodes raw text, so we restore them as explicit features.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        out_dim: int = QUESTION_DIM,
        freeze_backbone: bool = True,
        device: str | None = None,
        use_structured: bool = True,
        text_noise_std: float = 0.0,
        text_dropout: float = 0.0,
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_device = device
        self.use_structured = bool(use_structured)
        self.text_noise_std = float(text_noise_std)
        self.text_dropout = float(text_dropout)

        self.st_model = SentenceTransformer(model_name, device=self.st_device)
        st_dim = self.st_model.get_embedding_dimension()
        if st_dim is None:
            raise RuntimeError(
                f"SentenceTransformer model '{model_name}' returned no embedding dimension"
            )

        if freeze_backbone:
            for p in self.st_model.parameters():
                p.requires_grad = False

        self.node_ip_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.no_node_emb = nn.Parameter(torch.zeros(16))
        self.scope_emb = nn.Embedding(len(SCOPES), 16)

        if self.use_structured:
            self.intent_emb = nn.Embedding(len(INTENTS), 32)
            self.metric_emb = nn.Embedding(len(METRICS), 32)
            self.horizon_emb = nn.Embedding(len(HORIZONS), 16)
            concat_dim = st_dim + 32 + 32 + 16 + 16 + 16
        else:
            concat_dim = st_dim + 16 + 16
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
        )

        logger.info(
            "HybridQuestionEncoder initialized model=%s st_dim=%d use_structured=%s → %d",
            model_name,
            st_dim,
            self.use_structured,
            out_dim,
        )

    def _encode_ip(self, ip: str, device: torch.device) -> torch.Tensor | None:
        parts = ip.split(".")
        if len(parts) != 4:
            return None
        octets: list[float] = []
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                return None
            if value < 0 or value > 255:
                return None
            octets.append(float(value) / 255.0)
        return torch.tensor(octets, dtype=torch.float32, device=device)

    def forward(
        self,
        questions: list[dict],
        node_ips: list[str] | None = None,
    ) -> torch.Tensor:

        texts = [q.get("raw", "") if isinstance(q, dict) else str(q) for q in questions]
        with torch.no_grad():
            st_embs = self.st_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.st_device,
            )
        proj_device = next(self.proj.parameters()).device
        st_embs = st_embs.detach().to(proj_device).clone()

        if self.training:
            if self.text_noise_std > 0.0:
                st_norms = st_embs.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                st_embs = st_embs + torch.randn_like(st_embs) * (
                    st_norms * self.text_noise_std
                )
            if self.text_dropout > 0.0:
                mask = torch.bernoulli(
                    torch.full_like(st_embs, 1.0 - self.text_dropout)
                )
                st_embs = st_embs * mask / (1.0 - self.text_dropout)

        ip_feats: list[torch.Tensor] = []
        scope_idxs: list[int] = []
        intent_idxs: list[int] = []
        metric_idxs: list[int] = []
        horizon_idxs: list[int] = []
        for q in questions:
            if not isinstance(q, dict):
                ip_feats.append(self.no_node_emb.to(proj_device))
                scope_idxs.append(SCOPE_INDEX["cluster"])
                intent_idxs.append(INTENT_INDEX["query_value"])
                metric_idxs.append(METRIC_INDEX["overall"])
                horizon_idxs.append(HORIZON_INDEX["current"])
                continue
            node_ref = q.get("node_ref")
            if node_ref:
                ip_vec = self._encode_ip(node_ref, proj_device)
                if ip_vec is not None:
                    ip_feats.append(self.node_ip_proj(ip_vec))
                else:
                    ip_feats.append(self.no_node_emb.to(proj_device))
            else:
                ip_feats.append(self.no_node_emb.to(proj_device))
            scope = q.get("scope", "cluster")
            scope_idxs.append(SCOPE_INDEX.get(scope, SCOPE_INDEX["cluster"]))
            intent_idxs.append(
                INTENT_INDEX.get(
                    str(q.get("intent") or ""), INTENT_INDEX["query_value"]
                )
            )
            metric_idxs.append(
                METRIC_INDEX.get(str(q.get("metric") or ""), METRIC_INDEX["overall"])
            )
            horizon_idxs.append(
                HORIZON_INDEX.get(str(q.get("horizon") or ""), HORIZON_INDEX["current"])
            )

        def idx_t(vals):
            return torch.tensor(vals, dtype=torch.long, device=proj_device)

        ip_tensor = torch.stack(ip_feats, dim=0)
        scope_tensor = self.scope_emb(idx_t(scope_idxs))
        if self.use_structured:
            intent_tensor = self.intent_emb(idx_t(intent_idxs))
            metric_tensor = self.metric_emb(idx_t(metric_idxs))
            horizon_tensor = self.horizon_emb(idx_t(horizon_idxs))
            combined = torch.cat(
                [
                    st_embs,
                    intent_tensor,
                    metric_tensor,
                    horizon_tensor,
                    scope_tensor,
                    ip_tensor,
                ],
                dim=-1,
            )
        else:
            combined = torch.cat([st_embs, scope_tensor, ip_tensor], dim=-1)
        return self.proj(combined)

    def token_states(self, texts: list[str]) -> torch.Tensor:
        """Per-token MiniLM hidden states (frozen) -> [B, n_tokens, 384].

        Gives the fusion token-level access to the question, so paraphrases
        sharing words/concepts with their source can be attended to at the
        token level instead of relying solely on the pooled embedding.
        """
        with torch.no_grad():
            transformer: Any = self.st_model[0]
            tokenized = transformer.preprocess([str(t) for t in texts])
            dev = next(transformer.parameters()).device
            tokenized = {
                k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                for k, v in tokenized.items()
            }
            out = transformer(tokenized)
            if isinstance(out, dict):
                states = out["token_embeddings"]
            else:
                states = out.last_hidden_state
            states = states.detach()
        proj_device = next(self.proj.parameters()).device
        return states.to(proj_device)


def build_question_encoder(
    backend: str = "sentence_transformer",
    out_dim: int = QUESTION_DIM,
    **kwargs,
) -> nn.Module:
    """
    Build a question encoder.

    backend: "sentence_transformer" (default), "hybrid", or "keyword"
    """
    if backend == "keyword":
        return KeywordQuestionEncoder(out_dim=out_dim, **kwargs)
    elif backend == "sentence_transformer":
        return SentenceTransformerQuestionEncoder(out_dim=out_dim, **kwargs)
    elif backend == "hybrid":
        return HybridQuestionEncoder(out_dim=out_dim, **kwargs)
    else:
        raise ValueError(f"Unknown question encoder backend: {backend}")


def encode_questions(
    encoder: nn.Module,
    questions,
    node_ips: list[str] | None = None,
) -> torch.Tensor:
    """
    Encode questions with any backend.

    - Keyword backend expects parsed dicts.
    - SentenceTransformer backend expects raw strings (we pull `raw` from dicts).
    - Hybrid backend expects parsed dicts (uses both `raw` and `node_ref`/`scope`).
    """
    if isinstance(encoder, KeywordQuestionEncoder):
        return encoder(questions, node_ips)
    if isinstance(encoder, HybridQuestionEncoder):
        return encoder(questions, node_ips)
    if isinstance(encoder, SentenceTransformerQuestionEncoder):
        texts = []
        for q in questions:
            if isinstance(q, dict):
                texts.append(q.get("raw", ""))
            else:
                texts.append(str(q))
        return encoder(texts)

    try:
        return encoder(questions, node_ips)
    except TypeError:
        return encoder(questions)


def encode_question_tokens(
    encoder: nn.Module,
    questions: list[dict],
) -> torch.Tensor:
    """Per-token MiniLM hidden states for a batch of questions -> [B, n, 384].

    Only the hybrid encoder exposes token states; other backends raise a
    TypeError, so callers should gate this on the encoder type / model flag.
    """
    texts = [q.get("raw", "") if isinstance(q, dict) else str(q) for q in questions]
    return getattr(encoder, "token_states")(texts)
