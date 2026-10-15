"""Scrape /metrics, parse, summarise per family, pack into a fixed per-node feature tensor."""

import logging
import math
import os
import re

import torch

from .metrics_shared import cpu_util_per_core, fetch_node_exporter_metrics

logger = logging.getLogger("prometheus_ingest")


_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{([^}]*)\})?"
    r"\s+([0-9eE\+\-\.]+(?:NaN|Inf)?)"
    r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_labels(label_blob: str) -> dict[str, str]:
    return {k: v for k, v in _LABEL_RE.findall(label_blob)}


def _compute_cpu_utilization_from_text(ip: str, text: str) -> float | None:
    """Estimate mean CPU utilization fraction from node_cpu_seconds_total
    deltas. Returns None until at least one prior sample exists.

    Per-core deltas and their shared counter state live in
    metrics_shared.cpu_util_per_core so every consumer sees the same values.
    """
    utils = cpu_util_per_core(ip, text)
    return (sum(utils) / len(utils)) if utils else None


TRACKED_FAMILIES = [
    "node_cpu_seconds_total",
    "node_memory_MemTotal_bytes",
    "node_memory_MemAvailable_bytes",
    "node_memory_MemFree_bytes",
    "node_memory_Buffers_bytes",
    "node_memory_Cached_bytes",
    "node_memory_SwapTotal_bytes",
    "node_memory_SwapFree_bytes",
    "node_network_receive_bytes_total",
    "node_network_transmit_bytes_total",
    "node_network_receive_packets_total",
    "node_network_transmit_packets_total",
    "node_network_receive_errs_total",
    "node_network_transmit_errs_total",
    "node_network_receive_drop_total",
    "node_network_transmit_drop_total",
    "node_network_speed_bytes",
    "node_disk_read_bytes_total",
    "node_disk_written_bytes_total",
    "node_disk_reads_completed_total",
    "node_disk_writes_completed_total",
    "node_disk_io_time_seconds_total",
    "node_disk_read_time_seconds_total",
    "node_disk_write_time_seconds_total",
    "node_filesystem_size_bytes",
    "node_filesystem_avail_bytes",
    "node_filesystem_free_bytes",
    "node_filesystem_files",
    "node_filesystem_files_free",
    "node_load1",
    "node_load5",
    "node_load15",
    "node_context_switches_total",
    "node_forks_total",
    "node_procs_running",
    "node_procs_blocked",
    "node_entropy_available_bits",
    "node_pressure_cpu_waiting_seconds_total",
    "node_pressure_memory_waiting_seconds_total",
    "node_pressure_io_waiting_seconds_total",
    "node_sockstat_TCP_alloc",
    "node_sockstat_TCP_inuse",
    "node_sockstat_sockets_used",
    "node_netstat_Tcp_CurrEstab",
    "node_netstat_Tcp_ActiveOpens",
    "node_netstat_Tcp_PassiveOpens",
    "node_netstat_Tcp_RetransSegs",
    "node_time_seconds",
    "node_boot_time_seconds",
    "node_hwmon_temp_celsius",
]

STATS_PER_FAMILY = 6
FEATURES_PER_NODE = len(TRACKED_FAMILIES) * STATS_PER_FAMILY
MAX_NODE_SLOTS = int(os.getenv("MAX_NODE_SLOTS", "16"))
NODE_ID_FEATURES = 4
NODE_PRESENCE_FEATURES = 1
NODE_SLOT_FEATURES = FEATURES_PER_NODE + NODE_ID_FEATURES + NODE_PRESENCE_FEATURES
CLUSTER_TOKEN_DIM = MAX_NODE_SLOTS * NODE_SLOT_FEATURES + 1
_FAMILY_INDEX = {name: i for i, name in enumerate(TRACKED_FAMILIES)}


def _cluster_identity_indices() -> list[int]:
    indices: list[int] = []
    for slot_idx in range(MAX_NODE_SLOTS):
        base = slot_idx * NODE_SLOT_FEATURES + FEATURES_PER_NODE
        indices.extend(range(base, base + NODE_ID_FEATURES + NODE_PRESENCE_FEATURES))
    indices.append(CLUSTER_TOKEN_DIM - 1)
    return indices


CLUSTER_IDENTITY_INDICES = _cluster_identity_indices()


def parse_prometheus_text(text: str) -> dict[str, list[float]]:
    """Parse raw Prometheus exposition text into {family_name: [values]}."""
    families: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        val_str = m.group(3)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        families.setdefault(name, []).append(val)
    return families


def _summarise(values: list[float]) -> list[float]:
    """Return [count, min, max, mean, sum, std] for a list of values."""
    if not values:
        return [0.0] * STATS_PER_FAMILY
    n = float(len(values))
    s = sum(values)
    mean = s / n
    mn = min(values)
    mx = max(values)
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
    else:
        std = 0.0
    return [n, mn, mx, mean, s, std]


def node_metrics_to_tensor(raw_text: str) -> torch.Tensor:
    """
    Convert raw node_exporter text → 1-D feature tensor [FEATURES_PER_NODE].

    Each tracked family gets 6 summary statistics.  Families not present in
    the scrape get zeros.
    """
    families = parse_prometheus_text(raw_text)
    feats: list[float] = []
    for family_name in TRACKED_FAMILIES:
        vals = families.get(family_name, [])
        feats.extend(_summarise(vals))
    return torch.tensor(feats, dtype=torch.float32)


def collect_full_cluster_snapshot(
    node_ips: list[str],
) -> tuple[torch.Tensor, dict[str, dict[str, list[float]]]]:
    """
    Scrape every node, return:
      - stacked tensor [N, FEATURES_PER_NODE]
      - raw parsed families per node (for QA answer generation)
    """
    tensors = []
    raw_parsed: dict[str, dict[str, list[float]]] = {}

    for ip in node_ips:
        text = fetch_node_exporter_metrics(ip)
        if text is None:
            tensors.append(torch.zeros(FEATURES_PER_NODE))
            raw_parsed[ip] = {}
        else:
            tensors.append(node_metrics_to_tensor(text))
            raw_parsed[ip] = parse_prometheus_text(text)
            cpu_util = _compute_cpu_utilization_from_text(ip, text)
            if cpu_util is not None:
                raw_parsed[ip]["__cpu_util__"] = [cpu_util]

    return torch.stack(tensors, dim=0).cpu(), raw_parsed


def _ip_to_vec(ip: str) -> torch.Tensor:
    parts = ip.split(".")
    values: list[float] = []
    for part in parts[:4]:
        try:
            value = int(part)
        except ValueError:
            value = 0
        values.append(max(0.0, min(1.0, float(value) / 255.0)))
    while len(values) < 4:
        values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def cluster_snapshot_to_token(
    node_tensors: torch.Tensor,
    node_ips: list[str] | None = None,
    max_node_slots: int = MAX_NODE_SLOTS,
) -> torch.Tensor:
    """
    Convert a variable-size node token set [N, F_node] into a fixed cluster token.

    Unlike pure mean/std pooling, this keeps node identity: nodes are sorted by
    IP, placed into stable slots, and each occupied slot stores node features,
    a normalized IP vector, and a presence bit. Empty slots are all zeros.
    """
    slot_features = FEATURES_PER_NODE + NODE_ID_FEATURES + NODE_PRESENCE_FEATURES
    if node_tensors is None or node_tensors.numel() == 0:
        return torch.zeros(max_node_slots * slot_features + 1, dtype=torch.float32)

    x = node_tensors.float().detach().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)

    if x.size(1) != FEATURES_PER_NODE:
        raise ValueError(
            f"Expected node feature dim {FEATURES_PER_NODE}, got {x.size(1)}"
        )

    if node_ips is not None and len(node_ips) >= x.size(0):
        paired = list(zip(node_ips[: x.size(0)], x))
        paired.sort(key=lambda item: item[0])
    else:
        paired = [(str(i), x[i]) for i in range(x.size(0))]

    slots = torch.zeros(max_node_slots, slot_features, dtype=torch.float32)
    for slot_idx, (ip, feats) in enumerate(paired[:max_node_slots]):
        slots[slot_idx, :FEATURES_PER_NODE] = feats
        slots[slot_idx, FEATURES_PER_NODE : FEATURES_PER_NODE + NODE_ID_FEATURES] = (
            _ip_to_vec(ip)
        )
        slots[slot_idx, FEATURES_PER_NODE + NODE_ID_FEATURES] = 1.0

    active_frac = torch.tensor(
        [float(min(len(paired), max_node_slots)) / float(max(1, max_node_slots))],
        dtype=torch.float32,
    )

    return torch.cat([slots.reshape(-1), active_frac], dim=0)


def flatten_cluster_tensor(node_tensors: torch.Tensor) -> torch.Tensor:
    """Flatten [N, F_node] → [N * F_node] for the temporal encoder."""
    return node_tensors.reshape(-1)


class RunningNormalizer:
    """
    Online normaliser — keeps running mean/variance per feature so the encoder
    always sees roughly standard-normal inputs.
    """

    def __init__(self, dim: int, momentum: float = 0.01):
        self.dim = dim
        self.momentum = momentum
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = 0

    def update(self, x: torch.Tensor):
        """x: [D] or [B, D] — always computed on CPU."""
        x = x.detach().cpu()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected normalizer dim {self.dim}, got {x.size(-1)}")

        batch_mean = x.mean(dim=0)
        batch_var = (
            x.var(dim=0, unbiased=False)
            if x.size(0) > 1
            else torch.zeros_like(batch_mean)
        )

        if self.count == 0:
            self.mean = batch_mean.clone()

            self.var = torch.where(
                batch_var > 1e-6,
                batch_var,
                torch.ones_like(batch_mean),
            ).clone()
        else:
            old_mean = self.mean.cpu()
            old_var = self.var.cpu()
            delta = batch_mean - old_mean

            self.mean = old_mean + self.momentum * delta

            self.var = (
                (1.0 - self.momentum) * (old_var + self.momentum * delta.pow(2))
                + self.momentum * batch_var
            ).clamp(min=1e-6)

        self.count += int(x.size(0))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x. Always operates on CPU, returns on x's original device."""
        orig_device = x.device
        x_cpu = x.detach().cpu()
        m = self.mean.cpu()
        s = self.var.cpu().clamp(min=1e-6).sqrt()
        result = (x_cpu - m) / s
        result = result.to(orig_device)

        if result.shape[-1] == CLUSTER_TOKEN_DIM:
            idx = torch.tensor(
                CLUSTER_IDENTITY_INDICES, dtype=torch.long, device=orig_device
            )
            raw = x_cpu.to(orig_device)
            result[..., idx] = raw[..., idx]

        return result

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        mean = d["mean"].cpu()
        var = d["var"].cpu().clamp(min=1e-6)
        count = int(d.get("count", 0))

        if mean.numel() != self.dim or var.numel() != self.dim:
            raise ValueError(
                f"Normalizer state dim mismatch: expected {self.dim}, "
                f"got mean={mean.numel()} var={var.numel()}"
            )

        if count > 1 and float(d["var"].cpu().max()) <= 1e-7:
            logger.warning(
                "Ignoring degenerate normalizer variance from checkpoint; "
                "normalizer will burn in again."
            )
            self.mean = mean
            self.var = torch.ones_like(mean)
            self.count = 0
            return

        self.mean = mean
        self.var = var
        self.count = count


def compute_semantic_features(
    raw_parsed: dict[str, dict[str, list[float]]],
    node_ips: list[str],
) -> dict[str, dict[str, float | None]]:
    """
    For each node, compute human-interpretable features that the QA generator
    can turn into ground-truth answers.

    Returns {ip: {feature_name: value_or_None}}.
    """
    result: dict[str, dict[str, float | None]] = {}

    for ip in node_ips:
        fam = raw_parsed.get(ip, {})
        d: dict[str, float | None] = {}

        cpu_vals = fam.get("node_cpu_seconds_total", [])
        d["cpu_total_count"] = float(len(cpu_vals)) if cpu_vals else None
        cpu_util = fam.get("__cpu_util__", [None])
        d["cpu_utilization_fraction"] = cpu_util[0] if cpu_util else None

        mem_total = fam.get("node_memory_MemTotal_bytes", [None])
        mem_avail = fam.get("node_memory_MemAvailable_bytes", [None])
        mt = mem_total[0] if mem_total else None
        ma = mem_avail[0] if mem_avail else None
        if mt and ma and mt > 0:
            d["mem_used_fraction"] = max(0.0, min(1.0, 1.0 - ma / mt))
            d["mem_total_bytes"] = mt
            d["mem_available_bytes"] = ma
        else:
            d["mem_used_fraction"] = None
            d["mem_total_bytes"] = None
            d["mem_available_bytes"] = None

        for la in ("node_load1", "node_load5", "node_load15"):
            vals = fam.get(la, [None])
            d[la] = vals[0] if vals else None

        disk_read = fam.get("node_disk_read_bytes_total", [])
        disk_write = fam.get("node_disk_written_bytes_total", [])
        d["disk_read_bytes_sum"] = sum(disk_read) if disk_read else None
        d["disk_write_bytes_sum"] = sum(disk_write) if disk_write else None

        net_rx = fam.get("node_network_receive_bytes_total", [])
        net_tx = fam.get("node_network_transmit_bytes_total", [])
        d["net_rx_bytes_sum"] = sum(net_rx) if net_rx else None
        d["net_tx_bytes_sum"] = sum(net_tx) if net_tx else None

        fs_size = fam.get("node_filesystem_size_bytes", [])
        fs_avail = fam.get("node_filesystem_avail_bytes", [])
        if fs_size and fs_avail:
            total_sz = sum(v for v in fs_size if v > 0)
            total_av = sum(v for v in fs_avail if v > 0)
            d["fs_used_fraction"] = (
                max(0.0, min(1.0, 1.0 - total_av / total_sz)) if total_sz > 0 else None
            )
        else:
            d["fs_used_fraction"] = None

        tcp_estab = fam.get("node_netstat_Tcp_CurrEstab", [None])
        d["tcp_established"] = tcp_estab[0] if tcp_estab else None

        tcp_retrans = fam.get("node_netstat_Tcp_RetransSegs", [None])
        d["tcp_retransmits"] = tcp_retrans[0] if tcp_retrans else None

        procs_run = fam.get("node_procs_running", [None])
        procs_blk = fam.get("node_procs_blocked", [None])
        d["procs_running"] = procs_run[0] if procs_run else None
        d["procs_blocked"] = procs_blk[0] if procs_blk else None

        ctx = fam.get("node_context_switches_total", [None])
        d["context_switches_total"] = ctx[0] if ctx else None

        temps = fam.get("node_hwmon_temp_celsius", [])
        d["max_temp_celsius"] = max(temps) if temps else None

        result[ip] = d

    return result
