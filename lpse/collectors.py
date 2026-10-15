import logging
import os
import time
from typing import Any

import requests
from kubernetes import client, config

from .metrics_shared import (
    compute_capacity_features,
    cpu_util_per_core,
    fetch_node_exporter_metrics,
    filter_node_exporter_text,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("collectors")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

KUBECONFIG_PATH = os.getenv("KUBECONFIG_PATH", "./config")


_v1: client.CoreV1Api | None = None


def _get_api() -> client.CoreV1Api:
    global _v1
    if _v1 is None:
        config.load_kube_config(config_file=KUBECONFIG_PATH)
        _v1 = client.CoreV1Api()
    return _v1


def get_node_ips(api: client.CoreV1Api | None = None) -> list[str]:
    api = api or _get_api()
    ips: list[str] = []
    nodes = api.list_node().items
    for node in nodes:
        for addr in node.status.addresses:
            if addr.type == "InternalIP":
                ips.append(addr.address)
    return ips


_CORE_CACHE_TTL_S = float(os.getenv("CORE_COUNT_CACHE_TTL_S", "300"))
_core_cache: dict[str, tuple[float, int]] = {}


def get_num_cores_from_node_exporter(
    ip: str, port: int = 9100, session: requests.Session | None = None
) -> int:
    now = time.time()
    cached = _core_cache.get(ip)
    if cached is not None and now - cached[0] < _CORE_CACHE_TTL_S:
        return cached[1]
    text = fetch_node_exporter_metrics(ip, port=port, session=session)
    if text is None:
        return 0
    num_cores = 0
    for line in text.splitlines():
        if line.startswith("node_cpu_seconds_total") and 'mode="idle"' in line:
            num_cores += 1
    if num_cores > 0:
        _core_cache[ip] = (now, num_cores)
    return num_cores


def cpu_util_from_text(
    ip: str, text: str, expected_cores: int | None = None
) -> list[float]:
    """Per-core CPU utilization vector, padded/truncated to expected_cores.

    The per-core deltas (and their shared counter state) come from
    metrics_shared.cpu_util_per_core so this stays consistent with
    prometheus_ingest's mean utilization.
    """
    cpu_util = cpu_util_per_core(ip, text)

    if expected_cores is not None:
        if not cpu_util:
            return [0.0] * expected_cores
        if len(cpu_util) < expected_cores:
            cpu_util.extend([0.0] * (expected_cores - len(cpu_util)))
        elif len(cpu_util) > expected_cores:
            cpu_util = cpu_util[:expected_cores]

    return cpu_util


def query_node_exporter_cpu_util(
    ip: str,
    port: int = 9100,
    expected_cores: int | None = None,
    session: requests.Session | None = None,
    raw_metrics: str | None = None,
):
    raw = raw_metrics
    if raw is None:
        raw = fetch_node_exporter_metrics(ip, port=port, session=session)
    if raw is None:
        logger.warning("Failed to query %s node_exporter", ip)
        return [0.0] * (expected_cores if expected_cores is not None else 10)

    filtered = filter_node_exporter_text(raw)
    return cpu_util_from_text(ip, filtered, expected_cores=expected_cores)


def collect_node_exporter_snapshots(
    node_ips: list[str],
    node_cores: list[int],
    session: requests.Session | None = None,
):
    per_node_cpu: dict[str, list[float]] = {}
    per_node_filtered: dict[str, str] = {}
    per_node_features: dict[str, dict[str, Any]] = {}

    for ip, cores in zip(node_ips, node_cores):
        raw = fetch_node_exporter_metrics(ip, session=session)
        if raw is None:
            per_node_cpu[ip] = [0.0] * int(cores)
            per_node_filtered[ip] = ""
            per_node_features[ip] = {
                "mem_available_fraction": None,
                "mem_used_fraction": None,
                "net_bytes_per_s": None,
                "net_link_bps": None,
                "net_utilization": None,
                "disk_bytes_per_s": None,
                "disk_utilization": None,
            }
            continue

        filtered = filter_node_exporter_text(raw)
        per_node_filtered[ip] = filtered
        per_node_cpu[ip] = cpu_util_from_text(ip, filtered, expected_cores=cores)
        per_node_features[ip] = compute_capacity_features(ip, filtered)

    return per_node_cpu, per_node_filtered, per_node_features
