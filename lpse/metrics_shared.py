import logging
import math
import os
import re
import signal
import threading
import time
from typing import Any

import requests

logger = logging.getLogger("metrics_shared")


NET_SAMPLE_INTERVAL_S = float(os.getenv("NET_SAMPLE_INTERVAL_S", "1"))
DISK_SAMPLE_INTERVAL_S = float(os.getenv("DISK_SAMPLE_INTERVAL_S", "1"))


SCRAPE_TIMEOUT_S = float(os.getenv("SCRAPE_TIMEOUT_S", "20"))

SCRAPE_RETRIES = int(os.getenv("SCRAPE_RETRIES", "3"))


_NODE_SCRAPE_OVERRIDES: dict[str, str] = {}
_env_override = os.getenv("NODE_SCRAPE_URL_MAP", "").strip()
if _env_override:
    try:
        import json as _json

        _NODE_SCRAPE_OVERRIDES = {
            k: v
            for k, v in _json.loads(_env_override).items()
            if isinstance(v, str) and v
        }
        if _NODE_SCRAPE_OVERRIDES:
            logger.info(
                "node_exporter scrape overrides active for %d nodes",
                len(_NODE_SCRAPE_OVERRIDES),
            )
    except Exception as e:
        logger.warning("Ignoring malformed NODE_SCRAPE_URL_MAP: %s", e)


_MEM_AVAIL_RE = re.compile(r"^node_memory_MemAvailable_bytes\s+([0-9eE\+\-\.]+)\s*$")
_MEM_TOTAL_RE = re.compile(r"^node_memory_MemTotal_bytes\s+([0-9eE\+\-\.]+)\s*$")

_NET_RX_RE = re.compile(
    r"^node_network_receive_bytes_total\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$"
)
_NET_TX_RE = re.compile(
    r"^node_network_transmit_bytes_total\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$"
)
_NET_SPEED_RE = re.compile(
    r"^node_network_speed_bytes\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$"
)

_DISK_READ_RE = re.compile(
    r"^node_disk_read_bytes_total\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$"
)
_DISK_WRITE_RE = re.compile(
    r"^node_disk_written_bytes_total\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$"
)

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


_prev_net_counters: dict[str, dict[tuple[str, str], float]] = {}
_prev_net_ts: dict[str, float] = {}
_max_net_Bps: dict[str, dict[str, float]] = {}

_prev_disk_counters: dict[str, dict[tuple[str, str], float]] = {}
_prev_disk_ts: dict[str, float] = {}
_max_disk_Bps: dict[str, dict[str, float]] = {}


_IGNORE_NET_IF_PREFIXES = ("lo", "docker", "veth", "cni", "flannel", "kube", "br-")
_IGNORE_DISK_PREFIXES = ("loop", "dm-", "sr", "ram", "md", "rbd")


def _parse_labels(label_blob: str) -> dict:
    return {k: v for k, v in _LABEL_RE.findall(label_blob)}


_prev_cpu_counters: dict[str, dict[tuple[str, str], float]] = {}
_CPU_LINE_RE = re.compile(r"^node_cpu_seconds_total\{([^}]*)\}\s+([0-9eE\+\-\.]+)\s*$")


def cpu_util_per_core(ip: str, text: str) -> list[float]:
    """Compute per-core CPU utilization fractions from node_cpu_seconds_total
    deltas for `ip`.

    Returns a list of per-core utilization in [0, 1] (cores sorted by index),
    containing only cores with a positive delta in this window. Returns an
    empty list until a prior sample exists for the node.
    """
    last = _prev_cpu_counters.get(ip, {})
    cur: dict[tuple[str, str], float] = {}
    idle_delta: dict[str, float] = {}
    total_delta: dict[str, float] = {}

    for line in text.splitlines():
        m = _CPU_LINE_RE.match(line)
        if not m:
            continue
        labels_blob, value_s = m.group(1), m.group(2)
        labels = _parse_labels(labels_blob)
        cpu = labels.get("cpu")
        mode = labels.get("mode")
        if cpu is None or mode is None:
            continue
        try:
            value = float(value_s)
        except ValueError:
            continue
        cur[(cpu, mode)] = value
        prev = last.get((cpu, mode))
        if prev is None:
            continue
        d = value - prev
        if d < 0:
            continue
        total_delta[cpu] = total_delta.get(cpu, 0.0) + d
        if mode == "idle":
            idle_delta[cpu] = idle_delta.get(cpu, 0.0) + d

    _prev_cpu_counters[ip] = cur

    cpus = sorted(total_delta.keys(), key=lambda x: int(x) if x.isdigit() else x)
    utils: list[float] = []
    for cpu in cpus:
        td = total_delta.get(cpu, 0.0)
        if td <= 0:
            continue
        idl = idle_delta.get(cpu, 0.0)
        utils.append(float(max(0.0, min(1.0, 1.0 - (idl / td)))))
    return utils


def filter_node_exporter_text(raw: str) -> str:
    """Keep only relevant metric families to reduce downstream parsing cost."""
    keep_prefixes = (
        "node_cpu_seconds_total",
        "node_memory_MemAvailable_bytes",
        "node_memory_MemTotal_bytes",
        "node_network_receive_bytes_total",
        "node_network_transmit_bytes_total",
        "node_network_speed_bytes",
        "node_disk_read_bytes_total",
        "node_disk_written_bytes_total",
    )
    out_lines = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(keep_prefixes):
            out_lines.append(line)
    return "\n".join(out_lines)


def fetch_node_exporter_metrics(
    ip: str, port: int = 9100, session: requests.Session | None = None
) -> str | None:
    base = _NODE_SCRAPE_OVERRIDES.get(ip) or f"http://{ip}:{port}"
    url = base.rstrip("/") + "/metrics"
    sess = session or requests

    for attempt in range(SCRAPE_RETRIES):
        try:
            deadline = SCRAPE_TIMEOUT_S
            old_handler = None
            if threading is not None and _is_main_thread():
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, deadline)
            try:
                r = sess.get(url, timeout=deadline)
                r.raise_for_status()
                return r.text
            finally:
                if old_handler is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
        except Exception as e:
            if attempt == SCRAPE_RETRIES - 1:
                logger.warning(
                    "Failed to fetch node_exporter metrics from %s: %s", ip, e
                )
                return None
            time.sleep(0.5)
    return None


def _timeout_handler(signum, frame):
    raise TimeoutError("node_exporter scrape exceeded wall-clock deadline")


def _is_main_thread() -> bool:
    try:
        return threading.current_thread() is threading.main_thread()
    except Exception:
        return True


def _net_dev_ok(dev: str) -> bool:
    return not any(dev.startswith(p) for p in _IGNORE_NET_IF_PREFIXES)


def _is_ethernet_dev(dev: str) -> bool:
    return dev.startswith(("eth", "en", "eno", "ens", "enp"))


def _disk_dev_ok(dev: str) -> bool:
    return not any(dev.startswith(p) for p in _IGNORE_DISK_PREFIXES)


def compute_capacity_features(
    ip: str,
    raw_metrics: str,
    net_sample_interval_s: float = NET_SAMPLE_INTERVAL_S,
    disk_sample_interval_s: float = DISK_SAMPLE_INTERVAL_S,
) -> dict[str, Any]:
    """Compute memory, network, and disk capacity features from node-exporter text."""
    now = time.time()

    mem_avail = None
    mem_total = None
    for line in raw_metrics.splitlines():
        m = _MEM_AVAIL_RE.match(line)
        if m:
            mem_avail = float(m.group(1))
            continue
        m = _MEM_TOTAL_RE.match(line)
        if m:
            mem_total = float(m.group(1))
            continue

    mem_avail_frac = None
    mem_used_frac = None
    if mem_avail is not None and mem_total and mem_total > 0:
        mem_avail_frac = max(0.0, min(1.0, mem_avail / mem_total))
        mem_used_frac = max(0.0, min(1.0, 1.0 - mem_avail_frac))

    last_net = _prev_net_counters.get(ip, {})
    cur_net: dict[tuple[str, str], float] = {}
    rx_delta_by_dev: dict[str, float] = {}
    tx_delta_by_dev: dict[str, float] = {}
    speed_bps_by_dev: dict[str, float] = {}

    for line in raw_metrics.splitlines():
        m = _NET_RX_RE.match(line)
        if m:
            labels = _parse_labels(m.group(1))
            dev = labels.get("device")
            if not dev or not _net_dev_ok(dev):
                continue
            val = float(m.group(2))
            cur_net[(dev, "rx")] = val
            prev = last_net.get((dev, "rx"))
            if prev is not None:
                d = val - prev
                if d >= 0:
                    rx_delta_by_dev[dev] = rx_delta_by_dev.get(dev, 0.0) + d
            continue

        m = _NET_TX_RE.match(line)

        if m:
            labels = _parse_labels(m.group(1))
            dev = labels.get("device")
            if not dev or not _net_dev_ok(dev):
                continue
            val = float(m.group(2))
            cur_net[(dev, "tx")] = val
            prev = last_net.get((dev, "tx"))
            if prev is not None:
                d = val - prev
                if d >= 0:
                    tx_delta_by_dev[dev] = tx_delta_by_dev.get(dev, 0.0) + d
            continue

        m = _NET_SPEED_RE.match(line)
        if m:
            labels = _parse_labels(m.group(1))
            dev = labels.get("device")
            if not dev or not _net_dev_ok(dev):
                continue
            speed_bps_by_dev[dev] = float(m.group(2)) * 8.0

    _prev_net_counters[ip] = cur_net

    prev_net_ts = _prev_net_ts.get(ip)
    _prev_net_ts[ip] = now
    dt_net = (now - prev_net_ts) if prev_net_ts is not None else None
    if dt_net is None or not math.isfinite(dt_net) or dt_net <= 0:
        dt_net = net_sample_interval_s

    net_Bps_by_dev: dict[str, float] = {}
    for dev in set(list(rx_delta_by_dev.keys()) + list(tx_delta_by_dev.keys())):
        rx_Bps = rx_delta_by_dev.get(dev, 0.0) / max(0.001, float(dt_net))
        tx_Bps = tx_delta_by_dev.get(dev, 0.0) / max(0.001, float(dt_net))
        net_Bps_by_dev[dev] = rx_Bps + tx_Bps

    eth_devs = [d for d in net_Bps_by_dev.keys() if _is_ethernet_dev(d)]
    candidate_devs = eth_devs or list(net_Bps_by_dev.keys())
    chosen_dev = None
    if candidate_devs:
        chosen_dev = max(
            candidate_devs,
            key=lambda d: (speed_bps_by_dev.get(d, 0.0), net_Bps_by_dev.get(d, 0.0)),
        )

    net_Bps = net_Bps_by_dev.get(chosen_dev, 0.0) if chosen_dev else 0.0
    max_combined_bps = None
    if chosen_dev and speed_bps_by_dev.get(chosen_dev):
        max_combined_bps = float(speed_bps_by_dev[chosen_dev]) * 2.0

    max_by_dev = _max_net_Bps.setdefault(ip, {})
    if chosen_dev:
        prev_max = max_by_dev.get(chosen_dev, 0.0)
        if math.isfinite(net_Bps) and net_Bps > prev_max:
            max_by_dev[chosen_dev] = float(net_Bps)

    net_max_Bps = None
    if max_combined_bps is not None and max_combined_bps > 0:
        net_max_Bps = max_combined_bps / 8.0
    elif chosen_dev:
        net_max_Bps = max(0.0, float(max_by_dev.get(chosen_dev, 0.0)))

    net_util = None
    if net_max_Bps is not None and net_max_Bps > 0:
        net_util = min(1.0, float(net_Bps) / float(net_max_Bps))

    last_disk = _prev_disk_counters.get(ip, {})
    cur_disk: dict[tuple[str, str], float] = {}
    read_delta_by_dev: dict[str, float] = {}
    write_delta_by_dev: dict[str, float] = {}

    for line in raw_metrics.splitlines():
        m = _DISK_READ_RE.match(line)
        if m:
            labels = _parse_labels(m.group(1))
            dev = labels.get("device")
            if not dev or not _disk_dev_ok(dev):
                continue
            val = float(m.group(2))
            cur_disk[(dev, "read")] = val
            prev = last_disk.get((dev, "read"))
            if prev is not None:
                d = val - prev
                if d >= 0:
                    read_delta_by_dev[dev] = read_delta_by_dev.get(dev, 0.0) + d
            continue

        m = _DISK_WRITE_RE.match(line)
        if m:
            labels = _parse_labels(m.group(1))
            dev = labels.get("device")
            if not dev or not _disk_dev_ok(dev):
                continue
            val = float(m.group(2))
            cur_disk[(dev, "write")] = val
            prev = last_disk.get((dev, "write"))
            if prev is not None:
                d = val - prev
                if d >= 0:
                    write_delta_by_dev[dev] = write_delta_by_dev.get(dev, 0.0) + d
            continue

    _prev_disk_counters[ip] = cur_disk

    prev_disk_ts = _prev_disk_ts.get(ip)
    _prev_disk_ts[ip] = now
    dt_disk = (now - prev_disk_ts) if prev_disk_ts is not None else None
    if dt_disk is None or not math.isfinite(dt_disk) or dt_disk <= 0:
        dt_disk = disk_sample_interval_s

    disk_Bps_by_dev: dict[str, float] = {}
    for dev in set(list(read_delta_by_dev.keys()) + list(write_delta_by_dev.keys())):
        rd_Bps = read_delta_by_dev.get(dev, 0.0) / max(0.001, float(dt_disk))
        wr_Bps = write_delta_by_dev.get(dev, 0.0) / max(0.001, float(dt_disk))
        disk_Bps_by_dev[dev] = rd_Bps + wr_Bps

    disk_candidates = list(disk_Bps_by_dev.keys())
    chosen_disk = (
        max(disk_candidates, key=lambda d: disk_Bps_by_dev.get(d, 0.0))
        if disk_candidates
        else None
    )
    disk_Bps = disk_Bps_by_dev.get(chosen_disk, 0.0) if chosen_disk else 0.0

    max_disk_by_dev = _max_disk_Bps.setdefault(ip, {})
    if chosen_disk:
        prev_max = max_disk_by_dev.get(chosen_disk, 0.0)
        if math.isfinite(disk_Bps) and disk_Bps > prev_max:
            max_disk_by_dev[chosen_disk] = float(disk_Bps)

    disk_max_Bps = None
    if chosen_disk:
        disk_max_Bps = max(0.0, float(max_disk_by_dev.get(chosen_disk, 0.0)))

    disk_util = None
    if disk_max_Bps is not None and disk_max_Bps > 0:
        disk_util = min(1.0, float(disk_Bps) / float(disk_max_Bps))

    return {
        "mem_available_fraction": mem_avail_frac,
        "mem_used_fraction": mem_used_frac,
        "net_bytes_per_s": net_Bps,
        "net_link_bps": (speed_bps_by_dev.get(chosen_dev) if chosen_dev else None),
        "net_device": chosen_dev,
        "net_max_bytes_per_s": net_max_Bps,
        "net_utilization": net_util,
        "disk_bytes_per_s": disk_Bps,
        "disk_max_bytes_per_s": disk_max_Bps,
        "disk_utilization": disk_util,
    }
