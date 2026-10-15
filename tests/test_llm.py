import json
import logging
import math
import os
import socket
import sys
import time
from typing import Any
from urllib.parse import urlparse

import requests

from lpse.collectors import (
    collect_node_exporter_snapshots,
    get_node_ips,
    get_num_cores_from_node_exporter,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("node_utilization")


USE_LLM = os.getenv("USE_LLM", "0") == "1"


SAMPLE_INTERVAL_S = float(os.getenv("SAMPLE_INTERVAL_S", "1"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))

AFTER_PRINT_SLEEP_S = float(os.getenv("AFTER_PRINT_SLEEP_S", "0"))


CAPABILITY_BUCKETS = [100, 90, 75, 50, 25, 10, 2]
NET_SAMPLE_INTERVAL_S = float(os.getenv("NET_SAMPLE_INTERVAL_S", "1"))


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://10.68.184.17:8000")
VLLM_CHAT_URL = os.getenv("VLLM_CHAT_URL", f"{VLLM_BASE_URL}/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "openai/gpt-oss-120b")
VLLM_TIMEOUT_S = float(os.getenv("VLLM_TIMEOUT_S", "60"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
RETURN_FULL_RESPONSE_JSON = os.getenv("RETURN_FULL_RESPONSE_JSON", "0") == "1"


def _vllm_endpoint_reachable() -> bool:
    u = urlparse(VLLM_CHAT_URL)
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=1).close()
        return True
    except OSError:
        return False


if "pytest" in sys.modules and not _vllm_endpoint_reachable():
    import pytest

    pytest.skip(f"vLLM endpoint unreachable: {VLLM_CHAT_URL}", allow_module_level=True)


def _bucket_percent(fraction: float) -> int:
    """Map a fraction [0,1] to the nearest conservative bucket (high-to-low)."""
    if not math.isfinite(fraction):
        return 2
    pct = max(0.0, min(1.0, fraction)) * 100.0
    for b in CAPABILITY_BUCKETS:
        if pct >= b:
            return b
    return 2


def _format_bucket(b: int) -> str:
    return "1-2%" if b <= 2 else f"{b}%"


def _parse_bucket_str(s: Any) -> int | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        v = int(s)
        return v
    if not isinstance(s, str):
        return None
    t = s.strip()
    if t == "1-2%":
        return 2
    if t.endswith("%"):
        t = t[:-1].strip()
    try:
        return int(float(t))
    except Exception:
        return None


def _compute_baseline_utilization(
    node_ips: list[str],
    per_node_cpu: dict[str, list[float]],
    per_node_features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    baseline: dict[str, dict[str, str]] = {}
    for ip in node_ips:
        cpu_vec = per_node_cpu.get(ip) or []
        cpu_util = (sum(cpu_vec) / len(cpu_vec)) if cpu_vec else 0.0
        cpu_used = max(0.0, min(1.0, float(cpu_util)))

        mem_used = per_node_features.get(ip, {}).get("mem_used_fraction")
        if mem_used is None:
            mem_avail = per_node_features.get(ip, {}).get("mem_available_fraction")
            mem_used = (
                None
                if mem_avail is None
                else max(0.0, min(1.0, 1.0 - float(mem_avail)))
            )

        net_util = per_node_features.get(ip, {}).get("net_utilization")
        net_used = None if net_util is None else max(0.0, min(1.0, float(net_util)))

        cpu_b = _bucket_percent(cpu_used)
        ram_b = _bucket_percent(float(mem_used)) if mem_used is not None else 2

        if net_used is None:
            net_Bps = per_node_features.get(ip, {}).get("net_bytes_per_s")
            if net_Bps is None:
                net_b = 2
            else:
                mbps = (float(net_Bps) * 8.0) / 1_000_000.0
                if mbps < 1:
                    net_b = 2
                elif mbps < 10:
                    net_b = 10
                elif mbps < 50:
                    net_b = 25
                elif mbps < 100:
                    net_b = 50
                elif mbps < 500:
                    net_b = 75
                elif mbps < 1000:
                    net_b = 90
                else:
                    net_b = 100
        else:
            net_b = _bucket_percent(net_used)

        baseline[ip] = {
            "CPU": _format_bucket(cpu_b),
            "RAM": _format_bucket(ram_b),
            "network bandwidth": _format_bucket(net_b),
        }
    return baseline


def call_vllm_capacity(
    node_exporter_filtered: dict[str, str], summary_features: dict[str, dict[str, Any]]
):
    """Ask the LLM to bucket per-node current utilization.

    Returns JSON mapping:
      ip -> {"CPU": "...%", "RAM": "...%", "network bandwidth": "...%"}
    """
    body = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only JSON. No thinking. No explanations. "
                    "Output must be a single JSON object mapping each node IP to an object with keys: "
                    "CPU, RAM, network bandwidth. Values must be one of: 100%, 90%, 75%, 50%, 25%, 10%, 1-2%."
                ),
            },
            {
                "role": "user",
                "content": (
                    "For each node below, estimate its current utilization based on the node-exporter metrics. "
                    "Ignore comments/system info; metrics are already filtered. "
                    "Provide utilization buckets for each resource:\n"
                    "CPU utilization: 100%, 90%, 75%, 50%, 25%, 10%, 1-2%\n"
                    "RAM utilization: 100%, 90%, 75%, 50%, 25%, 10%, 1-2%\n"
                    "network bandwidth utilization: 100%, 90%, 75%, 50%, 25%, 10%, 1-2%\n\n"
                    "You also get computed summary features (may be null): cpu_utilization_fraction, mem_used_fraction, mem_available_fraction, net_bytes_per_s, net_link_bps, net_device, net_max_bytes_per_s, net_utilization. "
                    "Use these plus the filtered metrics as evidence.\n"
                    "If baseline_buckets are provided, you should usually output the same buckets unless the raw metrics strongly contradict them.\n\n"
                    "summary_features:\n"
                    + json.dumps(summary_features, separators=(",", ":"))
                    + "\n\n"
                    "filtered_node_exporter_metrics:\n"
                    + json.dumps(node_exporter_filtered, separators=(",", ":"))
                ),
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    r = requests.post(VLLM_CHAT_URL, json=body, timeout=VLLM_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()

    logger.debug("Full vLLM response: %s", json.dumps(data, indent=2)[:1000])

    if RETURN_FULL_RESPONSE_JSON:
        return json.dumps(data, indent=2)

    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning") or ""

    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "")
    if finish_reason == "length":
        logger.warning(
            "Response truncated due to max_tokens limit. Increase MAX_TOKENS."
        )

    logger.debug(
        "Raw content from vLLM (len=%d, finish=%s): %r",
        len(content),
        finish_reason,
        content[:500] if content else content,
    )

    if not content:
        logger.warning("Empty content from vLLM")
        return "{}"

    content = content.strip()

    while "<think>" in content:
        think_start = content.find("<think>")
        think_end = content.find("</think>", think_start)
        if think_end == -1:
            content = content[:think_start]
            break
        content = content[:think_start] + content[think_end + 8 :]

    content = content.strip()

    json_start = content.find("{")
    json_end = content.rfind("}")
    if json_start >= 0 and json_end >= 0:
        json_str = content[json_start : json_end + 1]

        try:
            parsed = json.loads(json_str)

            return json.dumps(parsed, separators=(",", ":"))
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from vLLM: %s", e)
            return "{}"

    logger.warning("No JSON found in vLLM response")
    logger.debug("Content was: %s", content[:500])
    return "{}"


def main():
    node_ips = get_node_ips()
    if not node_ips:
        raise RuntimeError("No Kubernetes nodes found via API")

    node_cores = []
    for ip in node_ips:
        cores = get_num_cores_from_node_exporter(ip)
        if cores <= 0:
            logger.warning("Could not detect core count for %s; falling back to 10", ip)
            cores = 10
        node_cores.append(int(cores))

    logger.info("Nodes: %s (total cores: %d)", node_ips, sum(node_cores))
    logger.info("vLLM: %s model=%s", VLLM_CHAT_URL, VLLM_MODEL)

    output_file = os.getenv("OUTPUT_FILE", "llm_utilization.txt")
    input_file = os.getenv("INPUT_FILE", "node_exporter_filtered.txt")

    while True:
        per_node_cpu, per_node_filtered, per_node_features = (
            collect_node_exporter_snapshots(node_ips, node_cores)
        )
        observed_at = time.time()

        input_json = json.dumps(per_node_filtered, separators=(",", ":"))
        with open(input_file, "a") as f:
            f.write(input_json)
            f.write("\n")

        baseline = _compute_baseline_utilization(
            node_ips, per_node_cpu, per_node_features
        )

        if not USE_LLM:
            output_obj: dict[str, dict[str, str]] = baseline
        else:
            cpu_util_frac: dict[str, float | None] = {}
            for ip in node_ips:
                cpu_vec = per_node_cpu.get(ip) or []
                cpu_util = (sum(cpu_vec) / len(cpu_vec)) if cpu_vec else None
                cpu_util_frac[ip] = (
                    None if cpu_util is None else float(max(0.0, min(1.0, cpu_util)))
                )

            summary_for_llm = {
                ip: {
                    **per_node_features.get(ip, {}),
                    "cpu_utilization_fraction": cpu_util_frac.get(ip),
                    "baseline_buckets": baseline.get(ip),
                    "observed_at": observed_at,
                }
                for ip in node_ips
            }
            try:
                llm_json = call_vllm_capacity(per_node_filtered, summary_for_llm)
                parsed = json.loads(llm_json)
            except Exception as e:
                logger.warning("vLLM call/parse failed; using baseline: %s", e)
                parsed = None

            merged: dict[str, dict[str, str]] = {}
            if isinstance(parsed, dict):
                for ip in node_ips:
                    node_obj = parsed.get(ip)
                    if not isinstance(node_obj, dict):
                        merged[ip] = baseline[ip]
                        continue

                    out: dict[str, str] = {}
                    for k in ("CPU", "RAM", "network bandwidth"):
                        b_str = baseline[ip].get(k)
                        l_str = node_obj.get(k)
                        b_val = _parse_bucket_str(b_str)
                        l_val = _parse_bucket_str(l_str)
                        if b_val is not None and l_val is not None:
                            out[k] = _format_bucket(int(max(b_val, l_val)))
                        elif b_val is not None:
                            out[k] = _format_bucket(int(b_val))
                        elif l_val is not None:
                            out[k] = _format_bucket(int(l_val))
                        else:
                            out[k] = str(b_str or "1-2%")

                    merged[ip] = out
                output_obj = merged
            else:
                output_obj = baseline

        output = json.dumps(output_obj, separators=(",", ":"))

        sys.stdout.write(output)
        sys.stdout.write("\n")
        sys.stdout.flush()

        with open(output_file, "a") as f:
            f.write(output)
            f.write("\n")

        time.sleep(AFTER_PRINT_SLEEP_S if AFTER_PRINT_SLEEP_S > 0 else 0.0)


if __name__ == "__main__":
    main()
