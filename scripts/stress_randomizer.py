#!/usr/bin/env python3
"""Cycle random stress-ng profiles across worker nodes via Kubernetes Jobs to diversify training signal.

Usage:
  python scripts/stress_randomizer.py            # run continuously
  python scripts/stress_randomizer.py --once     # one profile, then exit
  python scripts/stress_randomizer.py --cleanup  # delete all stress jobs
"""

import argparse
import logging
import os
import random
import sys
import time

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("stress_randomizer")

NAMESPACE = "stress2"
STRESS_IMAGE = "ghcr.io/colinianking/stress-ng:latest"


STRESS_PROFILES = [
    {
        "name": "idle",
        "description": "No stress — everything idle",
        "cpu_load": 0,
        "vm_bytes": "0",
        "vm_workers": 0,
        "io_workers": 0,
        "duration_s": None,
        "weight": 1.5,
    },
    {
        "name": "cpu_low",
        "description": "Light CPU load (10-25%)",
        "cpu_load_range": (10, 25),
        "vm_bytes": "0",
        "vm_workers": 0,
        "io_workers": 0,
        "weight": 2.0,
    },
    {
        "name": "cpu_medium",
        "description": "Medium CPU load (40-60%)",
        "cpu_load_range": (40, 60),
        "vm_bytes": "0",
        "vm_workers": 0,
        "io_workers": 0,
        "weight": 2.0,
    },
    {
        "name": "cpu_high",
        "description": "High CPU load (75-95%)",
        "cpu_load_range": (75, 95),
        "vm_bytes": "0",
        "vm_workers": 0,
        "io_workers": 0,
        "weight": 1.5,
    },
    {
        "name": "cpu_spike",
        "description": "CPU spike to 100%",
        "cpu_load": 100,
        "vm_bytes": "0",
        "vm_workers": 0,
        "io_workers": 0,
        "duration_range_s": (15, 45),
        "weight": 1.0,
    },
    {
        "name": "memory_low",
        "description": "Light memory pressure (128-512M)",
        "cpu_load": 5,
        "vm_bytes_range": ("128M", "512M"),
        "vm_workers": 2,
        "io_workers": 0,
        "weight": 1.5,
    },
    {
        "name": "memory_medium",
        "description": "Medium memory pressure (512M-2G)",
        "cpu_load": 5,
        "vm_bytes_range": ("512M", "2G"),
        "vm_workers": 2,
        "io_workers": 0,
        "weight": 1.5,
    },
    {
        "name": "memory_high",
        "description": "High memory pressure (2G-4G)",
        "cpu_load": 5,
        "vm_bytes_range": ("2G", "4G"),
        "vm_workers": 3,
        "io_workers": 0,
        "weight": 1.0,
    },
    {
        "name": "io_bound",
        "description": "Disk I/O stress",
        "cpu_load": 5,
        "vm_bytes": "64M",
        "vm_workers": 1,
        "io_workers_range": (2, 8),
        "weight": 1.5,
    },
    {
        "name": "mixed_cpu_mem",
        "description": "CPU + memory stress together",
        "cpu_load_range": (40, 80),
        "vm_bytes_range": ("256M", "2G"),
        "vm_workers": 2,
        "io_workers": 0,
        "weight": 2.0,
    },
    {
        "name": "mixed_all",
        "description": "CPU + memory + I/O stress",
        "cpu_load_range": (30, 70),
        "vm_bytes_range": ("256M", "1G"),
        "vm_workers": 2,
        "io_workers_range": (1, 4),
        "weight": 1.0,
    },
    {
        "name": "asymmetric",
        "description": "High load on some nodes, idle on others",
        "cpu_load_range": (60, 95),
        "vm_bytes_range": ("256M", "1G"),
        "vm_workers": 2,
        "io_workers": 0,
        "node_fraction": 0.5,
        "weight": 2.0,
    },
]


CYCLE_MIN_S = int(os.getenv("STRESS_CYCLE_MIN_S", "120"))
CYCLE_MAX_S = int(os.getenv("STRESS_CYCLE_MAX_S", "900"))


def _load_kubeconfig() -> None:
    try:
        config.load_kube_config()
    except Exception as e:
        logger.error("Failed to load kubeconfig: %s", e)
        sys.exit(1)


def _kube_apis() -> tuple[client.CoreV1Api, client.BatchV1Api, client.AppsV1Api]:
    _load_kubeconfig()
    return client.CoreV1Api(), client.BatchV1Api(), client.AppsV1Api()


def _parse_bytes(s: str) -> int:
    """Parse '256M', '2G' etc. to bytes for comparison."""
    s = s.strip().upper()
    if s == "0":
        return 0
    if s.endswith("G"):
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def _random_bytes_in_range(low: str, high: str) -> str:
    """Return a random byte string between low and high."""
    low_b = _parse_bytes(low)
    high_b = _parse_bytes(high)
    val = random.randint(low_b, high_b)
    if val >= 1024 * 1024 * 1024:
        return f"{val // (1024 * 1024 * 1024)}G"
    elif val >= 1024 * 1024:
        return f"{val // (1024 * 1024)}M"
    return str(val)


def get_worker_nodes(core_v1: client.CoreV1Api) -> list[str]:
    """Get names of worker nodes."""
    try:
        nodes = core_v1.list_node(label_selector="dedicated=worker").items
        if not nodes:
            all_nodes = core_v1.list_node().items
            nodes = []
            for node in all_nodes:
                labels = node.metadata.labels or {}
                if "node-role.kubernetes.io/control-plane" in labels:
                    continue
                if "node-role.kubernetes.io/master" in labels:
                    continue
                nodes.append(node)
        return [n.metadata.name for n in nodes]
    except Exception as e:
        logger.error("Failed to get nodes: %s", e)
        return []


def cleanup_stress_jobs(batch_v1: client.BatchV1Api, apps_v1: client.AppsV1Api):
    """Delete all stress jobs in the namespace."""
    logger.info("Cleaning up stress jobs in namespace %s...", NAMESPACE)
    try:
        batch_v1.delete_collection_namespaced_job(
            namespace=NAMESPACE,
            propagation_policy="Background",
        )
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete jobs: %s", e)

    try:
        apps_v1.delete_namespaced_daemon_set(
            name="node-stress",
            namespace=NAMESPACE,
            propagation_policy="Background",
        )
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete daemonset: %s", e)
    logger.info("Cleanup complete")


def resolve_profile(profile: dict) -> dict:
    """Resolve randomised ranges in a profile to concrete values."""
    p = dict(profile)

    if "cpu_load_range" in p:
        lo, hi = p.pop("cpu_load_range")
        p["cpu_load"] = random.randint(lo, hi)
    elif "cpu_load" not in p:
        p["cpu_load"] = 0

    if "vm_bytes_range" in p:
        lo, hi = p.pop("vm_bytes_range")
        p["vm_bytes"] = _random_bytes_in_range(lo, hi)
    elif "vm_bytes" not in p:
        p["vm_bytes"] = "0"

    if "vm_workers" not in p:
        p["vm_workers"] = 0

    if "io_workers_range" in p:
        lo, hi = p.pop("io_workers_range")
        p["io_workers"] = random.randint(lo, hi)
    elif "io_workers" not in p:
        p["io_workers"] = 0

    if "duration_range_s" in p:
        lo, hi = p.pop("duration_range_s")
        p["duration_s"] = random.randint(lo, hi)
    elif p.get("duration_s") is None:
        p["duration_s"] = random.randint(CYCLE_MIN_S, CYCLE_MAX_S)

    return p


def deploy_stress_job(
    batch_v1: client.BatchV1Api,
    node_name: str,
    profile: dict,
    job_suffix: str = "",
) -> str | None:
    """Deploy a stress-ng job to a specific node."""
    p = profile
    job_name = f"stress-{p['name']}-{node_name}{job_suffix}".lower()

    job_name = job_name.replace(".", "-").replace("_", "-")[:63]

    duration = p.get("duration_s", 60)
    cpu_load = p.get("cpu_load", 0)
    vm_bytes = p.get("vm_bytes", "0")
    vm_workers = p.get("vm_workers", 0)
    io_workers = p.get("io_workers", 0)

    stress_args = []
    if cpu_load > 0:
        stress_args += ["--cpu", "$(nproc)", "--cpu-load", str(cpu_load)]
    if vm_workers > 0 and vm_bytes != "0":
        stress_args += ["--vm", str(vm_workers), "--vm-bytes", vm_bytes, "--vm-keep"]
    if io_workers > 0:
        stress_args += ["--io", str(io_workers)]

    if not stress_args:
        return None

    stress_args += [
        "--temp-path",
        "/tmp",
        "--timeout",
        f"{duration}s",
        "--metrics-brief",
    ]
    stress_cmd = " ".join(["exec stress-ng"] + stress_args)

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": NAMESPACE,
            "labels": {
                "app": "stress-randomizer",
                "profile": p["name"],
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": 10,
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 5,
                    "nodeName": node_name,
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "stress",
                            "image": STRESS_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-c"],
                            "args": [stress_cmd],
                            "workingDir": "/tmp",
                            "env": [
                                {"name": "TMPDIR", "value": "/tmp"},
                            ],
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "64Mi"},
                            },
                            "volumeMounts": [
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                        }
                    ],
                    "volumes": [
                        {"name": "tmp", "emptyDir": {}},
                    ],
                },
            },
        },
    }

    try:
        batch_v1.create_namespaced_job(namespace=NAMESPACE, body=manifest)
        logger.info("Deployed job %s → node %s", job_name, node_name)
        return job_name
    except ApiException as e:
        if e.status == 409:
            logger.info("Job %s already exists, skipping", job_name)
            return job_name
        logger.warning("Failed to create job %s: %s", job_name, e)
        return None


def _collect_failed_pod_logs(core_v1: client.CoreV1Api, limit: int = 5) -> None:
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="app=stress-randomizer",
        ).items
    except ApiException as e:
        logger.warning("Failed to list pods for log collection: %s", e)
        return

    failed = []
    for pod in pods:
        status = pod.status
        if status and status.phase == "Failed":
            failed.append(pod)
            continue
        for cs in status.container_statuses or []:
            state = cs.state
            if state and state.terminated and state.terminated.exit_code != 0:
                failed.append(pod)
                break

    for pod in failed[:limit]:
        name = pod.metadata.name
        reason = pod.status.reason or ""
        message = pod.status.message or ""
        logger.warning("Pod failed: %s reason=%s message=%s", name, reason, message)
        for cs in pod.status.container_statuses or []:
            if cs.state and cs.state.terminated:
                term = cs.state.terminated
                logger.warning(
                    "  container=%s exit=%s reason=%s message=%s",
                    cs.name,
                    term.exit_code,
                    term.reason or "",
                    term.message or "",
                )
        try:
            logs = core_v1.read_namespaced_pod_log(
                name=name,
                namespace=NAMESPACE,
                container="stress",
                tail_lines=100,
            )
            if logs:
                logger.warning("Logs from %s:\n%s", name, logs.rstrip())
        except ApiException as e:
            logger.warning("Failed to read logs for %s: %s", name, e)


def run_cycle(
    core_v1: client.CoreV1Api,
    batch_v1: client.BatchV1Api,
    apps_v1: client.AppsV1Api,
    nodes: list[str],
):
    """Pick a random profile, deploy to random nodes, wait, cleanup."""

    weights = [p.get("weight", 1.0) for p in STRESS_PROFILES]
    profile_template = random.choices(STRESS_PROFILES, weights=weights, k=1)[0]
    profile = resolve_profile(profile_template)

    duration = profile.get("duration_s", 60)
    node_fraction = profile.get("node_fraction", 1.0)

    if node_fraction < 1.0:
        n = max(1, int(len(nodes) * node_fraction))
        target_nodes = random.sample(nodes, n)
    else:
        target_nodes = list(nodes)

    logger.info("═══════════════════════════════════════════════════════════════")
    logger.info(
        "Stress cycle: profile=%s  duration=%ds  nodes=%d/%d",
        profile["name"],
        duration,
        len(target_nodes),
        len(nodes),
    )
    logger.info(
        "  cpu_load=%d%%  vm_bytes=%s  vm_workers=%d  io_workers=%d",
        profile.get("cpu_load", 0),
        profile.get("vm_bytes", "0"),
        profile.get("vm_workers", 0),
        profile.get("io_workers", 0),
    )
    logger.info("═══════════════════════════════════════════════════════════════")

    if profile["name"] == "idle":
        cleanup_stress_jobs(batch_v1, apps_v1)
        logger.info("Idle period for %ds", duration)
        time.sleep(duration)
        return

    cleanup_stress_jobs(batch_v1, apps_v1)
    time.sleep(2)

    suffix = f"-{int(time.time()) % 10000}"
    for node in target_nodes:
        deploy_stress_job(batch_v1, node, profile, job_suffix=suffix)

    logger.info("Stress running for %ds...", duration)
    time.sleep(duration)

    _collect_failed_pod_logs(core_v1)

    cleanup_stress_jobs(batch_v1, apps_v1)

    cooldown = random.randint(5, 15)
    logger.info("Cooldown for %ds before next cycle", cooldown)
    time.sleep(cooldown)


def main():
    parser = argparse.ArgumentParser(
        description="Randomized cluster stress for LPSE training"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run one stress cycle and exit"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete all stress jobs and exit"
    )
    parser.add_argument(
        "--cycles", type=int, default=0, help="Run N cycles then exit (0=infinite)"
    )
    args = parser.parse_args()

    core_v1, batch_v1, apps_v1 = _kube_apis()

    try:
        core_v1.read_namespace(NAMESPACE)
    except ApiException as e:
        if e.status == 404:
            ns_body = client.V1Namespace(metadata=client.V1ObjectMeta(name=NAMESPACE))
            core_v1.create_namespace(ns_body)
        else:
            logger.error("Failed to ensure namespace %s: %s", NAMESPACE, e)
            sys.exit(1)

    if args.cleanup:
        cleanup_stress_jobs(batch_v1, apps_v1)
        return

    nodes = get_worker_nodes(core_v1)
    if not nodes:
        logger.error("No worker nodes found. Exiting.")
        sys.exit(1)

    logger.info("Found %d worker nodes: %s", len(nodes), nodes)
    logger.info("Cycle duration range: %d–%ds", CYCLE_MIN_S, CYCLE_MAX_S)
    logger.info("Profiles available: %s", [p["name"] for p in STRESS_PROFILES])

    cycle_count = 0
    try:
        while True:
            run_cycle(core_v1, batch_v1, apps_v1, nodes)
            cycle_count += 1
            if args.once or (args.cycles > 0 and cycle_count >= args.cycles):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted. Cleaning up...")
        cleanup_stress_jobs(batch_v1, apps_v1)


if __name__ == "__main__":
    main()
