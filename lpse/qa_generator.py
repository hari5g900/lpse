"""Generate (question, answer) pairs from a Prometheus snapshot. Answers are indices into a fixed template codebook."""

import logging
import math
import random

from .question_encoder import parse_question

logger = logging.getLogger("qa_generator")


_CPU_ANSWERS = [
    "CPU usage is very low (under 10%)",
    "CPU usage is low (10-30%)",
    "CPU usage is moderate (30-50%)",
    "CPU usage is elevated (50-70%)",
    "CPU usage is high (70-90%)",
    "CPU usage is critical (over 90%)",
    "CPU is idle",
    "This node is CPU-bound",
    "This node is not CPU-bound",
]


_MEM_ANSWERS = [
    "Memory usage is very low (under 20%)",
    "Memory usage is low (20-40%)",
    "Memory usage is moderate (40-60%)",
    "Memory usage is elevated (60-80%)",
    "Memory usage is high (over 80%)",
    "Memory usage is critical (over 95%)",
    "This node is memory-bound",
    "This node is not memory-bound",
    "Plenty of memory available",
    "Memory is under pressure",
]


_NET_ANSWERS = [
    "Network traffic is minimal",
    "Network traffic is low",
    "Network traffic is moderate",
    "Network traffic is high",
    "Network is congested",
    "Network has retransmissions (possible packet loss)",
    "Network looks healthy",
    "Many TCP connections established",
    "Few active network connections",
]


_DISK_ANSWERS = [
    "Disk I/O is minimal",
    "Disk I/O is low",
    "Disk I/O is moderate",
    "Disk I/O is high",
    "Disk I/O is very high",
    "This node is I/O-bound",
    "This node is not I/O-bound",
    "Filesystem usage is low (under 50%)",
    "Filesystem usage is moderate (50-80%)",
    "Filesystem usage is high (over 80%)",
    "Filesystem is nearly full (over 90%)",
]


_LOAD_ANSWERS = [
    "Load average is low (system underutilised)",
    "Load average is normal",
    "Load average is elevated (system is busy)",
    "Load average is high (system may be overloaded)",
    "Load average is very high (system is overloaded)",
    "Load is increasing (1m > 5m > 15m)",
    "Load is decreasing (1m < 5m < 15m)",
    "Load is stable",
]


_PROC_ANSWERS = [
    "Few processes running",
    "Many processes running",
    "Some processes are blocked on I/O",
    "No blocked processes",
    "High context switch rate",
    "Normal context switch rate",
]


_TEMP_ANSWERS = [
    "Temperature is normal",
    "Temperature is elevated",
    "Temperature is high (thermal throttling may occur)",
]


_CLUSTER_ANSWERS = [
    "Cluster is healthy overall",
    "Cluster has some nodes under load",
    "Cluster is under heavy load",
    "All nodes are idle",
    "Load is evenly distributed across nodes",
    "Load is unevenly distributed (some nodes are hot)",
    "One node is significantly more loaded than others",
]


_COMPARE_ANSWERS = [
    "Node {node_a} has higher CPU usage",
    "Node {node_b} has higher CPU usage",
    "Both nodes have similar CPU usage",
    "Node {node_a} has higher memory usage",
    "Node {node_b} has higher memory usage",
    "Both nodes have similar memory usage",
]


_EXTREME_ANSWERS = [
    "Node {node} has the highest CPU usage",
    "Node {node} has the lowest CPU usage",
    "Node {node} has the highest memory usage",
    "Node {node} has the lowest memory usage",
    "Node {node} has the highest disk I/O",
    "Node {node} has the highest network traffic",
    "Node {node} has the highest load average",
]


_PREDICT_ANSWERS = [
    "CPU usage is likely to increase",
    "CPU usage is likely to stay stable",
    "CPU usage is likely to decrease",
    "Memory pressure is likely to increase",
    "Memory usage will likely stay stable",
    "Overall load is likely to increase",
    "Overall load is likely to stay stable",
    "Overall load is likely to decrease",
]


_CONDITION_ANSWERS = [
    "Yes, this node appears healthy",
    "No, this node shows signs of stress",
    "Yes, there is a potential bottleneck",
    "No bottleneck detected",
    "Yes, the node is overloaded",
    "No, the node is not overloaded",
    "Some nodes are approaching capacity",
    "All nodes are operating normally",
]


_GENERIC_ANSWERS = [
    "The system is operating within normal parameters",
    "Unable to determine with high confidence",
]


def build_answer_templates() -> list[str]:
    """Return the full ordered list of answer templates."""
    return (
        _CPU_ANSWERS
        + _MEM_ANSWERS
        + _NET_ANSWERS
        + _DISK_ANSWERS
        + _LOAD_ANSWERS
        + _PROC_ANSWERS
        + _TEMP_ANSWERS
        + _CLUSTER_ANSWERS
        + _COMPARE_ANSWERS
        + _EXTREME_ANSWERS
        + _PREDICT_ANSWERS
        + _CONDITION_ANSWERS
        + _GENERIC_ANSWERS
    )


ANSWER_TEMPLATES = build_answer_templates()
NUM_ANSWERS = len(ANSWER_TEMPLATES)


_CATEGORY_RANGES: dict[str, tuple[int, int]] = {}
_offset = 0
for _cat, _lst in [
    ("cpu", _CPU_ANSWERS),
    ("memory", _MEM_ANSWERS),
    ("network", _NET_ANSWERS),
    ("disk", _DISK_ANSWERS),
    ("load", _LOAD_ANSWERS),
    ("process", _PROC_ANSWERS),
    ("temperature", _TEMP_ANSWERS),
    ("cluster", _CLUSTER_ANSWERS),
    ("compare", _COMPARE_ANSWERS),
    ("extreme", _EXTREME_ANSWERS),
    ("predict", _PREDICT_ANSWERS),
    ("condition", _CONDITION_ANSWERS),
    ("generic", _GENERIC_ANSWERS),
]:
    _CATEGORY_RANGES[_cat] = (_offset, _offset + len(_lst))
    _offset += len(_lst)


def _cat_start_end(cat: str) -> tuple[int, int]:
    return _CATEGORY_RANGES.get(cat, (0, NUM_ANSWERS))


def _unknown_idx() -> int:
    """Index of the 'Unable to determine with high confidence' template.

    Used when a node's telemetry is missing entirely (unreachable/down), so
    we never assert a *healthy* answer about a node whose features are all
    zeros. This keeps labels consistent with the zeroed token features.
    """
    start, _ = _cat_start_end("generic")
    return start + 1


_SEMANTIC_ANSWER_GROUPS = [
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8],
    [9, 10, 11, 12, 13, 14, 17, 18],
    [15, 16],
    [19, 20, 21, 22, 23, 25, 26, 27],
    [24],
    [28, 29, 30, 31, 32],
    [33, 34],
    [35, 36, 37, 38],
    [39, 40, 41, 42, 43],
    [44, 45, 46],
    [47, 48],
    [49, 50],
    [51, 52],
    [53, 54, 55],
    [56, 57, 58, 59],
    [60, 61, 62],
    [63, 64, 65],
    [66, 67, 68],
    [69, 70],
    [71, 72],
    [73, 74, 75],
    [76, 77, 78],
    [79, 80],
    [81, 82, 83],
    [84, 85],
    [86, 87],
    [88, 89],
    [90, 91],
    [92, 93],
]


_seen_group_idx: set[int] = set()
for _group in _SEMANTIC_ANSWER_GROUPS:
    for _idx in _group:
        if not (0 <= _idx < NUM_ANSWERS):
            raise AssertionError(
                f"semantic soft-label group index {_idx} out of range [0, {NUM_ANSWERS})"
            )
        if _idx in _seen_group_idx:
            raise AssertionError(
                f"semantic soft-label index {_idx} listed in multiple groups"
            )
        _seen_group_idx.add(_idx)


def build_answer_soft_targets(
    num_answers: int = NUM_ANSWERS,
    soft_mass: float = 0.30,
    tau: float = 1.0,
    uniform_smoothing: float = 0.0,
) -> list[list[float]]:
    """
    Build semantic soft labels over answer templates.

    The true template keeps most of the probability mass, while nearby templates
    in the same semantic group receive partial mass according to index distance.
    """
    soft_mass = max(0.0, min(0.95, float(soft_mass)))
    uniform_smoothing = max(0.0, min(0.95, float(uniform_smoothing)))
    tau = max(1e-6, float(tau))

    rows: list[list[float]] = []
    for true_idx in range(num_answers):
        similarities = [0.0] * num_answers

        for group in _SEMANTIC_ANSWER_GROUPS:
            if true_idx not in group:
                continue
            true_pos = group.index(true_idx)
            for pos, answer_idx in enumerate(group):
                if answer_idx == true_idx or answer_idx >= num_answers:
                    continue
                distance = abs(pos - true_pos)
                similarities[answer_idx] = max(
                    similarities[answer_idx], math.exp(-distance / tau)
                )

        row = [0.0] * num_answers
        sim_sum = sum(similarities)
        if sim_sum > 0.0:
            row[true_idx] = 1.0 - soft_mass
            for answer_idx, sim in enumerate(similarities):
                if sim > 0.0:
                    row[answer_idx] = soft_mass * sim / sim_sum
        else:
            row[true_idx] = 1.0

        if uniform_smoothing > 0.0:
            uniform = uniform_smoothing / max(1, num_answers)
            row = [(1.0 - uniform_smoothing) * value + uniform for value in row]

        rows.append(row)

    return rows


_QUESTION_TEMPLATES = {
    "cpu": [
        "What is the CPU usage?",
        "How much CPU is being used?",
        "Is CPU utilisation high?",
        "What is the CPU load on this node?",
        "Tell me about CPU usage",
        "Is this node CPU-bound?",
        "How busy is the processor?",
        "What's the compute usage?",
        "Give me the CPU stats",
        "How are the cores doing?",
    ],
    "memory": [
        "What is the memory usage?",
        "How much memory is being used?",
        "Is memory under pressure?",
        "How much RAM is available?",
        "Is this node memory-bound?",
        "Tell me about memory usage",
        "What's the memory utilisation?",
        "How much memory is free?",
        "Are we running low on memory?",
    ],
    "network": [
        "What is the network traffic?",
        "How much bandwidth is being used?",
        "Is the network congested?",
        "Are there any network issues?",
        "How many connections are established?",
        "Is there packet loss?",
        "Tell me about network utilisation",
        "What's the network throughput?",
    ],
    "disk": [
        "What is the disk I/O?",
        "How busy are the disks?",
        "Is this node I/O-bound?",
        "What's the disk utilisation?",
        "How much disk space is available?",
        "Is the filesystem getting full?",
        "Tell me about storage usage",
        "What's the disk activity?",
    ],
    "load": [
        "What is the load average?",
        "Is the system overloaded?",
        "What's the system load?",
        "Is load average high?",
        "How busy is the system?",
        "Is the load increasing or decreasing?",
        "What does the load trend look like?",
    ],
    "process": [
        "How many processes are running?",
        "Are any processes blocked?",
        "What's the context switch rate?",
        "How many processes are in the run queue?",
    ],
    "temperature": [
        "What is the temperature?",
        "Is the system overheating?",
        "Are temperatures normal?",
    ],
    "health": [
        "Is the cluster healthy?",
        "Is everything running normally?",
        "Are all nodes OK?",
        "Give me a cluster overview",
        "Summarize the cluster status",
        "What's the overall system health?",
        "Is there anything wrong?",
    ],
    "overall": [
        "Give me a status summary",
        "What's the overall cluster state?",
        "How is the system doing?",
        "System health report",
        "Any issues to worry about?",
    ],
}


_SCOPED_TEMPLATES = {
    "single_node": [
        "What is the {metric} on node {node}?",
        "How is node {node} doing in terms of {metric}?",
        "Tell me about {metric} on {node}",
        "Is {metric} high on node {node}?",
    ],
    "cluster": [
        "Which node has the highest {metric}?",
        "Which node has the lowest {metric}?",
        "Compare all nodes by {metric}",
        "Is {metric} balanced across the cluster?",
    ],
    "predict": [
        "What will {metric} be in the next few minutes?",
        "Is {metric} trending up or down?",
        "Will {metric} become a problem soon?",
    ],
}


def _sample_question_templates(
    category: str, limit: int, exclude_phrases: tuple[str, ...] = ()
) -> list[str]:
    templates = _QUESTION_TEMPLATES[category]
    if exclude_phrases:
        templates = [
            qt
            for qt in templates
            if not any(phrase in qt.lower() for phrase in exclude_phrases)
        ]
    return random.sample(templates, min(limit, len(templates)))


def _classify_cpu(mean_cpu: float) -> int:
    """Return answer index within _CPU_ANSWERS for a CPU utilisation level."""
    start, _ = _cat_start_end("cpu")
    if mean_cpu < 0.01:
        return start + 6
    elif mean_cpu < 0.10:
        return start + 0
    elif mean_cpu < 0.30:
        return start + 1
    elif mean_cpu < 0.50:
        return start + 2
    elif mean_cpu < 0.70:
        return start + 3
    elif mean_cpu < 0.90:
        return start + 4
    else:
        return start + 5


def _classify_mem(mem_used_frac: float | None) -> int:
    start, _ = _cat_start_end("memory")
    if mem_used_frac is None:
        return _unknown_idx()
    if mem_used_frac < 0.20:
        return start + 0
    elif mem_used_frac < 0.40:
        return start + 1
    elif mem_used_frac < 0.60:
        return start + 2
    elif mem_used_frac < 0.80:
        return start + 3
    elif mem_used_frac < 0.95:
        return start + 4
    else:
        return start + 5


def _classify_fs(fs_used_frac: float | None) -> int:
    start, _ = _cat_start_end("disk")
    if fs_used_frac is None:
        return _unknown_idx()
    if fs_used_frac < 0.50:
        return start + 7
    elif fs_used_frac < 0.80:
        return start + 8
    elif fs_used_frac < 0.90:
        return start + 9
    else:
        return start + 10


def _classify_load(
    load1: float | None,
    load5: float | None,
    load15: float | None,
    num_cores: int,
) -> int:
    start, _ = _cat_start_end("load")
    if load1 is None:
        return _unknown_idx()

    ratio = load1 / max(1, num_cores)
    if ratio < 0.3:
        return start + 0
    elif ratio < 0.7:
        return start + 1
    elif ratio < 1.0:
        return start + 2
    elif ratio < 2.0:
        return start + 3
    else:
        return start + 4


def _classify_load_trend(
    load1: float | None, load5: float | None, load15: float | None
) -> int:
    start, _ = _cat_start_end("load")
    if load1 is None or load5 is None or load15 is None:
        return _unknown_idx()
    if load1 > load5 > load15:
        return start + 5
    elif load1 < load5 < load15:
        return start + 6
    else:
        return start + 7


def generate_qa_pairs(
    semantic_features: dict[str, dict[str, float | None]],
    node_ips: list[str],
    node_cores: list[int] | None = None,
    max_pairs: int = 50,
) -> list[tuple[dict, int]]:
    """
    From a single metric snapshot's semantic features, generate training pairs.

    Returns: list of (parsed_question_dict, answer_template_index)
    """
    pairs: list[tuple[dict, int]] = []

    if node_cores is None:
        node_cores = [10] * len(node_ips)

    for node_idx, ip in enumerate(node_ips):
        sf = semantic_features.get(ip, {})

        node_down = not sf
        cores = node_cores[node_idx] if node_idx < len(node_cores) else 10

        cpu_util = sf.get("cpu_utilization_fraction")
        if cpu_util is not None:
            cpu_proxy = max(0.0, min(1.0, float(cpu_util)))
        else:
            load1 = sf.get("node_load1")
            if load1 is not None:
                cpu_proxy = min(1.0, load1 / max(1, cores))
            else:
                cpu_proxy = 0.0

        for qt in _sample_question_templates("cpu", 2, exclude_phrases=("cpu-bound",)):
            q = parse_question(f"{qt} on node {ip}", node_ips)
            pairs.append((q, _unknown_idx() if node_down else _classify_cpu(cpu_proxy)))

        cpu_start, _ = _cat_start_end("cpu")
        q = parse_question(f"Is node {ip} CPU-bound?", node_ips)
        if node_down:
            pairs.append((q, _unknown_idx()))
        else:
            is_cpu_bound = cpu_proxy >= 0.7
            pairs.append((q, cpu_start + 7 if is_cpu_bound else cpu_start + 8))

        mem_frac = sf.get("mem_used_fraction")
        for qt in _sample_question_templates(
            "memory", 2, exclude_phrases=("memory-bound",)
        ):
            q = parse_question(f"{qt} on node {ip}", node_ips)
            pairs.append((q, _classify_mem(mem_frac)))

        mem_start, _ = _cat_start_end("memory")
        q = parse_question(f"Is node {ip} memory-bound?", node_ips)
        if node_down:
            pairs.append((q, _unknown_idx()))
        else:
            is_mem_bound = mem_frac is not None and mem_frac >= 0.8
            pairs.append((q, mem_start + 6 if is_mem_bound else mem_start + 7))

        fs_frac = sf.get("fs_used_fraction")
        q = parse_question(f"How much disk space is available on node {ip}?", node_ips)
        pairs.append((q, _classify_fs(fs_frac)))

        load1 = sf.get("node_load1")
        load5 = sf.get("node_load5")
        load15 = sf.get("node_load15")
        q = parse_question(f"What is the load average on node {ip}?", node_ips)
        pairs.append((q, _classify_load(load1, load5, load15, cores)))

        q = parse_question(f"Is load increasing or decreasing on node {ip}?", node_ips)
        pairs.append((q, _classify_load_trend(load1, load5, load15)))

        proc_start, _ = _cat_start_end("process")
        procs_blocked = sf.get("procs_blocked")
        if procs_blocked is not None:
            q = parse_question(f"Are any processes blocked on node {ip}?", node_ips)
            pairs.append((q, proc_start + 2 if procs_blocked > 0 else proc_start + 3))

        temp_start, _ = _cat_start_end("temperature")
        max_temp = sf.get("max_temp_celsius")
        if max_temp is not None:
            q = parse_question(f"What is the temperature on node {ip}?", node_ips)
            if max_temp < 70:
                pairs.append((q, temp_start + 0))
            elif max_temp < 85:
                pairs.append((q, temp_start + 1))
            else:
                pairs.append((q, temp_start + 2))

    cluster_start, _ = _cat_start_end("cluster")

    all_loads = []
    all_mem = []
    for ip in node_ips:
        sf = semantic_features.get(ip, {})
        l1 = sf.get("node_load1")
        cores = node_cores[node_ips.index(ip)] if ip in node_ips else 10
        if l1 is not None:
            all_loads.append(l1 / max(1, cores))
        mf = sf.get("mem_used_fraction")
        if mf is not None:
            all_mem.append(mf)

    for qt in random.sample(
        _QUESTION_TEMPLATES["health"], min(3, len(_QUESTION_TEMPLATES["health"]))
    ):
        q = parse_question(qt, node_ips)
        if all_loads:
            avg_load = sum(all_loads) / len(all_loads)
            max_load = max(all_loads)
            if max_load < 0.3 and (not all_mem or max(all_mem) < 0.5):
                pairs.append((q, cluster_start + 3))
            elif avg_load < 0.5:
                pairs.append((q, cluster_start + 0))
            elif avg_load < 0.8:
                pairs.append((q, cluster_start + 1))
            else:
                pairs.append((q, cluster_start + 2))
        else:
            pairs.append((q, cluster_start + 0))

    if len(all_loads) > 1:
        load_std = (
            sum((load - sum(all_loads) / len(all_loads)) ** 2 for load in all_loads)
            / len(all_loads)
        ) ** 0.5
        q = parse_question("Is load balanced across the cluster?", node_ips)
        if load_std < 0.1:
            pairs.append((q, cluster_start + 4))
        elif load_std < 0.3:
            pairs.append((q, cluster_start + 5))
        else:
            pairs.append((q, cluster_start + 6))

    extreme_start, _ = _cat_start_end("extreme")
    if all_loads and len(node_ips) > 1:
        q = parse_question("Which node has the highest CPU usage?", node_ips)
        pairs.append((q, extreme_start + 0))

        q = parse_question("Which node has the lowest CPU usage?", node_ips)
        pairs.append((q, extreme_start + 1))

    predict_start, _ = _cat_start_end("predict")
    if all_loads:
        avg_load = sum(all_loads) / len(all_loads)

        all_l1 = [semantic_features.get(ip, {}).get("node_load1") for ip in node_ips]
        all_l5 = [semantic_features.get(ip, {}).get("node_load5") for ip in node_ips]
        l1_avg = sum(v for v in all_l1 if v is not None) / max(
            1, sum(1 for v in all_l1 if v is not None)
        )
        l5_avg = sum(v for v in all_l5 if v is not None) / max(
            1, sum(1 for v in all_l5 if v is not None)
        )

        q = parse_question("Will load increase?", node_ips)
        if l1_avg > l5_avg * 1.1:
            pairs.append((q, predict_start + 5))
        elif l1_avg < l5_avg * 0.9:
            pairs.append((q, predict_start + 7))
        else:
            pairs.append((q, predict_start + 6))

    random.shuffle(pairs)

    balanced_pairs = []

    mem_start, _ = _cat_start_end("memory")
    cpu_start, _ = _cat_start_end("cpu")
    fs_start, _ = _cat_start_end("disk")
    load_start, _ = _cat_start_end("load")
    proc_start, _ = _cat_start_end("process")
    cluster_start, _ = _cat_start_end("cluster")

    default_indices = {
        cpu_start + 0,
        cpu_start + 6,
        cpu_start + 8,
        mem_start + 8,
        mem_start + 0,
        mem_start + 7,
        fs_start + 7,
        load_start + 0,
        load_start + 1,
        proc_start + 3,
        cluster_start + 0,
        cluster_start + 3,
    }

    default_count = 0
    max_defaults = max(1, max_pairs // 3)

    for p in pairs:
        if p[1] in default_indices:
            if default_count < max_defaults:
                balanced_pairs.append(p)
                default_count += 1
        else:
            balanced_pairs.append(p)

    return balanced_pairs[:max_pairs]


def fill_answer_template(
    template_idx: int, node_ips: list[str], semantic_features: dict
) -> str:
    """
    Fill in template placeholders with actual values.
    Used at inference time to produce human-readable answers.
    """
    if template_idx < 0 or template_idx >= NUM_ANSWERS:
        return "Unknown answer"

    template = ANSWER_TEMPLATES[template_idx]

    def _metric_value(ip: str, metric: str) -> float | None:
        sf = (
            semantic_features.get(ip, {}) if isinstance(semantic_features, dict) else {}
        )
        if metric == "cpu":
            return sf.get("node_load1")
        if metric == "memory":
            return sf.get("mem_used_fraction")
        if metric == "disk":
            dr = sf.get("disk_read_bytes_sum")
            dw = sf.get("disk_write_bytes_sum")
            if dr is None and dw is None:
                return None
            return float((dr or 0.0) + (dw or 0.0))
        if metric == "network":
            rx = sf.get("net_rx_bytes_sum")
            tx = sf.get("net_tx_bytes_sum")
            if rx is None and tx is None:
                return None
            return float((rx or 0.0) + (tx or 0.0))
        if metric == "load":
            return sf.get("node_load1")
        return None

    def _metric_from_template(t: str) -> str | None:
        tl = t.lower()
        if "cpu" in tl:
            return "cpu"
        if "memory" in tl or "ram" in tl:
            return "memory"
        if "disk" in tl or "i/o" in tl or "filesystem" in tl:
            return "disk"
        if "network" in tl or "tcp" in tl:
            return "network"
        if "load" in tl:
            return "load"
        return None

    def _select_node(metric: str | None, prefer_max: bool = True) -> str | None:
        if not node_ips:
            return None
        if metric is None:
            return node_ips[0]
        scored = []
        for ip in node_ips:
            val = _metric_value(ip, metric)
            if val is not None:
                scored.append((val, ip))
        if not scored:
            return node_ips[0]
        scored.sort(key=lambda x: x[0], reverse=prefer_max)
        return scored[0][1]

    if "{node}" in template:
        metric = _metric_from_template(template)
        prefer_max = "highest" in template.lower()
        if "lowest" in template.lower():
            prefer_max = False
        node = _select_node(metric, prefer_max=prefer_max) or "unknown"
        template = template.replace("{node}", node)

    if "{node_a}" in template:
        metric = _metric_from_template(template)
        if node_ips:
            if metric:
                scored = []
                for ip in node_ips:
                    val = _metric_value(ip, metric)
                    if val is not None:
                        scored.append((val, ip))
                scored.sort(key=lambda x: x[0], reverse=True)
                node_a = scored[0][1] if scored else node_ips[0]
                node_b = (
                    scored[1][1]
                    if len(scored) > 1
                    else (node_ips[1] if len(node_ips) > 1 else node_ips[0])
                )
            else:
                node_a = node_ips[0]
                node_b = node_ips[1] if len(node_ips) > 1 else node_ips[0]
        else:
            node_a, node_b = "A", "B"
        template = template.replace("{node_a}", node_a)
        template = template.replace("{node_b}", node_b)

    return template
