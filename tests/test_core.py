"""Unit tests for the LPSE core math and data pipeline.

Run from the repo root:
    python -m pytest tests/test_core.py -q
    (or) python tests/test_core.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lpse.metrics_shared import cpu_util_per_core, fetch_node_exporter_metrics
from lpse.prometheus_ingest import (
    CLUSTER_IDENTITY_INDICES,
    CLUSTER_TOKEN_DIM,
    FEATURES_PER_NODE,
    NODE_SLOT_FEATURES,
    RunningNormalizer,
    cluster_snapshot_to_token,
)
from lpse.qa_generator import (
    _CATEGORY_RANGES,
    _SEMANTIC_ANSWER_GROUPS,
    ANSWER_TEMPLATES,
    NUM_ANSWERS,
    _classify_cpu,
    _classify_fs,
    _classify_load,
    _classify_load_trend,
    _classify_mem,
    _unknown_idx,
    build_answer_soft_targets,
    generate_qa_pairs,
)
from lpse.question_encoder import parse_question
from lpse.replay_buffer import QAReplayBuffer


def test_answer_inventory_size():

    assert NUM_ANSWERS == 94
    assert len(ANSWER_TEMPLATES) == NUM_ANSWERS


def test_category_ranges_cover_all_templates():
    seen = set()
    for start, end in _CATEGORY_RANGES.values():
        seen.update(range(start, end))
    assert seen == set(range(NUM_ANSWERS))


def test_semantic_groups_in_range():
    flat = [i for g in _SEMANTIC_ANSWER_GROUPS for i in g]
    assert all(0 <= i < NUM_ANSWERS for i in flat)
    assert len(flat) == len(set(flat))


def test_unknown_idx_is_the_unable_to_determine_template():
    idx = _unknown_idx()
    assert ANSWER_TEMPLATES[idx] == "Unable to determine with high confidence"


def test_soft_targets_are_distributions():
    rows = build_answer_soft_targets()
    assert len(rows) == NUM_ANSWERS
    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-6
        assert all(v >= 0.0 for v in row)

    assert rows[0][0] == max(rows[0])


def test_soft_targets_group_mass():

    rows = build_answer_soft_targets(soft_mass=0.3)
    assert rows[0][1] > 0.0

    assert rows[0][9] == 0.0


def test_classifiers_unknown_on_missing_telemetry():

    assert (
        ANSWER_TEMPLATES[_classify_mem(None)]
        == "Unable to determine with high confidence"
    )
    assert (
        ANSWER_TEMPLATES[_classify_fs(None)]
        == "Unable to determine with high confidence"
    )
    assert (
        ANSWER_TEMPLATES[_classify_load(None, None, None, 8)]
        == "Unable to determine with high confidence"
    )
    assert (
        ANSWER_TEMPLATES[_classify_load_trend(None, None, None)]
        == "Unable to determine with high confidence"
    )


def test_classify_cpu_bounds():
    start, _ = _CATEGORY_RANGES["cpu"]
    assert _classify_cpu(0.99) == start + 5
    assert _classify_cpu(0.0) == start + 6
    assert _classify_cpu(0.05) == start + 0


def test_generate_qa_pairs_down_node_labels_unknown():
    ips = ["10.0.0.1", "10.0.0.2"]
    semantic = {
        "10.0.0.1": {},
        "10.0.0.2": {
            "cpu_utilization_fraction": 0.5,
            "mem_used_fraction": 0.4,
            "node_load1": 1.0,
            "node_load5": 1.0,
            "node_load15": 1.0,
            "fs_used_fraction": 0.3,
        },
    }
    random.seed(0)
    pairs = generate_qa_pairs(semantic, ips, [8, 8], max_pairs=50)
    assert pairs

    for q, ans in pairs:
        if q.get("node_ref") == "10.0.0.1":
            assert ANSWER_TEMPLATES[ans] == "Unable to determine with high confidence"

    for q, ans in pairs:
        if q.get("node_ref") == "10.0.0.2":
            assert ANSWER_TEMPLATES[ans] != "Unable to determine with high confidence"


def test_parse_question():
    q = parse_question("What is the CPU usage of node 10.0.0.1?", ["10.0.0.1"])
    assert q["metric"] == "cpu"
    assert q["node_ref"] == "10.0.0.1"
    assert q["scope"] == "single_node"
    q2 = parse_question("Is any node memory-bound?")
    assert q2["intent"] == "is_condition"
    assert q2["metric"] == "memory"


def test_cluster_token_dimensions_and_identity():
    n_nodes = 3
    feats = torch.randn(n_nodes, 300)
    tok = cluster_snapshot_to_token(feats, ["10.0.0.3", "10.0.0.1", "10.0.0.2"])
    assert tok.shape == (CLUSTER_TOKEN_DIM,)

    ip0 = tok[FEATURES_PER_NODE : FEATURES_PER_NODE + 4]
    assert torch.allclose(ip0, torch.tensor([10 / 255.0, 0.0, 0.0, 1 / 255.0]))

    for slot in range(3):
        base = slot * NODE_SLOT_FEATURES
        assert tok[base + FEATURES_PER_NODE + 4].item() == 1.0

    base = 3 * NODE_SLOT_FEATURES
    assert tok[base + FEATURES_PER_NODE + 4].item() == 0.0


def test_normalizer_identity_features_preserved():
    norm = RunningNormalizer(CLUSTER_TOKEN_DIM)

    for _ in range(5):
        t = torch.randn(CLUSTER_TOKEN_DIM) * 10
        for i in CLUSTER_IDENTITY_INDICES:
            t[i] = 1.0
        norm.update(t)
    out = norm.normalize(torch.ones(CLUSTER_TOKEN_DIM) * 7)
    for i in CLUSTER_IDENTITY_INDICES:
        assert out[i].item() == 7.0


def test_normalizer_degnerate_variance_resets():
    norm = RunningNormalizer(4)
    norm.load_state_dict(
        {
            "mean": torch.zeros(4),
            "var": torch.ones(4) * 1e-8,
            "count": 100,
        }
    )

    assert norm.count == 0
    assert torch.allclose(norm.var, torch.ones(4))


def test_cpu_util_per_core_needs_two_samples():
    t0 = 'node_cpu_seconds_total{cpu="0",mode="idle"} 100\n'
    assert cpu_util_per_core("h", t0) == []
    t1 = 'node_cpu_seconds_total{cpu="0",mode="idle"} 140\n'
    utils = cpu_util_per_core("h", t1)
    assert len(utils) == 1
    assert abs(utils[0] - 0.0) < 1e-9


def test_node_exporter_scrape_override():
    os.environ["NODE_SCRAPE_URL_MAP"] = '{"10.0.0.9": "http://127.0.0.1:1"}'

    try:
        import importlib

        import lpse.metrics_shared as ms

        importlib.reload(ms)
        assert ms._NODE_SCRAPE_OVERRIDES.get("10.0.0.9") == "http://127.0.0.1:1"
        assert fetch_node_exporter_metrics("10.0.0.9") is None
    finally:
        os.environ.pop("NODE_SCRAPE_URL_MAP", None)


def test_replay_buffer_live_count():
    rb = QAReplayBuffer(qa_capacity=1000, context_capacity=2)
    c1 = rb.add_context(torch.zeros(2, 3))
    rb.add_qa(c1, {"raw": "q1"}, 0)
    rb.add_qa(c1, {"raw": "q2"}, 1)
    assert len(rb) == 2
    c2 = rb.add_context(torch.ones(2, 3))
    rb.add_qa(c2, {"raw": "q3"}, 0)
    assert len(rb) == 3

    c3 = rb.add_context(torch.ones(2, 3) * 2)
    rb.add_qa(c3, {"raw": "q4"}, 2)
    assert len(rb) == 2


def test_replay_buffer_deque_cap_drop():
    rb = QAReplayBuffer(qa_capacity=2, context_capacity=10)
    c = rb.add_context(torch.zeros(2, 3))
    rb.add_qa(c, {"raw": "a"}, 0)
    rb.add_qa(c, {"raw": "b"}, 1)
    assert len(rb) == 2
    rb.add_qa(c, {"raw": "c"}, 2)
    assert len(rb) == 2


def test_replay_buffer_len_never_negative_after_eviction():

    rb = QAReplayBuffer(qa_capacity=2, context_capacity=2)
    c1 = rb.add_context(torch.zeros(2, 3))
    rb.add_qa(c1, {"raw": "a"}, 0)
    rb.add_qa(c1, {"raw": "b"}, 1)
    rb.add_qa(c1, {"raw": "c"}, 2)
    assert len(rb) == 2
    c2 = rb.add_context(torch.ones(2, 3))
    assert len(rb) == 2
    rb.add_qa(c2, {"raw": "d"}, 3)
    assert len(rb) == 2
    rb.add_context(torch.ones(2, 3) * 2)
    assert len(rb) == 1
    assert len(rb) >= 0


def _import_benchmark():
    import benchmark_llm_forecast as b

    return b


def test_parse_llm_answer_index_open():
    b = _import_benchmark()
    assert b._parse_llm_answer_index(5, ANSWER_TEMPLATES) == 5
    assert b._parse_llm_answer_index({"answer_index": 7}, ANSWER_TEMPLATES) == 7
    assert b._parse_llm_answer_index({"index": 2}, ANSWER_TEMPLATES) == 2

    assert b._parse_llm_answer_index(999, ANSWER_TEMPLATES) is None
    assert b._parse_llm_answer_index(True, ANSWER_TEMPLATES) is None


def test_paraphrase_rewrites():
    b = _import_benchmark()
    out = b._paraphrase_question("Is node 10.0.0.1 CPU-bound?")
    assert out != "Is node 10.0.0.1 CPU-bound?"
    assert "10.0.0.1" in out

    out2 = b._paraphrase_question("What is the load average on node 10.0.0.1?")
    assert out2 != "What is the load average on node 10.0.0.1?"


def test_strip_thinking_and_extract_json():
    b = _import_benchmark()

    open_tag = chr(0x3C) + "think" + chr(0x3E)
    close_tag = chr(0x3C) + "/" + "think" + chr(0x3E)
    text = open_tag + "let me reason" + close_tag + ' {"answer_index": 3}'
    assert b._strip_thinking_blocks(text) == '{"answer_index": 3}'
    cands = b._extract_json_candidates(text)
    assert any(c.get("answer_index") == 3 for c in cands)


def test_category_option_indices_whole_category():
    b = _import_benchmark()
    idxs = b._category_option_indices(0)
    assert 0 in idxs
    assert len(idxs) >= 2


if __name__ == "__main__":
    failures = 0
    g = dict(globals())
    for name, fn in sorted(g.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e!r}")
    print(
        f"\n{sum(1 for k in g if k.startswith('test_') and callable(g[k])) - failures}/{sum(1 for k in g if k.startswith('test_') and callable(g[k]))} passed"
    )
    sys.exit(1 if failures else 0)
