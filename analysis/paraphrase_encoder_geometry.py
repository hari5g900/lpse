#!/usr/bin/env python3
"""Does the frozen all-MiniLM-L6-v2 encoder keep benchmark paraphrases near
their source templates? Category-level nearest-template retrieval over the
full training-time question vocabulary, using the repo's own template lists
(lpse/qa_generator.py) and paraphrase rules (benchmark_llm_forecast.py),
with a representative IP substituted.

Supports the corrected paraphrase-collapse diagnosis: raw
frozen-encoder embeddings retrieve a correct-category template for ~91% of
paraphrases (77.5% on the aggressive pattern rewrites, 100% on the generic
fallback; mean cos to source 0.755), so the 89%->39% benchmark drop lives in
the layers trained only on template strings, not in the encoder.

Template/pattern lists are inlined (not imported) so this runs in any env
with only `sentence-transformers` + `numpy` — no torch-cluster deps. If
qa_generator's templates or the benchmark's paraphrase rules change, update
the copies here.

Usage: python analysis/paraphrase_encoder_geometry.py
(downloads all-MiniLM-L6-v2 on first run; CPU is fine, takes seconds)"""

import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

IP = os.environ.get("VLLM_HOST", "10.68.184.22")


BASE = {
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


vocab, vocab_cat = [], []
for cat, qs in BASE.items():
    for q in qs:
        vocab.append(q)
        vocab_cat.append(cat)
        if cat in ("cpu", "memory", "disk", "load", "process", "temperature"):
            vocab.append(
                f"{q.rstrip('?')} on node {IP}" + ("?" if q.endswith("?") else "")
            )
            vocab_cat.append(cat)

for q, cat in [
    (f"Is node {IP} CPU-bound?", "cpu"),
    (f"Is node {IP} memory-bound?", "memory"),
    (f"How much disk space is available on node {IP}?", "disk"),
    (f"What is the load average on node {IP}?", "load"),
    (f"Is load increasing or decreasing on node {IP}?", "load"),
    (f"Are any processes blocked on node {IP}?", "process"),
    (f"What is the temperature on node {IP}?", "temperature"),
]:
    vocab.append(q)
    vocab_cat.append(cat)


PATTERNS = [
    (
        re.compile(r"^Is node (\S+) CPU-bound\?$"),
        lambda m: f"Has the processor on box {m.group(1)} hit its ceiling?",
        "cpu",
    ),
    (
        re.compile(r"^Is node (\S+) memory-bound\?$"),
        lambda m: f"Is RAM the bottleneck for {m.group(1)} right now?",
        "memory",
    ),
    (
        re.compile(r"^How much disk space is available on node (\S+)\?$"),
        lambda m: f"How much room is left on {m.group(1)}'s filesystem?",
        "disk",
    ),
    (
        re.compile(r"^What is the load average on node (\S+)\?$"),
        lambda m: f"What load is {m.group(1)} carrying?",
        "load",
    ),
    (
        re.compile(r"^Is load increasing or decreasing on node (\S+)\?$"),
        lambda m: f"Which way is {m.group(1)}'s load trending?",
        "load",
    ),
    (
        re.compile(r"^Are any processes blocked on node (\S+)\?$"),
        lambda m: f"Has anything stalled in {m.group(1)}'s run queue?",
        "process",
    ),
    (
        re.compile(r"^What is the temperature on node (\S+)\?$"),
        lambda m: f"How hot is {m.group(1)} running?",
        "temperature",
    ),
    (
        re.compile(
            r"^(?:What is the CPU usage|How much CPU is being used|Is CPU utilisation high|What is the CPU load on this node|Tell me about CPU usage|How busy is the processor|What's the compute usage|Give me the CPU stats|How are the cores doing) on node ([\d\.]+)\??$"
        ),
        lambda m: f"How hard is {m.group(1)}'s processor working at the moment?",
        "cpu",
    ),
    (
        re.compile(
            r"^(?:What is the memory usage|How much memory is being used|Is memory under pressure|How much RAM is available|Tell me about memory usage|What's the memory utilisation|How much memory is free|Are we running low on memory) on node ([\d\.]+)\??$"
        ),
        lambda m: f"What's the RAM picture for {m.group(1)} right now?",
        "memory",
    ),
    (
        re.compile(
            r"^(?:Is the cluster healthy|Is everything running normally|Are all nodes OK|Give me a cluster overview|Summarize the cluster status|What's the overall system health|Is there anything wrong)\??$"
        ),
        lambda m: "From a 30,000-foot view, is anything off across the fleet?",
        "health",
    ),
    (
        re.compile(
            r"^(?:Give me a status summary|What's the overall cluster state|How is the system doing|System health report|Any issues to worry about)\??$"
        ),
        lambda m: "Wrap up the cluster's vibe in one go.",
        "overall",
    ),
]


def paraphrase(text):
    s = text.strip()
    for pat, fn, cat in PATTERNS:
        m = pat.match(s)
        if m:
            return fn(m), cat, "pattern"
    return f"Quick question: {s.rstrip('?').lower()}, can you tell?", None, "fallback"


paras, para_cat, para_src, para_kind = [], [], [], []
for q, cat in zip(vocab, vocab_cat):
    p, pcat, kind = paraphrase(q)
    paras.append(p)
    para_cat.append(cat)
    para_src.append(q)
    para_kind.append(kind)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ev = np.asarray(model.encode(vocab, normalize_embeddings=True, show_progress_bar=False))
ep = np.asarray(model.encode(paras, normalize_embeddings=True, show_progress_bar=False))

sims = ep @ ev.T
top1 = sims.argmax(axis=1)

hit_cat = hit_cat_pat = hit_cat_fb = 0
n_pat = n_fb = 0
cos_to_src = []


def health_overall_merge(c):
    return "health" if c in ("health", "overall") else c


for i, t in enumerate(top1):
    same = health_overall_merge(vocab_cat[t]) == health_overall_merge(para_cat[i])
    hit_cat += same
    if para_kind[i] == "pattern":
        n_pat += 1
        hit_cat_pat += same
    else:
        n_fb += 1
        hit_cat_fb += same
    cos_to_src.append(sims[i, vocab.index(para_src[i])])

print(f"vocab={len(vocab)} paraphrases={len(paras)} (pattern={n_pat}, fallback={n_fb})")
print(f"top-1 nearest-template SAME-CATEGORY retrieval: {hit_cat / len(paras):.1%}")
print(f"  pattern rewrites : {hit_cat_pat / max(1, n_pat):.1%}")
print(f"  generic fallback : {hit_cat_fb / max(1, n_fb):.1%}")
print(
    f"cos(paraphrase, its source template): mean={np.mean(cos_to_src):.3f} "
    f"min={np.min(cos_to_src):.3f}"
)

print("\nmisses (paraphrase -> retrieved template):")
for i, t in enumerate(top1):
    if health_overall_merge(vocab_cat[t]) != health_overall_merge(para_cat[i]):
        print(
            f"  [{para_cat[i]}->{vocab_cat[t]}] '{paras[i][:60]}' -> '{vocab[t][:60]}'"
        )
