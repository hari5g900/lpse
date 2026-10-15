#!/usr/bin/env python3
"""Summarize paraphrase-robustness experiment results into a comparison table."""

import csv
import json
import os
import sys

RESULT_DIR = os.getenv("RESULT_DIR", "artifacts/results/para_exp")


def load(path):
    if not os.path.isfile(path):
        return None
    return json.load(open(path))


def summary_row(prefix, variant):
    clean = load(os.path.join(RESULT_DIR, f"{variant}_clean.json"))
    para = load(os.path.join(RESULT_DIR, f"{variant}_para.json"))
    if not clean and not para:
        return None
    row = {"variant": variant}
    for suffix, src in [("clean", clean), ("para", para)]:
        if not src:
            row[f"{suffix}_jepa"] = None
            continue
        row[f"{suffix}_jepa"] = round(src["jepa"].get("accuracy", 0) * 100, 1)

    para_csv = os.path.join(RESULT_DIR, f"{variant}_para.csv")
    if os.path.isfile(para_csv):
        seen = {}
        for r in csv.DictReader(open(para_csv)):
            if r["question"] not in seen:
                seen[r["question"]] = r
        rows = list(seen.values())
        if rows:
            hits = sum(
                1 for r in rows if r.get("jepa_correct") in ("1", "True", "true")
            )
            row["unique_para_jepa"] = round(hits / len(rows) * 100, 1)
            row["unique_n"] = len(rows)
    return row


variants = sys.argv[1:] or ["baseline", "A", "B", "C", "D", "ALL"]
print(
    f"{'variant':10s} {'clean':>7s} {'para':>7s} {'uniq-para':>10s}  delta(clean-para)"
)
for v in variants:
    row = summary_row(RESULT_DIR, v)
    if row is None:
        print(f"{v:10s}  (no results yet)")
        continue
    c = row.get("clean_jepa")
    p = row.get("para_jepa")
    u = row.get("unique_para_jepa")
    d = (float(c) - float(p)) if (c is not None and p is not None) else None
    delta = f"{d:.1f}" if d is not None else "n/a"
    print(f"{v:10s} {str(c):>7s} {str(p):>7s} {str(u):>10s}  {delta}")
