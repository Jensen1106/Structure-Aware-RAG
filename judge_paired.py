"""Paired bootstrap test on LLM-Judge composite (bge-m3), StructStdRAG vs Naive/Dense."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.statistical_test import bootstrap_paired_test

DIMS = ["factual_correctness", "source_grounding", "constraint_accuracy", "overall_quality"]

# item_id -> method -> composite
by_item = defaultdict(dict)
for line in open("results/s1/llm_judge_detailed.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    sc = d["scores"]
    comp = sum(sc[k] for k in DIMS) / len(DIMS)
    by_item[d["item_id"]][d["method"]] = comp

methods = ["Naive RAG", "Dense RAG", "StructStdRAG"]
# aligned per-item vectors (only items with all 3 methods)
vecs = {m: [] for m in methods}
for it, mm in by_item.items():
    if all(m in mm for m in methods):
        for m in methods:
            vecs[m].append(mm[m])

n = len(vecs["StructStdRAG"])
print("paired items n =", n)
for m in methods:
    print("  %-10s composite mean = %.3f" % (m, sum(vecs[m]) / n))
print()
for b in ["Naive RAG", "Dense RAG"]:
    bp = bootstrap_paired_test(vecs["StructStdRAG"], vecs[b], n_bootstrap=10000)
    diff = bp["mean_diff"]
    p = bp["p_value"]
    # two-sided interpretation: is StructStdRAG different from baseline?
    print("StructStdRAG vs %-10s  dComposite = %+.3f  p = %.4f  CI=[%.3f, %.3f]" % (
        b, diff, p, bp["ci_lower"], bp["ci_upper"]))

# also per-dimension factual + overall (the two where StructStdRAG looks lower)
print()
for dim in ["factual_correctness", "overall_quality"]:
    dv = {m: [] for m in methods}
    for it, mm in by_item.items():
        # need raw dim per method; re-read
        pass
