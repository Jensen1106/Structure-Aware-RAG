"""Stratified analysis of e1_n500 by gold-answer numeric content.

Splits the 500 items into two strata based on whether the gold answer
contains a (number + unit) tuple, then re-runs the paired bootstrap
test on each stratum. Rationale: Tuple-F1 only has discriminative
power when the gold answer contains numerical values; on free-text
clauses all methods score ~0.49 (noise floor).

Reads:  results/e1_n500/e1_n500_per_item.jsonl
        data/maintqa/test_mil470a_500.jsonl
Writes: results/e1_n500/e1_n500_numeric_stratified.json + .md
"""

from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MAINTQA_DIR, RESULTS_DIR
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test


NUMERIC_RE = re.compile(
    r"(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|hours?|minutes?|min|s|seconds?|%|°|degrees?|°C|kg/m|"
    r"in|inch|ft|lb|psi|bar|kpa|mpa|hz|khz|mhz|w|kw|v|kv|a|ma)\b",
    re.IGNORECASE,
)


def has_numeric(text: str) -> bool:
    """True if gold answer text contains any (number + recognised unit)."""
    if not text:
        return False
    return NUMERIC_RE.search(text) is not None


def load_per_item():
    p = RESULTS_DIR / "e1_n500" / "e1_n500_per_item.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    return by_method


def load_test500_items():
    p = MAINTQA_DIR / "test_mil470a_500.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def aggregate(items, key):
    vals = [it[key] for it in items]
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    ci = bootstrap_ci(vals, n_bootstrap=2000) if n else {"ci_lower": 0, "ci_upper": 0}
    return {"n": n, "mean": mean, "ci": [ci["ci_lower"], ci["ci_upper"]]}


def paired(items_a, items_b, key):
    a_by = {it["id"]: it[key] for it in items_a}
    b_by = {it["id"]: it[key] for it in items_b}
    common = sorted(set(a_by) & set(b_by))
    if not common:
        return {"n": 0, "diff": 0, "p": 1.0, "ci": [0, 0]}
    a = [a_by[i] for i in common]
    b = [b_by[i] for i in common]
    bp = bootstrap_paired_test(a, b, n_bootstrap=10000)
    return {"n": len(common), "diff": bp["mean_diff"], "p": bp["p_value"],
            "ci": [bp["ci_lower"], bp["ci_upper"]]}


def main():
    test_items = load_test500_items()
    numeric_ids = {iid for iid, it in test_items.items() if has_numeric(it.get("answer", ""))}
    text_ids = set(test_items) - numeric_ids
    print(f"test_500: numeric={len(numeric_ids)}, free_text={len(text_ids)}")

    by_method = load_per_item()
    methods = ["Naive RAG", "Dense RAG", "StructStdRAG"]

    out = {
        "stratification": {
            "rule": "gold answer contains (number + recognised unit) regex",
            "n_numeric": len(numeric_ids),
            "n_text": len(text_ids),
        },
        "strata": {},
    }

    for sname, id_set in [("numeric", numeric_ids), ("free_text", text_ids)]:
        rec = {"per_method": {}}
        per_method_items = {}
        for m in methods:
            sub = [it for it in by_method[m] if it["id"] in id_set]
            per_method_items[m] = sub
            rec["per_method"][m] = {
                "n": len(sub),
                "Tuple-F1": aggregate(sub, "tuple_f1"),
                "Partial Match": aggregate(sub, "pm"),
                "Source Traceability": aggregate(sub, "src"),
                "Hallucination Rate": aggregate(sub, "hal"),
            }
        rec["paired"] = {}
        for base in ["Naive RAG", "Dense RAG"]:
            rec["paired"][f"StructStdRAG_vs_{base.split()[0]}"] = {
                "Tuple-F1": paired(per_method_items["StructStdRAG"], per_method_items[base], "tuple_f1"),
                "Partial Match": paired(per_method_items["StructStdRAG"], per_method_items[base], "pm"),
                "Source Traceability": paired(per_method_items["StructStdRAG"], per_method_items[base], "src"),
                "Hallucination Rate": paired(per_method_items["StructStdRAG"], per_method_items[base], "hal"),
            }
        out["strata"][sname] = rec

    out_dir = RESULTS_DIR / "e1_n500"
    with open(out_dir / "e1_n500_numeric_stratified.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # markdown summary
    lines = ["# test_500 numeric / free-text stratified analysis", ""]
    lines.append(f"Stratification rule: gold answer contains (number + recognised unit).")
    lines.append(f"  numeric stratum: n={len(numeric_ids)}")
    lines.append(f"  free-text stratum: n={len(text_ids)}")
    lines.append("")
    for sname in ["numeric", "free_text"]:
        rec = out["strata"][sname]
        lines.append(f"## Stratum: {sname}")
        lines.append("")
        lines.append("| Method | n | Tuple-F1 [CI] | PM [CI] | SrcTrace | Hal Rate |")
        lines.append("|---|---|---|---|---|---|")
        for m in methods:
            r = rec["per_method"][m]
            tf = r["Tuple-F1"]; pm = r["Partial Match"]; st = r["Source Traceability"]; hr = r["Hallucination Rate"]
            lines.append(
                f"| {m} | {r['n']} | {tf['mean']:.4f} [{tf['ci'][0]:.4f}, {tf['ci'][1]:.4f}] | "
                f"{pm['mean']:.4f} [{pm['ci'][0]:.4f}, {pm['ci'][1]:.4f}] | "
                f"{st['mean']:.4f} | {hr['mean']:.4f} |"
            )
        lines.append("")
        lines.append("**Paired tests (StructStdRAG vs):**")
        for k, v in rec["paired"].items():
            tf = v["Tuple-F1"]; pm = v["Partial Match"]; st = v["Source Traceability"]; hr = v["Hallucination Rate"]
            lines.append(
                f"- {k}: ΔTF={tf['diff']:+.4f} (p={tf['p']:.4f}); "
                f"ΔPM={pm['diff']:+.4f} (p={pm['p']:.4f}); "
                f"ΔSrc={st['diff']:+.4f}; ΔHal={hr['diff']:+.4f} (p={hr['p']:.4f})"
            )
        lines.append("")

    md_path = out_dir / "e1_n500_numeric_stratified.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
