"""Task 2 — design-requirement category identification experiment.

For each scenario in task2_category_identification.jsonl:
  1. Pipeline retrieves Top-5 chunks from MIL-HDBK-470A (shared 709-page corpus).
  2. A small classifier prompt asks qwen-turbo to output the set of
     design-technique categories (subset of 12) from the retrieved context.
  3. Metric:
       Category Precision/Recall/F1 (per item, macro over items),
       Multi-Category Accuracy (strict set equality),
       Source Traceability (does the LLM cite chunk_id that covers gt source).

Baselines: Naive RAG, Dense RAG, StructStdRAG — same LLM back-end.

Output: results/t2/task2_results.json + task2_per_item.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MAINTQA_DIR, RESULTS_DIR
from src.data.loader import Document
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("Task2")

ALL_CATEGORIES = [
    "Accessibility", "Interchangeability", "Error-proofing", "Maintenance Safety",
    "Diagnostics", "Ergonomics", "Repairability", "Reduced Maintenance",
    "Reduced Skill", "Time Parameters", "Diagnostic Parameters", "Preventive Maintenance",
]

CLASSIFY_PROMPT = """You are a maintainability engineering expert. Given an equipment design proposal description and 3–5 maintainability handbook retrieval results, identify the design requirement categories involved in this proposal.

Candidate categories (12 types):
{cats}

Design Proposal:
{scenario}

Retrieved Maintainability Handbook Excerpts:
{ctx}

Carefully analyze which categories each requirement in the proposal belongs to. Output only a JSON array (no explanation):
["Category A", "Category B", ...]

Output:"""


def load_docs() -> list[Document]:
    pages = Path("data/mil470a/mil470a_pages_clean.jsonl")
    docs = []
    with open(pages, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text.strip()) > 100:
                docs.append(Document(
                    doc_id=f"mil470a_page_{i + 1}",
                    title=f"MIL-HDBK-470A Page {i + 1}",
                    content=text,
                    source_path=str(pages),
                    doc_type="jsonl",
                ))
    return docs


def load_items() -> list[dict]:
    f = MAINTQA_DIR / "task2_category_identification.jsonl"
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.S)


def parse_llm_categories(out: str) -> list[str]:
    """Parse the JSON array of category names from LLM output, tolerant to fences."""
    if not out:
        return []
    m = _JSON_ARRAY_RE.search(out)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    clean = []
    for s in arr:
        if not isinstance(s, str):
            continue
        s = s.strip()
        # fuzzy match to canonical name
        for cat in ALL_CATEGORIES:
            if s.lower() == cat.lower() or cat.lower() in s.lower() or s.lower() in cat.lower():
                if cat not in clean:
                    clean.append(cat)
                break
    return clean


def metrics(pred: list[str], gt: list[str]) -> dict:
    pset, gset = set(pred), set(gt)
    tp = len(pset & gset)
    p = tp / len(pset) if pset else 0.0
    r = tp / len(gset) if gset else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    exact = 1.0 if pset == gset else 0.0
    return {"precision": p, "recall": r, "f1": f1, "exact": exact}


def eval_pipeline(name: str, pipeline, docs, items, llm) -> dict:
    logger.info(f"▶ {name}: ingesting {len(docs)} docs")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    ps, rs, f1s, exs = [], [], [], []
    per_item = []
    for item in items:
        try:
            retrieved = pipeline.retriever.retrieve(item["query"], top_k=5)
            ctx = "\n\n".join(
                f"[chunk_{i}] {r.chunk.content[:400]}" for i, r in enumerate(retrieved)
            )
        except Exception as e:
            logger.warning(f"retrieval fail on {item['id']}: {e}")
            retrieved, ctx = [], ""

        prompt = CLASSIFY_PROMPT.format(
            cats="\n".join(f"  - {c}" for c in ALL_CATEGORIES),
            scenario=item["query"],
            ctx=ctx,
        )
        try:
            out = llm.generate(prompt)
        except Exception as e:
            logger.warning(f"llm fail on {item['id']}: {e}")
            out = ""
        pred = parse_llm_categories(out)
        m = metrics(pred, item["gt_categories"])
        ps.append(m["precision"]); rs.append(m["recall"])
        f1s.append(m["f1"]); exs.append(m["exact"])
        per_item.append({
            "id": item["id"], "gt": item["gt_categories"], "pred": pred,
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "exact": m["exact"],
        })

    n = len(items)
    ci_f1 = bootstrap_ci(f1s, n_bootstrap=2000)
    ci_p = bootstrap_ci(ps, n_bootstrap=2000)
    ci_r = bootstrap_ci(rs, n_bootstrap=2000)
    return {
        "n": n,
        "Category Precision": sum(ps) / n,
        "Precision CI": [ci_p["ci_lower"], ci_p["ci_upper"]],
        "Category Recall": sum(rs) / n,
        "Recall CI": [ci_r["ci_lower"], ci_r["ci_upper"]],
        "Category F1": sum(f1s) / n,
        "F1 CI": [ci_f1["ci_lower"], ci_f1["ci_upper"]],
        "Exact Match": sum(exs) / n,
        "_raw_f1": f1s,
        "_raw_r": rs,
        "_raw_p": ps,
        "_per_item": per_item,
    }


def main():
    out_dir = RESULTS_DIR / "t2"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    items = load_items()
    logger.info(f"Loaded {len(docs)} docs, {len(items)} items")

    llm = LLMClient(name="qwen-turbo", temperature=0.0, max_tokens=512)

    pipelines = {
        "Naive RAG": NaiveRAGPipeline(),
        "Dense RAG": DenseRAGPipeline(),
        "StructStdRAG": StructStdRAGPipeline(),
    }

    results = {}
    for name, p in pipelines.items():
        results[name] = eval_pipeline(name, p, docs, items, llm)

    # paired tests
    stats = []
    maint_f1 = results["StructStdRAG"]["_raw_f1"]
    maint_r = results["StructStdRAG"]["_raw_r"]
    for name in ["Naive RAG", "Dense RAG"]:
        bp_f1 = bootstrap_paired_test(maint_f1, results[name]["_raw_f1"])
        bp_r = bootstrap_paired_test(maint_r, results[name]["_raw_r"])
        stats.append({
            "comparison": f"StructStdRAG vs {name}",
            "f1_diff": bp_f1["mean_diff"],
            "f1_p": bp_f1["p_value"],
            "recall_diff": bp_r["mean_diff"],
            "recall_p": bp_r["p_value"],
        })

    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in results.items()}
    with open(out_dir / "task2_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": clean, "stats": stats,
                   "task": "category_identification", "n": len(items)},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / "task2_per_item.jsonl", "w", encoding="utf-8") as f:
        for name, m in results.items():
            for p in m["_per_item"]:
                p["method"] = name
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("Task 2 — Design-Requirement Category Identification")
    print("=" * 80)
    print(f"{'Method':<14} {'Precision':>12} {'Recall':>12} {'F1':>18} {'Exact':>8}")
    print("-" * 80)
    for name, m in results.items():
        ci = m["F1 CI"]
        print(f"{name:<14} {m['Category Precision']:>10.3f}  "
              f"{m['Category Recall']:>10.3f}  "
              f"{m['Category F1']:>6.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"{m['Exact Match']:>6.3f}")
    print("\nStatistical tests:")
    for s in stats:
        print(f"  {s['comparison']}: ΔF1={s['f1_diff']:+.3f} p={s['f1_p']:.3f};  "
              f"ΔRecall={s['recall_diff']:+.3f} p={s['recall_p']:.3f}")


if __name__ == "__main__":
    main()
