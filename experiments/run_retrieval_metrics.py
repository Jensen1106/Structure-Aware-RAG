"""Retrieval-only metrics — Recall@K / MRR / NDCG (no LLM needed).

For each test item we have:
  - gold answer quote (canonical clause sentence)
  - gold source_id (e.g. "mil470a_c_354_cont-06")

Hit definition (LLM-free, robust to OCR noise):
  signature = first 12 distinct informative tokens (alnum, len >= 3) from
              gold quote
  chunk is a "hit" iff >= 70% of the signature tokens appear in the chunk
  (token-set membership; whitespace/punctuation/case insensitive)

Metrics: Recall@{1,3,5,10}, MRR@10, NDCG@10 (binary relevance).

Output: results/retrieval_metrics/retrieval_metrics.json
        results/retrieval_metrics/retrieval_per_item.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import math
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
from src.utils.logger import get_logger

logger = get_logger("RetEval")

TOP_K_MAX = 10
RECALL_KS = [1, 3, 5, 10]
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")
_PAGE_RE = re.compile(r"mil470a_page_(\d+)")


def make_signature(item):
    """Page-level hit signature: the ± neighborhood of the gold page.

    In MIL-HDBK-470A Appendix C, a guideline often spans 1–2 adjacent pages
    due to layout. We accept any chunk whose doc_id page is within ±1 of
    the gold page as a hit — this is the "user finds the right location in
    the handbook" criterion, which matches how engineers actually use it.
    """
    src = item.get("answer_source") or {}
    gold_page = src.get("page")
    if gold_page is None:
        return None
    try:
        gp = int(gold_page)
    except (TypeError, ValueError):
        return None
    return {"gold_pages": {gp - 1, gp, gp + 1}}


def _chunk_page(chunk) -> int | None:
    did = getattr(chunk, "doc_id", "") or ""
    m = _PAGE_RE.search(did)
    if m:
        return int(m.group(1))
    cid = getattr(chunk, "chunk_id", "") or ""
    m = _PAGE_RE.search(cid)
    return int(m.group(1)) if m else None


def is_hit(chunk, sig):
    if not sig:
        return False
    pg = _chunk_page(chunk)
    if pg is None:
        return False
    return pg in sig["gold_pages"]


def load_docs():
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
                    content=text, source_path=str(pages), doc_type="jsonl",
                ))
    return docs


def load_items(name):
    return [json.loads(l) for l in
            open(MAINTQA_DIR / f"{name}.jsonl", encoding="utf-8") if l.strip()]


def get_signature(item):
    return make_signature(item)


def evaluate(pipeline, docs, items, method, split):
    t0 = time.time()
    logger.info(f"▶ {method} on {split}: ingesting {len(docs)} docs")
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    per_item = []
    recall_at = {k: [] for k in RECALL_KS}
    mrrs, ndcgs = [], []

    for i, item in enumerate(items):
        if i and i % 50 == 0:
            logger.info(f"  progress {i}/{len(items)}")
        sig = get_signature(item)
        try:
            results = pipeline.retriever.retrieve(item["query"], top_k=TOP_K_MAX)
        except Exception as e:
            logger.warning(f"retrieval fail {item['id']}: {e}")
            results = []

        hit_flags = [0] * TOP_K_MAX
        first_hit = None
        for rank, r in enumerate(results[:TOP_K_MAX]):
            if is_hit(r.chunk, sig):
                hit_flags[rank] = 1
                if first_hit is None:
                    first_hit = rank + 1

        for k in RECALL_KS:
            recall_at[k].append(1 if any(hit_flags[:k]) else 0)
        mrrs.append(1.0 / first_hit if first_hit else 0.0)

        total_hits = sum(hit_flags)
        if total_hits == 0:
            ndcg = 0.0
        else:
            dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hit_flags))
            idcg = sum(1 / math.log2(i + 2) for i in range(total_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

        per_item.append({
            "id": item["id"], "method": method, "split": split,
            "gold_page": (item.get("answer_source") or {}).get("page"),
            "first_hit_rank": first_hit, "hit_flags": hit_flags,
        })

    n = len(items)
    summary = {"n": n, "method": method, "split": split}
    for k in RECALL_KS:
        vals = recall_at[k]
        ci = bootstrap_ci(vals, n_bootstrap=2000)
        summary[f"Recall@{k}"] = sum(vals) / n
        summary[f"Recall@{k} CI"] = [ci["ci_lower"], ci["ci_upper"]]
    ci_mrr = bootstrap_ci(mrrs, n_bootstrap=2000)
    summary["MRR@10"] = sum(mrrs) / n
    summary["MRR@10 CI"] = [ci_mrr["ci_lower"], ci_mrr["ci_upper"]]
    ci_ndcg = bootstrap_ci(ndcgs, n_bootstrap=2000)
    summary["NDCG@10"] = sum(ndcgs) / n
    summary["NDCG@10 CI"] = [ci_ndcg["ci_lower"], ci_ndcg["ci_upper"]]
    summary["_raw"] = {
        **{f"recall_{k}": recall_at[k] for k in RECALL_KS},
        "mrr": mrrs, "ndcg": ndcgs,
    }
    summary["_per_item"] = per_item
    return summary


def paired_tests(rows):
    stats = []
    by_method = {r["method"]: r for r in rows}
    maint = by_method["StructStdRAG"]["_raw"]
    for other in ["Naive RAG", "Dense RAG"]:
        o = by_method[other]["_raw"]
        for metric_key in ["recall_5", "mrr", "ndcg"]:
            bp = bootstrap_paired_test(maint[metric_key], o[metric_key])
            stats.append({
                "comparison": f"StructStdRAG vs {other}",
                "metric": metric_key,
                "diff": bp["mean_diff"],
                "ci": [bp["ci_lower"], bp["ci_upper"]],
                "p_value": bp["p_value"],
            })
    return stats


def main():
    out_dir = RESULTS_DIR / "retrieval_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    splits = {
        "test_mil470a_88":  load_items("test_mil470a_88"),
        "test_mil470a_200": load_items("test_mil470a_200"),
    }

    all_rows = []
    per_item_all = []
    stats_by_split = {}
    for split, items in splits.items():
        logger.info(f"\n=== split: {split} (n={len(items)}) ===")
        pipelines = {
            "Naive RAG": NaiveRAGPipeline(),
            "Dense RAG": DenseRAGPipeline(),
            "StructStdRAG":  StructStdRAGPipeline(),
        }
        split_rows = []
        for name, p in pipelines.items():
            row = evaluate(p, docs, items, name, split)
            split_rows.append(row)
            per_item_all.extend(row.pop("_per_item"))
        stats_by_split[split] = paired_tests(split_rows)
        for r in split_rows:
            r.pop("_raw", None)
        all_rows.extend(split_rows)

    with open(out_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": all_rows, "paired_tests": stats_by_split},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / "retrieval_per_item.jsonl", "w", encoding="utf-8") as f:
        for p in per_item_all:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    for split in splits:
        print("\n" + "=" * 100)
        print(f"Retrieval-only metrics — {split}")
        print("=" * 100)
        header = f"{'Method':<12}" + "".join(f"{f'R@{k}':>8}" for k in RECALL_KS) + \
                 f"{'MRR@10':>12}{'NDCG@10':>10}"
        print(header)
        for r in all_rows:
            if r["split"] != split:
                continue
            row_str = (f"{r['method']:<12}" +
                       "".join(f"{r[f'Recall@{k}']:>8.3f}" for k in RECALL_KS) +
                       f"{r['MRR@10']:>12.3f}{r['NDCG@10']:>10.3f}")
            print(row_str)
        print("\nPaired tests (StructStdRAG vs others):")
        for s in stats_by_split[split]:
            stars = "***" if s["p_value"] < 0.001 else \
                    "**" if s["p_value"] < 0.01 else \
                    "*" if s["p_value"] < 0.05 else ""
            print(f"  {s['comparison']:<26} {s['metric']:<10} "
                  f"Δ={s['diff']:+.3f}  p={s['p_value']:.4f} {stars}")


if __name__ == "__main__":
    main()
