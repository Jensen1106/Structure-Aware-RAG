"""S3b: Cross-Document Generalization Experiment

Evaluates StructStdRAG + 2 baselines on GJB 451B (Chinese military standard
for maintainability terminology, different from the training domain MIL-HDBK-470A)
to demonstrate cross-document / cross-standard generalization.

Pipeline requires no retraining:
- Replace document source with GJB 451B
- Directly ingest + query 30 items from test_gjb451b_30.jsonl
- Compare StructStdRAG / Naive RAG / Dense RAG on Tuple-F1 / Hallucination Rate / Answer Coverage
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, MAINTQA_DIR
from src.data.loader import Document
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.evaluation.metrics import tuple_f1, extract_answer_tuples, partial_match
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.utils.logger import get_logger

logger = get_logger("S3b")

NUMERIC_RE = re.compile(r"(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|%|°)")


def is_hallucinated(pred: str, retrieved: list) -> bool:
    if not pred: return True
    pred_nums = set(m[0] for m in NUMERIC_RE.findall(pred))
    if not pred_nums: return False
    combined = " ".join(r.chunk.content for r in retrieved[:3])
    return any(n not in combined for n in pred_nums)


def answer_coverage(gold: str, retrieved: list) -> float:
    if not gold: return 1.0
    tokens = re.findall(r"[a-zA-Z0-9]+|[一-鿿]", gold.lower())
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens: return 1.0
    combined = " ".join(r.chunk.content for r in retrieved).lower()
    return sum(1 for t in tokens if t in combined) / len(tokens)


def _normalize(answer: str) -> str:
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


def load_gjb_docs() -> list[Document]:
    pages_file = Path("data/gjb451b/gjb451b_pages.jsonl")
    docs = []
    with open(pages_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if text and len(text.strip()) > 100:
                docs.append(Document(
                    doc_id=f"gjb451b_page_{i+1}",
                    title=f"GJB 451B Page {i+1}",
                    content=text,
                    source_path=str(pages_file),
                    doc_type="jsonl",
                ))
    logger.info(f"Loaded {len(docs)} GJB 451B pages")
    return docs


def load_test():
    test_file = MAINTQA_DIR / "test_gjb451b_30.jsonl"
    items = [json.loads(l) for l in open(test_file, encoding="utf-8") if l.strip()]
    logger.info(f"Loaded {len(items)} test items")
    return items


def eval_pipeline(pipeline, docs, items, name) -> dict:
    logger.info(f"▶ {name}: ingesting {len(docs)} docs...")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time()-t0:.0f}s")

    t_f1s, pms, hals, covs = [], [], [], []
    per_item = []
    for idx, item in enumerate(items):
        try:
            result = pipeline.run(item["query"])
            pred = _normalize(result.generation.answer)
            retrieved = result.retrieved
        except Exception as e:
            logger.warning(f"  failed on {item['id']}: {e}")
            pred, retrieved = "", []

        t_f1 = tuple_f1(
            extract_answer_tuples(pred, parameter=item.get("constraint_type", "")),
            extract_answer_tuples(item["answer"], parameter=item.get("constraint_type", "")),
        )
        pm = partial_match(pred, item["answer"])
        hal = 1 if is_hallucinated(pred, retrieved) else 0
        cov = answer_coverage(item["answer"], retrieved)
        t_f1s.append(t_f1)
        pms.append(pm)
        hals.append(hal)
        covs.append(cov)
        per_item.append({
            "id": item["id"],
            "pred": pred[:500],
            "gold": item["answer"][:200],
            "tuple_f1": t_f1, "hal": hal, "cov": cov,
            "constraint_type": item.get("constraint_type"),
        })

    n = len(items)
    ci_t = bootstrap_ci(t_f1s, n_bootstrap=2000)
    ci_cov = bootstrap_ci(covs, n_bootstrap=2000)
    return {
        "n": n,
        "Tuple-F1": sum(t_f1s) / n,
        "Tuple-F1 CI": [ci_t["ci_lower"], ci_t["ci_upper"]],
        "Partial Match": sum(pms) / n,
        "Hallucination Rate": sum(hals) / n,
        "Answer Coverage": sum(covs) / n,
        "Answer Coverage CI": [ci_cov["ci_lower"], ci_cov["ci_upper"]],
        "_raw_tuple": t_f1s,
        "_raw_cov": covs,
        "_per_item": per_item,
        "_elapsed_s": time.time() - t0,
    }


def main():
    out_dir = RESULTS_DIR / "s3"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_gjb_docs()
    items = load_test()

    pipelines = {
        "Naive RAG": NaiveRAGPipeline(),
        "Dense RAG": DenseRAGPipeline(),
        "StructStdRAG": StructStdRAGPipeline(),
    }

    results = {}
    for name, p in pipelines.items():
        m = eval_pipeline(p, docs, items, name)
        results[name] = m

    # Statistical tests: StructStdRAG vs others
    maint_t = results["StructStdRAG"]["_raw_tuple"]
    maint_c = results["StructStdRAG"]["_raw_cov"]
    stats = []
    for name in ["Naive RAG", "Dense RAG"]:
        bp_t = bootstrap_paired_test(maint_t, results[name]["_raw_tuple"])
        bp_c = bootstrap_paired_test(maint_c, results[name]["_raw_cov"])
        stats.append({
            "comparison": f"StructStdRAG vs {name}",
            "tuple_f1_diff": bp_t["mean_diff"],
            "tuple_f1_ci": [bp_t["ci_lower"], bp_t["ci_upper"]],
            "tuple_f1_p": bp_t["p_value"],
            "cov_diff": bp_c["mean_diff"],
            "cov_p": bp_c["p_value"],
        })

    # Save (strip raw fields)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in results.items()}

    with open(out_dir / "s3b_cross_document.json", "w", encoding="utf-8") as f:
        json.dump({"summary": clean, "stats": stats, "domain": "GJB 451B (zero-shot)"},
                  f, ensure_ascii=False, indent=2)

    # Per-item details
    with open(out_dir / "s3b_per_item.jsonl", "w", encoding="utf-8") as f:
        for name, m in results.items():
            for p in m["_per_item"]:
                p["method"] = name
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Print results
    print("\n" + "=" * 80)
    print("S3b: Cross-Document (GJB 451B Zero-Shot) Results")
    print("=" * 80)
    print(f"{'Method':<18} {'Tuple-F1':>14} {'PM':>8} {'Hal.Rate':>10} {'Cov':>8}")
    print("-" * 80)
    for name, m in results.items():
        ci = m["Tuple-F1 CI"]
        print(f"{name:<18} {m['Tuple-F1']:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"{m['Partial Match']:>6.3f} {m['Hallucination Rate']:>10.3f} "
              f"{m['Answer Coverage']:>8.3f}")

    print("\nStatistical tests:")
    for s in stats:
        print(f"  {s['comparison']}: Δ Tuple-F1 = {s['tuple_f1_diff']:+.3f} "
              f"[{s['tuple_f1_ci'][0]:+.3f},{s['tuple_f1_ci'][1]:+.3f}] "
              f"p={s['tuple_f1_p']:.3f}")

    print(f"\nResults saved to: {out_dir}/s3b_cross_document.json")


if __name__ == "__main__":
    main()
