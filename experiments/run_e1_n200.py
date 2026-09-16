"""E1-n200: Primary experiment on the 200-item independent test set.

test_mil470a_200 is disjoint from train_mil470a_highconf (and from
test_mil470a_88). This addresses the reviewer concern about the 88-item
test being small and potentially sharing source_ids with train.

Methods: Naive RAG, Dense RAG, StructStdRAG (3 primary methods only to
keep runtime bounded; we refer to full 8-baseline E1-n88 for the
extended comparison).

Output: results/e1_n200/e1_n200_results.json + per_item.jsonl
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
from src.evaluation.metrics import (
    exact_match,
    extract_answer_tuples,
    partial_match,
    tuple_f1,
)
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.utils.logger import get_logger

logger = get_logger("E1-n200")

NUMERIC_RE = re.compile(r"(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|%|°)")


def is_hallucinated(pred: str, retrieved: list) -> bool:
    if not pred:
        return True
    pred_nums = set(m[0] for m in NUMERIC_RE.findall(pred))
    if not pred_nums:
        return False
    combined = " ".join(r.chunk.content for r in retrieved[:3])
    return any(n not in combined for n in pred_nums)


def source_traceability(pred: str, retrieved: list) -> float:
    if not pred or not retrieved:
        return 0.0
    tokens = re.findall(r"[a-zA-Z0-9]+", pred.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return 0.0
    combined = " ".join(r.chunk.content for r in retrieved[:3]).lower()
    return sum(1 for t in tokens if t in combined) / len(tokens)


def _normalize(answer: str) -> str:
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


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
    return [json.loads(l) for l in open(MAINTQA_DIR / "test_mil470a_200.jsonl", encoding="utf-8") if l.strip()]


def eval_pipeline(name, pipeline, docs, items):
    logger.info(f"▶ {name}: ingesting {len(docs)} docs")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    em_s, pm_s, tf_s, hal_s, src_s = [], [], [], [], []
    per_item = []
    for i, item in enumerate(items):
        if i and i % 50 == 0:
            logger.info(f"  progress {i}/{len(items)}")
        try:
            res = pipeline.run(item["query"])
            pred = _normalize(res.generation.answer)
            retrieved = res.retrieved
        except Exception as e:
            logger.warning(f"  fail {item['id']}: {e}")
            pred, retrieved = "", []
        gold = item["answer"]
        em = exact_match(pred, gold)
        pm = partial_match(pred, gold)
        tf = tuple_f1(
            extract_answer_tuples(pred, parameter=item.get("constraint_type", "")),
            extract_answer_tuples(gold, parameter=item.get("constraint_type", "")),
        )
        hal = 1 if is_hallucinated(pred, retrieved) else 0
        src = source_traceability(pred, retrieved)
        em_s.append(em); pm_s.append(pm); tf_s.append(tf); hal_s.append(hal); src_s.append(src)
        per_item.append({
            "id": item["id"], "constraint_type": item.get("constraint_type"),
            "difficulty": item.get("difficulty"),
            "pred": pred[:400], "gold": gold[:200],
            "em": em, "pm": pm, "tuple_f1": tf, "hal": hal, "src": src,
        })

    n = len(items)
    ci_tf = bootstrap_ci(tf_s, n_bootstrap=2000)
    ci_pm = bootstrap_ci(pm_s, n_bootstrap=2000)
    ci_hal = bootstrap_ci(hal_s, n_bootstrap=2000)
    return {
        "n": n,
        "Exact Match": sum(em_s) / n,
        "Partial Match": sum(pm_s) / n,
        "Partial Match CI": [ci_pm["ci_lower"], ci_pm["ci_upper"]],
        "Tuple-F1": sum(tf_s) / n,
        "Tuple-F1 CI": [ci_tf["ci_lower"], ci_tf["ci_upper"]],
        "Hallucination Rate": sum(hal_s) / n,
        "Hallucination Rate CI": [ci_hal["ci_lower"], ci_hal["ci_upper"]],
        "Source Traceability": sum(src_s) / n,
        "_raw_tf": tf_s,
        "_raw_pm": pm_s,
        "_raw_hal": hal_s,
        "_per_item": per_item,
    }


def main():
    out_dir = RESULTS_DIR / "e1_n200"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    items = load_items()
    logger.info(f"Loaded {len(docs)} docs, {len(items)} items")

    pipelines = {
        "Naive RAG": NaiveRAGPipeline(),
        "Dense RAG": DenseRAGPipeline(),
        "StructStdRAG": StructStdRAGPipeline(),
    }

    results = {}
    for name, p in pipelines.items():
        results[name] = eval_pipeline(name, p, docs, items)

    # paired tests
    stats = []
    for name in ["Naive RAG", "Dense RAG"]:
        bp_tf = bootstrap_paired_test(results["StructStdRAG"]["_raw_tf"], results[name]["_raw_tf"])
        bp_hal = bootstrap_paired_test(results["StructStdRAG"]["_raw_hal"], results[name]["_raw_hal"])
        bp_pm = bootstrap_paired_test(results["StructStdRAG"]["_raw_pm"], results[name]["_raw_pm"])
        stats.append({
            "comparison": f"StructStdRAG vs {name}",
            "tuple_f1_diff": bp_tf["mean_diff"],
            "tuple_f1_ci": [bp_tf["ci_lower"], bp_tf["ci_upper"]],
            "tuple_f1_p": bp_tf["p_value"],
            "hal_diff": bp_hal["mean_diff"],
            "hal_p": bp_hal["p_value"],
            "pm_diff": bp_pm["mean_diff"],
            "pm_p": bp_pm["p_value"],
        })

    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in results.items()}
    with open(out_dir / "e1_n200_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": clean, "stats": stats, "n": len(items)},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / "e1_n200_per_item.jsonl", "w", encoding="utf-8") as f:
        for name, m in results.items():
            for p in m["_per_item"]:
                p["method"] = name
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print(f"E1 (n=200 independent test set, disjoint from train + test_88)")
    print("=" * 90)
    print(f"{'Method':<14} {'Tuple-F1':>18} {'PM':>14} {'Hal.Rate':>14} {'SrcTrace':>10}")
    for name, m in results.items():
        ci = m["Tuple-F1 CI"]
        pmci = m["Partial Match CI"]
        halci = m["Hallucination Rate CI"]
        print(f"{name:<14} {m['Tuple-F1']:>6.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"{m['Partial Match']:>4.3f} [{pmci[0]:.3f},{pmci[1]:.3f}]  "
              f"{m['Hallucination Rate']:>4.3f} [{halci[0]:.3f},{halci[1]:.3f}]  "
              f"{m['Source Traceability']:>8.3f}")
    print("\nStatistical tests:")
    for s in stats:
        print(f"  {s['comparison']}:  ΔTF={s['tuple_f1_diff']:+.3f} p={s['tuple_f1_p']:.3f};  "
              f"ΔHal={s['hal_diff']:+.3f} p={s['hal_p']:.3f};  ΔPM={s['pm_diff']:+.3f} p={s['pm_p']:.3f}")


if __name__ == "__main__":
    main()
