"""Cross-LLM backend verification.

Replaces the generation LLM from qwen-turbo to deepseek-v3 (different
vendor from Alibaba's Qwen), and re-runs the 3 primary methods on
test_mil470a_88. Verifies the main conclusions are *not* dependent on
a single LLM family — directly addresses the "backend bias" reviewer
concern at Q2.

Output: results/cross_llm/cross_llm_results.json
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
from src.evaluation.metrics import extract_answer_tuples, partial_match, tuple_f1
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("CrossLLM")

NUMERIC_RE = re.compile(r"(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|%|°)")


def is_hallucinated(pred, retrieved):
    if not pred:
        return True
    pred_nums = set(m[0] for m in NUMERIC_RE.findall(pred))
    if not pred_nums:
        return False
    combined = " ".join(r.chunk.content for r in retrieved[:3])
    return any(n not in combined for n in pred_nums)


def _normalize(answer):
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


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


def load_items():
    return [json.loads(l) for l in open(MAINTQA_DIR / "test_mil470a_88.jsonl", encoding="utf-8") if l.strip()]


def eval_pipeline(name, pipeline, docs, items):
    logger.info(f"▶ {name}: ingesting")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    tfs, pms, hals = [], [], []
    for i, item in enumerate(items):
        if i and i % 30 == 0:
            logger.info(f"  {name} progress {i}/{len(items)}")
        try:
            res = pipeline.run(item["query"])
            pred = _normalize(res.generation.answer)
            retrieved = res.retrieved
        except Exception as e:
            logger.warning(f"  fail {item['id']}: {e}")
            pred, retrieved = "", []
        tfs.append(tuple_f1(
            extract_answer_tuples(pred, parameter=item.get("constraint_type", "")),
            extract_answer_tuples(item["answer"], parameter=item.get("constraint_type", "")),
        ))
        pms.append(partial_match(pred, item["answer"]))
        hals.append(1 if is_hallucinated(pred, retrieved) else 0)
    n = len(items)
    ci_tf = bootstrap_ci(tfs, n_bootstrap=2000)
    return {
        "n": n,
        "Tuple-F1": sum(tfs) / n,
        "Tuple-F1 CI": [ci_tf["ci_lower"], ci_tf["ci_upper"]],
        "Partial Match": sum(pms) / n,
        "Hallucination Rate": sum(hals) / n,
        "_tfs": tfs, "_pms": pms, "_hals": hals,
    }


def main():
    out_dir = RESULTS_DIR / "cross_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    items = load_items()

    # Cross-LLM: use deepseek-v3 as the generation model
    llm = LLMClient(name="deepseek-v3", temperature=0.0, max_tokens=2048)

    pipelines = {
        "Naive RAG": NaiveRAGPipeline(llm_client=llm),
        "Dense RAG": DenseRAGPipeline(llm_client=llm),
        "StructStdRAG":  StructStdRAGPipeline(llm_client=llm),
    }

    results = {}
    for name, p in pipelines.items():
        results[name] = eval_pipeline(name, p, docs, items)

    stats = []
    for name in ["Naive RAG", "Dense RAG"]:
        bp_tf = bootstrap_paired_test(results["StructStdRAG"]["_tfs"], results[name]["_tfs"])
        bp_hal = bootstrap_paired_test(results["StructStdRAG"]["_hals"], results[name]["_hals"])
        stats.append({
            "comparison": f"StructStdRAG vs {name}",
            "tuple_f1_diff": bp_tf["mean_diff"],
            "tuple_f1_p": bp_tf["p_value"],
            "hal_diff": bp_hal["mean_diff"],
            "hal_p": bp_hal["p_value"],
        })

    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in results.items()}
    with open(out_dir / "cross_llm_results.json", "w", encoding="utf-8") as f:
        json.dump({"backend": "deepseek-v3", "n": len(items),
                   "summary": clean, "stats": stats},
                  f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Cross-LLM verification (deepseek-v3 backend, test_mil470a_88, n=88)")
    print("=" * 80)
    print(f"{'Method':<12} {'Tuple-F1':>18} {'PM':>8} {'Hal.Rate':>10}")
    for name, m in results.items():
        ci = m["Tuple-F1 CI"]
        print(f"{name:<12} {m['Tuple-F1']:>6.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"{m['Partial Match']:>6.3f}  {m['Hallucination Rate']:>8.3f}")
    print("\nStatistical tests (vs StructStdRAG):")
    for s in stats:
        print(f"  {s['comparison']}:  ΔTF={s['tuple_f1_diff']:+.3f} p={s['tuple_f1_p']:.3f};  "
              f"ΔHal={s['hal_diff']:+.3f} p={s['hal_p']:.3f}")


if __name__ == "__main__":
    main()
