"""CatRetriever β / γ ablation — maps out the boundary condition.

We sweep (β, γ) ∈ {(0,0), (0.1,-0.05), (0.3,-0.2), (0.5,-0.3)} on:
  • test_mil470a_88 — Task 1 single-clause retrieval (StructStdRAG's design target)
  • task2_category_identification — multi-label scenario
  (both shared MIL-HDBK-470A 709-page index)

Objective: show that StructStdRAG's category re-ranking is a **controllable
trade-off** (β=0 recovers pure Dense+BM25; higher β helps Task 1 but
hurts Task 2 multi-label). This converts the Task 2 negative result
into a honestly-discussed, parameterised boundary.

Output: results/ablation_bgamma/bgamma_results.json
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
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("AblBGamma")

ALL_CATEGORIES = [
    "Accessibility", "Interchangeability", "Error-proofing", "Maintenance Safety",
    "Diagnostics", "Ergonomics", "Repairability", "Reduced Maintenance",
    "Reduced Skill", "Time Parameters", "Diagnostic Parameters", "Preventive Maintenance",
]

SETTINGS = [
    # (beta, type_boost, type_penalty, label)
    (0.0, 0.0,  0.0,  "off"),       # no category signal at all
    (0.1, 0.05, 0.02, "weak"),
    (0.3, 0.3,  0.2,  "default"),
    (0.5, 0.5,  0.3,  "strong"),
]

CLASSIFY_PROMPT = """You are a maintainability engineering expert. Given an equipment design proposal description and 3–5 maintainability handbook retrieval results, identify the design requirement categories involved in this proposal.

Candidate categories (12 types):
{cats}

Design Proposal:
{scenario}

Retrieved Handbook Excerpts:
{ctx}

Please output a JSON array (no explanation):
["Category A", "Category B", ...]

Output:"""

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.S)


def parse_categories(out: str) -> list[str]:
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
        for cat in ALL_CATEGORIES:
            if s.lower() == cat.lower() or cat.lower() in s.lower() or s.lower() in cat.lower():
                if cat not in clean:
                    clean.append(cat)
                break
    return clean


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


def _normalize(answer: str) -> str:
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


def run_task1(pipeline, items):
    tfs, pms = [], []
    for item in items:
        try:
            res = pipeline.run(item["query"])
            pred = _normalize(res.generation.answer)
        except Exception as e:
            logger.warning(f"t1 fail {item['id']}: {e}")
            pred = ""
        gold = item["answer"]
        tfs.append(tuple_f1(
            extract_answer_tuples(pred, parameter=item.get("constraint_type", "")),
            extract_answer_tuples(gold, parameter=item.get("constraint_type", "")),
        ))
        pms.append(partial_match(pred, gold))
    n = len(items)
    return {"tuple_f1": sum(tfs) / n, "partial_match": sum(pms) / n, "n": n}


def run_task2(pipeline, items, llm):
    ps, rs, f1s = [], [], []
    for item in items:
        try:
            retrieved = pipeline.retriever.retrieve(item["query"], top_k=5)
            ctx = "\n\n".join(f"[chunk_{i}] {r.chunk.content[:400]}"
                              for i, r in enumerate(retrieved))
        except Exception as e:
            logger.warning(f"t2 retr fail {item['id']}: {e}")
            ctx = ""
        prompt = CLASSIFY_PROMPT.format(
            cats="\n".join(f"  - {c}" for c in ALL_CATEGORIES),
            scenario=item["query"], ctx=ctx,
        )
        try:
            out = llm.generate(prompt)
        except Exception:
            out = ""
        pred = parse_categories(out)
        gt = item["gt_categories"]
        tp = len(set(pred) & set(gt))
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt) if gt else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f1)
    n = len(items)
    return {
        "precision": sum(ps) / n, "recall": sum(rs) / n, "f1": sum(f1s) / n,
        "n": n,
    }


def main():
    out_dir = RESULTS_DIR / "ablation_bgamma"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    t1_items = [json.loads(l) for l in open(MAINTQA_DIR / "test_mil470a_88.jsonl", encoding="utf-8") if l.strip()]
    t2_items = [json.loads(l) for l in open(MAINTQA_DIR / "task2_category_identification.jsonl", encoding="utf-8") if l.strip()]

    logger.info(f"Task1 items: {len(t1_items)}; Task2 items: {len(t2_items)}")

    llm = LLMClient(name="qwen-turbo", temperature=0.0, max_tokens=512)

    all_results = []
    pipeline = StructStdRAGPipeline()
    pipeline.ingest(docs)
    logger.info("Pipeline ingested once — β/γ sweep without re-indexing.")

    for beta, tboost, tpen, label in SETTINGS:
        logger.info(f"▶ setting={label}  β={beta}  type_boost={tboost}  type_pen={tpen}")
        pipeline.retriever.beta = beta
        pipeline.retriever.type_boost = tboost
        pipeline.retriever.type_penalty = tpen

        t0 = time.time()
        t1 = run_task1(pipeline, t1_items)
        t1_time = time.time() - t0

        t0 = time.time()
        t2 = run_task2(pipeline, t2_items, llm)
        t2_time = time.time() - t0

        rec = {
            "setting": label, "beta": beta,
            "type_boost": tboost, "type_penalty": tpen,
            "task1": t1, "task2": t2,
            "task1_time_s": round(t1_time, 1),
            "task2_time_s": round(t2_time, 1),
        }
        all_results.append(rec)
        logger.info(f"  Task1 TF={t1['tuple_f1']:.3f}  Task2 F1={t2['f1']:.3f}")

    with open(out_dir / "bgamma_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("CatRetriever β/γ sweep")
    print("=" * 90)
    print(f"{'setting':<10} {'β':>6} {'boost':>6} {'pen':>6}   "
          f"{'T1 TF':>8} {'T1 PM':>8}   {'T2 P':>6} {'T2 R':>6} {'T2 F1':>6}")
    for r in all_results:
        t1, t2 = r["task1"], r["task2"]
        print(f"{r['setting']:<10} {r['beta']:>6.2f} {r['type_boost']:>6.2f} "
              f"{r['type_penalty']:>6.2f}   {t1['tuple_f1']:>8.3f} {t1['partial_match']:>8.3f}   "
              f"{t2['precision']:>6.3f} {t2['recall']:>6.3f} {t2['f1']:>6.3f}")
    print(f"\nSaved → {out_dir}/bgamma_results.json")


if __name__ == "__main__":
    main()
