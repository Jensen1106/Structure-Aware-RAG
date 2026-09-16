"""Task 3 — open-ended maintainability design suggestion generation.

Pipeline: Naive RAG / Dense RAG / StructStdRAG produce N suggestions per scenario
(retrieve from MIL-HDBK-470A). Then qwen-max does blind 5-dimension Likert
scoring over anonymised method labels M1/M2/M3.

Dimensions (1–5 scale): coverage, correctness, source_grounding,
actionability, practicality.

Output: results/t3/task3_results.json + task3_per_item.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
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

logger = get_logger("Task3")

DIMS = ["coverage", "correctness", "source_grounding", "actionability", "practicality"]

JUDGE_PROMPT = """You are a senior maintainability engineering review expert. Please score the following design suggestions on a scale of 1–5 (integer) across the five dimensions below.

Scoring Dimensions (1–5 each):
- coverage (Coverage): How many maintainability design categories the suggestions cover (1=single category, 5=covers 5+ categories)
- correctness (Correctness): Factual correctness of the suggestions (1=mostly incorrect, 5=fully correct)
- source_grounding (Source Grounding): Accuracy of cited standard clauses (1=no/misattributed citations, 5=all accurately traceable)
- actionability (Actionability): Concreteness of the suggestions (1=vague principles, 5=specific to numerical values/methods)
- practicality (Practicality): Usability for direct engineering adoption (1=unusable, 5=directly executable)

Equipment Scenario:
{scenario}

Design Suggestions:
{answer}

Please output JSON only, no explanation:
{{"coverage": X, "correctness": X, "source_grounding": X, "actionability": X, "practicality": X}}

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
    f = MAINTQA_DIR / "task3_open_generation.jsonl"
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def _normalize(answer: str) -> str:
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


def gen_outputs(pipeline, docs, items, name) -> list[dict]:
    logger.info(f"▶ {name}: ingesting {len(docs)} docs")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    outs = []
    for it in items:
        try:
            res = pipeline.run(it["query"])
            ans = _normalize(res.generation.answer)
        except Exception as e:
            logger.warning(f"{name} failed on {it['id']}: {e}")
            ans = ""
        outs.append({"id": it["id"], "method": name, "answer": ans[:3000]})
    return outs


_SCORE_RE = re.compile(r"\{[^{}]+\}", re.S)


def parse_scores(out: str) -> dict:
    if not out:
        return {d: 1 for d in DIMS}
    m = _SCORE_RE.search(out)
    if not m:
        return {d: 1 for d in DIMS}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {d: 1 for d in DIMS}
    clean = {}
    for d in DIMS:
        v = obj.get(d, 1)
        try:
            v = int(round(float(v)))
        except Exception:
            v = 1
        v = max(1, min(5, v))
        clean[d] = v
    return clean


def judge_blind(answers_by_method: dict[str, list[dict]], items: list[dict],
                judge: LLMClient, seed: int = 42) -> dict:
    """For each item, shuffle method order → judge → store mapping back."""
    rng = random.Random(seed)
    scores_by_method = {m: {d: [] for d in DIMS} for m in answers_by_method}
    per_item_scores = []

    # index answers by (method, id)
    a_idx = {}
    for m, lst in answers_by_method.items():
        for rec in lst:
            a_idx[(m, rec["id"])] = rec["answer"]

    methods = list(answers_by_method.keys())
    for it in items:
        ordered = methods[:]
        rng.shuffle(ordered)
        for alias_i, real in enumerate(ordered):
            label = f"M{alias_i + 1}"
            ans = a_idx.get((real, it["id"]), "")
            prompt = JUDGE_PROMPT.format(
                scenario=f"{it['title']} — {it['scenario_desc']}",
                answer=f"(Output from System {label})\n{ans}",
            )
            try:
                out = judge.generate(prompt)
            except Exception as e:
                logger.warning(f"judge fail {real}/{it['id']}: {e}")
                out = ""
            s = parse_scores(out)
            for d in DIMS:
                scores_by_method[real][d].append(s[d])
            per_item_scores.append({
                "id": it["id"], "method": real, "alias": label,
                **s,
            })
    return {"scores": scores_by_method, "per_item": per_item_scores}


def summarise(scores: dict[str, list[int]]) -> dict:
    n = len(scores[DIMS[0]])
    row = {"n": n}
    for d in DIMS:
        vals = scores[d]
        row[d] = sum(vals) / n if n else 0.0
        ci = bootstrap_ci(vals, n_bootstrap=2000)
        row[f"{d}_ci"] = [ci["ci_lower"], ci["ci_upper"]]
    row["composite"] = sum(row[d] for d in DIMS) / len(DIMS)
    return row


def main():
    out_dir = RESULTS_DIR / "t3"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    items = load_items()
    logger.info(f"Loaded {len(docs)} docs, {len(items)} items")

    pipelines = {
        "Naive RAG": NaiveRAGPipeline(),
        "Dense RAG": DenseRAGPipeline(),
        "StructStdRAG": StructStdRAGPipeline(),
    }

    outputs = {}
    for name, p in pipelines.items():
        outputs[name] = gen_outputs(p, docs, items, name)

    # save raw answers
    with open(out_dir / "task3_answers.jsonl", "w", encoding="utf-8") as f:
        for name, lst in outputs.items():
            for rec in lst:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    judge = LLMClient(name="qwen-max", temperature=0.0, max_tokens=256)
    logger.info("▶ qwen-max blind judging...")
    res = judge_blind(outputs, items, judge)

    summary = {m: summarise(s) for m, s in res["scores"].items()}

    # paired tests on composite
    stats = []
    for name in ["Naive RAG", "Dense RAG"]:
        # composite per-item
        def composite(method):
            n = len(res["scores"][method][DIMS[0]])
            return [sum(res["scores"][method][d][i] for d in DIMS) / len(DIMS)
                    for i in range(n)]
        bp = bootstrap_paired_test(composite("StructStdRAG"), composite(name))
        stats.append({
            "comparison": f"StructStdRAG vs {name}",
            "composite_diff": bp["mean_diff"],
            "composite_p": bp["p_value"],
        })

    with open(out_dir / "task3_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "stats": stats, "n": len(items), "dims": DIMS},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / "task3_per_item.jsonl", "w", encoding="utf-8") as f:
        for rec in res["per_item"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print("Task 3 — Open Generation (qwen-max blind 5-dim judge, n=20)")
    print("=" * 90)
    hdr = f"{'Method':<12}" + "".join(f"{d:>10}" for d in DIMS) + f"{'Comp':>10}"
    print(hdr)
    for m, row in summary.items():
        line = f"{m:<12}" + "".join(f"{row[d]:>10.2f}" for d in DIMS) + f"{row['composite']:>10.2f}"
        print(line)
    print("\nStatistical tests:")
    for s in stats:
        print(f"  {s['comparison']}: Δcomposite={s['composite_diff']:+.3f} p={s['composite_p']:.3f}")


if __name__ == "__main__":
    main()
