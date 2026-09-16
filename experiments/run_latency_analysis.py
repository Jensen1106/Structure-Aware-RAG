"""Latency / token-cost analysis — E7.

For each primary pipeline (Naive / Dense / CRAG-lite / RankRAG-lite /
HiChunk-lite / StructStdRAG) we measure on test_mil470a_88:
  - ingest time (chunking + indexing)
  - per-query wall-time (retrieval + generation), median / mean / p95
  - total LLM calls per query
  - prompt + completion token estimate (4 chars/token approximation)

Total-cost per query and relative overhead vs Naive are reported.

Output: results/latency/latency_results.json + latency_per_query.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MAINTQA_DIR, RESULTS_DIR
from src.data.loader import Document
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.crag_lite_rag import CRAGLiteRAGPipeline
from src.pipeline.rankrag_lite_rag import RankRAGLiteRAGPipeline
from src.pipeline.hichunk_lite_rag import HiChunkLiteRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("Latency")


class InstrumentedLLMClient(LLMClient):
    """Wraps LLMClient to count calls and approximate tokens."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.reset()

    def reset(self):
        self._n_calls = 0
        self._prompt_chars = 0
        self._completion_chars = 0

    def generate(self, prompt):
        self._n_calls += 1
        self._prompt_chars += len(prompt or "")
        out = super().generate(prompt)
        self._completion_chars += len(out or "")
        return out


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
                    doc_id=f"mil470a_page_{i+1}",
                    title=f"p{i+1}", content=text,
                    source_path=str(pages), doc_type="jsonl",
                ))
    return docs


def load_items():
    return [json.loads(l) for l in
            open(MAINTQA_DIR / "test_mil470a_88.jsonl", encoding="utf-8") if l.strip()]


def measure(name, make_pipeline, docs, items):
    logger.info(f"▶ {name}")
    llm = InstrumentedLLMClient(temperature=0.0, max_tokens=2048)
    # ingest
    pipeline = make_pipeline(llm)
    t0 = time.perf_counter()
    llm.reset()
    pipeline.ingest(docs)
    ingest_s = time.perf_counter() - t0
    ingest_calls = llm._n_calls
    ingest_prompt = llm._prompt_chars
    ingest_completion = llm._completion_chars

    # per-query
    per_q_latency = []
    per_q_calls = []
    per_q_prompt = []
    per_q_completion = []
    per_rec = []
    for i, item in enumerate(items):
        llm.reset()
        t_q = time.perf_counter()
        try:
            pipeline.run(item["query"])
        except Exception as e:
            logger.warning(f"  fail {item['id']}: {e}")
        dt = time.perf_counter() - t_q
        per_q_latency.append(dt)
        per_q_calls.append(llm._n_calls)
        per_q_prompt.append(llm._prompt_chars)
        per_q_completion.append(llm._completion_chars)
        per_rec.append({"id": item["id"], "method": name,
                         "latency_s": round(dt, 3),
                         "llm_calls": llm._n_calls,
                         "prompt_chars": llm._prompt_chars,
                         "completion_chars": llm._completion_chars})

    n = len(items)
    def _p(vals, q):
        return round(statistics.quantiles(vals, n=100)[q - 1], 3) if len(vals) >= 2 else round(vals[0], 3)

    report = {
        "method": name,
        "n_items": n,
        "ingest_s": round(ingest_s, 1),
        "ingest_llm_calls": ingest_calls,
        "ingest_prompt_tokens_approx": ingest_prompt // 4,
        "ingest_completion_tokens_approx": ingest_completion // 4,
        "per_query": {
            "mean_s": round(statistics.mean(per_q_latency), 3),
            "median_s": round(statistics.median(per_q_latency), 3),
            "p95_s": _p(per_q_latency, 95),
            "stdev_s": round(statistics.pstdev(per_q_latency), 3),
            "mean_llm_calls": round(statistics.mean(per_q_calls), 2),
            "mean_prompt_tokens_approx": round(statistics.mean(per_q_prompt) / 4),
            "mean_completion_tokens_approx": round(statistics.mean(per_q_completion) / 4),
        },
    }
    return report, per_rec


def main():
    out_dir = RESULTS_DIR / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    items = load_items()

    factories = {
        "Naive RAG":      lambda llm: NaiveRAGPipeline(llm_client=llm),
        "Dense RAG":      lambda llm: DenseRAGPipeline(llm_client=llm),
        "CRAG-lite":      lambda llm: CRAGLiteRAGPipeline(llm_client=llm),
        "RankRAG-lite":   lambda llm: RankRAGLiteRAGPipeline(llm_client=llm),
        "HiChunk-lite":   lambda llm: HiChunkLiteRAGPipeline(llm_client=llm),
        "StructStdRAG":       lambda llm: StructStdRAGPipeline(llm_client=llm),
    }

    reports = {}
    all_recs = []
    for name, make in factories.items():
        rep, recs = measure(name, make, docs, items)
        reports[name] = rep
        all_recs.extend(recs)

    # relative overhead vs Naive
    naive = reports["Naive RAG"]["per_query"]
    for name in reports:
        pq = reports[name]["per_query"]
        pq["overhead_vs_naive_x"] = round(pq["mean_s"] / naive["mean_s"], 2)

    with open(out_dir / "latency_results.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    with open(out_dir / "latency_per_query.jsonl", "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print(f"Latency / token-cost analysis on test_mil470a_88 (n=88)")
    print("=" * 100)
    hdr = (f"{'Method':<14} {'Ingest(s)':>10} {'Mean/q(s)':>10} {'p95(s)':>8} "
           f"{'LLM/q':>7} {'Prompt tok/q':>14} {'Compl tok/q':>13} {'×Naive':>8}")
    print(hdr)
    for name, r in reports.items():
        pq = r["per_query"]
        print(f"{name:<14} {r['ingest_s']:>10.1f} {pq['mean_s']:>10.3f} "
              f"{pq['p95_s']:>8.3f} {pq['mean_llm_calls']:>7.2f} "
              f"{pq['mean_prompt_tokens_approx']:>14} "
              f"{pq['mean_completion_tokens_approx']:>13} "
              f"{pq['overhead_vs_naive_x']:>7.2f}×")


if __name__ == "__main__":
    main()
