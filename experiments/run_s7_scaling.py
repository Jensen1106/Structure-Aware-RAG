"""S7: 语料规模 scaling 实验（返修 R3-4 / E-4）

对 mil470a_pages_clean 语料按页序取 {25%, 50%, 75%, 100%} 子集，测量：
  ingest 分解（分块时间 / 索引时间[嵌入+BM25]）、chunk 数、
  平均/p95 检索延迟（LLM-free，top-5，含类别重排，88 条真实查询）。

只测效率不测质量：小子集缺 gold 页，质量指标会误导。
输出：results/s7_scaling/scaling.json + 控制台表
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_e5_ablation_v2 import load_docs, load_items
from src.chunk.std_chunk import StdChunker
from src.retriever.cat_retriever import CatRetriever
from src.utils.logger import get_logger

logger = get_logger("S7-Scaling")

OUT = ROOT / "results" / "s7_scaling"
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
EMB_DIM = 1024


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs_full = load_docs()
    items = load_items()
    queries = [it["query"] for it in items]
    logger.info(f"Corpus: {len(docs_full)} pages; queries: {len(queries)}")

    results = {}
    for frac in FRACTIONS:
        n = max(1, int(round(len(docs_full) * frac)))
        docs = docs_full[:n]

        chunker = StdChunker()
        retriever = CatRetriever()

        t0 = time.perf_counter()
        chunks = chunker.chunk_documents(docs)
        t_chunk = time.perf_counter() - t0

        t0 = time.perf_counter()
        retriever.index(chunks)
        t_index = time.perf_counter() - t0

        # 预热（不计入）
        for q in queries[:3]:
            retriever.retrieve(q, top_k=5)

        lat = []
        for q in queries:
            t0 = time.perf_counter()
            retriever.retrieve(q, top_k=5)
            lat.append(time.perf_counter() - t0)
        lat.sort()
        mean_lat = sum(lat) / len(lat)
        p95 = lat[int(0.95 * (len(lat) - 1))]

        row = {
            "fraction": frac,
            "n_pages": n,
            "n_chunks": len(chunks),
            "chunk_s": round(t_chunk, 2),
            "index_s": round(t_index, 2),
            "ingest_s": round(t_chunk + t_index, 2),
            "mean_retrieve_s": round(mean_lat, 4),
            "p95_retrieve_s": round(p95, 4),
            "est_dense_mem_mb": round(len(chunks) * EMB_DIM * 4 / 1e6, 1),
        }
        results[f"{int(frac*100)}pct"] = row
        logger.info(f"[{int(frac*100)}%] {row}")

        # 释放 GPU 显存（每次新建 dense 索引）
        import gc
        import torch
        del retriever
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT / "scaling.json", "w", encoding="utf-8") as f:
        json.dump({"corpus": "mil470a_pages_clean", "n_queries": len(queries),
                   "retriever": "CatRetriever (dense bge-m3 + BM25 + category rerank)",
                   "results": results}, f, ensure_ascii=False, indent=2)

    print("\n=== S7: Corpus scaling (LLM-free) ===")
    print(f"{'Corpus':>8}{'Pages':>7}{'Chunks':>8}{'Chunk(s)':>10}{'Index(s)':>10}"
          f"{'MeanRet(s)':>12}{'P95Ret(s)':>11}{'Mem(MB)':>9}")
    for k, r in results.items():
        print(f"{k:>8}{r['n_pages']:>7}{r['n_chunks']:>8}{r['chunk_s']:>10.1f}"
              f"{r['index_s']:>10.1f}{r['mean_retrieve_s']:>12.3f}"
              f"{r['p95_retrieve_s']:>11.3f}{r['est_dense_mem_mb']:>9.1f}")


if __name__ == "__main__":
    main()
