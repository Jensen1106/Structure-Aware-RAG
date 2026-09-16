"""Baseline: Dense + BM25 hybrid retrieval"""

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.dense_retriever import DenseRetriever
from src.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Dense + BM25 weighted fusion retrieval"""

    name = "Dense + BM25"

    def __init__(
        self,
        w_dense: float | None = None,
        w_bm25: float | None = None,
        dense_model: str | None = None,
    ):
        cfg = DEFAULT_CONFIG.retriever
        self.w_dense = w_dense if w_dense is not None else cfg.w_dense
        self.w_bm25 = w_bm25 if w_bm25 is not None else cfg.w_bm25
        self.dense = DenseRetriever(model_name=dense_model)
        self.bm25 = BM25Retriever()

    def index(self, chunks: list[Chunk]):
        self.dense.index(chunks)
        self.bm25.index(chunks)
        logger.info(f"Hybrid indexed {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # Fetch more candidates from both retrievers
        recall_k = max(top_k * 4, 50)
        dense_results = self.dense.retrieve(query, top_k=recall_k)
        bm25_results = self.bm25.retrieve(query, top_k=recall_k)

        # Normalize scores
        dense_scores = self._normalize_scores(dense_results)
        bm25_scores = self._normalize_scores(bm25_results)

        # Merge: use chunk_id as key
        merged: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for result, norm_score in zip(dense_results, dense_scores):
            cid = result.chunk.chunk_id
            merged[cid] = merged.get(cid, 0) + self.w_dense * norm_score
            chunk_map[cid] = result.chunk

        for result, norm_score in zip(bm25_results, bm25_scores):
            cid = result.chunk.chunk_id
            merged[cid] = merged.get(cid, 0) + self.w_bm25 * norm_score
            chunk_map[cid] = result.chunk

        # Sort and take top_k
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (cid, score) in enumerate(ranked):
            results.append(RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                rank=rank + 1,
                metadata={"method": "hybrid"},
            ))
        return results

    @staticmethod
    def _normalize_scores(results: list[RetrievalResult]) -> list[float]:
        if not results:
            return []
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        return [(s - min_s) / rng for s in scores]
