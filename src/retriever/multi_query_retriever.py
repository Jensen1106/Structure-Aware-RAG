"""Multi-Query Retriever (2024 Query Decomposition RAG)

Method:
1. Use LLM to decompose the original query into 3 sub-queries (expressed from different angles)
2. Each sub-query independently retrieves top_k chunks
3. Merge results using Reciprocal Rank Fusion (RRF)

Motivation: a single query often misses relevant chunks due to phrasing issues;
multi-angle queries expand recall.

References:
- LangChain MultiQueryRetriever (2024)
- Simplified variant of RAG-Fusion (Rackauckas 2023)
- RRF: Cormack et al. 2009
"""

import re
import numpy as np
from collections import defaultdict

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.config import EMBEDDING_MODEL
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


MQ_PROMPT = """You are a maintainability engineering expert. To retrieve more relevant content from technical documents, rewrite the original question below into 3 sub-questions from different angles.

Requirements:
- Restate from different angles (technical terminology, constraint conditions, numerical parameters, etc.)
- Each sub-question on its own line, starting with "1.", "2.", "3."
- Keep the same language as the original question
- Do not output anything else

Original Question: {query}

3 Sub-questions:"""


class MultiQueryRetriever(BaseRetriever):
    """Multi-query retrieval: LLM decomposition + multiple retrievals + RRF fusion"""

    name = "Multi-Query RAG"

    def __init__(self, model_name: str | None = None, llm_client=None,
                 n_subqueries: int = 3, rrf_k: int = 60):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.llm = llm_client or get_llm_client()
        self.n_subqueries = n_subqueries
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"MultiQuery loaded embedding model: {self.model_name}")

    def index(self, chunks: list[Chunk]):
        self._load_model()
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False,
                                             normalize_embeddings=True)
        logger.info(f"MultiQuery indexed {len(chunks)} chunks")

    def _decompose(self, query: str) -> list[str]:
        """LLM decomposes query into multiple sub-queries"""
        prompt = MQ_PROMPT.format(query=query)
        try:
            text = self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"Query decomposition failed, using original: {e}")
            return [query]
        # Parse numbered lines
        subqs = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\d+[.:、）)]\s*(.+)", line)
            if m:
                sub = m.group(1).strip()
                if len(sub) > 4:
                    subqs.append(sub)
        if not subqs:
            subqs = [query]
        # At least include the original query
        if query not in subqs:
            subqs = [query] + subqs
        return subqs[: self.n_subqueries + 1]

    def _retrieve_single(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._load_model()
        subqueries = self._decompose(query)

        # Each sub-query retrieves top_k*2 candidates
        per_query_k = max(top_k * 2, 10)
        all_candidates = defaultdict(list)  # idx -> list of (rank, score, subquery_idx)
        for qi, subq in enumerate(subqueries):
            hits = self._retrieve_single(subq, per_query_k)
            for rank, (idx, score) in enumerate(hits):
                all_candidates[idx].append((rank + 1, score, qi))

        # RRF fusion: score = sum(1 / (rrf_k + rank))
        fused = {}
        for idx, hits in all_candidates.items():
            rrf_score = sum(1.0 / (self.rrf_k + rank) for rank, _, _ in hits)
            fused[idx] = rrf_score

        # Sort
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (idx, score) in enumerate(ranked):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank + 1,
                metadata={"n_subqueries": len(subqueries)},
            ))
        return results
