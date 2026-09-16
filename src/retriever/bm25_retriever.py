"""Baseline: BM25 retrieval"""

import math
from collections import Counter

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 keyword retrieval"""

    name = "BM25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0
        self.tokenized_docs: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization (mixed Chinese/English)"""
        import re
        # English tokenized by spaces/punctuation, Chinese by character
        tokens = re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())
        return tokens

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenized_docs = [self._tokenize(c.content) for c in chunks]
        self.doc_lens = [len(d) for d in self.tokenized_docs]
        self.avg_dl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1

        # Compute document frequency
        self.doc_freqs = {}
        for doc_tokens in self.tokenized_docs:
            for token in set(doc_tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        logger.info(f"BM25 indexed {len(chunks)} chunks, vocab size: {len(self.doc_freqs)}")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = self._tokenize(query)
        n = len(self.chunks)
        scores = []

        for i, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            tf_counter = Counter(doc_tokens)
            dl = self.doc_lens[i]

            for qt in query_tokens:
                if qt not in self.doc_freqs:
                    continue
                df = self.doc_freqs[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf = tf_counter.get(qt, 0)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                score += idf * tf_norm
            scores.append(score)

        # Sort and take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (idx, score) in enumerate(ranked):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                rank=rank + 1,
            ))
        return results
