"""Baseline: Dense Retrieval (BGE)"""

import numpy as np

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.config import EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense vector retrieval (BGE embedding model)"""

    name = "Dense Retrieval (BGE)"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")

    def index(self, chunks: list[Chunk]):
        self._load_model()
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        logger.info(f"Dense indexed {len(chunks)} chunks, embedding shape: {self.embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._load_model()
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # Cosine similarity (already normalized, dot product suffices)
        scores = np.dot(self.embeddings, query_emb.T).flatten()

        # Sort and take top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query (for external use)"""
        self._load_model()
        return self.model.encode([query], normalize_embeddings=True)[0]
