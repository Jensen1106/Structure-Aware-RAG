"""Baseline: Adaptive Chunking
Reference: https://github.com/ekimetrics/adaptive-chunking
"""

from src.chunk.base import BaseChunker, Chunk
from src.config import DEFAULT_CONFIG


class AdaptiveChunker(BaseChunker):
    """Adaptive chunking: dynamically adjust chunk size based on content density"""

    name = "Adaptive Chunking"

    def __init__(
        self,
        min_size: int | None = None,
        max_size: int | None = None,
        embed_model=None,
    ):
        self.min_size = min_size or DEFAULT_CONFIG.chunk.adaptive_min_size
        self.max_size = max_size or DEFAULT_CONFIG.chunk.adaptive_max_size
        self.embed_model = embed_model

    def _get_embedder(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self.embed_model

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        """
        Adaptive chunking:
        1. First split by paragraphs
        2. Compute information density per paragraph (embedding vector change rate)
        3. High-density regions use smaller chunks, low-density regions use larger chunks
        """
        import re
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        if not paragraphs:
            return []

        # Compute paragraph embeddings
        embedder = self._get_embedder()
        embeddings = embedder.encode(paragraphs, show_progress_bar=False)

        # Compute information density (embedding change rate between adjacent paragraphs)
        import numpy as np
        densities = [1.0]  # first paragraph default density 1
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i - 1], embeddings[i]) / (
                np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            density = 1.0 - sim  # larger change → higher density
            densities.append(density)

        # Adaptive merging
        chunks = []
        current = ""
        idx = 0
        pos = 0

        for i, para in enumerate(paragraphs):
            target_size = self._adaptive_size(densities[i])

            if len(current) + len(para) <= target_size:
                current += "\n\n" + para if current else para
            else:
                if current.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_adaptive_{idx:04d}",
                        content=current.strip(),
                        doc_id=doc_id,
                        start_pos=pos,
                        end_pos=pos + len(current),
                    ))
                    idx += 1
                pos += len(current)
                current = para

        if current.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_adaptive_{idx:04d}",
                content=current.strip(),
                doc_id=doc_id,
                start_pos=pos,
                end_pos=pos + len(current),
            ))
        return chunks

    def _adaptive_size(self, density: float) -> int:
        """Compute target chunk size based on information density"""
        # Higher density → smaller chunks; lower density → larger chunks
        ratio = 1.0 - min(density, 1.0)  # 0~1, 0=minimum chunk, 1=maximum chunk
        return int(self.min_size + ratio * (self.max_size - self.min_size))
