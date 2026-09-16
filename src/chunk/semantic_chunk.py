"""Baseline: Semantic Chunking (split based on semantic similarity)"""

import numpy as np

from src.chunk.base import BaseChunker, Chunk
from src.config import DEFAULT_CONFIG


class SemanticChunker(BaseChunker):
    """Semantic chunking: split at semantic breakpoints based on sentence embedding similarity"""

    name = "Semantic Chunking"

    def __init__(self, threshold: float | None = None, embed_model=None):
        self.threshold = threshold or DEFAULT_CONFIG.chunk.semantic_threshold
        self.embed_model = embed_model  # lazy loading

    def _get_embedder(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self.embed_model

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting"""
        import re
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(
                chunk_id=f"{doc_id}_semantic_0000",
                content=text,
                doc_id=doc_id,
            )]

        embedder = self._get_embedder()
        embeddings = embedder.encode(sentences, show_progress_bar=False)

        # Compute cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(sim)

        # Split where similarity is below threshold
        chunks = []
        current_sentences = [sentences[0]]
        idx = 0
        pos = 0

        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                chunk_text = "".join(current_sentences)
                if chunk_text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_semantic_{idx:04d}",
                        content=chunk_text.strip(),
                        doc_id=doc_id,
                        start_pos=pos,
                        end_pos=pos + len(chunk_text),
                    ))
                    idx += 1
                pos += len(chunk_text)
                current_sentences = []
            current_sentences.append(sentences[i + 1])

        # Last segment
        if current_sentences:
            chunk_text = "".join(current_sentences)
            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_semantic_{idx:04d}",
                    content=chunk_text.strip(),
                    doc_id=doc_id,
                    start_pos=pos,
                    end_pos=pos + len(chunk_text),
                ))
        return chunks
