"""Baseline: Fixed-size chunking (512 tokens)"""

from src.chunk.base import BaseChunker, Chunk
from src.config import DEFAULT_CONFIG


class FixedChunker(BaseChunker):
    """Fixed-size chunking"""

    name = "Fixed-512"

    def __init__(self, size: int | None = None, overlap: int | None = None):
        self.size = size or DEFAULT_CONFIG.chunk.fixed_size
        self.overlap = overlap or DEFAULT_CONFIG.chunk.fixed_overlap

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_fixed_{idx:04d}",
                    content=chunk_text,
                    doc_id=doc_id,
                    start_pos=start,
                    end_pos=min(end, len(text)),
                ))
                idx += 1
            start += self.size - self.overlap
        return chunks
