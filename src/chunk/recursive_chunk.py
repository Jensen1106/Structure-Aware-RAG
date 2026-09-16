"""Baseline: RecursiveCharacterTextSplitter (LangChain-style)"""

from src.chunk.base import BaseChunker, Chunk
from src.config import DEFAULT_CONFIG


class RecursiveChunker(BaseChunker):
    """Recursive character splitter"""

    name = "RecursiveCharacterTextSplitter"

    SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]

    def __init__(self, size: int | None = None, overlap: int | None = None):
        self.size = size or DEFAULT_CONFIG.chunk.recursive_size
        self.overlap = overlap or DEFAULT_CONFIG.chunk.recursive_overlap

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        pieces = self._split_recursive(text, self.SEPARATORS)
        # Merge into target-sized chunks
        chunks = []
        current = ""
        idx = 0
        start_pos = 0

        for piece in pieces:
            if len(current) + len(piece) <= self.size:
                current += piece
            else:
                if current.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_recursive_{idx:04d}",
                        content=current.strip(),
                        doc_id=doc_id,
                        start_pos=start_pos,
                        end_pos=start_pos + len(current),
                    ))
                    idx += 1
                # Preserve overlap
                overlap_text = current[-self.overlap:] if len(current) > self.overlap else ""
                start_pos += len(current) - len(overlap_text)
                current = overlap_text + piece

        if current.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_recursive_{idx:04d}",
                content=current.strip(),
                doc_id=doc_id,
                start_pos=start_pos,
                end_pos=start_pos + len(current),
            ))
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if not sep:
            return list(text)
        parts = text.split(sep)
        result = []
        for part in parts:
            if len(part) <= self.size:
                result.append(part + sep)
            else:
                result.extend(self._split_recursive(part, separators[1:]))
        return result
