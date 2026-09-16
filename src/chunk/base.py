"""Chunker base class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Chunk data structure"""
    chunk_id: str
    content: str
    doc_id: str                         # source document ID
    start_pos: int = 0                  # start position in source text
    end_pos: int = 0
    metadata: dict = field(default_factory=dict)
    # StdChunk-specific fields
    layer: int | None = None            # level: 1=Section, 2=Paragraph, 3=Constraint
    semantic_type: str | None = None    # maintenance_procedure / design_guideline / constraint_specification
    constraint_types: list[str] = field(default_factory=list)  # included constraint type labels
    parent_id: str | None = None        # parent chunk ID (cross-layer reference)
    child_ids: list[str] = field(default_factory=list)         # child chunk IDs
    cross_refs: list[str] = field(default_factory=list)        # cross-referenced chunk IDs


class BaseChunker(ABC):
    """Chunker base class"""

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        """Chunk the text"""
        ...

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """Batch process documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc.content, doc_id=doc.doc_id)
            all_chunks.extend(chunks)
        return all_chunks
