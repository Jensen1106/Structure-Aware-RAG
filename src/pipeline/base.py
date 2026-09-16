"""RAG pipeline base class: chunk → retrieve → generate"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.chunk.base import BaseChunker, Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.generator.base import BaseGenerator, GenerationResult
from src.data.loader import Document


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    query: str
    generation: GenerationResult
    retrieved: list[RetrievalResult]
    chunks_count: int = 0
    metadata: dict = field(default_factory=dict)


class BasePipeline(ABC):
    """RAG pipeline base class"""

    name: str = "base"

    def __init__(self, chunker: BaseChunker, retriever: BaseRetriever, generator: BaseGenerator):
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self._indexed = False

    def ingest(self, documents: list[Document]):
        """Ingest documents: chunk → build index"""
        chunks = self.chunker.chunk_documents(documents)
        self.retriever.index(chunks)
        self._indexed = True
        return chunks

    def run(self, query: str, top_k: int = 5) -> PipelineResult:
        """Run full pipeline (with per-stage timing for latency analysis)"""
        import time as _time
        _t0 = _time.perf_counter()
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        _t1 = _time.perf_counter()
        generation = self.generator.generate(query, retrieved)
        _t2 = _time.perf_counter()
        return PipelineResult(
            query=query,
            generation=generation,
            retrieved=retrieved,
            metadata={
                "pipeline": self.name,
                "t_retrieve": _t1 - _t0,
                "t_generate": _t2 - _t1,
            },
        )

    def batch_run(self, queries: list[str], top_k: int = 5) -> list[PipelineResult]:
        """Batch run"""
        return [self.run(q, top_k=top_k) for q in queries]
