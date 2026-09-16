"""Generator base class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.retriever.base import RetrievalResult


@dataclass
class GenerationResult:
    """Generation result"""
    answer: str
    sources: list[str] = field(default_factory=list)  # referenced chunk IDs
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class BaseGenerator(ABC):
    """Generator base class"""

    name: str = "base"

    @abstractmethod
    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        """Generate answer based on query and retrieval results"""
        ...

    def _build_context(self, retrieved: list[RetrievalResult]) -> str:
        """Concatenate retrieval results into context"""
        parts = []
        for i, r in enumerate(retrieved):
            parts.append(f"[Source {i+1}] (chunk_id: {r.chunk.chunk_id})\n{r.chunk.content}")
        return "\n\n".join(parts)
