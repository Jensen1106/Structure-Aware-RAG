"""Retriever base class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.chunk.base import Chunk


@dataclass
class RetrievalResult:
    """Retrieval result"""
    chunk: Chunk
    score: float
    rank: int = 0
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Retriever base class"""

    name: str = "base"

    @abstractmethod
    def index(self, chunks: list[Chunk]):
        """Build index"""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve"""
        ...
