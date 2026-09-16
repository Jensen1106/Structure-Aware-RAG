"""Baseline: Source-only (return retrieved source text only, no generation)"""

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult


class SourceOnlyGenerator(BaseGenerator):
    """Return only the retrieved source text without generation"""

    name = "Source-only"

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        if not retrieved:
            return GenerationResult(answer="No relevant content found", sources=[])

        # Directly concatenate retrieval results as the answer
        parts = []
        sources = []
        for r in retrieved:
            parts.append(r.chunk.content)
            sources.append(r.chunk.chunk_id)

        answer = "\n\n---\n\n".join(parts)
        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=max(r.score for r in retrieved) if retrieved else 0,
        )
