"""Baseline: Direct Generation (generate without retrieval)"""

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult
from src.utils.logger import get_logger
from src.utils.llm_client import get_llm_client

logger = get_logger(__name__)


class DirectGenerator(BaseGenerator):
    """Generate directly from retrieval context without additional verification or reranking"""

    name = "Direct Generation"

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_llm_client()

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        if retrieved:
            context = self._build_context(retrieved)
            prompt = f"""You are a maintainability engineering QA assistant. Answer the question directly and strictly based on the reference materials.

Requirements:
1. Answer only from the reference materials; do not add facts or values not present in them.
2. Preserve the original meaning of term definitions, constraint values, and units.
3. If the reference materials are insufficient, state explicitly: "The reference materials are insufficient to answer."
4. When citing materials, use [Source n] notation.

Reference Materials:
{context}

Question: {query}

Answer:"""
            sources = [r.chunk.chunk_id for r in retrieved]
            confidence = max(r.score for r in retrieved)
        else:
            prompt = f"""Answer the following maintainability engineering question. If uncertain, state so explicitly.

Question: {query}

Answer:"""
            sources = []
            confidence = 0.3

        answer = self.llm.generate(prompt)
        return GenerationResult(answer=answer, sources=sources, confidence=confidence)
