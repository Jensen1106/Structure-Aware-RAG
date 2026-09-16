"""Baseline: CRAG (Corrective RAG)
Reference: https://github.com/suryayalavarthi/crag-reproduction
"""

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult
from src.utils.logger import get_logger
from src.utils.llm_client import get_llm_client

logger = get_logger(__name__)


class CRAGGenerator(BaseGenerator):
    """Corrective RAG: assess retrieval quality and correct as needed"""

    name = "CRAG"

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_llm_client()

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        # Step 1: Assess relevance of retrieved results
        relevance = self._assess_relevance(query, retrieved)

        if relevance == "correct":
            # Retrieved results are reliable, generate directly
            context = self._build_context(retrieved)
            answer = self._generate_with_context(query, context)
            sources = [r.chunk.chunk_id for r in retrieved]
        elif relevance == "ambiguous":
            # Partially relevant, refine then generate
            refined = self._refine_results(query, retrieved)
            context = self._build_context(refined)
            answer = self._generate_with_context(query, context)
            sources = [r.chunk.chunk_id for r in refined]
        else:
            # Not relevant, attempt to regenerate (without retrieval)
            answer = self._generate_without_context(query)
            sources = []

        return GenerationResult(
            answer=answer,
            sources=sources,
            metadata={"relevance_assessment": relevance},
        )

    def _assess_relevance(self, query: str, retrieved: list[RetrievalResult]) -> str:
        """Evaluate the relevance of retrieved results to the query"""
        if not retrieved:
            return "incorrect"

        context = self._build_context(retrieved[:3])
        prompt = f"""Evaluate the relevance of the following retrieval results to the question.

Question: {query}

Retrieval Results:
{context}

Answer "correct" (fully relevant), "ambiguous" (partially relevant), or "incorrect" (irrelevant).
Answer with one word only:"""

        result = self.llm.generate(prompt).strip().lower()
        if "correct" in result:
            return "correct"
        elif "ambiguous" in result:
            return "ambiguous"
        return "incorrect"

    def _refine_results(self, query: str, retrieved: list[RetrievalResult]) -> list[RetrievalResult]:
        """Refine retrieval results, keeping only the relevant parts"""
        refined = []
        for r in retrieved:
            if r.score > 0.3:  # simple threshold filter
                refined.append(r)
        return refined if refined else retrieved[:1]

    def _generate_with_context(self, query: str, context: str) -> str:
        prompt = f"""Answer the question based on the following reference materials.

Reference Materials:
{context}

Question: {query}

Answer:"""
        return self.llm.generate(prompt)

    def _generate_without_context(self, query: str) -> str:
        prompt = f"""Answer the following question. If uncertain, state so explicitly.

Question: {query}

Answer:"""
        return self.llm.generate(prompt)
