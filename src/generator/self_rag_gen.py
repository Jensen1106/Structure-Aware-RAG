"""Baseline: Self-RAG
Reference: https://github.com/AkariAsai/self-rag
"""

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult
from src.utils.logger import get_logger
from src.utils.llm_client import get_llm_client

logger = get_logger(__name__)


class SelfRAGGenerator(BaseGenerator):
    """Self-RAG: adaptive retrieval and self-reflective generation"""

    name = "Self-RAG"

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_llm_client()

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        # Step 1: Determine whether retrieval is needed
        need_retrieval = self._assess_need_retrieval(query)

        if need_retrieval and retrieved:
            # Step 2: Generate candidate answers (per-segment)
            candidates = []
            for r in retrieved[:3]:
                candidate = self._generate_segment(query, r)
                # Step 3: Self-critique — evaluate generation quality
                critique = self._self_critique(query, candidate, r)
                candidates.append((candidate, critique, r))

            # Step 4: Select the best answer
            best = max(candidates, key=lambda x: x[1]["score"])
            answer = best[0]
            sources = [best[2].chunk.chunk_id]
            metadata = {"critique": best[1], "need_retrieval": True}
        else:
            # No retrieval needed, generate directly
            answer = self._generate_direct(query)
            sources = []
            metadata = {"need_retrieval": False}

        return GenerationResult(
            answer=answer,
            sources=sources,
            metadata=metadata,
        )

    def _assess_need_retrieval(self, query: str) -> bool:
        """Determine whether retrieval is needed"""
        prompt = f"""Determine whether the following question requires consulting external materials to answer.
Answer "yes" or "no".

Question: {query}

Retrieval needed:"""
        result = self.llm.generate(prompt).strip().lower()
        return "yes" in result

    def _generate_segment(self, query: str, result: RetrievalResult) -> str:
        """Generate an answer segment based on a single retrieval result"""
        prompt = f"""Answer the question based on the following reference materials.

Reference Materials:
{result.chunk.content}

Question: {query}

Answer:"""
        return self.llm.generate(prompt)

    def _self_critique(self, query: str, answer: str, result: RetrievalResult) -> dict:
        """Self-critique: evaluate generation quality"""
        prompt = f"""Evaluate the quality of the following answer.

Question: {query}
Reference Materials: {result.chunk.content}
Answer: {answer}

Rate the following dimensions (1-5 points):
1. Relevance (IsRel): whether the answer is relevant to the question
2. Support (IsSup): whether the answer is supported by the reference materials
3. Usefulness (IsUse): whether the answer is useful to the user

Format: IsRel=X, IsSup=X, IsUse=X"""

        result_text = self.llm.generate(prompt)

        # Parse scores
        import re
        scores = {"IsRel": 3, "IsSup": 3, "IsUse": 3}
        for key in scores:
            match = re.search(rf"{key}\s*=\s*(\d)", result_text)
            if match:
                scores[key] = int(match.group(1))

        scores["score"] = sum(scores.values()) / len(scores)
        return scores

    def _generate_direct(self, query: str) -> str:
        prompt = f"Answer the following question: {query}\n\nAnswer:"
        return self.llm.generate(prompt)
