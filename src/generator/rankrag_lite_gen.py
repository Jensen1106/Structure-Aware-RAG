"""RankRAG-lite generator — NeurIPS 2024 (Yu et al.).

Core idea of RankRAG: a *single* LLM call both (a) re-ranks the retrieved
passages and (b) generates the answer, using a single instruction-tuned
prompt. The original paper fine-tunes a LLaMA-8B on rank+gen data; here
we replicate the inference-time behaviour with prompt engineering on a
general instruction-following LLM (qwen-turbo/deepseek-v3).

Prompt contract:
  - Input: query + Top-K retrieved passages (labelled P1..PK)
  - Output:  `RANK: [<passage-ids-in-descending-relevance>]`
             `ANSWER: <final answer, drawing ONLY from the top-ranked passages>`

Reference: Yu et al. "RankRAG: Unifying Context Ranking with
Retrieval-Augmented Generation." NeurIPS 2024.
"""

from __future__ import annotations

import re

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


PROMPT = """You are a maintainability engineering expert. Given a query and {k} retrieved passages (P1..P{k}),
first rerank them in descending order of relevance to the query, then generate an answer using only the top-3 ranked passages.

# Query
{query}

# Candidate Passages
{passages}

Output strictly in the following format (two lines):
RANK: [P_id1, P_id2, ...]
ANSWER: <answer supported only by the top-3 ranked passages; if unanswerable, state so directly>"""


class RankRAGLiteGenerator(BaseGenerator):
    """Single-LLM rank-then-generate (RankRAG inference-time replication)."""

    name = "RankRAG-lite"

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_llm_client()

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        if not retrieved:
            return GenerationResult(answer="The reference materials are insufficient to answer.", sources=[], metadata={})

        k = min(len(retrieved), 8)
        passages_txt = "\n\n".join(
            f"P{i+1}: {r.chunk.content[:500]}"
            for i, r in enumerate(retrieved[:k])
        )
        prompt = PROMPT.format(k=k, query=query, passages=passages_txt)
        try:
            out = self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"RankRAG LLM call failed: {e}")
            out = ""

        rank_order, answer = self._parse(out, k)
        # reorder retrieved per LLM rank
        reordered = [retrieved[i - 1] for i in rank_order if 1 <= i <= k]
        reordered += [r for i, r in enumerate(retrieved[:k], 1) if i not in rank_order]
        sources = [r.chunk.chunk_id for r in reordered[:3]]
        return GenerationResult(
            answer=answer or "The reference materials are insufficient to answer.",
            sources=sources,
            metadata={"rank_order": rank_order},
        )

    @staticmethod
    def _parse(text: str, k: int) -> tuple[list[int], str]:
        if not text:
            return list(range(1, k + 1)), ""
        m_rank = re.search(r"RANK:\s*\[([^\]]*)\]", text)
        order: list[int] = []
        if m_rank:
            for tok in re.findall(r"P?(\d+)", m_rank.group(1)):
                i = int(tok)
                if 1 <= i <= k and i not in order:
                    order.append(i)
        if not order:
            order = list(range(1, k + 1))
        m_ans = re.search(r"ANSWER:\s*(.*)", text, re.DOTALL)
        answer = m_ans.group(1).strip() if m_ans else text.strip()
        return order, answer
