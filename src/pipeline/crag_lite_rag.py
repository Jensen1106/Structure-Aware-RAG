"""CRAG-lite — Corrective RAG (Yan et al., arXiv 2024) without web fallback.

Original CRAG pipeline:
  1. Retrieve Top-K
  2. LLM-based retrieval evaluator scores relevance: correct / ambiguous / incorrect
  3a. correct → direct generation
  3b. ambiguous → knowledge refinement (decompose + filter top-strips)
  3c. incorrect → web search fallback

We replace 3c with "fallback to full Top-10 concatenation" since web search
is not available in the offline engineering-document setting. This keeps
the core corrective mechanism intact.

Reference: Yan et al. "Corrective Retrieval Augmented Generation." arXiv:2401.15884
"""

from __future__ import annotations

from src.pipeline.base import BasePipeline
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.dense_retriever import DenseRetriever
from src.generator.crag_gen import CRAGGenerator


class CRAGLiteRAGPipeline(BasePipeline):
    """CRAG without web fallback — the corrective mechanism only."""

    name = "CRAG-lite"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=RecursiveChunker(),
            retriever=DenseRetriever(),
            generator=CRAGGenerator(llm_client=llm_client),
        )
