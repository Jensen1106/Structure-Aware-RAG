"""Baseline: Multi-Query RAG (2024 Query Decomposition)"""

from src.pipeline.base import BasePipeline
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.multi_query_retriever import MultiQueryRetriever
from src.generator.direct_gen import DirectGenerator


class MultiQueryRAGPipeline(BasePipeline):
    name = "Multi-Query RAG"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=RecursiveChunker(),
            retriever=MultiQueryRetriever(llm_client=llm_client),
            generator=DirectGenerator(llm_client=llm_client),
        )
