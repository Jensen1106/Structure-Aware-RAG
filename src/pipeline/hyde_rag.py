"""Baseline: HyDE RAG (Gao et al., 2022)"""

from src.pipeline.base import BasePipeline
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.hyde_retriever import HyDERetriever
from src.generator.direct_gen import DirectGenerator


class HyDERAGPipeline(BasePipeline):
    name = "HyDE RAG"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=RecursiveChunker(),
            retriever=HyDERetriever(llm_client=llm_client),
            generator=DirectGenerator(llm_client=llm_client),
        )
