"""Baseline: Dense Retrieval + Generation"""

from src.pipeline.base import BasePipeline
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.dense_retriever import DenseRetriever
from src.generator.direct_gen import DirectGenerator


class DenseRAGPipeline(BasePipeline):
    name = "Dense Retrieval + Gen"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=RecursiveChunker(),
            retriever=DenseRetriever(),
            generator=DirectGenerator(llm_client=llm_client),
        )
