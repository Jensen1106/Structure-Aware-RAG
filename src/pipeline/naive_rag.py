"""Baseline: Naive RAG（Fixed-512 + Dense + Direct Gen）"""

from src.pipeline.base import BasePipeline
from src.chunk.fixed_chunk import FixedChunker
from src.retriever.dense_retriever import DenseRetriever
from src.generator.direct_gen import DirectGenerator


class NaiveRAGPipeline(BasePipeline):
    name = "Naive RAG"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=FixedChunker(),
            retriever=DenseRetriever(),
            generator=DirectGenerator(llm_client=llm_client),
        )
