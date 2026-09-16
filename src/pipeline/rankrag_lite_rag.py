"""RankRAG-lite pipeline — Dense retriever + RankRAGLiteGenerator."""

from __future__ import annotations

from src.pipeline.base import BasePipeline
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.dense_retriever import DenseRetriever
from src.generator.rankrag_lite_gen import RankRAGLiteGenerator


class RankRAGLiteRAGPipeline(BasePipeline):
    name = "RankRAG-lite"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=RecursiveChunker(),
            retriever=DenseRetriever(),
            generator=RankRAGLiteGenerator(llm_client=llm_client),
        )
