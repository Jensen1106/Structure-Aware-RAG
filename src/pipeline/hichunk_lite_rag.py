"""HiChunk-lite pipeline — HiChunkLiteChunker + Dense + DirectGen."""

from __future__ import annotations

from src.pipeline.base import BasePipeline
from src.chunk.hichunk_lite import HiChunkLiteChunker
from src.retriever.dense_retriever import DenseRetriever
from src.generator.direct_gen import DirectGenerator


class HiChunkLiteRAGPipeline(BasePipeline):
    name = "HiChunk-lite"

    def __init__(self, llm_client=None):
        super().__init__(
            chunker=HiChunkLiteChunker(llm_client=llm_client),
            retriever=DenseRetriever(),
            generator=DirectGenerator(llm_client=llm_client),
        )
