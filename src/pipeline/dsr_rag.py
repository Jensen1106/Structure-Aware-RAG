"""Baseline: DSR (Dual-Stage Retrieval)"""

from src.pipeline.base import BasePipeline, PipelineResult
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.generator.direct_gen import DirectGenerator
from src.data.loader import Document


class DSRPipeline(BasePipeline):
    """Dual-Stage Retrieval: coarse screening followed by fine ranking"""

    name = "DSR"

    def __init__(self, llm_client=None):
        self.chunker = RecursiveChunker()
        self.stage1 = BM25Retriever()   # coarse screening
        self.stage2 = DenseRetriever()  # fine ranking
        self.generator = DirectGenerator(llm_client=llm_client)
        self.retriever = self.stage1    # compatible with base class
        self._indexed = False

    def ingest(self, documents: list[Document]):
        chunks = self.chunker.chunk_documents(documents)
        self.stage1.index(chunks)
        self.stage2.index(chunks)
        self._indexed = True
        return chunks

    def run(self, query: str, top_k: int = 5) -> PipelineResult:
        # Stage 1: BM25 coarse screening top-50
        coarse = self.stage1.retrieve(query, top_k=50)

        # Stage 2: Dense fine ranking on the coarse-screened results
        coarse_chunks = [r.chunk for r in coarse]
        self.stage2.index(coarse_chunks)
        fine = self.stage2.retrieve(query, top_k=top_k)

        generation = self.generator.generate(query, fine)
        return PipelineResult(
            query=query,
            generation=generation,
            retrieved=fine,
            metadata={"pipeline": self.name},
        )
