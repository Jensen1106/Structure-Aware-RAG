"""StructStdRAG: Complete pipeline (StdChunker + CatRetriever + BoundGrounder)"""

from src.pipeline.base import BasePipeline, PipelineResult
from src.chunk.std_chunk import StdChunker
from src.retriever.cat_retriever import CatRetriever
from src.generator.bound_grounder import BoundGrounder


class StructStdRAGPipeline(BasePipeline):
    """StructStdRAG complete pipeline: three innovative modules combined"""

    name = "StructStdRAG"

    def __init__(self, llm_client=None, equipment_type: str | None = None):
        super().__init__(
            chunker=StdChunker(),
            retriever=CatRetriever(equipment_type=equipment_type),
            generator=BoundGrounder(llm_client=llm_client, equipment_type=equipment_type),
        )
        self.equipment_type = equipment_type
