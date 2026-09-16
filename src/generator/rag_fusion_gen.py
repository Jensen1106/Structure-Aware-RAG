"""Baseline: RAG-Fusion (multi-query retrieval fusion generation)"""

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import BaseRetriever, RetrievalResult
from src.utils.logger import get_logger
from src.utils.llm_client import get_llm_client

logger = get_logger(__name__)


class RAGFusionGenerator(BaseGenerator):
    """RAG-Fusion: generate multiple query variants, fuse retrieval results for generation"""

    name = "RAG-Fusion"

    def __init__(self, llm_client=None, retriever: BaseRetriever | None = None):
        self.llm = llm_client or get_llm_client()
        self.retriever = retriever

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        # Step 1: Generate query variants
        query_variants = self._generate_variants(query)

        # Step 2: Retrieve for each variant (if an independent retriever is available)
        all_results = list(retrieved)  # use existing results as the base
        if self.retriever:
            for variant in query_variants:
                variant_results = self.retriever.retrieve(variant, top_k=3)
                all_results.extend(variant_results)

        # Step 3: Reciprocal Rank Fusion (RRF) for dedup and re-ranking
        fused = self._reciprocal_rank_fusion(all_results)

        # Step 4: Generate based on fused results
        context = self._build_context(fused[:5])
        prompt = f"""Answer the question based on the following reference materials. Cite sources.

Reference Materials:
{context}

Question: {query}

Answer:"""

        answer = self.llm.generate(prompt)
        sources = [r.chunk.chunk_id for r in fused[:5]]
        return GenerationResult(answer=answer, sources=sources)

    def _generate_variants(self, query: str, n: int = 3) -> list[str]:
        """Generate query variants"""
        prompt = f"""Generate {n} different query variants for the following question to retrieve more relevant information.
One variant per line, no numbering.

Original Question: {query}

Variants:"""
        result = self.llm.generate(prompt)
        variants = [line.strip() for line in result.split("\n") if line.strip()]
        return variants[:n]

    def _reciprocal_rank_fusion(self, results: list[RetrievalResult], k: int = 60) -> list[RetrievalResult]:
        """RRF fusion ranking"""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for r in results:
            cid = r.chunk.chunk_id
            rrf_score = 1.0 / (k + r.rank)
            scores[cid] = scores.get(cid, 0) + rrf_score
            if cid not in chunk_map:
                chunk_map[cid] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for rank, (cid, score) in enumerate(ranked):
            result = chunk_map[cid]
            fused.append(RetrievalResult(
                chunk=result.chunk,
                score=score,
                rank=rank + 1,
            ))
        return fused
