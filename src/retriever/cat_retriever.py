"""CatRetriever: Constraint-type-aware hybrid retrieval mechanism (Innovation 2)

Two-stage retrieval:
  Stage 1: Dense recall top-50
  Stage 2: Constraint-type re-ranking + equipment model filtering

Core innovation:
  - Query constraint type identification (rules + classifier)
  - Constraint type match bonus / mismatch penalty
  - Equipment model adaptation filtering
"""

import numpy as np

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.config import DEFAULT_CONFIG
from src.utils.constraint_taxonomy import (
    classify_query_constraint,
    ConstraintType,
    EquipmentProfile,
    EQUIPMENT_PROFILES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatRetriever(BaseRetriever):
    """Constraint-type-aware hybrid retrieval"""

    name = "CatRetriever"

    def __init__(self, equipment_type: str | None = None):
        cfg = DEFAULT_CONFIG.retriever
        self.recall_k = cfg.recall_k
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.type_boost = cfg.type_match_boost
        self.type_penalty = cfg.type_mismatch_penalty
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.chunks: list[Chunk] = []
        self.equipment_profile: EquipmentProfile | None = None

        if equipment_type and equipment_type in EQUIPMENT_PROFILES:
            self.equipment_profile = EQUIPMENT_PROFILES[equipment_type]
            logger.info(f"Loaded equipment profile: {self.equipment_profile.name}")

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.dense.index(chunks)
        self.bm25.index(chunks)

        # Annotate constraint types for each chunk (if not already annotated)
        for chunk in chunks:
            if not chunk.constraint_types:
                matched = classify_query_constraint(chunk.content)
                chunk.constraint_types = [ct.label for ct in matched]

        logger.info(f"CatRetriever indexed {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Two-stage retrieval"""

        # === Query constraint type identification ===
        query_types = classify_query_constraint(query)
        query_type_labels = {ct.label for ct in query_types}
        logger.info(f"Query constraint types: {query_type_labels}")

        # === Stage 1: Dense + BM25 recall ===
        dense_results = self.dense.retrieve(query, top_k=self.recall_k)
        bm25_results = self.bm25.retrieve(query, top_k=self.recall_k)

        # Merge candidate set
        candidates: dict[str, dict] = {}
        chunk_map: dict[str, Chunk] = {}

        for r in dense_results:
            cid = r.chunk.chunk_id
            candidates.setdefault(cid, {"dense": 0, "bm25": 0})
            candidates[cid]["dense"] = r.score
            chunk_map[cid] = r.chunk

        for r in bm25_results:
            cid = r.chunk.chunk_id
            candidates.setdefault(cid, {"dense": 0, "bm25": 0})
            candidates[cid]["bm25"] = r.score
            chunk_map[cid] = r.chunk

        # === Stage 2: Constraint-type re-ranking ===
        final_scores: dict[str, float] = {}

        for cid, scores in candidates.items():
            chunk = chunk_map[cid]

            # Base score = α·dense + (1-α)·bm25
            base_score = self.alpha * scores["dense"] + (1 - self.alpha) * scores["bm25"]

            # Constraint type match score
            type_score = self._compute_type_match_score(chunk, query_type_labels)

            # Equipment model filtering
            equipment_factor = self._equipment_relevance(chunk)

            # Final score
            final_scores[cid] = (base_score + self.beta * type_score) * equipment_factor

        # Sort and take top_k
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (cid, score) in enumerate(ranked):
            chunk = chunk_map[cid]
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                metadata={
                    "method": "cat_retriever",
                    "query_types": list(query_type_labels),
                    "chunk_types": chunk.constraint_types,
                    "type_matched": bool(query_type_labels & set(chunk.constraint_types)),
                },
            ))

        # Automatically attach parent chunk context
        results = self._attach_parent_context(results)

        return results

    def _compute_type_match_score(self, chunk: Chunk, query_type_labels: set[str]) -> float:
        """Compute constraint type match score"""
        if not query_type_labels:
            return 0.0

        chunk_types = set(chunk.constraint_types)
        if not chunk_types:
            return 0.0

        # Overlap → bonus, no overlap → penalty
        overlap = query_type_labels & chunk_types
        if overlap:
            return self.type_boost * len(overlap) / len(query_type_labels)
        else:
            return self.type_penalty

    def _equipment_relevance(self, chunk: Chunk) -> float:
        """Equipment model relevance filtering"""
        if self.equipment_profile is None:
            return 1.0

        # Check if the chunk is related to the target equipment model
        content_lower = chunk.content.lower()
        profile = self.equipment_profile

        # If the chunk explicitly mentions other equipment models, down-weight
        other_profiles = [p for k, p in EQUIPMENT_PROFILES.items()
                          if p.name != profile.name]
        for other in other_profiles:
            if other.name.lower() in content_lower:
                if profile.name.lower() not in content_lower:
                    return 0.5  # Down-weight but not fully exclude

        # If the chunk contains constraint types emphasized by the target equipment, up-weight
        chunk_types = set(chunk.constraint_types)
        emphasized = {
            ct.label
            for ct_id in profile.emphasized_qualitative
            for ct in [__import__('src.utils.constraint_taxonomy', fromlist=['ID_TO_TYPE']).ID_TO_TYPE.get(ct_id)]
            if ct is not None
        }
        if chunk_types & emphasized:
            return 1.2  # Up-weight

        return 1.0

    def _attach_parent_context(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """When a Layer 3 (Constraint Block) chunk is retrieved, automatically attach
        the context of its Layer 2 parent chunk"""
        chunk_map = {c.chunk_id: c for c in self.chunks}

        for result in results:
            chunk = result.chunk
            if chunk.layer == 3 and chunk.parent_id:
                parent = chunk_map.get(chunk.parent_id)
                if parent:
                    result.metadata["parent_context"] = parent.content
                    result.metadata["parent_id"] = parent.chunk_id

        return results
