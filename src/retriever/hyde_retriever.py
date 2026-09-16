"""HyDE Retriever (Gao et al., 2022, ACL 2023)

HyDE = Hypothetical Document Embeddings

Method:
1. Use LLM to generate a "hypothetical answer/document" (hypothetical document)
2. Encode this hypothetical document with BGE (instead of the user's original query)
3. Use the hypothetical document's embedding for dense retrieval

Motivation: original queries are often short and abstract; hypothetical documents
are closer to the target chunk distribution, making retrieval similarity more robust.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL 2023
"""

import numpy as np

from src.chunk.base import Chunk
from src.retriever.base import BaseRetriever, RetrievalResult
from src.config import EMBEDDING_MODEL
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


HYDE_PROMPT_TEMPLATE = """You are a maintainability engineering expert. For the question below, write a short hypothetical answer passage to be used for retrieving similar documents. Keep the answer to 80–150 words and do not cite sources.

Question: {query}

Hypothetical Answer:"""


class HyDERetriever(BaseRetriever):
    """HyDE retrieval: LLM hypothetical answer → embedding → dense retrieval"""

    name = "HyDE"

    def __init__(self, model_name: str | None = None, llm_client=None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.llm = llm_client or get_llm_client()
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"HyDE loaded embedding model: {self.model_name}")

    def index(self, chunks: list[Chunk]):
        self._load_model()
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False,
                                             normalize_embeddings=True)
        logger.info(f"HyDE indexed {len(chunks)} chunks")

    def _generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical answer"""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"HyDE generation failed, fallback to query: {e}")
            return query

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._load_model()
        hypothetical = self._generate_hypothetical(query)

        # Retrieve using hypothetical answer embedding (can also be weighted fusion with original query emb)
        query_emb = self.model.encode([query], normalize_embeddings=True)
        hyde_emb = self.model.encode([hypothetical], normalize_embeddings=True)
        # 0.5 query + 0.5 hypothetical average (robust variant mentioned in the paper)
        fused_emb = 0.5 * query_emb + 0.5 * hyde_emb
        fused_emb = fused_emb / (np.linalg.norm(fused_emb) + 1e-8)

        scores = np.dot(self.embeddings, fused_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                metadata={"hypothetical": hypothetical[:200]},
            ))
        return results
