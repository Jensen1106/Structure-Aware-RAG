"""HiChunk-lite — LLM-based hierarchical chunking (arXiv 2025, simplified).

Original HiChunk uses a fine-tuned LLM to iteratively decide chunk
boundaries at multiple granularities. We replicate the inference-time
behaviour with a prompt-engineered approach using qwen-turbo: the LLM
is asked to output boundary markers for each document, producing a
two-level hierarchy (section-level + paragraph-level).

To keep cost bounded we limit to ONE LLM call per document (not per
paragraph) and cap input at ~4k tokens per doc.

Reference: "HiChunk: Hierarchical Chunking for RAG via LLM Segmentation."
arXiv:2501.xxxxx (2025).
"""

from __future__ import annotations

import re
import json

from src.chunk.base import BaseChunker, Chunk
from src.data.loader import Document
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


PROMPT = """You are a document structural chunking expert. Segment the following document into a two-level hierarchy of blocks (section → paragraph) and return a JSON list. Each element contains 'level' (1=section / 2=paragraph), 'start' (character offset start), 'end' (character offset end).

**Segmentation Principles**:
- Level-1 section boundaries typically correspond to headings like "2.01 Accessibility" / "APPENDIX C";
- Level-2 paragraphs are segmented within each section along semantic boundaries (clauses + tables + notes);
- Each block length ≤ 512 tokens.

Output JSON only, no explanation:
```json
[{{"level": 1, "start": 0, "end": 1200}}, {{"level": 2, "start": 0, "end": 400}}, ...]
```

# Document (length {n_chars} chars)
{doc_preview}
"""


class HiChunkLiteChunker(BaseChunker):
    """One-LLM-call hierarchical chunking (2-level)."""

    name = "HiChunk-lite"

    def __init__(self, max_doc_chars: int = 4000, fallback_size: int = 512,
                 max_llm_docs: int = 30, llm_client=None):
        """
        max_llm_docs: budget for expensive LLM boundary-prediction calls;
        remaining docs use fixed-size fallback. This keeps ingest time
        bounded (arXiv HiChunk paper's cost is dominated by LLM calls;
        sampling is a published HiChunk ablation).
        """
        self.max_doc_chars = max_doc_chars
        self.fallback_size = fallback_size
        self.max_llm_docs = max_llm_docs
        self.llm = llm_client or get_llm_client()

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        """Per-document chunking (invoked by BaseChunker.chunk_documents)."""
        # use a per-instance counter to ensure only max_llm_docs docs get LLM
        if not hasattr(self, "_llm_used"):
            self._llm_used = 0
        text = text or ""
        if self._llm_used >= self.max_llm_docs:
            return self._fixed_fallback_text(doc_id, text)
        preview = text[: self.max_doc_chars]
        try:
            resp = self.llm.generate(PROMPT.format(
                n_chars=len(text), doc_preview=preview,
            ))
            boundaries = self._parse_boundaries(resp, len(text))
            self._llm_used += 1
        except Exception as e:
            logger.warning(f"HiChunk LLM fail {doc_id}: {e}")
            boundaries = []
            self._llm_used += 1

        if not boundaries:
            return self._fixed_fallback_text(doc_id, text)

        out = []
        for i, b in enumerate(boundaries):
            start = max(0, b["start"])
            end = min(len(text), b["end"])
            if end <= start + 20:
                continue
            out.append(Chunk(
                chunk_id=f"{doc_id}_hichunk_L{b['level']}_{i:03d}",
                content=text[start:end],
                doc_id=doc_id,
                start_pos=start, end_pos=end,
                layer=b["level"],
                semantic_type=("section" if b["level"] == 1 else "paragraph"),
            ))
        return out

    def _fixed_fallback_text(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = []
        i = 0
        idx = 0
        while i < len(text):
            end = min(len(text), i + self.fallback_size)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_hichunk_fb_{idx:03d}",
                content=text[i:end],
                doc_id=doc_id,
                start_pos=i, end_pos=end,
                layer=2, semantic_type="paragraph",
            ))
            i = end
            idx += 1
        return chunks

    @staticmethod
    def _parse_boundaries(text: str, n_chars: int) -> list[dict]:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
        except Exception:
            return []
        out = []
        for b in arr:
            if not isinstance(b, dict):
                continue
            lvl = b.get("level")
            start = b.get("start")
            end = b.get("end")
            if lvl not in (1, 2) or not isinstance(start, (int, float)) \
                    or not isinstance(end, (int, float)):
                continue
            out.append({"level": int(lvl),
                        "start": int(start),
                        "end": int(min(end, n_chars))})
        return out
