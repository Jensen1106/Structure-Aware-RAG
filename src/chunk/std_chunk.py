"""StdChunk: Structure-aware chunking strategy for standards/maintainability documents (Innovation 1)

Three-layer chunking architecture:
  Layer 1 - Section Block (coarse-grained): corresponds to document sections
  Layer 2 - Paragraph Block (medium-grained): corresponds to clauses within sections
  Layer 3 - Constraint Block (fine-grained): corresponds to table/numeric constraints

Features:
  - Preserves table integrity (no splitting within tables)
  - Preserves maintenance procedure step sequences
  - Semantic type labeling (maintenance_procedure / design_guideline / constraint_specification)
  - Constraint type tags (12 categories)
  - Cross-layer reference linking ("see Table 3-1" → auto-association)
"""

import re
from src.chunk.base import BaseChunker, Chunk
from src.config import DEFAULT_CONFIG
from src.data.preprocessor import Preprocessor
from src.utils.constraint_taxonomy import classify_query_constraint, CONSTRAINT_TYPES


class StdChunker(BaseChunker):
    """Structure-aware chunker for standards/maintainability documents"""

    name = "StdChunk"

    # Semantic type keywords
    PROCEDURE_KEYWORDS = ["procedure", "step", "operation", "disassembly", "installation", "replacement"]
    GUIDELINE_KEYWORDS = ["design guideline", "design requirement", "guideline", "requirement", "shall", "should"]
    CONSTRAINT_KEYWORDS = ["≥", "≤", "≮", "≯", "%", "mm", "kg", "N", "h", "metric", "parameter", "threshold"]

    def __init__(
        self,
        section_max: int | None = None,
        paragraph_max: int | None = None,
        constraint_max: int | None = None,
    ):
        cfg = DEFAULT_CONFIG.chunk
        self.section_max = section_max or cfg.maint_section_max
        self.paragraph_max = paragraph_max or cfg.maint_paragraph_max
        self.constraint_max = constraint_max or cfg.maint_constraint_max
        self.preprocessor = Preprocessor()

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        """Three-layer structure-aware chunking"""
        all_chunks: list[Chunk] = []

        # Layer 1: Section Blocks
        section_chunks = self._split_sections(text, doc_id)

        for sec_chunk in section_chunks:
            all_chunks.append(sec_chunk)

            # Layer 2: Paragraph Blocks
            para_chunks = self._split_paragraphs(sec_chunk)
            for para_chunk in para_chunks:
                all_chunks.append(para_chunk)

                # Layer 3: Constraint Blocks
                constraint_chunks = self._extract_constraints(para_chunk)
                for con_chunk in constraint_chunks:
                    all_chunks.append(con_chunk)
                    para_chunk.child_ids.append(con_chunk.chunk_id)

                sec_chunk.child_ids.append(para_chunk.chunk_id)

        # Build cross-layer references
        self._build_cross_references(all_chunks)

        # Label constraint types
        for chunk in all_chunks:
            chunk.constraint_types = [
                ct.label for ct in classify_query_constraint(chunk.content)
            ]

        return all_chunks

    def _split_sections(self, text: str, doc_id: str) -> list[Chunk]:
        """Layer 1: Split by section headings"""
        section_pattern = re.compile(
            r"(?=(?:^|\n)(?:§?\d+(?:\.\d+)*|第[一二三四五六七八九十]+[章节条]|#{1,4})\s+)",
        )
        parts = section_pattern.split(text)
        parts = [p.strip() for p in parts if p.strip()]

        chunks = []
        for i, part in enumerate(parts):
            # Truncate overly long sections
            content = part[:self.section_max] if len(part) > self.section_max else part
            semantic_type = self._classify_semantic_type(content)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_L1_{i:03d}",
                content=content,
                doc_id=doc_id,
                layer=1,
                semantic_type=semantic_type,
            ))
        return chunks

    def _split_paragraphs(self, section_chunk: Chunk) -> list[Chunk]:
        """Layer 2: Split by paragraphs/clauses within a section"""
        text = section_chunk.content
        # Split by double newlines or clause numbering
        para_pattern = re.compile(r"\n\n+|(?=\n\s*[a-z]\)|(?=\n\s*\d+\)))")
        parts = para_pattern.split(text)
        parts = [p.strip() for p in parts if p.strip()]

        chunks = []
        for i, part in enumerate(parts):
            if len(part) > self.paragraph_max:
                # Further split by sentence if too long
                sub_parts = self._split_by_sentence(part, self.paragraph_max)
                for j, sp in enumerate(sub_parts):
                    chunks.append(Chunk(
                        chunk_id=f"{section_chunk.chunk_id}_L2_{i:03d}_{j:02d}",
                        content=sp,
                        doc_id=section_chunk.doc_id,
                        layer=2,
                        semantic_type=self._classify_semantic_type(sp),
                        parent_id=section_chunk.chunk_id,
                    ))
            else:
                chunks.append(Chunk(
                    chunk_id=f"{section_chunk.chunk_id}_L2_{i:03d}",
                    content=part,
                    doc_id=section_chunk.doc_id,
                    layer=2,
                    semantic_type=self._classify_semantic_type(part),
                    parent_id=section_chunk.chunk_id,
                ))
        return chunks

    def _extract_constraints(self, para_chunk: Chunk) -> list[Chunk]:
        """Layer 3: Extract constraint value blocks from paragraphs (table rows, numeric constraints)"""
        text = para_chunk.content
        chunks = []

        # Detect table content (containing | delimiters or consecutive numeric lines)
        table_pattern = re.compile(r"(\|.+\|(?:\n\|.+\|)*)", re.MULTILINE)
        for i, match in enumerate(table_pattern.finditer(text)):
            table_text = match.group(1).strip()
            if len(table_text) <= self.constraint_max:
                chunks.append(Chunk(
                    chunk_id=f"{para_chunk.chunk_id}_L3_tbl_{i:02d}",
                    content=table_text,
                    doc_id=para_chunk.doc_id,
                    layer=3,
                    semantic_type="constraint_specification",
                    parent_id=para_chunk.chunk_id,
                ))

        # Detect numeric constraint sentences (containing ≥ ≤ etc. symbols and numeric+unit)
        numeric_pattern = re.compile(
            r"[^。\n]*(?:≥|≤|≮|≯|>=|<=|不[超大小]于|至少|最[大小多少])\s*\d+[.\d]*\s*(?:mm|cm|m|kg|N|h|%|°|步)[^。\n]*[。]?",
        )
        for i, match in enumerate(numeric_pattern.finditer(text)):
            constraint_text = match.group(0).strip()
            if constraint_text and len(constraint_text) <= self.constraint_max:
                chunks.append(Chunk(
                    chunk_id=f"{para_chunk.chunk_id}_L3_num_{i:02d}",
                    content=constraint_text,
                    doc_id=para_chunk.doc_id,
                    layer=3,
                    semantic_type="constraint_specification",
                    parent_id=para_chunk.chunk_id,
                ))

        return chunks

    def _build_cross_references(self, chunks: list[Chunk]):
        """Build cross-layer reference links"""
        chunk_map = {c.chunk_id: c for c in chunks}

        cross_ref_pattern = re.compile(
            r"见表\s*(\d+[-–]?\d*)|see\s+Table\s+(\d+[-–]?\d*)|"
            r"见\s*§?(\d+(?:\.\d+)*)|see\s+§?(\d+(?:\.\d+)*)",
            re.IGNORECASE,
        )

        for chunk in chunks:
            refs = cross_ref_pattern.findall(chunk.content)
            for ref_groups in refs:
                ref_id = next((r for r in ref_groups if r), None)
                if ref_id:
                    # Search across all chunks for matching reference target
                    for target in chunks:
                        if ref_id in target.content[:100]:  # Match at the beginning
                            if target.chunk_id != chunk.chunk_id:
                                chunk.cross_refs.append(target.chunk_id)
                                break

    def _classify_semantic_type(self, text: str) -> str:
        """Determine semantic type"""
        text_lower = text.lower()
        proc_score = sum(1 for kw in self.PROCEDURE_KEYWORDS if kw in text_lower)
        guide_score = sum(1 for kw in self.GUIDELINE_KEYWORDS if kw in text_lower)
        const_score = sum(1 for kw in self.CONSTRAINT_KEYWORDS if kw in text)

        if const_score >= 2:
            return "constraint_specification"
        elif proc_score > guide_score:
            return "maintenance_procedure"
        else:
            return "design_guideline"

    def _split_by_sentence(self, text: str, max_size: int) -> list[str]:
        """Split long text by sentence"""
        sentences = re.split(r"(?<=[。！？.!?])", text)
        parts = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= max_size:
                current += sent
            else:
                if current:
                    parts.append(current)
                current = sent
        if current:
            parts.append(current)
        return parts
