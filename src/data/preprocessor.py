"""Text preprocessing: cleaning, segmentation, table detection"""

import re
from dataclasses import dataclass

from src.data.loader import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Section:
    """Document section"""
    section_id: str  # e.g., "§4.6.2"
    title: str
    level: int  # hierarchy depth 1-4
    content: str
    parent_id: str | None = None
    children_ids: list[str] | None = None
    tables: list[dict] | None = None


class Preprocessor:
    """Document preprocessor"""

    # Section numbering patterns (support Chinese and English standards)
    SECTION_PATTERNS = [
        re.compile(r"^(§?\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE),  # §4.6.2 Title
        re.compile(r"^(第[一二三四五六七八九十]+[章节条])\s*(.+)$", re.MULTILINE),  # Chapter X
        re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE),  # 4.6.2 Title
        re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),  # Markdown heading
    ]

    # Table detection
    TABLE_PATTERN = re.compile(
        r"(表\s*\d+[-–]\d+|Table\s+\d+[-–]\d+|表\s*\d+|Table\s+\d+)",
        re.IGNORECASE,
    )

    # Cross reference
    CROSS_REF_PATTERN = re.compile(
        r"(见表\s*\d+[-–]?\d*|see\s+Table\s+\d+[-–]?\d*|"
        r"见\s*§?\d+(?:\.\d+)*|see\s+§?\d+(?:\.\d+)*|"
        r"按照?\s*§?\d+(?:\.\d+)*|per\s+§?\d+(?:\.\d+)*)",
        re.IGNORECASE,
    )

    def clean_text(self, text: str) -> str:
        """Text cleaning"""
        text = re.sub(r"\n{3,}", "\n\n", text)  # extra blank lines
        text = re.sub(r"[ \t]+", " ", text)  # extra spaces
        text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)  # blank lines
        return text.strip()

    def extract_sections(self, doc: Document) -> list[Section]:
        """Extract section structure from document"""
        content = self.clean_text(doc.content)
        sections = []

        # Find all section headings
        matches = []
        for pattern in self.SECTION_PATTERNS:
            for m in pattern.finditer(content):
                level = self._detect_level(m.group(1))
                matches.append((m.start(), m.group(1), m.group(2), level))

        # Sort by position
        matches.sort(key=lambda x: x[0])

        # Split content
        for i, (start, sec_id, title, level) in enumerate(matches):
            end = matches[i + 1][0] if i + 1 < len(matches) else len(content)
            sec_content = content[start:end].strip()

            sections.append(Section(
                section_id=sec_id.strip(),
                title=title.strip(),
                level=level,
                content=sec_content,
            ))

        if not sections:
            sections.append(Section(
                section_id="root",
                title=doc.title,
                level=0,
                content=content,
            ))

        logger.info(f"Extracted {len(sections)} sections from {doc.doc_id}")
        return sections

    def extract_cross_references(self, text: str) -> list[str]:
        """Extract cross references from text"""
        return self.CROSS_REF_PATTERN.findall(text)

    def detect_tables(self, text: str) -> list[str]:
        """Detect table references in text"""
        return self.TABLE_PATTERN.findall(text)

    def _detect_level(self, section_id: str) -> int:
        """Determine hierarchy depth from section number"""
        if section_id.startswith("#"):
            return len(section_id)
        parts = section_id.replace("§", "").split(".")
        return len(parts)
