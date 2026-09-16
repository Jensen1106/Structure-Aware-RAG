"""MaintQA evaluation benchmark dataset construction and management"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import Counter

from src.config import MAINTQA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnswerSource:
    document: str
    section: str
    page: int | None = None
    quote: str = ""


@dataclass
class MaintQAItem:
    """MaintQA single QA data item"""
    id: str  # e.g., "maintqa_access_001"
    query: str  # Chinese query
    query_en: str  # English query
    constraint_type: str  # constraint type label (one of 12 categories)
    constraint_type_id: int  # constraint type ID (1-12)
    answer: str  # standard answer
    answer_source: AnswerSource = field(default_factory=lambda: AnswerSource("", ""))
    difficulty: str = "medium"  # easy / medium / hard
    equipment_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    related_constraints: list[str] = field(default_factory=list)
    # Ground truth
    relevant_chunk_ids: list[str] = field(default_factory=list)
    num_relevant_chunks: int = 0
    relevant_chunk_keywords: list[str] = field(default_factory=list)  # keywords for GT computation
    maint_relevant_ids: list[str] = field(default_factory=list)  # StdChunk-specific GT


class MaintQADataset:
    """MaintQA dataset manager"""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or MAINTQA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.items: list[MaintQAItem] = []

    def add(self, item: MaintQAItem):
        self.items.append(item)

    def load(self, split: str = "test") -> list[MaintQAItem]:
        """Load from JSONL"""
        path = self.data_dir / f"{split}.jsonl"
        if not path.exists():
            # Try v2
            path = self.data_dir / f"{split}_v2.jsonl"
        if not path.exists():
            logger.warning(f"Dataset file not found: {split}.jsonl or {split}_v2.jsonl")
            return []

        items = []
        known_fields = set(MaintQAItem.__dataclass_fields__.keys())
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    source = data.pop("answer_source", {})
                    # Only pass known fields
                    filtered = {k: v for k, v in data.items() if k in known_fields}
                    item = MaintQAItem(
                        **filtered,
                        answer_source=AnswerSource(**source) if source else AnswerSource("", ""),
                    )
                    items.append(item)
        self.items = items
        logger.info(f"Loaded {len(items)} items from {path}")
        return items

    def save(self, split: str = "test"):
        """Save as JSONL"""
        path = self.data_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                data = asdict(item)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.items)} items to {path}")

    def stats(self) -> dict:
        """Dataset statistics"""
        if not self.items:
            return {}
        return {
            "total": len(self.items),
            "by_type": dict(Counter(item.constraint_type for item in self.items)),
            "by_difficulty": dict(Counter(item.difficulty for item in self.items)),
            "by_category": {
                "qualitative": sum(1 for item in self.items if item.constraint_type_id <= 9),
                "quantitative": sum(1 for item in self.items if item.constraint_type_id > 9),
            },
        }
