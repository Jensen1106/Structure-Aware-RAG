"""Simple MaintQA dataset generation tool (based on existing documents)

Extract typical QA pairs from GJB 368B and GJB 451B
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_directory
from src.data.maintqa_builder import MaintQADataset, MaintQAItem, AnswerSource
from src.config import RAW_DIR, MAINTQA_DIR
from src.utils.logger import get_logger

logger = get_logger("build_maintqa")


def build_sample_dataset():
    """Build sample MaintQA dataset (manually annotated seed data)"""

    # Sample QA pairs (based on GJB 368B)
    sample_items = [
        MaintQAItem(
            id="maintqa_access_001",
            query="What is the minimum requirement for maintenance passage width?",
            query_en="What is the minimum maintenance passage width requirement?",
            constraint_type="Accessibility",
            constraint_type_id=1,
            answer="500mm",
            answer_source=AnswerSource(
                document="GJB 368B",
                section="§4.6.2",
                quote="Maintenance passage width shall be no less than 500 mm",
            ),
            difficulty="easy",
            equipment_types=["aircraft_engine", "naval_power"],
            keywords=["accessibility", "passage", "width"],
        ),
        MaintQAItem(
            id="maintqa_access_002",
            query="What is the minimum requirement for hand-access clearance?",
            query_en="What is the minimum hand reach gap requirement?",
            constraint_type="Accessibility",
            constraint_type_id=1,
            answer="50 mm (finger operation) or 100 mm (tool operation)",
            answer_source=AnswerSource(
                document="GJB 368B",
                section="§4.6.2",
                quote="Hand-access clearance shall be no less than 50 mm (finger operation) or 100 mm (tool operation)",
            ),
            difficulty="easy",
            equipment_types=["aircraft_engine"],
            keywords=["accessibility", "clearance", "hand access"],
        ),
        MaintQAItem(
            id="maintqa_time_001",
            query="What is the definition of MTTR?",
            query_en="What is the definition of MTTR?",
            constraint_type="Time Parameters",
            constraint_type_id=10,
            answer="Mean Time To Repair — the average time required from the start of maintenance to restoration of normal operating condition after a product failure",
            answer_source=AnswerSource(
                document="GJB 368B",
                section="§3.1",
            ),
            difficulty="easy",
            equipment_types=["aircraft_engine", "naval_power", "armored_vehicle"],
            keywords=["MTTR", "mean time to repair", "definition"],
        ),
        MaintQAItem(
            id="maintqa_diag_001",
            query="What is the requirement for the fault detection rate (FDR)?",
            query_en="What is the requirement for Fault Detection Rate (FDR)?",
            constraint_type="Diagnostic Parameters",
            constraint_type_id=11,
            answer="≥ 95% (typical value); the specific value is specified by contract",
            answer_source=AnswerSource(
                document="GJB 368B",
                section="§4.6.1",
            ),
            difficulty="medium",
            equipment_types=["aircraft_engine"],
            keywords=["FDR", "fault detection rate", "diagnostics"],
        ),
        MaintQAItem(
            id="maintqa_ergo_001",
            query="What is the maximum weight limit for an LRU?",
            query_en="What is the maximum weight limit for LRU?",
            constraint_type="Ergonomics",
            constraint_type_id=6,
            answer="≤ 25 kg (general requirement); aircraft engines typically require ≤ 20 kg",
            answer_source=AnswerSource(
                document="DOD-HDBK-791",
                section="§3.5",
            ),
            difficulty="medium",
            equipment_types=["aircraft_engine"],
            keywords=["LRU", "weight", "human factors engineering"],
        ),
    ]

    # Save
    dataset = MaintQADataset()
    dataset.items = sample_items
    dataset.save("test")

    logger.info(f"Created sample MaintQA dataset with {len(sample_items)} items")
    logger.info(f"Saved to {MAINTQA_DIR / 'test.jsonl'}")
    logger.info(f"Stats: {dataset.stats()}")


if __name__ == "__main__":
    build_sample_dataset()
