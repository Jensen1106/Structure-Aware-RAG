"""Filter high-confidence MaintQA-style samples from MIL-HDBK-470A candidates."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.constraint_taxonomy import LABEL_TO_TYPE


BASE_DIR = Path("data/mil470a")
INPUT_CANDIDATES = BASE_DIR / "mil470a_maintqa_candidates.jsonl"

OUTPUT_HIGHCONF = BASE_DIR / "mil470a_maintqa_highconf.jsonl"
OUTPUT_BALANCED = BASE_DIR / "mil470a_maintqa_balanced_30.jsonl"
OUTPUT_REJECTED = BASE_DIR / "mil470a_maintqa_rejected.jsonl"
OUTPUT_SUMMARY = BASE_DIR / "mil470a_maintqa_highconf_summary.json"

TYPE_PATTERNS = {
    "Accessibility": [
        r"\baccess\b",
        r"accessible",
        r"clearance",
        r"viewable",
        r"visible",
        r"access opening",
        r"inspection",
        r"inspect",
        r"replacement",
        r"remove",
        r"installed",
        r"ground level",
        r"percentile male hand",
    ],
    "Interchangeability": [
        r"interchange",
        r"interchangeability",
        r"standardi[sz]",
        r"standardized",
        r"identical",
        r"same connector",
        r"same requirements",
        r"common",
    ],
    "Error-proofing": [
        r"identification",
        r"marking",
        r"label",
        r"color code",
        r"visual indication",
        r"keyed",
        r"alignment",
        r"improper",
        r"wrong",
        r"reverse",
    ],
    "Maintenance Safety": [
        r"safety",
        r"hazard",
        r"warning",
        r"protect",
        r"protective",
        r"catastrophic",
        r"fire",
        r"explosive",
        r"chemical",
        r"biological",
        r"radiation",
        r"fod",
        r"safe/arm",
        r"lock",
        r"interlock",
    ],
    "Diagnostics": [
        r"\bbit\b",
        r"\bbite\b",
        r"diagnostic",
        r"testability",
        r"test point",
        r"test points",
        r"fault isolate",
        r"fault isolation",
        r"sensor",
        r"monitor",
        r"self-test",
        r"indicator",
    ],
    "Ergonomics": [
        r"human",
        r"gloved",
        r"hand",
        r"lift",
        r"personnel",
        r"operator",
        r"crew",
        r"readable",
        r"percentile",
        r"anthropometric",
        r"95 percentile",
        r"50 percentile",
    ],
    "Repairability": [
        r"repair",
        r"repairable",
        r"repaired",
        r"module",
        r"modular",
        r"\blru\b",
        r"\bsru\b",
        r"\bwra\b",
        r"\bsra\b",
    ],
    "Reduced Maintenance": [
        r"should not require",
        r"no scheduled",
        r"maintenance-free",
        r"no time-change",
        r"no functional check",
        r"should not be planned solely",
        r"without removal",
    ],
    "Reduced Skill": [
        r"by hand",
        r"one hand",
        r"one tool",
        r"no tools required",
        r"no tool required",
        r"no torquing",
        r"no torque",
        r"no rigging",
        r"no calibration",
        r"quick release",
        r"simple",
    ],
    "Time Parameters": [
        r"\bmttr\b",
        r"repair time",
        r"mean time to repair",
        r"elapsed time",
        r"\bminutes?\b",
        r"\bmin\.\b",
        r"\bhours?\b",
        r"\b\d+\s*minutes?\b",
        r"\b\d+\s*min\.\b",
        r"\b\d+\s*hours?\b",
    ],
    "Diagnostic Parameters": [
        r"\bfdr\b",
        r"\bfir\b",
        r"\bfar\b",
        r"false alarm rate",
        r"fault detection rate",
        r"fault isolation capability",
        r"fault isolation rate",
        r"\b98%\b",
        r"\b99%\b",
        r"\b100%\b fault detection",
    ],
    "Preventive Maintenance": [
        r"preventive maintenance",
        r"scheduled maintenance",
        r"periodic maintenance",
        r"inspection",
        r"time-change",
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def answer_text(row: dict) -> str:
    return f"{row.get('answer', '')} {row.get('query', '')} {row.get('query_en', '')}".lower()


def is_highconf(row: dict) -> tuple[bool, str]:
    if row["source_record_type"] not in {"appendix_c", "appendix_d"}:
        return False, "source_not_primary"

    if "ocrmiss" in row["source_id"].lower():
        return False, "ocr_missing_guideline_id"

    answer = row.get("answer", "").strip()
    if len(answer) < 20:
        return False, "answer_too_short"
    if len(answer) > 500:
        return False, "answer_too_long"

    text = answer_text(row)
    constraint = row["constraint_type"]
    patterns = TYPE_PATTERNS.get(constraint, [])
    if patterns and not any(re.search(p, text, re.I) for p in patterns):
        return False, "weak_constraint_signal"

    if row["source_record_type"] == "appendix_d" and row["constraint_type"] not in {
        "Reduced Skill",
        "Time Parameters",
        "Diagnostics",
        "Repairability",
    }:
        return False, "appendix_d_type_mismatch"

    return True, "accepted"


def sort_key(row: dict) -> tuple:
    source_rank = 0 if row["source_record_type"] == "appendix_d" else 1
    answer_len = len(row["answer"])
    mapping_rank = 0 if len(row.get("mapping_candidates", [])) <= 2 else 1
    return (source_rank, mapping_rank, answer_len, row["id"])


def main() -> None:
    rows = load_jsonl(INPUT_CANDIDATES)

    highconf: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        ok, reason = is_highconf(row)
        if ok:
            highconf.append(row)
        else:
            rejected.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "source_record_type": row["source_record_type"],
                "constraint_type": row["constraint_type"],
                "reason": reason,
            })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in highconf:
        groups[row["constraint_type"]].append(row)
    for rows_in_type in groups.values():
        rows_in_type.sort(key=sort_key)

    balanced: list[dict] = []
    for label in LABEL_TO_TYPE:
        balanced.extend(groups.get(label, [])[:30])

    with OUTPUT_HIGHCONF.open("w", encoding="utf-8") as f:
        for row in highconf:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUTPUT_BALANCED.open("w", encoding="utf-8") as f:
        for row in balanced:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUTPUT_REJECTED.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_input": len(rows),
        "total_highconf": len(highconf),
        "total_balanced_30": len(balanced),
        "total_rejected": len(rejected),
        "highconf_by_constraint_type": dict(Counter(r["constraint_type"] for r in highconf)),
        "balanced_by_constraint_type": dict(Counter(r["constraint_type"] for r in balanced)),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
    }

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
