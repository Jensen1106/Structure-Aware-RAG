"""Build MaintQA-style candidate items from extracted MIL-HDBK-470A data."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.data.maintqa_builder import AnswerSource
from src.utils.constraint_taxonomy import LABEL_TO_TYPE, classify_query_constraint


BASE_DIR = Path("data/mil470a")

OUTPUT_CANDIDATES = BASE_DIR / "mil470a_maintqa_candidates.jsonl"
OUTPUT_UNMAPPED = BASE_DIR / "mil470a_unmapped_records.jsonl"
OUTPUT_SUMMARY = BASE_DIR / "mil470a_maintqa_summary.json"

CUSTOM_KEYWORDS = {
    "Accessibility": [
        "access",
        "accessible",
        "clearance",
        "viewable",
        "visible",
        "reach",
        "opening",
        "inspect",
        "inspection",
        "removal",
        "replace",
        "replacement",
        "installed",
        "from the ground",
    ],
    "Interchangeability": [
        "interchangeable",
        "interchangeability",
        "standardize",
        "standardization",
        "standardized",
        "same connector",
        "identical positions",
        "common",
        "keyed differently",
    ],
    "Error-proofing": [
        "color code",
        "label",
        "identification",
        "marking",
        "prevent",
        "improper",
        "misalignment",
        "keyed",
        "correct",
        "wrong",
        "visual indication",
    ],
    "Maintenance Safety": [
        "safety",
        "hazard",
        "warning",
        "protect",
        "protective",
        "catastrophic",
        "fire",
        "chemical",
        "biological",
        "radiation",
        "fod",
        "safe/arm",
        "lock",
        "interlock",
    ],
    "Diagnostics": [
        "bit",
        "bite",
        "diagnostic",
        "diagnostics",
        "testability",
        "test point",
        "test points",
        "fault isolate",
        "fault isolation",
        "indicator",
        "sensor",
        "monitor",
        "troubleshooting",
    ],
    "Ergonomics": [
        "human",
        "operator",
        "crew",
        "gloved",
        "hand",
        "lifting",
        "personnel",
        "readable",
        "8th grade",
        "95 percentile",
        "50 percentile",
        "anthropometric",
    ],
    "Repairability": [
        "repair",
        "repairable",
        "repaired",
        "module",
        "modular",
        "lru",
        "sru",
        "wra",
        "sra",
    ],
    "Reduced Maintenance": [
        "no scheduled",
        "maintenance-free",
        "no time-change",
        "no inspection",
        "no functional check flight",
        "no rigging or calibration should be required",
        "should not require scheduled",
    ],
    "Reduced Skill": [
        "by hand",
        "no tools required",
        "no tool required",
        "one hand",
        "one tool",
        "simple",
        "quick release",
        "no torquing",
        "no torque",
        "no special",
        "no calibration",
        "no rigging",
    ],
    "Time Parameters": [
        "mttr",
        "repair time",
        "elapsed time",
        "minutes",
        "min.",
        "hours",
        "mmh/oh",
        "mean time to repair",
    ],
    "Diagnostic Parameters": [
        "fdr",
        "fir",
        "far",
        "false alarm rate",
        "fault detection rate",
        "fault isolation capability",
        "fault isolation rate",
    ],
    "Preventive Maintenance": [
        "preventive maintenance",
        "scheduled maintenance",
        "periodic maintenance",
    ],
}

CATEGORY_FALLBACK = {
    "2.01": "Accessibility",
    "2.02": "Reduced Skill",
    "2.03": "Ergonomics",
    "2.04": "Interchangeability",
    "2.05": "Interchangeability",
    "2.06": "Reduced Maintenance",
    "2.07": "Repairability",
    "2.08": "Diagnostics",
    "2.09": "Diagnostics",
    "2.10": "Diagnostics",
    "2.11": "Preventive Maintenance",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:80] or "item"


def detect_constraint(text: str, category_id: str | None = None) -> tuple[str | None, str, list[str]]:
    scores: Counter[str] = Counter()
    text_lower = text.lower()

    for label, keywords in CUSTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1

    for matched in classify_query_constraint(text):
        scores[matched.label] += 2

    if not scores and category_id:
        for key, label in CATEGORY_FALLBACK.items():
            if category_id == key or category_id.startswith(f"{key}."):
                return label, "category_fallback", []

    if not scores:
        return None, "unmapped", []

    best_label, _score = scores.most_common(1)[0]
    candidates = [label for label, score in scores.most_common(4) if score > 0]
    return best_label, "keyword_match", candidates


def infer_equipment_types(text: str) -> list[str]:
    text_lower = text.lower()
    result = set()
    if "aircraft" in text_lower or "airframe" in text_lower or "engine" in text_lower:
        result.add("aircraft_engine")
    if "ground vehicle" in text_lower or "vehicle" in text_lower or "tank" in text_lower:
        result.add("armored_vehicle")
    if "ship" in text_lower or "naval" in text_lower or "carrier" in text_lower:
        result.add("naval_power")
    if not result:
        result.update(["aircraft_engine", "armored_vehicle", "naval_power"])
    return sorted(result)


def build_keywords(*texts: str) -> list[str]:
    joined = " ".join(t for t in texts if t)
    tokens = re.findall(r"[A-Z]{2,}(?:-[A-Z0-9]+)?|[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", joined)
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= 8:
            break
    return seen


def make_candidate(
    *,
    source_type: str,
    record: dict,
    source_text: str,
    title: str,
    section: str,
    answer: str,
) -> tuple[dict | None, dict | None]:
    constraint_label, method, candidates = detect_constraint(source_text, record.get("category_id"))
    if constraint_label is None:
        return None, {
            "source_type": source_type,
            "source_id": record.get("id"),
            "title": title,
            "section": section,
            "page": record.get("page"),
            "reason": "unmapped_constraint",
        }

    constraint = LABEL_TO_TYPE[constraint_label]
    base = f"{source_type}_{record.get('id', slug(title))}"
    name = constraint.name
    query = f"What {name}-related content does {title} under {section} in MIL-HDBK-470A contain?"
    query_en = f"What does MIL-HDBK-470A say about {title} in {section}?"
    difficulty = "easy" if len(answer) < 180 else "medium"

    candidate = {
        "id": base,
        "query": query,
        "query_en": query_en,
        "constraint_type": constraint.label,
        "constraint_type_id": constraint.id,
        "answer": answer,
        "answer_source": {
            "document": "MIL-HDBK-470A",
            "section": section,
            "page": record.get("page"),
            "quote": answer[:400],
        },
        "difficulty": difficulty,
        "equipment_types": infer_equipment_types(f"{title} {answer} {section}"),
        "keywords": build_keywords(title, answer, section),
        "related_constraints": [],
        "source_record_type": source_type,
        "source_id": record.get("id"),
        "mapping_method": method,
        "mapping_candidates": candidates,
    }
    return candidate, None


def build_all() -> tuple[list[dict], list[dict]]:
    appendix_a = load_jsonl(BASE_DIR / "mil470a_appendix_a_templates.jsonl")
    appendix_c = load_jsonl(BASE_DIR / "mil470a_appendix_c_guidelines.jsonl")
    appendix_d = load_jsonl(BASE_DIR / "mil470a_appendix_d_actions.jsonl")
    clauses = load_jsonl(BASE_DIR / "mil470a_clauses.jsonl")

    candidates: list[dict] = []
    unmapped: list[dict] = []

    for row in appendix_a:
        section = row["section"]
        title = row["title"]
        answer = row["text"].strip()
        if not answer:
            continue
        cand, miss = make_candidate(
            source_type="appendix_a",
            record=row,
            source_text=f"{section} {title} {answer}",
            title=title,
            section=section,
            answer=answer,
        )
        if cand:
            candidates.append(cand)
        if miss:
            unmapped.append(miss)

    for row in appendix_c:
        title = row["guideline_id"]
        section = f"{row['category_id']} {row['category_title']}"
        answer = row["guideline_text"].strip()
        cand, miss = make_candidate(
            source_type="appendix_c",
            record=row,
            source_text=f"{row['category_title']} {row['guideline_id']} {answer}",
            title=title,
            section=section,
            answer=answer,
        )
        if cand:
            candidates.append(cand)
        if miss:
            unmapped.append(miss)

    for row in appendix_d:
        title = row["title"]
        section = row.get("figure_id") or "Appendix D"
        answer = row["text"].strip()
        cand, miss = make_candidate(
            source_type="appendix_d",
            record=row,
            source_text=f"{title} {answer} repair time maintenance action tools",
            title=title,
            section=section,
            answer=answer,
        )
        if cand:
            candidates.append(cand)
        if miss:
            unmapped.append(miss)

    for row in clauses:
        title = row["title"]
        section = row["clause_id"]
        answer = row["text"].strip()
        if len(answer) < 40:
            continue
        cand, miss = make_candidate(
            source_type="clause",
            record=row,
            source_text=f"{title} {answer}",
            title=title,
            section=section,
            answer=answer,
        )
        if cand:
            candidates.append(cand)
        if miss:
            unmapped.append(miss)

    return candidates, unmapped


def main() -> None:
    candidates, unmapped = build_all()

    with OUTPUT_CANDIDATES.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUTPUT_UNMAPPED.open("w", encoding="utf-8") as f:
        for row in unmapped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_counts = Counter(row["constraint_type"] for row in candidates)
    source_counts = Counter(row["source_record_type"] for row in candidates)
    summary = {
        "total_candidates": len(candidates),
        "total_unmapped": len(unmapped),
        "by_constraint_type": dict(type_counts),
        "by_source_type": dict(source_counts),
    }
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
