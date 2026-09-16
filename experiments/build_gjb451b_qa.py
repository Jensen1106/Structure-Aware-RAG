"""S3b Data Preparation: Construct 30 cross-domain test QA pairs from GJB 451B text.

GJB 451B "General Quality Characteristics Terminology for Equipment" is the Chinese military standard
for maintainability terminology, which differs from the training domain MIL-HDBK-470A
(in language, standard system, and document structure).

Construction method: Automatically extract QA pairs based on the "term + definition" structure in the document:
1. Parse sections (e.g., "6.1 Characteristics", "6.2 Parameters")
2. Match the "term: definition" pattern within each section
3. Construct queries like "What is the definition of {term} in GJB 451B?", with the answer being the definition text
4. Classify and annotate according to 12 constraint types (based on keyword matching)

Output: data/maintqa/test_gjb451b_30.jsonl
"""

from __future__ import annotations

import json
import sys
import re
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MAINTQA_DIR
from src.utils.constraint_taxonomy import classify_query_constraint
from src.utils.logger import get_logger

logger = get_logger("GJBQA")


# --- Parsing -------------------------------------------------------------------

# Section title pattern: "§6.1 Characteristics" / "6.1 Characteristics" / "6.1"
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([^\n]{2,40})\s*$", re.MULTILINE)

# Term: Definition matching pattern (Chinese term + English/definition)
# Common formats:
#   "3.1.1 Reliability reliability  Definition..."
#   "3.1.2 Maintainability  Definition..."
TERM_DEF_RE = re.compile(
    r"(\d+\.\d+(?:\.\d+)*)\s+"          # Number
    r"([一-鿿][一-鿿 A-Za-z（）()\-]{1,15})"  # Chinese term (with parentheses and English)
    r"\s+"
    r"([^0-9\n]{5,})",                 # Definition (does not start with a number)
    re.MULTILINE,
)


def load_pages(path: Path) -> str:
    """Concatenate all page texts into a single document."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            blocks.append(row.get("text", ""))
    return "\n".join(blocks)


def clean_text(text: str) -> str:
    """Remove common PDF noise: dot lines, continuous dots, extra spaces."""
    text = re.sub(r"\.{3,}\s*\d+", "", text)      # Table of contents dot lines "........... 12"
    text = re.sub(r"·{3,}\s*\d+", "", text)      # Chinese table of contents dots
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[•·]+", " ", text)
    return text


# --- Keyword -> Constraint Type Classification ---------------------------------

# 12 constraint types
CONSTRAINT_KEYWORDS = {
    "Accessibility": ["accessibility", "passage", "clearance", "maintenance space"],
    "Interchangeability": ["interchangeability", "standardization", "standard parts"],
    "Error-proofing": ["error-proofing", "identification", "error", "identification marking"],
    "Maintenance Safety": ["maintenance safety", "safety", "protection", "warning"],
    "Diagnostics": ["detection", "diagnostics", "diagnostic", "BIT", "fault detection"],
    "Ergonomics": ["human factors", "man-machine", "ergonomics", "ergonomic"],
    "Repairability": ["repairable", "repair", "repair", "modularity"],
    "Reduced Maintenance": ["reduced maintenance", "preventive", "maintenance content"],
    "Reduced Skill": ["reduced skill", "skill", "training"],
    "Time Parameters": ["time", "MTTR", "Mmax", "MTBF", "mean time to repair"],
    "Diagnostic Parameters": ["fault detection rate", "FDR", "FIR", "isolation rate"],
    "Preventive Maintenance": ["preventive", "servicing", "scheduled maintenance"],
}


def classify_constraint(text: str) -> tuple[str, int]:
    """Classify into one of the 12 types based on keywords."""
    scores = {}
    text_lower = text.lower()
    for ctype, kws in CONSTRAINT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw.lower() in text_lower)
        if score > 0:
            scores[ctype] = score
    if not scores:
        return "Unknown", 0
    best = max(scores.items(), key=lambda x: x[1])
    ctype_ids = {k: i + 1 for i, k in enumerate(CONSTRAINT_KEYWORDS.keys())}
    return best[0], ctype_ids[best[0]]


# --- Main Pipeline -------------------------------------------------------------

def build_qa(pages_file: Path, n_target: int = 30, seed: int = 42) -> list[dict]:
    """Build QA pairs."""
    raw_text = load_pages(pages_file)
    text = clean_text(raw_text)
    logger.info(f"Cleaned text: {len(text)} chars")

    # Extract term-definition pairs
    term_defs = []
    for match in TERM_DEF_RE.finditer(text):
        number = match.group(1).strip()
        term = match.group(2).strip()
        definition = match.group(3).strip()
        # Truncate definition to within 120 characters (take the first sentence)
        definition = re.split(r"[。；]", definition)[0]
        if len(definition) < 10 or len(definition) > 200:
            continue
        if len(term) < 2:
            continue
        term_defs.append({
            "number": number,
            "term": term,
            "definition": definition,
            "raw_match": match.group(0)[:200],
        })

    logger.info(f"Extracted {len(term_defs)} candidate term-def pairs")

    # Classify and filter
    classified = []
    for td in term_defs:
        ctype, ctype_id = classify_constraint(td["term"] + " " + td["definition"])
        if ctype == "Unknown":
            continue
        classified.append({**td, "constraint_type": ctype, "constraint_type_id": ctype_id})

    logger.info(f"Classified to 12 types: {len(classified)}")
    # Print sample count per type
    from collections import Counter
    ctr = Counter(c["constraint_type"] for c in classified)
    for k, v in ctr.most_common():
        logger.info(f"  {k}: {v}")

    # Stratified sampling of n_target items by type
    rng = random.Random(seed)
    by_type = {}
    for c in classified:
        by_type.setdefault(c["constraint_type"], []).append(c)

    sampled = []
    target_per_type = max(1, n_target // len(by_type))
    for ctype, items in by_type.items():
        pick = rng.sample(items, min(target_per_type, len(items)))
        sampled.extend(pick)

    # Pad to n_target
    rest = [c for c in classified if c not in sampled]
    rng.shuffle(rest)
    while len(sampled) < n_target and rest:
        sampled.append(rest.pop())
    sampled = sampled[:n_target]

    # Construct QA pairs
    qa_items = []
    for i, c in enumerate(sampled):
        qa_items.append({
            "id": f"gjb451b_{i+1:03d}",
            "query": f"What is the definition of {c['term']} in GJB 451B?",
            "query_en": f"What is the definition of {c['term']} in GJB 451B?",
            "constraint_type": c["constraint_type"],
            "constraint_type_id": c["constraint_type_id"],
            "answer": c["definition"],
            "answer_source": {
                "document": "GJB 451B",
                "section": c["number"],
                "quote": c["definition"],
            },
            "difficulty": "medium",
            "equipment_types": ["general"],
            "keywords": [c["term"]],
            "source_record_type": "term_def",
            "source_id": f"gjb451b_{c['number']}",
        })

    return qa_items


def main():
    pages_file = Path("data/gjb451b/gjb451b_pages.jsonl")
    out_file = MAINTQA_DIR / "test_gjb451b_30.jsonl"

    qa_items = build_qa(pages_file, n_target=30)
    logger.info(f"Built {len(qa_items)} QA items")

    with open(out_file, "w", encoding="utf-8") as f:
        for q in qa_items:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {out_file}")

    print("\n" + "=" * 70)
    print("Sample QA items:")
    print("=" * 70)
    for q in qa_items[:5]:
        print(f"\n[{q['id']}] Type: {q['constraint_type']}")
        print(f"  Q: {q['query']}")
        print(f"  A: {q['answer'][:100]}...")


if __name__ == "__main__":
    main()
