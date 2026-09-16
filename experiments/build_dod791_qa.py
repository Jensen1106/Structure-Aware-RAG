"""Build test_dod791_30.jsonl — 30-item cross-document test set from DOD-HDBK-791.

DOD-HDBK-791 (Maintainability Design Techniques) is a US military design
handbook. Structure:  `<clause-number> <TITLE>\n<body paragraphs>`.
We parse clauses, pick the first meaningful body sentence as `answer`,
classify each into one of the 12 maintainability constraint types using
keyword rules (reused from build_gjb451b_qa.py), then balance-sample 30.

Output: data/maintqa/test_dod791_30.jsonl  (same schema as test_mil470a_88.jsonl)
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PAGES_FILE = Path("data/dod791/dod791_pages.jsonl")
OUT_FILE = Path("data/maintqa/test_dod791_30.jsonl")

CLAUSE_HEAD_RE = re.compile(
    r"^(\d+-\d+(?:\.\d+)*)\s+([A-Z][A-Z /,\-\(\)&]{3,80})\s*$",
    re.MULTILINE,
)

CONSTRAINT_KEYWORDS = {
    "Accessibility":         ["access", "passage", "clearance", "opening", "reach", "removal"],
    "Interchangeability":    ["interchange", "standardi", "standard part", "modular", "module"],
    "Error-proofing":        ["keying", "error", "mispositioning", "misinstall", "orientation",
                              "marking", "identif"],
    "Maintenance Safety":    ["safety", "hazard", "warning", "shield", "guard", "protect",
                              "ground", "electrical shock"],
    "Diagnostics":           ["BIT", "built-in test", "test point", "fault detection",
                              "diagnos", "isolat", "monitor"],
    "Ergonomics":            ["human", "operator", "posture", "workspace",
                              "anthropom", "illumin"],
    "Repairability":         ["repair", "disassembly", "replace", "remove and", "replaceable"],
    "Reduced Maintenance":   ["reduce maintenance", "minimize maintenance", "reduction of"],
    "Reduced Skill":         ["skill", "training", "technician level"],
    "Time Parameters":       ["MTTR", "mean time", "repair time", "downtime", "time to repair"],
    "Diagnostic Parameters": ["fault detection rate", "FDR", "FIR", "isolation rate",
                              "percent of false", "detection coverage"],
    "Preventive Maintenance":["preventive", "inspection", "servicing", "scheduled maintenance",
                              "lubricat"],
}

CTYPE_IDS = {k: i + 1 for i, k in enumerate(CONSTRAINT_KEYWORDS.keys())}


def classify(text: str) -> tuple[str, int]:
    t = text.lower()
    scores = {}
    for ctype, kws in CONSTRAINT_KEYWORDS.items():
        s = sum(1 for kw in kws if kw.lower() in t)
        if s > 0:
            scores[ctype] = s
    if not scores:
        return "Unknown", 0
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], CTYPE_IDS[best[0]]


def load_full_text(path: Path) -> tuple[str, list[int]]:
    """Return concatenated text + page-start offsets."""
    pages = [json.loads(l) for l in open(path, encoding="utf-8")]
    chunks, offsets, cursor = [], [], 0
    for p in pages:
        t = p.get("text", "")
        chunks.append(t)
        offsets.append(cursor)
        cursor += len(t) + 1  # +1 for joining newline
    return "\n".join(chunks), offsets


def offset_to_page(offset: int, page_offsets: list[int]) -> int:
    # linear scan (len < 300); fine
    page = 1
    for i, po in enumerate(page_offsets, start=1):
        if po > offset:
            return page
        page = i
    return page


CAPS_CONTINUATION_RE = re.compile(
    r"^(?:[A-Z][A-Z /,\-\(\)&]{2,}\s*\n)+"
)


def pick_snippet(body: str, min_len: int = 40, max_len: int = 350) -> str | None:
    """Take a well-formed sentence from body (skip caps-continuation title lines)."""
    # strip continuation lines of a split title (e.g. "AND EVALUATION OF ...")
    body = CAPS_CONTINUATION_RE.sub("", body.lstrip())
    # strip leading sub-clause heading like "7-3.5.1 Importance "
    body = re.sub(r"^\d+-\d+(?:\.\d+)*\s+[A-Z][A-Za-z][^\n]{0,50}\n", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return None
    sents = re.split(r"(?<=[.])\s+", body)
    for s in sents:
        s = s.strip()
        if len(s) < min_len or len(s) > max_len:
            continue
        if s.endswith(":"):
            continue
        # reject sentences that are still mostly capitals (leftover heading)
        letters = [c for c in s if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
            continue
        return s
    if len(body) >= min_len:
        return body[:max_len].rsplit(" ", 1)[0] + "..."
    return None


def build(n_target: int = 30, seed: int = 42) -> list[dict]:
    text, page_offsets = load_full_text(PAGES_FILE)

    matches = list(CLAUSE_HEAD_RE.finditer(text))
    records = []
    for i, m in enumerate(matches):
        number = m.group(1)
        title = m.group(2).strip().rstrip(",-&/ ")
        # body = between this match and the next clause head
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        snippet = pick_snippet(body)
        if not snippet:
            continue
        ctype, ctype_id = classify(title + " " + snippet)
        if ctype == "Unknown":
            continue
        page = offset_to_page(m.start(), page_offsets)
        records.append({
            "number": number,
            "title": title,
            "answer": snippet,
            "constraint_type": ctype,
            "constraint_type_id": ctype_id,
            "page": page,
        })

    print(f"Parsed {len(matches)} clause heads, "
          f"{len(records)} classified to 12 types")
    print("Per-type:", dict(Counter(r["constraint_type"] for r in records)))

    rng = random.Random(seed)
    by_type = {}
    for r in records:
        by_type.setdefault(r["constraint_type"], []).append(r)

    per_type_target = max(1, n_target // len(by_type)) if by_type else 0
    sampled = []
    for ctype, items in by_type.items():
        rng.shuffle(items)
        sampled.extend(items[:per_type_target])
    # top up to n_target
    rest = [r for r in records if r not in sampled]
    rng.shuffle(rest)
    sampled.extend(rest[: max(0, n_target - len(sampled))])
    sampled = sampled[:n_target]

    # build QA
    qa_items = []
    for i, r in enumerate(sampled):
        qa_items.append({
            "id": f"dod791_{i + 1:03d}",
            "query": f"What maintainability design requirement does DOD-HDBK-791 "
                     f"§{r['number']} ({r['title'].title()}) state?",
            "query_en": f"What maintainability design requirement does DOD-HDBK-791 "
                         f"§{r['number']} ({r['title'].title()}) state?",
            "constraint_type": r["constraint_type"],
            "constraint_type_id": r["constraint_type_id"],
            "answer": r["answer"],
            "answer_source": {
                "document": "DOD-HDBK-791",
                "section": f"§{r['number']} {r['title']}",
                "page": r["page"],
                "quote": r["answer"],
            },
            "difficulty": "medium",
            "equipment_types": ["general"],
            "keywords": [r["title"].split()[0].lower()] if r["title"] else [],
            "source_record_type": "clause",
            "source_id": f"dod791_{r['number']}",
        })
    return qa_items


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    qa = build(n_target=30)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for q in qa:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(qa)} items → {OUT_FILE}")
    print("\n=== Preview ===")
    for q in qa[:5]:
        print(f"[{q['id']}] {q['constraint_type']}")
        print(f"   Q: {q['query']}")
        print(f"   A: {q['answer'][:150]}")
        print()
    print("Type distribution:",
          dict(Counter(q["constraint_type"] for q in qa)))


if __name__ == "__main__":
    main()
