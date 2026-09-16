"""Build test_mil470a_200 — 200-item independent main test set.

Motivation: The existing test_mil470a_88 shares source_ids with
train_mil470a_highconf (88/88 overlap). Reviewers will attack this. We
build a *disjoint* 200-item test set from the 7,892 Appendix-C guidelines
that were NEVER in train_highconf. 12-class balanced + difficulty tagged.

Output: data/maintqa/test_mil470a_200.jsonl
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

APPENDIX_C = Path("data/mil470a/mil470a_appendix_c_guidelines.jsonl")
TRAIN_FILE = Path("data/maintqa/train_mil470a_highconf.jsonl")
TEST88_FILE = Path("data/maintqa/test_mil470a_88.jsonl")
OUT_FILE = Path("data/maintqa/test_mil470a_200.jsonl")

# category_title (as found in appendix_c) → canonical constraint_type
# Covers all 12 classes — for categories that aren't obviously mapped,
# we use the *design-technique-prefix* of the guideline_id (e.g. BIT-* →
# Diagnostics, CONT-* → Accessibility-ish contact requirement, etc.).
CATEGORY_TO_CONSTRAINT = {
    "Accessibility":                          ("Accessibility", 1),
    "Access Doors, Panels, and Openings":     ("Accessibility", 1),
    "Weapon Bays, Racks, Compartments,":      ("Accessibility", 1),
    "External and Parasitic Tanks, Pods,":    ("Accessibility", 1),
    "Cargo Holds, Storage Bays, and Storage": ("Accessibility", 1),
    "Human Factors (including":               ("Ergonomics", 6),
    "Man-Machine Interfaces":                 ("Ergonomics", 6),
    "Controls":                               ("Ergonomics", 6),
    "Crew Stations, Crew Cabs, Cockpits, and": ("Ergonomics", 6),
    "Personnel Seats (Crew and Passenger),":  ("Ergonomics", 6),
    "Mating and Connections":                 ("Interchangeability", 2),
    "Standardization and Interchangeability": ("Interchangeability", 2),
    "Connections":                            ("Interchangeability", 2),
    "Simplification":                         ("Reduced Maintenance", 8),
    "Identification and Labeling":            ("Error-proofing", 3),
    "Environmental Factors":                  ("Preventive Maintenance", 12),
    "Support and Ground Handling":            ("Preventive Maintenance", 12),
    "Environmental Control, Air":             ("Preventive Maintenance", 12),
    "External Test Equipment":                ("Diagnostics", 5),
    "Avionics and Electronics":               ("Diagnostics", 5),
    "Antennas, Apertures, and Sensors":       ("Diagnostics", 5),
    "Power":                                  ("Maintenance Safety", 4),
    "Control Rods, Cables, and Controlex":    ("Maintenance Safety", 4),
    "Cartridge Actuated Devices, Shaped":     ("Maintenance Safety", 4),
    "Weapons, Guns, Flares, Chaff, and":      ("Maintenance Safety", 4),
    "Extinguishing Agents, Containers,":      ("Maintenance Safety", 4),
    "Safe/Arm Devices - Non Weapon":          ("Maintenance Safety", 4),
    "Fluid Systems":                          ("Maintenance Safety", 4),
    "Fuel Systems, Tanks, Containers, Pumps,": ("Maintenance Safety", 4),
    "Structures, Airframes, Bodies, Chassis,": ("Repairability", 7),
    "Structures":                             ("Repairability", 7),
    "Engines (Turbine-driven)":               ("Repairability", 7),
    "Engines (Gasoline and Diesel)":          ("Repairability", 7),
    "Transmissions, Clutches, and Rotors":    ("Repairability", 7),
    "Auxiliary, Secondary, and Emergency":    ("Repairability", 7),
    "Gear Boxes and Drives":                  ("Repairability", 7),
    "Bellcranks, Pivots, Mechanical":         ("Repairability", 7),
    "Steering and Directional Control":       ("Repairability", 7),
    "Flight Control Systems and Air Cushion": ("Repairability", 7),
    "Wheels, Tires, and Brakes":              ("Repairability", 7),
    "Landing Gear and Alighting Gear":        ("Repairability", 7),
    "Tracks":                                 ("Repairability", 7),
    "Hooks and Catapults":                    ("Repairability", 7),
}

# design-technique prefix → constraint_type (fallback + quantitative classes)
PREFIX_TO_CONSTRAINT = {
    "BIT":   ("Diagnostics", 5),
    "FI":    ("Diagnostics", 5),
    "FD":    ("Diagnostics", 5),
    "SURV":  ("Diagnostics", 5),
    "FDR":   ("Diagnostic Parameters", 11),
    "FIR":   ("Diagnostic Parameters", 11),
    "MP":    ("Repairability", 7),
    "MC":    ("Accessibility", 1),
    "CONT":  ("Accessibility", 1),
    "EC":    ("Error-proofing", 3),
    "ID":    ("Error-proofing", 3),
    "MARK":  ("Error-proofing", 3),
    "HF":    ("Ergonomics", 6),
    "ENV":   ("Maintenance Safety", 4),
    "SAFE":  ("Maintenance Safety", 4),
    "PM":    ("Preventive Maintenance", 12),
    "STD":   ("Interchangeability", 2),
    "STAN":  ("Interchangeability", 2),
    "SKILL": ("Reduced Skill", 9),
    "RED":   ("Reduced Maintenance", 8),
    "SIM":   ("Reduced Maintenance", 8),
    "TIME":  ("Time Parameters", 10),
    "MTTR":  ("Time Parameters", 10),
    "PYRO":  ("Maintenance Safety", 4),
}


def classify(row: dict) -> tuple[str, int] | None:
    cat = (row.get("category_title") or "").strip()
    mapped = CATEGORY_TO_CONSTRAINT.get(cat)
    if mapped:
        return mapped
    # fallback: guideline_prefix
    prefix = (row.get("guideline_prefix") or "").strip().upper()
    if prefix in PREFIX_TO_CONSTRAINT:
        return PREFIX_TO_CONSTRAINT[prefix]
    # partial match of category_title
    for k, v in CATEGORY_TO_CONSTRAINT.items():
        if k.lower()[:15] in cat.lower():
            return v
    return None


QUESTION_TEMPLATES = [
    "What maintainability design requirement does MIL-HDBK-470A state for {gid} under {cat}?",
    "What is the maintainability requirement proposed by {gid} in MIL-HDBK-470A §{cat_id}?",
    "What maintainability design principle does {gid} on page {page} of MIL-HDBK-470A Appendix C state?",
    "What content does {gid} in the {cat} section of MIL-HDBK-470A cover?",
]


def difficulty_tag(text: str) -> str:
    n = len(text)
    if n < 80:
        return "easy"
    if n < 180:
        return "medium"
    return "hard"


def main(n_target: int = 200, seed: int = 42):
    rows = [json.loads(l) for l in open(APPENDIX_C, encoding="utf-8")]
    train_sids = set(json.loads(l).get("source_id") for l in open(TRAIN_FILE, encoding="utf-8"))
    test88_sids = set(json.loads(l).get("source_id") for l in open(TEST88_FILE, encoding="utf-8"))
    excluded = train_sids | test88_sids
    print(f"Excluded source_ids: {len(excluded)} (train={len(train_sids)}, test88={len(test88_sids)})")

    candidates = []
    for r in rows:
        gid = r.get("guideline_id", "")
        if not gid or gid.startswith("OCRMISS"):
            continue
        text = (r.get("guideline_text") or "").strip()
        if not (40 <= len(text) <= 320):
            continue
        sid = r.get("id")
        if sid in excluded:
            continue
        mapped = classify(r)
        if not mapped:
            continue
        ctype, ctype_id = mapped
        candidates.append({
            "raw": r,
            "constraint_type": ctype,
            "constraint_type_id": ctype_id,
            "text": text,
        })
    print(f"Candidates (after filters + excl + classify): {len(candidates)}")
    print("Per-type:", dict(Counter(c["constraint_type"] for c in candidates).most_common()))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_type[c["constraint_type"]].append(c)

    # de-dup by guideline_id body prefix
    for ctype in list(by_type):
        seen = set()
        uniq = []
        for c in by_type[ctype]:
            key = c["text"][:60].lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        by_type[ctype] = uniq

    # balanced sampling: target 17 per class, top up with residuals until 200
    per_class_target = max(1, n_target // len(by_type))
    picked: list[dict] = []
    for ctype, items in by_type.items():
        rng.shuffle(items)
        picked.extend(items[:per_class_target])
    # residual pool
    used = {id(x) for x in picked}
    residual = [c for c in candidates if id(c) not in used]
    rng.shuffle(residual)
    while len(picked) < n_target and residual:
        picked.append(residual.pop())
    picked = picked[:n_target]

    # build records
    out = []
    for i, c in enumerate(picked):
        r = c["raw"]
        gid = r["guideline_id"]
        cat = (r.get("category_title") or "").strip().split(",")[0][:40]
        cat_id = r.get("category_id", "")
        page = r.get("page", "")
        tpl = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        query = tpl.format(cat=cat, cat_id=cat_id, gid=gid, page=page)

        out.append({
            "id": r["id"],
            "query": query,
            "query_en": f"What does MIL-HDBK-470A Appendix C {cat} §{cat_id} {gid} state?",
            "constraint_type": c["constraint_type"],
            "constraint_type_id": c["constraint_type_id"],
            "answer": c["text"],
            "answer_source": {
                "document": "MIL-HDBK-470A",
                "section": f"{cat_id} {cat}".strip(),
                "page": page,
                "quote": c["text"][:200],
            },
            "difficulty": difficulty_tag(c["text"]),
            "equipment_types": ["general"],
            "keywords": [gid, cat.split()[0].lower() if cat else ""],
            "source_record_type": "appendix_c",
            "source_id": r["id"],
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out)} items → {OUT_FILE}")
    print("Per-type in test_200:",
          dict(Counter(x["constraint_type"] for x in out).most_common()))
    print("Per-difficulty:",
          dict(Counter(x["difficulty"] for x in out).most_common()))
    # confirm no overlap
    overlap = set(x["id"] for x in out) & excluded
    print(f"Overlap with train/test88: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
