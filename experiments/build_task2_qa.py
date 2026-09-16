"""Build test dataset for Task 2 — design-requirement category identification.

Input: MIL-HDBK-470A Appendix C 9,014 guidelines, each with `category_title`
       (one of the 12 design techniques) + `guideline_text`.

Strategy (rule-based, no LLM cost):
  1. For each sample, randomly pick 2–3 distinct categories.
  2. From each chosen category, pick one guideline and take a short snippet.
  3. Paraphrase the snippets into a *design scenario* description: glue the
     snippets into a single paragraph with neutral connectives and strip
     category-revealing keywords where possible.
  4. Ground-truth category set T* = the categories of picked guidelines.

30 samples, balanced over possible multi-category combinations.

Output: data/maintqa/task2_category_identification.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

APPENDIX_C = Path("data/mil470a/mil470a_appendix_c_guidelines.jsonl")
OUT_FILE = Path("data/maintqa/task2_category_identification.jsonl")

# Category title → canonical constraint_type label (matches constraint_taxonomy.py)
CATEGORY_TO_CONSTRAINT = {
    "Accessibility":              ("Accessibility", 1),
    "Human Factors (including":   ("Ergonomics", 6),
    "Mating and Connections":     ("Interchangeability", 2),
    "Standardization and Interchangeability": ("Interchangeability", 2),
    "Simplification":             ("Reduced Maintenance", 8),
    "Identification and Labeling": ("Error-proofing", 3),
    "Connections":                ("Interchangeability", 2),
    "Power":                      ("Maintenance Safety", 4),
    "Control Rods, Cables, and Controlex": ("Maintenance Safety", 4),
    "Control":                    ("Maintenance Safety", 4),
    "Steering and Directional Control": ("Maintenance Safety", 4),
    "Flight Control Systems and Air Cushion": ("Maintenance Safety", 4),
    "Man-Machine Interfaces":     ("Ergonomics", 6),
    "Controls":                   ("Ergonomics", 6),
    "External Test Equipment":    ("Diagnostics", 5),
    "Support and Ground Handling": ("Preventive Maintenance", 12),
    "Tracks":                     ("Repairability", 7),
    "Wheels, Tires, and Brakes":  ("Repairability", 7),
    "Landing Gear and Alighting Gear": ("Repairability", 7),
    "Hooks and Catapults":        ("Repairability", 7),
    "Engines (Gasoline and Diesel)": ("Repairability", 7),
    "Engines (Turbine-driven)":   ("Repairability", 7),
    "Transmissions, Clutches, and Rotors": ("Repairability", 7),
    "Auxiliary, Secondary, and Emergency": ("Repairability", 7),
    "Gear Boxes and Drives":      ("Repairability", 7),
    "Bellcranks, Pivots, Mechanical": ("Repairability", 7),
    "Environmental Control, Air": ("Preventive Maintenance", 12),
    "Cartridge Actuated Devices, Shaped": ("Maintenance Safety", 4),
    "Crew Stations, Crew Cabs, Cockpits, and": ("Ergonomics", 6),
    "Access Doors, Panels, and Openings": ("Accessibility", 1),
    "Weapon Bays, Racks, Compartments,": ("Accessibility", 1),
    "External and Parasitic Tanks, Pods,": ("Accessibility", 1),
    "Personnel Seats (Crew and Passenger),": ("Ergonomics", 6),
    "Extinguishing Agents, Containers,": ("Maintenance Safety", 4),
    "Safe/Arm Devices - Non Weapon": ("Maintenance Safety", 4),
}


def load_guidelines(min_text_len: int = 60, max_text_len: int = 300) -> dict[str, list[dict]]:
    """Group guidelines by canonical constraint type."""
    by_ctype: dict[str, list[dict]] = defaultdict(list)
    with open(APPENDIX_C, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cat_title = (row.get("category_title") or "").strip()
            mapped = CATEGORY_TO_CONSTRAINT.get(cat_title)
            if not mapped:
                # try partial match
                for k, v in CATEGORY_TO_CONSTRAINT.items():
                    if k.lower() in cat_title.lower() and len(k) > 5:
                        mapped = v
                        break
            if not mapped:
                continue
            ctype, ctype_id = mapped
            text = (row.get("guideline_text") or "").strip()
            if not (min_text_len <= len(text) <= max_text_len):
                continue
            # strip lead-in keywords that reveal category
            clean = re.sub(r"^\s*(The |All )?", "", text)
            by_ctype[ctype].append({
                "guideline_id": row.get("guideline_id"),
                "category_title": cat_title,
                "text": clean,
                "page": row.get("page"),
                "source_id": f"mil470a_c_{row.get('page')}_{row.get('guideline_id')}",
                "constraint_type": ctype,
                "constraint_type_id": ctype_id,
            })
    # de-dup by text prefix
    for ctype, items in by_ctype.items():
        seen = set()
        uniq = []
        for it in items:
            key = it["text"][:60].lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
        by_ctype[ctype] = uniq
    return by_ctype


CONNECTORS = ["Furthermore, ", "Additionally, ", "The design also requires that ", "Moreover, ", "It is further specified that ", "Further, "]


def strip_reveal_terms(text: str) -> str:
    """Remove highly category-revealing terms so Task 2 is non-trivial."""
    # remove leading "Design ..." style imperatives (no meaning in scenario description)
    text = re.sub(r"^(Design|Use|Ensure|Provide|Incorporate|Avoid)\s+", "", text)
    text = text[:1].upper() + text[1:] if text else text
    return text


def build_scenario(seed: int, by_ctype: dict[str, list[dict]], n_cats: int = 3) -> dict:
    rng = random.Random(seed)
    cats = rng.sample([c for c in by_ctype if by_ctype[c]], n_cats)
    picks = []
    snippets = []
    for ctype in cats:
        g = rng.choice(by_ctype[ctype])
        picks.append(g)
        snippets.append(strip_reveal_terms(g["text"]))
    # assemble scenario description
    sent_0 = f"An equipment design proposal must consider the following maintainability requirements: {snippets[0]}"
    parts = [sent_0]
    for s, conn in zip(snippets[1:], CONNECTORS):
        parts.append(f"{conn}{s}")
    scenario = " ".join(parts)
    # shorten (truncate to ~600 chars)
    if len(scenario) > 700:
        scenario = scenario[:700].rsplit(".", 1)[0] + "."
    # ground-truth categories
    gt_set = sorted({p["constraint_type"] for p in picks})
    gt_ids = sorted({p["constraint_type_id"] for p in picks})
    return {
        "scenario": scenario,
        "gt_categories": gt_set,
        "gt_category_ids": gt_ids,
        "sources": [
            {
                "constraint_type": p["constraint_type"],
                "source_id": p["source_id"],
                "quote": p["text"][:160],
            }
            for p in picks
        ],
    }


def main(n_target: int = 30):
    by_ctype = load_guidelines()
    print("Per-category guideline counts:",
          {k: len(v) for k, v in sorted(by_ctype.items())})

    rng = random.Random(42)
    samples = []
    seen_combos = set()
    attempts = 0
    while len(samples) < n_target and attempts < 300:
        attempts += 1
        n_cats = rng.choice([2, 2, 3, 3, 3])
        seed = rng.randint(0, 10 ** 9)
        try:
            s = build_scenario(seed, by_ctype, n_cats=n_cats)
        except Exception:
            continue
        combo = tuple(s["gt_categories"])
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        samples.append(s)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            record = {
                "id": f"task2_{i + 1:03d}",
                "query": s["scenario"],
                "gt_categories": s["gt_categories"],
                "gt_category_ids": s["gt_category_ids"],
                "n_gt": len(s["gt_categories"]),
                "sources": s["sources"],
                "task": "category_identification",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} Task-2 items → {OUT_FILE}")
    print("n_gt distribution:",
          dict(Counter(len(s["gt_categories"]) for s in samples)))
    print("Category coverage (appearances):",
          dict(Counter(c for s in samples for c in s["gt_categories"])))
    print("\n=== Preview ===")
    for i in range(3):
        s = samples[i]
        print(f"[task2_{i + 1:03d}] GT={s['gt_categories']}")
        print(f"  scenario: {s['scenario'][:220]}...")
        print()


if __name__ == "__main__":
    main()
