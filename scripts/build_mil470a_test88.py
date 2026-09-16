"""Build MIL-470A 88-item balanced test set

Source: data/mil470a/mil470a_maintqa_balanced_30_dedup.jsonl (230 items)
Target: data/maintqa/test_mil470a_88.jsonl (88 items)

Strategy:
- 12 constraint types, stratified by data volume
- Large types (>=30 items): 8 per type = 8x9 = 72 items
- Small types supplement 16 items total (diversity fallback)
- Fixed seed=42 for reproducibility
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "mil470a" / "mil470a_maintqa_balanced_30_dedup.jsonl"
DST = PROJECT_ROOT / "data" / "maintqa" / "test_mil470a_88.jsonl"

# Large types (>=20 items) -- 8 per type
BIG_TYPES = {
    "Accessibility", "Diagnostics", "Ergonomics",
    "Interchangeability", "Maintenance Safety",
    "Preventive Maintenance", "Repairability",
}
BIG_PER = 8

# Medium types -- take all (when smaller than BIG_PER, include all)
MID_TYPES = {
    "Error-proofing",  # 12
    "Reduced Skill",   # 4
    "Reduced Maintenance",  # 2
    "Time Parameters",  # 1
    "Diagnostic Parameters",  # 1
}

TARGET = 88


def main():
    rng = random.Random(42)
    items = []
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    by_type: dict[str, list] = defaultdict(list)
    for item in items:
        by_type[item.get("constraint_type", "Unknown")].append(item)

    sampled: list = []
    for ct in sorted(BIG_TYPES):
        pool = by_type.get(ct, [])
        if len(pool) <= BIG_PER:
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, BIG_PER))

    for ct in sorted(MID_TYPES):
        sampled.extend(by_type.get(ct, []))

    if len(sampled) > TARGET:
        sampled = sampled[:TARGET]
    elif len(sampled) < TARGET:
        # Fill from Accessibility pool
        extra_pool = [
            x for x in by_type["Accessibility"] if x not in sampled
        ]
        sampled.extend(rng.sample(extra_pool, TARGET - len(sampled)))

    sampled.sort(key=lambda x: (x.get("constraint_type", ""), x.get("id", "")))
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total sampled: {len(sampled)}")
    cc = Counter(x.get("constraint_type", "?") for x in sampled)
    for k, v in sorted(cc.items()):
        print(f"  {k}: {v}")
    print(f"Saved to {DST}")


if __name__ == "__main__":
    main()
