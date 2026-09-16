"""Build test_mil470a_500 — 500-item independent main test set.

Extends test_mil470a_200 to 500 items to address reviewer concern about
small effect sizes on the headline Tuple-F1 metric. The 500 items include
the original 200 plus 300 additional items, with strict exclusion of
train + test_88 + test_200 source_ids to preserve zero-overlap guarantee.

Output: data/maintqa/test_mil470a_500.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Reuse classification logic from build_test200
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_test200 import (
    APPENDIX_C, TRAIN_FILE, TEST88_FILE, QUESTION_TEMPLATES,
    classify, difficulty_tag,
)

TEST200_FILE = Path("data/maintqa/test_mil470a_200.jsonl")
OUT_FILE = Path("data/maintqa/test_mil470a_500.jsonl")


def main(n_target: int = 500, seed: int = 42):
    rows = [json.loads(l) for l in open(APPENDIX_C, encoding="utf-8")]
    train_sids = set(json.loads(l).get("source_id") for l in open(TRAIN_FILE, encoding="utf-8"))
    test88_sids = set(json.loads(l).get("source_id") for l in open(TEST88_FILE, encoding="utf-8"))
    excluded_seen = train_sids | test88_sids

    # Carry forward the existing 200 items so test_500 ⊃ test_200
    test200_items = [json.loads(l) for l in open(TEST200_FILE, encoding="utf-8") if l.strip()]
    test200_sids = set(x["id"] for x in test200_items)
    print(f"Excluded (train+test88): {len(excluded_seen)} | carried forward (test200): {len(test200_sids)}")

    # Pool for new items: appendix_c minus everything seen
    excluded_all = excluded_seen | test200_sids
    candidates = []
    for r in rows:
        gid = r.get("guideline_id", "")
        if not gid or gid.startswith("OCRMISS"):
            continue
        text = (r.get("guideline_text") or "").strip()
        if not (40 <= len(text) <= 320):
            continue
        sid = r.get("id")
        if sid in excluded_all:
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
    print(f"New candidates: {len(candidates)}")
    print("Per-type:", dict(Counter(c["constraint_type"] for c in candidates).most_common()))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_type[c["constraint_type"]].append(c)

    # de-dup by text prefix
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

    # Need 300 new items, balanced across constraint types
    n_new = n_target - len(test200_items)
    per_class_target = max(1, n_new // len(by_type))
    picked: list[dict] = []
    for ctype, items in by_type.items():
        rng.shuffle(items)
        picked.extend(items[:per_class_target])
    used = {id(x) for x in picked}
    residual = [c for c in candidates if id(c) not in used]
    rng.shuffle(residual)
    while len(picked) < n_new and residual:
        picked.append(residual.pop())
    picked = picked[:n_new]
    print(f"Sampled {len(picked)} new items")

    # build new records (preserving the same template/format as test_200)
    new_records = []
    for i, c in enumerate(picked):
        r = c["raw"]
        gid = r["guideline_id"]
        cat = (r.get("category_title") or "").strip().split(",")[0][:40]
        cat_id = r.get("category_id", "")
        page = r.get("page", "")
        tpl = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        query = tpl.format(cat=cat, cat_id=cat_id, gid=gid, page=page)
        new_records.append({
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

    # combine: original 200 + new 300, in that order so old indices are stable
    out = list(test200_items) + new_records

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out)} items → {OUT_FILE}")
    print("Per-type in test_500:",
          dict(Counter(x["constraint_type"] for x in out).most_common()))
    print("Per-difficulty:",
          dict(Counter(x["difficulty"] for x in out).most_common()))
    overlap = set(x["id"] for x in out) & excluded_seen
    print(f"Overlap with train/test88: {len(overlap)} (must be 0)")
    contains = test200_sids.issubset(set(x["id"] for x in out))
    print(f"test_500 contains all test_200: {contains}")


if __name__ == "__main__":
    main()
