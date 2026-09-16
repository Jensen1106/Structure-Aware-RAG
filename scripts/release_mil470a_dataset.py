"""Build release-ready MIL-HDBK-470A dataset artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


MIL_DIR = Path("data/mil470a")
MAINTQA_DIR = Path("data/maintqa")

INPUT_HIGHCONF = MIL_DIR / "mil470a_maintqa_highconf.jsonl"

OUTPUT_DEDUP = MIL_DIR / "mil470a_maintqa_highconf_dedup.jsonl"
OUTPUT_BALANCED = MIL_DIR / "mil470a_maintqa_balanced_30_dedup.jsonl"
OUTPUT_SUMMARY = MIL_DIR / "mil470a_release_summary.json"
OUTPUT_MANIFEST = MIL_DIR / "mil470a_release_manifest.md"

OUTPUT_TRAIN = MAINTQA_DIR / "train_mil470a_highconf.jsonl"
OUTPUT_DEV = MAINTQA_DIR / "dev_mil470a_balanced_30.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def rank_row(row: dict) -> tuple:
    src_rank = {
        "appendix_d": 0,
        "appendix_c": 1,
        "appendix_a": 2,
        "clause": 3,
    }.get(row.get("source_record_type"), 9)
    mapping_rank = len(row.get("mapping_candidates", []))
    answer_len = len(row.get("answer", ""))
    return (src_rank, mapping_rank, answer_len, row.get("source_id", ""))


def dedup_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["constraint_type"], normalize_answer(row["answer"]))
        groups[key].append(row)

    deduped: list[dict] = []
    for _key, dup_rows in groups.items():
        dup_rows.sort(key=rank_row)
        base = dict(dup_rows[0])
        base["aliases"] = [
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "section": r["answer_source"]["section"],
                "page": r["answer_source"]["page"],
            }
            for r in dup_rows
        ]
        base["alias_count"] = len(dup_rows)
        deduped.append(base)

    deduped.sort(key=lambda r: (r["constraint_type"], r["id"]))
    return deduped


def build_balanced(rows: list[dict], per_type: int = 30) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["constraint_type"]].append(row)

    balanced: list[dict] = []
    for constraint_type in sorted(by_type):
        type_rows = sorted(
            by_type[constraint_type],
            key=lambda r: (-r.get("alias_count", 1), len(r.get("answer", "")), r["id"]),
        )
        balanced.extend(type_rows[:per_type])
    return balanced


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(summary: dict) -> None:
    OUTPUT_MANIFEST.write_text(
        "\n".join(
            [
                "# MIL470A Release Manifest",
                "",
                "Release files:",
                f"- `{OUTPUT_DEDUP.name}`",
                f"- `{OUTPUT_BALANCED.name}`",
                f"- `{OUTPUT_SUMMARY.name}`",
                "",
                "MaintQA-compatible files:",
                f"- `{OUTPUT_TRAIN}`",
                f"- `{OUTPUT_DEV}`",
                "",
                "Summary:",
                f"- High-confidence input: {summary['input_highconf']}",
                f"- After dedup: {summary['dedup_total']}",
                f"- Balanced subset: {summary['balanced_total']}",
                "",
                "Notes:",
                "- Dedup key is `(constraint_type, normalized_answer)`, used to eliminate duplicates of the same guideline across multiple category views.",
                "- The `aliases` field retains all original source pages and sections for traceability in the paper.",
                "- Recommended: use `train_mil470a_highconf.jsonl` for training, and `dev_mil470a_balanced_30.jsonl` for manual sampling or development.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    rows = load_jsonl(INPUT_HIGHCONF)
    deduped = dedup_rows(rows)
    balanced = build_balanced(deduped, per_type=30)

    write_jsonl(OUTPUT_DEDUP, deduped)
    write_jsonl(OUTPUT_BALANCED, balanced)
    write_jsonl(OUTPUT_TRAIN, deduped)
    write_jsonl(OUTPUT_DEV, balanced)

    summary = {
        "input_highconf": len(rows),
        "dedup_total": len(deduped),
        "balanced_total": len(balanced),
        "dedup_by_constraint_type": dict(Counter(r["constraint_type"] for r in deduped)),
        "balanced_by_constraint_type": dict(Counter(r["constraint_type"] for r in balanced)),
        "top_alias_counts": sorted(
            [
                {
                    "id": r["id"],
                    "constraint_type": r["constraint_type"],
                    "alias_count": r["alias_count"],
                    "answer": r["answer"][:200],
                }
                for r in deduped
            ],
            key=lambda x: x["alias_count"],
            reverse=True,
        )[:20],
    }

    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
