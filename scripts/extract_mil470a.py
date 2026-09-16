"""Extract structured datasets from MIL-HDBK-470A.

Outputs:
  - data/mil470a/mil470a_pages_clean.jsonl
  - data/mil470a/mil470a_clauses.jsonl
  - data/mil470a/mil470a_appendix_a_templates.jsonl
  - data/mil470a/mil470a_appendix_c_guidelines.jsonl
  - data/mil470a/mil470a_appendix_d_actions.jsonl
  - data/mil470a/mil470a_summary.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz


PDF_PATH = Path("data/mil470a/mil-hdbk-470a.pdf")
OUTPUT_DIR = Path("data/mil470a")

APPENDIX_NAMES = ["A", "B", "C", "D", "E", "F", "G"]

HEADER_LINE_RE = re.compile(r"^MIL-HDBK-470A$")
PRINTED_PAGE_RE = re.compile(r"^(?:[A-G]-\d+|\d+-\d+|[ivxlcdm]+)$", re.I)
CLAUSE_RE = re.compile(r"^((?:[A-G]\.\d+(?:\.\d+)*)|(?:\d+\.\d+(?:\.\d+)*))\s+(.+)$")
CATEGORY_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
GUIDELINE_RE = re.compile(r"^[A-Z][A-Z0-9()&/]*-\d{2}$")
TEMPLATE_ITEM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
FIGURE_D_RE = re.compile(r"^FIGURE\s+(D-\d+[A-Z]?)\.", re.I)

SKIP_C_LINES = {
    "APPENDIX C",
    "Guidelines by Category (ContÕd)",
    "Guidelines by Category",
    "Category No.",
    "Category Title",
    "Guideline No.",
    "Guideline",
}

SKIP_A_LINES = {
    "APPENDIX A",
}

SKIP_D_LINES = {
    "APPENDIX D",
}

QUESTION_SEEDS = {
    "appendix_a": "What maintainability solicitation or specification item is described here?",
    "appendix_c": "What maintainability design guideline does this record state?",
    "appendix_d": "What maintenance action, condition, or tool usage does this record describe?",
    "clause": "What maintainability concept, method, or requirement is described in this clause?",
}


@dataclass
class PageRecord:
    page_num: int
    printed_page: str
    appendix: str | None
    text: str


def normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = line.replace("¥", "-")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def detect_printed_page(lines: list[str]) -> str:
    for line in lines[:6]:
        if PRINTED_PAGE_RE.match(line):
            return line
    return ""


def detect_appendix(lines: list[str]) -> str | None:
    for line in lines[:6]:
        upper = line.upper()
        if upper.startswith("APPENDIX "):
            parts = upper.split()
            if len(parts) >= 2 and parts[1] in APPENDIX_NAMES:
                return parts[1]
    return None


def clean_page_lines(raw_lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in raw_lines:
        line = normalize_line(raw)
        if not line:
            continue
        if HEADER_LINE_RE.match(line):
            continue
        if PRINTED_PAGE_RE.match(line):
            continue
        cleaned.append(line)
    return cleaned


def load_pages(pdf_path: Path) -> list[PageRecord]:
    doc = fitz.open(pdf_path)
    pages: list[PageRecord] = []
    for idx in range(len(doc)):
        raw_lines = doc[idx].get_text("text").splitlines()
        norm_lines = [normalize_line(x) for x in raw_lines if normalize_line(x)]
        pages.append(
            PageRecord(
                page_num=idx + 1,
                printed_page=detect_printed_page(norm_lines),
                appendix=detect_appendix(norm_lines),
                text="\n".join(clean_page_lines(raw_lines)),
            )
        )
    return pages


def find_appendix_starts(pages: list[PageRecord]) -> dict[str, int]:
    starts: dict[str, int] = {}
    for appendix in APPENDIX_NAMES:
        target = f"APPENDIX {appendix}"
        for page in pages:
            raw_lines = page.text.splitlines()[:4]
            if any(normalize_line(line).upper() == target for line in raw_lines):
                starts[appendix] = page.page_num
                break
    # Fall back to original appendix detector if exact line was stripped
    for page in pages:
        if page.appendix and page.appendix not in starts:
            starts[page.appendix] = page.page_num
    return dict(sorted(starts.items(), key=lambda item: item[1]))


def appendix_ranges(starts: dict[str, int], total_pages: int) -> dict[str, tuple[int, int]]:
    ordered = sorted(starts.items(), key=lambda item: item[1])
    ranges: dict[str, tuple[int, int]] = {}
    for idx, (name, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_pages
        ranges[name] = (start, end)
    return ranges


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_clauses(pages: list[PageRecord], ranges: dict[str, tuple[int, int]]) -> list[dict]:
    c_start, c_end = ranges["C"]

    rows: list[dict] = []
    current: dict | None = None

    def flush():
        nonlocal current
        if not current:
            return
        text = " ".join(current["body"]).strip()
        if text:
            current["text"] = text
            current["question_seed"] = QUESTION_SEEDS["clause"]
            del current["body"]
            rows.append(current)
        current = None

    for page in pages:
        if page.page_num < 13:
            continue
        if c_start <= page.page_num <= c_end:
            continue

        for raw in page.text.splitlines():
            line = normalize_line(raw)
            if not line:
                continue
            if line.upper().startswith("APPENDIX "):
                continue
            match = CLAUSE_RE.match(line)
            if match and re.search(r"[A-Za-z]", match.group(2)):
                flush()
                current = {
                    "id": f"mil470a_clause_{page.page_num}_{match.group(1).replace('.', '_')}",
                    "clause_id": match.group(1),
                    "title": match.group(2),
                    "page": page.page_num,
                    "printed_page": page.printed_page,
                    "appendix": page.appendix,
                    "body": [],
                }
                continue

            if current:
                current["body"].append(line)

    flush()
    return rows


def extract_appendix_a_templates(pages: list[PageRecord], start: int, end: int) -> list[dict]:
    rows: list[dict] = []
    current_heading = "Appendix A"
    current: dict | None = None
    active = False

    def flush():
        nonlocal current
        if not current:
            return
        current["text"] = " ".join(current["body"]).strip()
        current["question_seed"] = QUESTION_SEEDS["appendix_a"]
        del current["body"]
        rows.append(current)
        current = None

    for page in pages:
        if not (start <= page.page_num <= end):
            continue
        for raw in page.text.splitlines():
            line = normalize_line(raw)
            if not line or line in SKIP_A_LINES:
                continue
            upper = line.upper()
            if upper.startswith("SECTION B.") or upper.startswith("SECTION C.") or upper in {
                "TEMPLATE FOR DEVELOPING MAINTAINABILITY PORTION OF A PROCUREMENT PACKAGE",
                "THE STATEMENT OF WORK",
                "THE SPECIFICATION",
            }:
                flush()
                current_heading = line
                active = True
                continue
            if not active:
                continue
            match = TEMPLATE_ITEM_RE.match(line)
            if match:
                flush()
                current = {
                    "id": f"mil470a_appendix_a_{page.page_num}_{match.group(1).replace('.', '_')}",
                    "section": current_heading,
                    "item_id": match.group(1),
                    "title": match.group(2),
                    "page": page.page_num,
                    "printed_page": page.printed_page,
                    "body": [],
                }
                continue
            if line.startswith("-") and current:
                current["body"].append(line)
                continue
            if current:
                current["body"].append(line)
    flush()
    return rows


def extract_appendix_c_guidelines(pages: list[PageRecord], start: int, end: int) -> list[dict]:
    rows: list[dict] = []
    category_id: str | None = None
    category_title: str = ""
    pending_category_id: str | None = None
    pending_title_lines: list[str] = []
    current_guideline_id: str | None = None
    current_guideline_lines: list[str] = []
    current_page: int = start
    active = False
    missing_counter = 0

    def flush_guideline():
        nonlocal current_guideline_id, current_guideline_lines
        if not current_guideline_id:
            return
        text = " ".join(current_guideline_lines).strip()
        if text:
            prefix = current_guideline_id.split("-", 1)[0]
            rows.append({
                "id": f"mil470a_c_{current_page}_{current_guideline_id.lower().replace('(', '').replace(')', '').replace('&', 'and')}",
                "category_id": category_id,
                "category_title": category_title,
                "guideline_id": current_guideline_id,
                "guideline_prefix": prefix,
                "guideline_text": text,
                "page": current_page,
                "printed_page": next((p.printed_page for p in pages if p.page_num == current_page), ""),
                "question_seed": QUESTION_SEEDS["appendix_c"],
            })
        current_guideline_id = None
        current_guideline_lines = []

    for page in pages:
        if not (start <= page.page_num <= end):
            continue
        current_page = page.page_num
        if (
            not active
            and "Guidelines by Category" in page.text
            and "TABLE OF CONTENTS" not in page.text
        ):
            active = True
            category_id = None
            category_title = ""
            pending_category_id = None
            pending_title_lines = []
        if not active and page.printed_page.startswith("C-"):
            try:
                if int(page.printed_page.split("-", 1)[1]) >= 47:
                    active = True
                    category_id = None
                    category_title = ""
                    pending_category_id = None
                    pending_title_lines = []
            except ValueError:
                pass
        for raw in page.text.splitlines():
            line = normalize_line(raw)
            if not line or line in SKIP_C_LINES:
                continue
            if not active:
                continue
            if line.upper().startswith("FIGURE C-") or line.upper().startswith("TABLE C-"):
                flush_guideline()
                continue
            if line.startswith("Guidelines by Category"):
                continue
            if CATEGORY_RE.match(line):
                flush_guideline()
                pending_category_id = line
                pending_title_lines = []
                continue
            if GUIDELINE_RE.match(line):
                flush_guideline()
                if pending_category_id:
                    category_id = pending_category_id
                    category_title = pending_title_lines[0].strip() if pending_title_lines else ""
                    pending_category_id = None
                    pending_title_lines = []
                current_guideline_id = line
                current_guideline_lines = []
                continue
            if pending_category_id and not current_guideline_id:
                if not pending_title_lines:
                    pending_title_lines.append(line)
                else:
                    if category_id != pending_category_id:
                        category_id = pending_category_id
                        category_title = pending_title_lines[0].strip()
                        pending_category_id = None
                        pending_title_lines = []
                    if not current_guideline_id:
                        missing_counter += 1
                        current_guideline_id = f"OCRMISS-{missing_counter:04d}"
                        current_guideline_lines = []
                    current_guideline_lines.append(line)
                continue
            if current_guideline_id:
                current_guideline_lines.append(line)
    flush_guideline()
    return rows


def extract_appendix_d_actions(pages: list[PageRecord], start: int, end: int) -> list[dict]:
    rows: list[dict] = []
    title_lines: list[str] = []
    bullets: list[str] = []
    page_num = start

    def flush(figure_id: str | None):
        nonlocal title_lines, bullets
        title = " ".join(title_lines).strip()
        if figure_id and title and bullets and len(title) <= 120:
            rows.append({
                "id": f"mil470a_d_{page_num}_{(figure_id or title).lower().replace(' ', '_').replace('.', '').replace('/', '_')[:80]}",
                "title": title,
                "figure_id": figure_id,
                "bullets": bullets[:],
                "text": " ".join(bullets).strip(),
                "page": page_num,
                "printed_page": next((p.printed_page for p in pages if p.page_num == page_num), ""),
                "question_seed": QUESTION_SEEDS["appendix_d"],
            })
        title_lines = []
        bullets = []

    for page in pages:
        if not (start <= page.page_num <= end):
            continue
        page_num = page.page_num
        if page.printed_page.startswith("D-"):
            try:
                if int(page.printed_page.split("-", 1)[1]) < 32:
                    continue
            except ValueError:
                pass
        for raw in page.text.splitlines():
            line = normalize_line(raw)
            if not line or line in SKIP_D_LINES:
                continue
            fig = FIGURE_D_RE.match(line)
            if fig:
                flush(fig.group(1))
                continue
            if line.upper().startswith("TABLE D-"):
                flush(None)
                continue
            if line.startswith("-"):
                bullets.append(line.lstrip("- ").strip())
                continue
            if bullets:
                bullets[-1] = f"{bullets[-1]} {line}".strip()
            else:
                title_lines.append(line)
    flush(None)
    return rows


def build_summary(
    pages: list[PageRecord],
    ranges: dict[str, tuple[int, int]],
    clauses: list[dict],
    appendix_a: list[dict],
    appendix_c: list[dict],
    appendix_d: list[dict],
) -> dict:
    return {
        "pdf_path": str(PDF_PATH),
        "total_pages": len(pages),
        "appendix_ranges": {k: {"start": v[0], "end": v[1]} for k, v in ranges.items()},
        "counts": {
            "pages_clean": len(pages),
            "clauses": len(clauses),
            "appendix_a_templates": len(appendix_a),
            "appendix_c_guidelines": len(appendix_c),
            "appendix_d_actions": len(appendix_d),
        },
    }


def main() -> None:
    pages = load_pages(PDF_PATH)
    starts = find_appendix_starts(pages)
    ranges = appendix_ranges(starts, len(pages))

    clauses = extract_clauses(pages, ranges)
    appendix_a = extract_appendix_a_templates(pages, *ranges["A"])
    appendix_c = extract_appendix_c_guidelines(pages, *ranges["C"])
    appendix_d = extract_appendix_d_actions(pages, *ranges["D"])
    summary = build_summary(pages, ranges, clauses, appendix_a, appendix_c, appendix_d)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "mil470a_pages_clean.jsonl", [asdict(p) for p in pages])
    write_jsonl(OUTPUT_DIR / "mil470a_clauses.jsonl", clauses)
    write_jsonl(OUTPUT_DIR / "mil470a_appendix_a_templates.jsonl", appendix_a)
    write_jsonl(OUTPUT_DIR / "mil470a_appendix_c_guidelines.jsonl", appendix_c)
    write_jsonl(OUTPUT_DIR / "mil470a_appendix_d_actions.jsonl", appendix_d)

    with (OUTPUT_DIR / "mil470a_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
