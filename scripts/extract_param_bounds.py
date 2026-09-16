"""从 MIL-HDBK-470A 语料抽取工程参数值域表（V-2 解决方案 / R2-1 半自动管线实证）

管线（对应论文 §6 "Adaptation Cost" 提出的 extraction→filtering→verification）：
  1) 预筛：附录 C 设计准则 + 正文条款中含数字的文本
  2) LLM 结构化（qwen-max, temperature=0, 批量 20 条/调用, JSON 输出）：
     {parameter, unit, values, quote}
  3) 规则过滤 + 聚合：单位归一 → 按参数名分组 → 值域 [min,max] → 退化/混杂单位剔除

输出（断点续传）：
  data/param_bounds/extraction_raw.jsonl      # 每条 LLM 抽取结果
  data/param_bounds/param_bounds_aggregated.json  # 聚合后参数表
用法：python scripts/extract_param_bounds.py [--limit N] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import LLM_CONFIG  # noqa: E402

OUT_DIR = ROOT / "data" / "param_bounds"
RAW = OUT_DIR / "extraction_raw_v2.jsonl"
AGG = OUT_DIR / "param_bounds_aggregated_v2.json"
MODEL = "qwen-max"
BATCH = 20

PROMPT = """You are extracting an engineering parameter-bound table from a US military maintainability design handbook (MIL-HDBK-470A).

For each numbered guideline/clause below, extract every ENGINEERING QUANTITY it mentions: physical dimensions, clearances, distances, weights, forces, torques, times, frequencies, percentages, temperature, pressure, noise, voltage, percentile body dimensions, etc.

Rules:
- parameter: the MOST SPECIFIC canonical English name, lowercase, including the equipment element or context being measured, e.g. "engine-to-firewall distance", "weapon bay door width", "75th-percentile male hand access", "structural fastener diameter", "mttr". NEVER use bare generic names such as "distance", "height", "width", "angle", "time", "clearance", "radius", "spacing" alone — always qualify WHAT is measured. Use the same canonical name for the same physical quantity across guidelines.
- unit: the unit exactly as implied (mm, cm, m, in, ft, kg, lb, N, lbf, in-lb, ft-lb, h, min, s, %, dB, V, A, W, psi, rpm, degrees, ...).
- values: the numeric value(s) mentioned for that parameter in this guideline (numbers only).
- quote: a short supporting snippet (<15 words) from the guideline.
- SKIP: section/clause/figure/table numbers, program task numbers (e.g. "Task 101"), dates, page numbers, purely procedural counts with no engineering unit, and qualitative-only guidelines.
- If a guideline has no engineering quantity, output nothing for it.

Guidelines:
{items}

Return STRICT JSON: {{"items": [{{"i": <index>, "parameter": "...", "unit": "...", "values": [..], "quote": "..."}}]}}"""


def load_records():
    recs = []
    for line in open(ROOT / "data/mil470a/mil470a_appendix_c_guidelines.jsonl", encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            recs.append({"src": "appendix_c", "id": d["guideline_id"],
                         "category": d.get("category_title", ""), "page": d.get("page"),
                         "text": d.get("guideline_text", "")})
    for line in open(ROOT / "data/mil470a/mil470a_clauses.jsonl", encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            recs.append({"src": "clauses", "id": d["clause_id"],
                         "category": d.get("title", ""), "page": d.get("page"),
                         "text": d.get("text", "")})
    # 预筛：含数字
    return [r for r in recs if re.search(r"\d", r["text"])]


def call_llm(prompt, retries=3):
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        LLM_CONFIG["base_url"].rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {LLM_CONFIG['api_key']}",
                 "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == retries - 1:
                print(f"  batch failed: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if args.limit:
        records = records[:args.limit]
    print(f"records to process (digit-containing): {len(records)}")

    # 断点续传：已处理的 (src,id) 跳过
    done_ids = set()
    if RAW.exists():
        for line in open(RAW, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                done_ids.add(d["rec_key"])
    todo = [(i, r) for i, r in enumerate(records) if f'{r["src"]}:{r["id"]}' not in done_ids]
    print(f"already done: {len(done_ids)}, todo: {len(todo)}")

    batches = []
    for s in range(0, len(todo), BATCH):
        batches.append(todo[s:s + BATCH])

    fout = open(RAW, "a", encoding="utf-8")
    t0 = time.time()
    n_ok = 0

    def run_batch(batch):
        items = "\n".join(f"[{i}] {r['text'][:400]}" for i, r in batch)
        return batch, call_llm(PROMPT.format(items=items))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_batch, b) for b in batches]
        for n, fut in enumerate(as_completed(futs), 1):
            batch, out = fut.result()
            if out and isinstance(out.get("items"), list):
                by_idx = {}
                for it in out["items"]:
                    try:
                        by_idx[int(it.get("i"))] = it
                    except (TypeError, ValueError):
                        continue
                for i, r in batch:
                    it = by_idx.get(i)
                    if not it:
                        continue
                    p = str(it.get("parameter", "")).strip().lower()
                    u = str(it.get("unit", "")).strip()
                    vals = it.get("values", [])
                    if isinstance(vals, (int, float)):
                        vals = [vals]
                    if not isinstance(vals, list):
                        continue
                    nums = []
                    for v in vals:
                        try:
                            nums.append(float(v))
                        except (TypeError, ValueError):
                            pass
                    if not nums:
                        continue
                    fout.write(json.dumps({
                        "rec_key": f'{r["src"]}:{r["id"]}', "src": r["src"],
                        "category": r["category"], "page": r["page"],
                        "parameter": p, "unit": u, "values": nums,
                        "quote": str(it.get("quote", ""))[:120]}, ensure_ascii=False) + "\n")
                n_ok += 1
            if n % 20 == 0:
                fout.flush()
                print(f"  {n}/{len(batches)} batches ({time.time()-t0:.0f}s)")
    fout.close()
    print(f"extraction done: {n_ok}/{len(batches)} batches ok in {time.time()-t0:.0f}s")
    aggregate()


UNIT_MAP = {
    "millimeters": "mm", "millimeter": "mm", "mm": "mm",
    "centimeters": "cm", "centimeter": "cm", "cm": "cm",
    "meters": "m", "meter": "m", "m": "m",
    "inches": "in", "inch": "in", "in": "in",
    "feet": "ft", "foot": "ft", "ft": "ft",
    "kilograms": "kg", "kilogram": "kg", "kg": "kg",
    "pounds": "lb", "lbs": "lb", "lb": "lb", "pound": "lb",
    "newtons": "n", "newton": "n", "n": "n",
    "lbf": "lbf", "in-lb": "in-lb", "ft-lb": "ft-lb", "in-lbs": "in-lb", "ft-lbs": "ft-lb",
    "hours": "h", "hour": "h", "hrs": "h", "h": "h",
    "minutes": "min", "minute": "min", "min": "min",
    "seconds": "s", "second": "s", "sec": "s", "s": "s",
    "percent": "%", "%": "%",
    "db": "db", "decibels": "db",
    "volts": "v", "volt": "v", "v": "v",
    "amps": "a", "amp": "a", "amperes": "a", "a": "a",
    "watts": "w", "watt": "w", "w": "w",
    "psi": "psi", "psig": "psi",
    "rpm": "rpm", "degrees": "deg", "degree": "deg", "°": "deg", "deg": "deg",
    "cycles": "cycles",
}

# 单位族：同族可互相换算到基准单位
UNIT_FAMILY = {
    "mm": ("length_mm", 1.0), "cm": ("length_mm", 10.0), "m": ("length_mm", 1000.0),
    "in": ("length_mm", 25.4), "ft": ("length_mm", 304.8),
    "kg": ("mass_kg", 1.0), "lb": ("mass_kg", 0.4536), "g": ("mass_kg", 0.001),
    "h": ("time_h", 1.0), "min": ("time_h", 1/60), "s": ("time_h", 1/3600),
}


def aggregate():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    # 1) 单位归一
    for r in rows:
        u = r["unit"].lower().strip()
        r["unit_n"] = UNIT_MAP.get(u)
    rows = [r for r in rows if r["unit_n"]]  # 未识别单位直接剔除（保守）

    # 2) 分组：参数名（含单位族归并）
    groups = defaultdict(list)
    for r in rows:
        fam = UNIT_FAMILY.get(r["unit_n"], (r["unit_n"], 1.0))
        r["fam"], r["factor"] = fam
        key = f'{r["parameter"]}|{r["fam"]}'
        groups[key].append(r)

    # 3) 聚合值域 + 过滤
    table = []
    for key, rs in groups.items():
        vals = [v * r["factor"] for r in rs for v in r["values"]]
        vals = [v for v in vals if 0 < v < 1e7]  # 剔除明显异常
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if lo == hi and len(rs) == 1:
            quality = "single-observation"
        elif lo > hi:
            continue
        else:
            quality = "aggregated"
        base_unit = rs[0]["unit_n"]
        table.append({
            "parameter": rs[0]["parameter"], "unit": base_unit,
            "family": rs[0]["fam"],
            "lower": round(lo / rs[0]["factor"], 4),
            "upper": round(hi / rs[0]["factor"], 4),
            "n_mentions": len(rs), "n_values": len(vals),
            "categories": sorted({r["category"] for r in rs if r["category"]})[:3],
            "pages": sorted({r["page"] for r in rs if r["page"]})[:6],
            "source": "MIL-HDBK-470A", "quality": quality,
            "example_quote": rs[0]["quote"],
        })
    table.sort(key=lambda t: (-t["n_mentions"], t["parameter"]))
    with open(AGG, "w", encoding="utf-8") as f:
        json.dump({"n_raw_rows": len(rows), "n_parameters": len(table),
                   "parameters": table}, f, ensure_ascii=False, indent=2)
    print(f"\naggregated: {len(rows)} raw rows -> {len(table)} distinct parameter|unit entries")
    from collections import Counter
    print("unit distribution:", Counter(t['unit'] for t in table).most_common(12))
    print("top by mentions:")
    for t in table[:15]:
        print(f"  {t['parameter']:<32} {t['unit']:<6} [{t['lower']}, {t['upper']}]  n={t['n_mentions']}")


if __name__ == "__main__":
    main()
