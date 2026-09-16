"""E5v2 的 LLM-as-Judge 重评（S-8：替换 build_sci_report_figures.py 中硬编码的消融 judge 数字）

对 run_e5_ablation_v2.py 的 7 个配置 × 88 条做盲评（qwen-max，v2 rubric），
复用 run_llm_judge_v2.call_judge 与 run_e5_ablation_v2 的组件构造。

流程：每个配置重建管线并 ingest（确定性），按 per_item 中的 query 重取 top-5 检索
（与生成时同口径），取前 3 片段作为 judge 的来源接地证据；同一 item 的 7 个配置
随机盲化编号后评分；聚合按配置输出五维均分。

输出：results/e5v2/judge_e5v2_results.json + judge_e5v2_detailed.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_e5_ablation_v2 import build_settings, load_docs, load_items
from experiments.run_llm_judge_v2 import call_judge
from src.utils.logger import get_logger

logger = get_logger("E5v2Judge")

OUT_DIR = ROOT / "results" / "e5v2"
MAX_WORKERS = 4


def collect_evidence(configs: list[str]) -> dict[str, dict[str, list[str]]]:
    """重建各配置的检索器，取每条 query 的 top-3 片段文本（与生成时同口径）。"""
    import gc
    import torch

    docs = load_docs()
    items = load_items()
    evidence: dict[str, dict[str, list[str]]] = {}
    settings = build_settings()
    for name in configs:
        pipe = settings[name]
        t0 = time.time()
        pipe.ingest(docs)
        logger.info(f"[{name}] ingested in {time.time()-t0:.0f}s")
        ev: dict[str, list[str]] = {}
        for it in items:
            try:
                retrieved = pipe.retriever.retrieve(it["query"], top_k=5)
                ev[it["id"]] = [r.chunk.content for r in retrieved[:3]]
            except Exception as e:
                logger.warning(f"[{name}] retrieve fail {it['id']}: {e}")
                ev[it["id"]] = []
        evidence[name] = ev
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return evidence


def main():
    per_item_files = sorted(OUT_DIR.glob("per_item_*.jsonl"))
    records = []  # (config, item_id, query, gold, source, pred, retrieved_texts)
    items_by_id = {it["id"]: it for it in load_items()}

    configs = []
    for f in per_item_files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            configs.append(r["config"]) if r["config"] not in configs else None
    logger.info(f"Configs from per_item files: {configs}")

    evidence = collect_evidence(configs)

    for f in per_item_files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            src = items_by_id[r["id"]].get("answer_source") or {}
            source = f"{src.get('section','')} (p.{src.get('page','?')})"
            records.append({
                "config": r["config"], "item_id": r["id"],
                "query": r["query"], "gold": r["gold"], "source": source,
                "pred": r["pred"],
                "retrieved": evidence[r["config"]].get(r["id"], []),
            })

    # 同一 item 内 7 配置盲化编号（seed 固定）
    by_item = defaultdict(list)
    for r in records:
        by_item[r["item_id"]].append(r)
    rng = random.Random(42)
    tasks = []
    for iid, group in by_item.items():
        order = list(range(len(group)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            r = group[idx]
            tasks.append((r, f"M{rank+1}"))

    logger.info(f"Judging {len(tasks)} items with {MAX_WORKERS} workers...")
    t0 = time.time()
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(lambda a: (a[0], a[1], call_judge(
            query=a[0]["query"], gold=a[0]["gold"], source=a[0]["source"],
            predicted=a[0]["pred"], retrieved=a[0]["retrieved"], method_code=a[1])), t)
            for t in tasks]
        for fut in as_completed(futs):
            r, code, scores = fut.result()
            if scores:
                results.append({"item_id": r["item_id"], "config": r["config"],
                                "method_code": code, "scores": scores})
            done += 1
            if done % 40 == 0:
                logger.info(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    with open(OUT_DIR / "judge_e5v2_detailed.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cfg = defaultdict(lambda: defaultdict(list))
    for j in results:
        for dim, v in j["scores"].items():
            if isinstance(v, (int, float)) and v >= 0:
                by_cfg[j["config"]][dim].append(v)
    summary = {}
    for cfg, dims in by_cfg.items():
        summary[cfg] = {
            dim: round(sum(vs) / len(vs), 3) for dim, vs in sorted(dims.items())
        }
        summary[cfg]["n"] = len(dims.get("overall_quality", []))
    with open(OUT_DIR / "judge_e5v2_results.json", "w", encoding="utf-8") as f:
        json.dump({"judge_model": "qwen-max", "rubric": "v2", "n_items": len(items_by_id),
                   "summary": summary}, f, ensure_ascii=False, indent=2)

    print("\n=== E5v2 LLM-Judge (qwen-max, blind, v2 rubric) ===")
    print(f"{'Config':<32}{'Factual':>9}{'Ground':>8}{'Constr':>8}{'Overall':>9}{'Abst':>7}{'n':>5}")
    for cfg, s in summary.items():
        print(f"{cfg:<32}{s.get('factual_correctness',0):>9.2f}"
              f"{s.get('source_grounding',0):>8.2f}{s.get('constraint_accuracy',0):>8.2f}"
              f"{s.get('overall_quality',0):>9.2f}{s.get('abstention_quality',0):>7.2f}{s.get('n',0):>5}")


if __name__ == "__main__":
    main()
