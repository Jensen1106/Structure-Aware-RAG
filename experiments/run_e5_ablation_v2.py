"""E5v2: 扩展消融实验（返修 R3-3 / D-2 决策）

在 test_mil470a_88 (n=88) + MIL470A 语料（mil470a_pages_clean.jsonl，709 页文档）上
跑 7 个配置 × 多指标，替代原 run_e5_ablation.py（其默认走 data/raw 冒烟路径）。

配置（与论文命名一致）：
  Full                      StdChunk + CatRetriever + BoundGrounder
  w/o StdChunk              Recursive  + CatRetriever + BoundGrounder
  w/o CatRetriever          StdChunk + Hybrid        + BoundGrounder
  w/o BoundGrounder         StdChunk + CatRetriever + Direct
  w/o StdChunk+CatRetriever Recursive  + Hybrid        + BoundGrounder
  w/o StdChunk+BoundGrounder Recursive + CatRetriever + Direct
  w/o CatRetriever+BoundGrounder StdChunk + Hybrid   + Direct

指标（与 run_all_baselines_test200.py 同口径）：
  生成：EM / PM / Tuple-F1(+95%CI) / Hallucination / SrcTrace
  检索：R@1 / R@5 / R@10 / MRR@10 / NDCG@10（页级 ±1 命中，独立 top-10 检索，不影响生成）
  延迟：t_retrieve / t_generate 均值（来自 PipelineResult.metadata）
  统计：Full vs 各配置 paired bootstrap（Tuple-F1 / PM / Hal）

输出：results/e5v2/results.json + per_item.jsonl（全量预测文本，供 LLM-Judge 重评）

用法：
  python experiments/run_e5_ablation_v2.py                    # 全量 7 配置 × 88 条
  python experiments/run_e5_ablation_v2.py --limit 5          # 冒烟
  python experiments/run_e5_ablation_v2.py --configs Full,"w/o BoundGrounder"
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MAINTQA_DIR, RESULTS_DIR
from src.data.loader import Document
from src.evaluation.metrics import (
    exact_match,
    extract_answer_tuples,
    partial_match,
    tuple_f1,
)
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.pipeline.base import BasePipeline
from src.chunk.std_chunk import StdChunker
from src.chunk.recursive_chunk import RecursiveChunker
from src.retriever.cat_retriever import CatRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.bound_grounder import BoundGrounder
from src.generator.direct_gen import DirectGenerator
from src.utils.logger import get_logger

logger = get_logger("E5v2")

PAGES = ROOT / "data" / "mil470a" / "mil470a_pages_clean.jsonl"
NUMERIC_RE = re.compile(r"(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|%|°)")
PAGE_RE = re.compile(r"page[_ ]?(\d+)", re.I)


# ── 与 run_all_baselines_test200.py 完全同口径的评测函数 ─────────────────────

def is_hallucinated(pred: str, retrieved: list) -> bool:
    if not pred:
        return True
    pred_nums = set(m[0] for m in NUMERIC_RE.findall(pred))
    if not pred_nums:
        return False
    combined = " ".join(r.chunk.content for r in retrieved[:3])
    return any(n not in combined for n in pred_nums)


def source_traceability(pred: str, retrieved: list) -> float:
    if not pred or not retrieved:
        return 0.0
    tokens = re.findall(r"[a-zA-Z0-9]+", pred.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return 0.0
    combined = " ".join(r.chunk.content for r in retrieved[:3]).lower()
    return sum(1 for t in tokens if t in combined) / len(tokens)


def _normalize(answer: str) -> str:
    marker = "\n\n---\n验证警告："
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


def load_docs() -> list[Document]:
    docs = []
    with open(PAGES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text.strip()) > 100:
                docs.append(Document(
                    doc_id=f"mil470a_page_{i + 1}",
                    title=f"MIL-HDBK-470A Page {i + 1}",
                    content=text,
                    source_path=str(PAGES),
                    doc_type="jsonl",
                ))
    return docs


def load_items(limit: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in
             open(MAINTQA_DIR / "test_mil470a_88.jsonl", encoding="utf-8") if l.strip()]
    return items[:limit] if limit else items


# ── 页级检索指标（±1 页命中，论文 Group C 口径） ────────────────────────────

def chunk_page(res) -> int | None:
    m = PAGE_RE.search(res.chunk.doc_id or "")
    return int(m.group(1)) if m else None


def retrieval_page_metrics(retrieved_top10: list, gold_page: int | None,
                           ks=(1, 5, 10)) -> dict:
    if gold_page is None:
        return {}
    pages = [chunk_page(r) for r in retrieved_top10]
    hits = [p is not None and abs(p - gold_page) <= 1 for p in pages]
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(any(hits[:k]))
    # MRR@10
    rr = 0.0
    for i, h in enumerate(hits[:10]):
        if h:
            rr = 1.0 / (i + 1)
            break
    out["MRR@10"] = rr
    # NDCG@10（二值相关性）
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits[:10]) if h)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(10, sum(hits))))
    out["NDCG@10"] = dcg / ideal if ideal > 0 else 0.0
    return out


# ── 消融配置 ────────────────────────────────────────────────────────────────

def build_settings() -> dict[str, BasePipeline]:
    return {
        "Full": BasePipeline(StdChunker(), CatRetriever(), BoundGrounder()),
        "w/o StdChunk": BasePipeline(RecursiveChunker(), CatRetriever(), BoundGrounder()),
        "w/o CatRetriever": BasePipeline(StdChunker(), HybridRetriever(), BoundGrounder()),
        "w/o BoundGrounder": BasePipeline(StdChunker(), CatRetriever(), DirectGenerator()),
        "w/o StdChunk+CatRetriever": BasePipeline(RecursiveChunker(), HybridRetriever(), BoundGrounder()),
        "w/o StdChunk+BoundGrounder": BasePipeline(RecursiveChunker(), CatRetriever(), DirectGenerator()),
        "w/o CatRetriever+BoundGrounder": BasePipeline(StdChunker(), HybridRetriever(), DirectGenerator()),
    }


def eval_config(name: str, pipeline: BasePipeline, docs, items, out_dir: Path) -> dict:
    logger.info(f"▶ {name}: ingesting {len(docs)} docs")
    t0 = time.time()
    pipeline.ingest(docs)
    logger.info(f"  ingested in {time.time() - t0:.0f}s")

    em_s, pm_s, tf_s, hal_s, src_s = [], [], [], [], []
    ret_rows: list[dict] = []
    per_item = []
    t_ret, t_gen = [], []

    for i, item in enumerate(items):
        if i and i % 20 == 0:
            logger.info(f"  {name} progress {i}/{len(items)}")
        try:
            res = pipeline.run(item["query"], top_k=5)
            pred = _normalize(res.generation.answer)
            retrieved = res.retrieved
            t_ret.append(res.metadata.get("t_retrieve", 0.0))
            t_gen.append(res.metadata.get("t_generate", 0.0))
        except Exception as e:
            logger.warning(f"  {name} fail {item['id']}: {e}")
            pred, retrieved = "", []
        gold = item["answer"]
        gold_page = (item.get("answer_source") or {}).get("page")

        # 独立 top-10 检索（不影响上面 top-5 生成口径）
        try:
            top10 = pipeline.retriever.retrieve(item["query"], top_k=10)
        except Exception:
            top10 = []
        rmet = retrieval_page_metrics(top10, gold_page)

        em = exact_match(pred, gold)
        pm = partial_match(pred, gold)
        tf = tuple_f1(
            extract_answer_tuples(pred, parameter=item.get("constraint_type", "")),
            extract_answer_tuples(gold, parameter=item.get("constraint_type", "")),
        )
        hal = 1 if is_hallucinated(pred, retrieved) else 0
        src = source_traceability(pred, retrieved)
        em_s.append(em); pm_s.append(pm); tf_s.append(tf); hal_s.append(hal); src_s.append(src)
        ret_rows.append(rmet)
        per_item.append({
            "config": name, "id": item["id"],
            "constraint_type": item.get("constraint_type"),
            "query": item["query"], "pred": pred, "gold": gold,
            "gold_page": gold_page,
            "retrieved_top10": [
                {"doc_id": r.chunk.doc_id, "score": round(r.score, 4)}
                for r in top10
            ],
            "em": em, "pm": pm, "tuple_f1": tf, "hal": hal, "src": src,
            **{k: v for k, v in rmet.items()},
        })

    n = len(items)
    ci_tf = bootstrap_ci(tf_s, n_bootstrap=2000)
    ci_pm = bootstrap_ci(pm_s, n_bootstrap=2000)
    metrics = {
        "n": n,
        "Exact Match": sum(em_s) / n,
        "Partial Match": sum(pm_s) / n,
        "Partial Match CI": [ci_pm["ci_lower"], ci_pm["ci_upper"]],
        "Tuple-F1": sum(tf_s) / n,
        "Tuple-F1 CI": [ci_tf["ci_lower"], ci_tf["ci_upper"]],
        "Hallucination Rate": sum(hal_s) / n,
        "Source Traceability": sum(src_s) / n,
        "mean_t_retrieve_s": sum(t_ret) / max(len(t_ret), 1),
        "mean_t_generate_s": sum(t_gen) / max(len(t_gen), 1),
        "_raw_tf": tf_s, "_raw_pm": pm_s, "_raw_hal": hal_s,
        "_per_item": per_item, "_ret_rows": ret_rows,
        "_elapsed_s": round(time.time() - t0, 1),
    }
    for k in ("R@1", "R@5", "R@10", "MRR@10", "NDCG@10"):
        vals = [r[k] for r in ret_rows if k in r]
        metrics[k] = sum(vals) / len(vals) if vals else None

    _dump_config(out_dir, name, metrics)
    logger.info(f"✓ {name}: TF={metrics['Tuple-F1']:.4f} PM={metrics['Partial Match']:.4f} "
                f"Hal={metrics['Hallucination Rate']:.3f} Src={metrics['Source Traceability']:.3f} "
                f"R@10={metrics['R@10']} in {metrics['_elapsed_s']:.0f}s")
    return metrics


def _dump_config(out_dir: Path, name: str, metrics: dict):
    """增量落盘：每个配置完成即写 per-config JSON + per-item JSONL（防中断丢结果）"""
    safe = name.replace("/", "_").replace(" ", "_")
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    clean["retrieval"] = {k: metrics.get(k) for k in ("R@1", "R@5", "R@10", "MRR@10", "NDCG@10")}
    with open(out_dir / f"config_{safe}.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"per_item_{safe}.jsonl", "w", encoding="utf-8") as f:
        for p in metrics["_per_item"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="限制条数（冒烟用）")
    ap.add_argument("--configs", type=str, default=None,
                    help="逗号分隔的配置子集，如 'Full,w/o BoundGrounder'")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else RESULTS_DIR / "e5v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    items = load_items(args.limit)
    logger.info(f"Loaded {len(docs)} docs, {len(items)} items")

    all_settings = build_settings()
    wanted = args.configs.split(",") if args.configs else list(all_settings)
    results = {}
    for name in wanted:
        name = name.strip()
        if name not in all_settings:
            logger.error(f"Unknown config: {name}; available: {list(all_settings)}")
            continue
        # 及时释放上一配置的 GPU/内存
        import gc
        import torch
        results[name] = eval_config(name, all_settings[name], docs, items, out_dir)
        gc.collect(); torch.cuda.empty_cache()

    # paired bootstrap: Full vs 每个消融配置
    stats = []
    if "Full" in results:
        full = results["Full"]
        for name, m in results.items():
            if name == "Full":
                continue
            if m["n"] != full["n"]:
                continue
            bp_tf = bootstrap_paired_test(full["_raw_tf"], m["_raw_tf"])
            bp_pm = bootstrap_paired_test(full["_raw_pm"], m["_raw_pm"])
            bp_hal = bootstrap_paired_test(full["_raw_hal"], m["_raw_hal"])
            stats.append({
                "comparison": f"Full vs {name}",
                "tuple_f1_diff": bp_tf["mean_diff"], "tuple_f1_p": bp_tf["p_value"],
                "pm_diff": bp_pm["mean_diff"], "pm_p": bp_pm["p_value"],
                "hal_diff": bp_hal["mean_diff"], "hal_p": bp_hal["p_value"],
            })

    payload = {
        "test_set": "test_mil470a_88",
        "corpus": str(PAGES),
        "n_docs": len(docs), "n_items": len(items),
        "summary": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in results.items()},
        "stats": stats,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 110)
    print(f"E5v2 扩展消融 · test_mil470a_88 (n={len(items)})")
    print("=" * 110)
    hdr = (f"{'Config':<30}{'TF':>7}{'PM':>7}{'Hal':>6}{'Src':>6}"
           f"{'R@1':>6}{'R@5':>6}{'R@10':>6}{'MRR':>6}{'NDCG':>6}{'tRet':>6}{'tGen':>6}")
    print(hdr)
    print("-" * 110)
    for name, m in results.items():
        def _f(v, w=6):
            return f"{v:.3f}"[:w].rjust(w) if isinstance(v, (int, float)) else "-".rjust(w)
        print(f"{name:<30}{m['Tuple-F1']:>7.3f}{m['Partial Match']:>7.3f}"
              f"{m['Hallucination Rate']:>6.3f}{m['Source Traceability']:>6.3f}"
              f"{_f(m.get('R@1'))}{_f(m.get('R@5'))}{_f(m.get('R@10'))}"
              f"{_f(m.get('MRR@10'))}{_f(m.get('NDCG@10'))}"
              f"{m['mean_t_retrieve_s']:>6.2f}{m['mean_t_generate_s']:>6.2f}")
    if stats:
        print("\nPaired bootstrap (Full vs each config):")
        for s in stats:
            star = "***" if s["tuple_f1_p"] < 0.001 else "**" if s["tuple_f1_p"] < 0.01 \
                else "*" if s["tuple_f1_p"] < 0.05 else "ns"
            print(f"  {s['comparison']:<42} ΔTF={s['tuple_f1_diff']:+.4f} "
                  f"p={s['tuple_f1_p']:.4f} {star:>3}")


if __name__ == "__main__":
    main()
