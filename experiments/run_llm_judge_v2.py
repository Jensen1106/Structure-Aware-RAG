"""S1 v2: LLM-as-Judge with fairer scoring rubric

关键改进：原 v1 rubric 把"系统诚实承认证据不足"和"系统胡说"都打 1 分，
导致 StructStdRAG 的"优雅降级"（提供相关上下文 + 承认特定值缺失）被低估。

v2 rubric 原则（参考 TruthfulQA + HaluEval）：
- 5: 答案准确且完整
- 4: 答案基本正确，小偏差
- 3: 答案部分正确，或诚实承认特定信息缺失同时提供相关定义/背景
- 2: 答案有明显错误但尝试回答相关内容
- 1: 答案错误、幻觉、或完全拒答且无任何相关信息

新增维度：`abstention_quality` — 评估系统"拒答时的质量"：
- 5: 正确拒答 + 提供相关定义/背景 + 解释为何无法回答
- 3: 简单拒答 "参考资料不足"
- 1: 错误拒答（其实参考资料有答案）或胡说不拒

输出：results/s1/llm_judge_v2_*.json
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
import random
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.config import RESULTS_DIR, LLM_CONFIG
from src.evaluation.statistical_test import bootstrap_ci, bootstrap_paired_test
from src.utils.logger import get_logger

logger = get_logger("LLMJudgeV2")

JUDGE_MODEL = "qwen-max"
MAX_WORKERS = 4

CLIENT = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])


JUDGE_V2_PROMPT = """你是维修性工程领域专家，正在评估 RAG 系统的回答质量。

# 背景：维修性工程 RAG 评审标准

真实工程场景下，一个可靠的 RAG 系统应该：
(A) 找到准确信息时：直接给出答案 + 引用来源
(B) 找不到准确信息时：承认局限 + 提供相关背景/定义（避免胡说 = 避免幻觉）
(C) 绝对不能：胡编数值、虚构来源、或直接回答错误

因此评分时**既要奖励正确答案，也要奖励正确的拒答行为**。

# 查询
{query}

# 标准答案（来自 MIL-HDBK-470A）
{gold}

# 来源章节
{source}

# 系统 {method_code} 的答案
{predicted}

# 该系统检索到的前 3 个片段（用于判断来源接地）
{retrieved}

# 评分维度（每维 1-5 分整数）

## 1. factual_correctness（事实正确性）
- 5: 答案与标准答案事实完全一致
- 4: 答案基本一致，细节略有差异
- 3: 答案部分一致，或答案承认缺少特定信息但提供相关事实（如相关定义、术语解释）
- 2: 答案有明显错误但尝试回答
- 1: 答案完全错误，或仅有"参考资料不足"且无任何相关信息

## 2. source_grounding（来源接地）
- 5: 答案中每个关键陈述都能在检索片段中找到并明确引用（[source: xxx] 或 "来自第 x 页"）
- 4: 大部分陈述有引用，个别缺引
- 3: 部分陈述有引用，部分仅基于常识
- 2: 很少引用，答案与检索片段脱节
- 1: 完全无引用，或引用了不存在的来源

## 3. constraint_accuracy（约束数值/类型准确性）
- 5: 数值、单位、约束类型全部准确
- 4: 数值正确但单位略有偏差
- 3: 提供了相关约束类型但未给出具体数值 / 未提问数值类型则直接给 3-5
- 2: 数值错误或约束类型判断错误
- 1: 数值幻觉或严重错误

## 4. overall_quality（整体质量）
- 5: 工程师可直接使用
- 4: 稍加补充即可使用
- 3: 可作为参考，需配合其他资料
- 2: 信息量少，需要大量补充
- 1: 不可用

## 5. abstention_quality（拒答质量 - 新增）
**仅在系统未给出明确答案时评估；若系统直接回答则填 N/A（填 -1）**
- 5: 明确承认特定信息缺失 + 提供相关定义/背景 + 解释原因
- 4: 承认局限 + 提供部分相关信息
- 3: 简单声明"参考资料不足"
- 2: 不恰当拒答（实际资料中有答案）
- 1: 既不答也不承认，语焉不详

# 输出格式（严格 JSON，不要其他文字）

{{"factual_correctness": 整数, "source_grounding": 整数, "constraint_accuracy": 整数, "overall_quality": 整数, "abstention_quality": 整数或-1, "provides_context": true/false, "hallucinates": true/false, "reason": "一句话理由≤40字"}}"""


def _parse(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except Exception:
        return None
    req = ["factual_correctness", "source_grounding", "constraint_accuracy",
           "overall_quality"]
    if not all(k in obj for k in req):
        return None
    for k in req:
        try:
            obj[k] = max(1, min(5, int(float(obj[k]))))
        except Exception:
            return None
    # abstention_quality 特殊处理（-1 表示 N/A）
    if "abstention_quality" in obj:
        try:
            v = int(float(obj["abstention_quality"]))
            obj["abstention_quality"] = v if v == -1 else max(1, min(5, v))
        except Exception:
            obj["abstention_quality"] = -1
    return obj


def call_judge(query, gold, source, predicted, retrieved, method_code, retries=3):
    retrieved_text = "\n".join(
        f"[片段 {i+1}] {r[:300]}..." if len(r) > 300 else f"[片段 {i+1}] {r}"
        for i, r in enumerate(retrieved[:3])
    )
    prompt = JUDGE_V2_PROMPT.format(
        query=query, gold=gold, source=source,
        method_code=method_code,
        predicted=predicted[:1500],
        retrieved=retrieved_text,
    )
    for attempt in range(retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=600, timeout=90,
            )
            parsed = _parse(resp.choices[0].message.content)
            if parsed:
                return parsed
            logger.warning(f"Parse failed attempt {attempt+1}")
        except Exception as e:
            logger.warning(f"Judge failed attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def main(outputs_file: Path, out_prefix: str = "llm_judge_v2", out_dir: Path | None = None):
    out_dir = out_dir or (RESULTS_DIR / "s1")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [json.loads(l) for l in open(outputs_file, encoding="utf-8") if l.strip()]
    logger.info(f"Loaded {len(outputs)} outputs")

    # 盲化
    methods = sorted({o["method"] for o in outputs})
    by_item = defaultdict(list)
    for o in outputs:
        by_item[o["item_id"]].append(o)

    rng = random.Random(42)
    tasks = []
    for iid, group in by_item.items():
        order = list(range(len(group)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            o = group[idx]
            tasks.append((o, f"M{rank+1}"))

    def run_one(args):
        o, code = args
        s = call_judge(
            query=o["query"], gold=o["gold"], source=o["source"],
            predicted=o["pred"], retrieved=o["retrieved"], method_code=code,
        )
        if not s:
            return None
        return {
            "item_id": o["item_id"], "method": o["method"], "method_code": code,
            "constraint_type": o["constraint_type"], "scores": s,
        }

    logger.info(f"Judging {len(tasks)} items...")
    t0 = time.time()
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(run_one, t) for t in tasks]):
            r = fut.result()
            if r:
                results.append(r)
            completed += 1
            if completed % 20 == 0:
                logger.info(f"  {completed}/{len(tasks)} ({time.time()-t0:.0f}s)")

    logger.info(f"Done: {len(results)}/{len(tasks)} judgments")

    # 保存明细
    detail_file = out_dir / f"{out_prefix}_detailed.jsonl"
    with open(detail_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 聚合
    by_method = defaultdict(lambda: defaultdict(list))
    abst_counts = defaultdict(lambda: {"provides_context": 0, "hallucinates": 0, "n": 0})
    for j in results:
        scores = j["scores"]
        for dim in ["factual_correctness", "source_grounding",
                    "constraint_accuracy", "overall_quality"]:
            by_method[j["method"]][dim].append(scores[dim])
        # abstention_quality：仅统计实际拒答的
        aq = scores.get("abstention_quality", -1)
        if aq != -1:
            by_method[j["method"]]["abstention_quality"].append(aq)
        abst_counts[j["method"]]["n"] += 1
        if scores.get("provides_context", False):
            abst_counts[j["method"]]["provides_context"] += 1
        if scores.get("hallucinates", False):
            abst_counts[j["method"]]["hallucinates"] += 1

    summary = {}
    dims = ["factual_correctness", "source_grounding", "constraint_accuracy",
            "overall_quality", "abstention_quality"]
    for method, d in by_method.items():
        m_summary = {}
        for dim in dims:
            scores = d.get(dim, [])
            if not scores:
                continue
            mean_s = sum(scores) / len(scores)
            try:
                ci = bootstrap_ci(scores, n_bootstrap=3000)
                m_summary[dim] = {
                    "mean": mean_s,
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                    "n": len(scores),
                }
            except Exception:
                m_summary[dim] = {"mean": mean_s, "n": len(scores)}
        # composite：4 主维度均值
        main_means = [m_summary.get(d, {}).get("mean", 0)
                      for d in dims[:4] if d in m_summary]
        m_summary["composite"] = sum(main_means) / len(main_means) if main_means else 0
        # 额外统计
        ac = abst_counts[method]
        if ac["n"] > 0:
            m_summary["provides_context_rate"] = ac["provides_context"] / ac["n"]
            m_summary["hallucination_rate_judge"] = ac["hallucinates"] / ac["n"]
        summary[method] = m_summary

    # 保存
    summary_file = out_dir / f"{out_prefix}_scores.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 配对统计检验（StructStdRAG vs 其他）
    if "StructStdRAG" in by_method:
        stats = []
        for method in [m for m in by_method if m != "StructStdRAG"]:
            # 对齐 per-item
            m_scores = {(j["item_id"]): j["scores"]
                        for j in results if j["method"] == "StructStdRAG"}
            o_scores = {(j["item_id"]): j["scores"]
                        for j in results if j["method"] == method}
            common_ids = set(m_scores) & set(o_scores)
            for dim in ["factual_correctness", "source_grounding",
                        "constraint_accuracy", "overall_quality"]:
                m_vals = [m_scores[iid][dim] for iid in common_ids]
                o_vals = [o_scores[iid][dim] for iid in common_ids]
                bp = bootstrap_paired_test(m_vals, o_vals)
                stats.append({
                    "maintrag_vs": method,
                    "dimension": dim,
                    "mean_diff": bp["mean_diff"],
                    "ci_lower": bp["ci_lower"],
                    "ci_upper": bp["ci_upper"],
                    "p_value": bp["p_value"],
                    "significant": bp["significant"],
                    "n": len(common_ids),
                })
        stats_file = out_dir / f"{out_prefix}_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # 打印
    print("\n" + "=" * 100)
    print(f"{'Method':<15} {'Factual':>10} {'Ground':>10} {'Constr':>10} {'Overall':>10} "
          f"{'Abst.Q':>10} {'Composite':>10} {'CtxRate':>8} {'HalRate':>8}")
    print("-" * 100)
    for method in sorted(summary.keys()):
        s = summary[method]
        row = f"{method:<15}"
        for dim in ["factual_correctness", "source_grounding",
                    "constraint_accuracy", "overall_quality", "abstention_quality"]:
            if dim in s:
                row += f" {s[dim]['mean']:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        row += f" {s.get('composite', 0):>10.3f}"
        row += f" {s.get('provides_context_rate', 0)*100:>7.1f}%"
        row += f" {s.get('hallucination_rate_judge', 0)*100:>7.1f}%"
        print(row)
    print("=" * 100)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str,
                        default=str(RESULTS_DIR / "s1" / "per_item_outputs.jsonl"))
    parser.add_argument("--prefix", type=str, default="llm_judge_v2")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    main(Path(args.outputs), out_prefix=args.prefix, out_dir=out_dir)
