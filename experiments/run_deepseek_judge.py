"""Cross-vendor Judge — DeepSeek-V3 independent scoring.

Uses the SAME 5-dim Likert rubric as S1 v2 (qwen-max) but replaces the
judge model with `deepseek-v3` (DeepSeek family, different vendor from
Alibaba's Qwen). Scores the same (item_id, method) pairs previously
produced by run_paper_eval / S1. Computes agreement with qwen-max via
Pearson r, Spearman ρ, and quadratic-weighted Cohen's κ.

This addresses the reviewer concern about "same-vendor judge bias" and
is a mandatory piece of Q2-level LLM-as-Judge methodology.

Output: results/s1/llm_judge_deepseek_detailed.jsonl
        results/s1/judge_agreement_qwen_vs_deepseek.json
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.config import LLM_CONFIG, RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger("DeepSeekJudge")

JUDGE_MODEL = "deepseek-v3"
QWEN_RESULTS = RESULTS_DIR / "s1" / "llm_judge_v2_detailed.jsonl"
OUTPUTS = RESULTS_DIR / "s1" / "per_item_outputs.jsonl"
OUT_DETAIL = RESULTS_DIR / "s1" / "llm_judge_deepseek_detailed.jsonl"
OUT_AGREE = RESULTS_DIR / "s1" / "judge_agreement_qwen_vs_deepseek.json"

CLIENT = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

JUDGE_PROMPT = """You are a maintainability engineering domain expert evaluating the answer quality of RAG systems.

In real-world engineering scenarios, a reliable RAG system should:
(A) When accurate information is found: directly provide the answer + cite sources
(B) When accurate information cannot be found: acknowledge limitations + provide relevant background/definitions (avoid hallucination)
(C) Must never: fabricate numerical values, invent sources, or provide directly incorrect answers

# Query
{query}

# Gold Answer
{gold}

# Source Section
{source}

# Answer from System {method_code}
{predicted}

# Top-3 Chunks Retrieved by This System
{retrieved}

# Scoring Dimensions (each scored 1–5 integer)

1. factual_correctness: Factual correctness (5=fully consistent; 3=partially correct/honest acknowledgment of missing info + relevant facts; 1=completely incorrect)
2. source_grounding: Source grounding (5=every claim has a citation; 3=partial citations; 1=no citations)
3. constraint_accuracy: Constraint value/type accuracy (5=all accurate; 3=type correct but no values; 1=numerical hallucination)
4. overall_quality: Overall quality (5=directly usable by an engineer; 1=unusable)
5. abstention_quality: Abstention quality (only score when no definitive answer is given; otherwise -1. 5=acknowledges missing info + provides background + explanation; 1=vague/non-committal)

# Output Format (strict JSON, no other text)

{{"factual_correctness": integer, "source_grounding": integer, "constraint_accuracy": integer, "overall_quality": integer, "abstention_quality": integer or -1, "provides_context": true/false, "hallucinates": true/false, "reason": "one-sentence rationale ≤40 words"}}
"""


def _parse(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except Exception:
        return None
    req = ["factual_correctness", "source_grounding", "constraint_accuracy", "overall_quality"]
    if not all(k in obj for k in req):
        return None
    for k in req:
        try:
            obj[k] = max(1, min(5, int(float(obj[k]))))
        except Exception:
            return None
    if "abstention_quality" in obj:
        try:
            v = int(float(obj["abstention_quality"]))
            obj["abstention_quality"] = v if v == -1 else max(1, min(5, v))
        except Exception:
            obj["abstention_quality"] = -1
    return obj


def call_judge(query, gold, source, predicted, retrieved, method_code, retries=3):
    retrieved_text = "\n".join(
        f"[Chunk {i+1}] {r[:300]}..." if len(r) > 300 else f"[Chunk {i+1}] {r}"
        for i, r in enumerate(retrieved[:3])
    )
    prompt = JUDGE_PROMPT.format(
        query=query, gold=gold, source=source, method_code=method_code,
        predicted=predicted[:1500], retrieved=retrieved_text,
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
        except Exception as e:
            logger.warning(f"attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def run_cross_judge(max_workers: int = 4):
    outputs = [json.loads(l) for l in open(OUTPUTS, encoding="utf-8") if l.strip()]
    logger.info(f"Loaded {len(outputs)} outputs for cross-judge")

    # blind alias via same seed as S1 v2
    rng = random.Random(42)
    by_item = defaultdict(list)
    for o in outputs:
        by_item[o["item_id"]].append(o)

    tasks = []
    for iid, group in by_item.items():
        order = list(range(len(group)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            o = group[idx]
            tasks.append((o, f"M{rank + 1}"))

    results = []
    done = 0

    def run_one(args):
        o, code = args
        scores = call_judge(o["query"], o["gold"], o["source"],
                            o["pred"], o.get("retrieved", []), code)
        return {
            "item_id": o["item_id"], "method": o["method"], "method_code": code,
            "constraint_type": o["constraint_type"], "scores": scores,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                logger.info(f"  cross-judge progress {done}/{len(tasks)}")

    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DETAIL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} records to {OUT_DETAIL}")
    return results


def agreement():
    q = [json.loads(l) for l in open(QWEN_RESULTS, encoding="utf-8") if l.strip()]
    d = [json.loads(l) for l in open(OUT_DETAIL, encoding="utf-8") if l.strip()]
    logger.info(f"Qwen judge: {len(q)};  DeepSeek judge: {len(d)}")

    qmap = {(r["item_id"], r["method"]): r["scores"] for r in q}
    dmap = {(r["item_id"], r["method"]): r["scores"] for r in d if r["scores"]}
    keys = sorted(qmap.keys() & dmap.keys())
    logger.info(f"Aligned pairs: {len(keys)}")

    dims = ["factual_correctness", "source_grounding", "constraint_accuracy",
            "overall_quality"]

    import statistics
    from math import sqrt

    def pearson(a, b):
        n = len(a)
        if n < 2:
            return 0.0
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        da = sqrt(sum((x - ma) ** 2 for x in a))
        db = sqrt(sum((x - mb) ** 2 for x in b))
        return num / (da * db) if da * db else 0.0

    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        for pos, i in enumerate(order):
            ranks[i] = pos + 1
        return ranks

    def spearman(a, b):
        return pearson(rank(a), rank(b))

    def weighted_kappa(a, b, n_cat=5):
        """Quadratic-weighted Cohen's kappa on discrete 1-5 ratings."""
        from collections import Counter
        assert all(1 <= x <= n_cat for x in a)
        assert all(1 <= x <= n_cat for x in b)
        n = len(a)
        obs = [[0] * n_cat for _ in range(n_cat)]
        for x, y in zip(a, b):
            obs[x - 1][y - 1] += 1
        # marginals
        hx = [sum(row) for row in obs]
        hy = [sum(obs[i][j] for i in range(n_cat)) for j in range(n_cat)]
        w = [[((i - j) ** 2) / ((n_cat - 1) ** 2) for j in range(n_cat)] for i in range(n_cat)]
        po = sum(w[i][j] * obs[i][j] for i in range(n_cat) for j in range(n_cat)) / n
        pe = sum(w[i][j] * hx[i] * hy[j] for i in range(n_cat) for j in range(n_cat)) / (n * n)
        return 1 - po / pe if pe > 0 else 0.0

    summary = {}
    for dim in dims:
        a = [qmap[k][dim] for k in keys]
        b = [dmap[k][dim] for k in keys]
        summary[dim] = {
            "pearson_r": round(pearson(a, b), 4),
            "spearman_rho": round(spearman(a, b), 4),
            "quadratic_kappa": round(weighted_kappa(a, b), 4),
            "qwen_mean": round(sum(a) / len(a), 3),
            "deepseek_mean": round(sum(b) / len(b), 3),
            "n": len(a),
        }

    # per-method composite agreement (average of 4 dims)
    by_method = defaultdict(lambda: {"qwen": [], "deepseek": []})
    for k in keys:
        qc = sum(qmap[k][d] for d in dims) / 4
        dc = sum(dmap[k][d] for d in dims) / 4
        by_method[k[1]]["qwen"].append(qc)
        by_method[k[1]]["deepseek"].append(dc)

    per_method = {}
    for m, v in by_method.items():
        per_method[m] = {
            "qwen_composite_mean": round(sum(v["qwen"]) / len(v["qwen"]), 3),
            "deepseek_composite_mean": round(sum(v["deepseek"]) / len(v["deepseek"]), 3),
            "pearson_r": round(pearson(v["qwen"], v["deepseek"]), 4),
            "n": len(v["qwen"]),
        }

    out = {
        "n_aligned_pairs": len(keys),
        "per_dimension": summary,
        "per_method_composite": per_method,
    }
    with open(OUT_AGREE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("qwen-max vs deepseek-v3 judge agreement")
    print("=" * 90)
    print(f"{'Dim':<22} {'r':>8} {'ρ':>8} {'κ(quad)':>10} {'qwen μ':>10} {'ds μ':>10}")
    for dim in dims:
        s = summary[dim]
        print(f"{dim:<22} {s['pearson_r']:>8.3f} {s['spearman_rho']:>8.3f} "
              f"{s['quadratic_kappa']:>10.3f} {s['qwen_mean']:>10.3f} {s['deepseek_mean']:>10.3f}")
    print("\nPer-method composite (4-dim mean):")
    print(f"{'Method':<16} {'qwen':>10} {'ds':>10} {'r':>8}")
    for m, v in per_method.items():
        print(f"{m:<16} {v['qwen_composite_mean']:>10.3f} "
              f"{v['deepseek_composite_mean']:>10.3f} {v['pearson_r']:>8.3f}")
    print(f"\nSaved → {OUT_AGREE}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agreement-only", action="store_true",
                   help="Skip judging; compute agreement from existing DeepSeek results")
    args = p.parse_args()
    if not args.agreement_only:
        run_cross_judge()
    agreement()
