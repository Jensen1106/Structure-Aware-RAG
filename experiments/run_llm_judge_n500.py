"""S1: LLM-as-Judge automatic evaluation (replacing human evaluation)

Reference methodology:
- G-Eval (Liu et al., 2023): GPT-4 as Judge with chain-of-thought
- MT-Bench (Zheng et al., 2023): LLM-judge aligns with human >80%
- Prometheus (Kim et al., 2024): fine-grained multi-dim scoring

Design:
- Judge model: qwen-max (stronger than the evaluated model qwen-turbo, avoiding self-preference bias)
- Scoring dimensions (4 dims, each 1-5 Likert):
  * factual_correctness   factual correctness
  * source_grounding      source citation accuracy
  * constraint_accuracy   constraint value/type accuracy
  * overall_quality       overall quality
- Blind evaluation: method names anonymized as M1/M2/.../Mn, randomized order to reduce position bias
- Repeat scoring 3 times and average to reduce variance (paper may use N=1 for simplicity)

Output:
- per_item_outputs.jsonl  generated answer per method per question
- llm_judge_scores.json   per-method average score + bootstrap 95% CI
- llm_judge_detailed.jsonl each judgment detail
"""

from __future__ import annotations

import os
# Force offline mode — avoid 9-min network timeout when loading HF models
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
from src.config import RESULTS_DIR, MAINTQA_DIR, LLM_CONFIG
from src.data.loader import Document
from src.pipeline.naive_rag import NaiveRAGPipeline
from src.pipeline.dense_rag import DenseRAGPipeline
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.evaluation.statistical_test import bootstrap_ci
from src.utils.logger import get_logger

logger = get_logger("LLMJudge")

# ── Judge configuration ───────────────────────────────────────────────────────

JUDGE_MODEL = "qwen-max"  # Stronger than qwen-turbo, avoiding self-preference bias
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 512
JUDGE_N_REPEATS = 1  # Number of scoring repetitions per item (use 1 to save cost for paper)
MAX_WORKERS = 4      # Concurrent calls

CLIENT = OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
)


JUDGE_PROMPT_TEMPLATE = """You are a senior expert in maintainability engineering, responsible for evaluating the answer quality of multiple RAG systems to the same maintainability question.

# Evaluation Task

**Query**:
{query}

**Gold Answer (from MIL-HDBK-470A)**:
{gold}

**Source Section**: {source}

**Answer to Evaluate** (System {method_code}):
{predicted}

**Top-3 Retrieved Chunks for This System** (for judging source citation):
{retrieved}

# Scoring Dimensions (each dimension 1-5, integer)

1. **factual_correctness (Factual Correctness)**
   - 5: The answer content is consistent with the gold answer; all facts are accurate
   - 3: Partially correct with some deviations, but core information is right
   - 1: The answer differs substantially from the gold answer or is completely incorrect

2. **source_grounding (Source Citation Accuracy)**
   - 5: Every key statement in the answer has direct support in the retrieved chunks with explicit source annotation
   - 3: Most statements are supported; a few lack citations
   - 1: The answer is disconnected from the retrieved chunks or entirely lacks source support

3. **constraint_accuracy (Constraint Value/Type Accuracy)**
   - 5: Numerical values, units, and constraint types (e.g., accessibility/diagnosability) are all accurate
   - 3: Numerical values are roughly correct but units or types have minor deviations
   - 1: Numerical errors or incorrect constraint type judgments

4. **overall_quality (Overall Quality)**
   - 5: The answer is clear, complete, and directly usable for engineering decisions
   - 3: Usable but requires human supplementation
   - 1: Not usable

# Output Requirements

**Strictly output JSON only**, without any extra text:

```json
{{"factual_correctness": integer, "source_grounding": integer, "constraint_accuracy": integer, "overall_quality": integer, "reason": "One-sentence justification (no more than 30 words)"}}
```"""


def _parse_judge_response(text: str) -> dict | None:
    """Extract JSON scores from LLM response."""
    # Try direct parsing
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the first JSON object
    match = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except Exception:
        return None
    # Validate required fields
    required = ["factual_correctness", "source_grounding",
                "constraint_accuracy", "overall_quality"]
    if not all(k in obj for k in required):
        return None
    # Force integer, clip to 1-5
    for k in required:
        try:
            obj[k] = max(1, min(5, int(float(obj[k]))))
        except Exception:
            return None
    return obj


def call_judge(query: str, gold: str, source: str,
               predicted: str, retrieved: list,
               method_code: str, retries: int = 3) -> dict | None:
    """Call qwen-max to produce scores."""
    retrieved_text = "\n".join(
        f"[Chunk {i+1}] {r[:300]}..." if len(r) > 300 else f"[Chunk {i+1}] {r}"
        for i, r in enumerate(retrieved[:3])
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        gold=gold,
        source=source,
        method_code=method_code,
        predicted=predicted[:1500],  # Truncate overly long answers
        retrieved=retrieved_text,
    )
    for attempt in range(retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                timeout=90,
            )
            text = resp.choices[0].message.content
            parsed = _parse_judge_response(text)
            if parsed:
                return parsed
            logger.warning(f"Parse failed (attempt {attempt+1}): {text[:200]}")
        except Exception as e:
            logger.warning(f"Judge call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


# ── Generate per-item outputs ────────────────────────────────────────────────

def _normalize(answer: str) -> str:
    marker = "\n\n---\nVerification Warnings:"
    return answer.split(marker, 1)[0].strip() if marker in answer else answer.strip()


def load_test_data(n_sample: int | None = None):
    """Load documents + test set."""
    # Documents
    pages_file = Path(MAINTQA_DIR).parent / "mil470a" / "mil470a_pages_clean.jsonl"
    docs = []
    with open(pages_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row.get("text", "") or row.get("content", "")
            if content and len(content.strip()) > 100:
                docs.append(Document(
                    doc_id=f"mil470a_page_{i+1}",
                    title=row.get("title", f"Page {i+1}"),
                    content=content,
                    source_path=str(pages_file),
                    doc_type="jsonl",
                ))

    # Test set
    test_file = Path(MAINTQA_DIR) / "test_mil470a_500.jsonl"
    items = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if n_sample and n_sample < len(items):
        random.Random(42).shuffle(items)
        items = items[:n_sample]
        items.sort(key=lambda x: x["id"])

    logger.info(f"Loaded {len(docs)} docs, {len(items)} test items")
    return docs, items


def generate_outputs(pipelines: dict, docs: list, items: list,
                     out_file: Path) -> list[dict]:
    """Run each pipeline over all items, producing per-item JSONL."""
    all_outputs = []

    for name, pipeline in pipelines.items():
        logger.info(f"▶ Running {name} on {len(items)} items...")
        t0 = time.time()
        pipeline.ingest(docs)

        for idx, item in enumerate(items):
            if (idx + 1) % 20 == 0:
                logger.info(f"  {name}: {idx+1}/{len(items)} ({time.time()-t0:.0f}s)")
            try:
                result = pipeline.run(item["query"])
                pred = _normalize(result.generation.answer)
                retrieved = [r.chunk.content for r in result.retrieved[:3]]
            except Exception as e:
                logger.warning(f"  {name} failed on {item['id']}: {e}")
                pred = ""
                retrieved = []

            all_outputs.append({
                "item_id": item["id"],
                "query": item["query"],
                "gold": item["answer"],
                "source": f"{item.get('answer_source', {}).get('document', '')} - "
                          f"{item.get('answer_source', {}).get('section', '')}",
                "constraint_type": item.get("constraint_type", "Unknown"),
                "equipment_types": item.get("equipment_types", []),
                "method": name,
                "pred": pred,
                "retrieved": retrieved,
            })

        logger.info(f"  {name} done in {time.time()-t0:.0f}s")

    # Save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for o in all_outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(all_outputs)} outputs to {out_file}")
    return all_outputs


# ── Judge scoring ────────────────────────────────────────────────────────────

def blind_judge(outputs: list[dict], out_file: Path,
                method_map_file: Path | None = None) -> list[dict]:
    """Blind evaluation: method name → M1/M2/.../Mn, group by item, shuffle method order within each group."""
    # 1. Blind method names
    methods = sorted({o["method"] for o in outputs})
    rng = random.Random(42)

    # 2. Group by item_id
    by_item = defaultdict(list)
    for o in outputs:
        by_item[o["item_id"]].append(o)

    # 3. Concurrent scoring
    tasks = []
    for item_id, group in by_item.items():
        # Shuffle method order for each item
        order = list(range(len(group)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            o = group[idx]
            method_code = f"M{rank+1}"  # Blind code
            tasks.append((o, method_code))

    results = []
    logger.info(f"▶ Calling {JUDGE_MODEL} on {len(tasks)} items × {JUDGE_N_REPEATS} reps "
                f"(~{len(tasks)*8/60:.0f} min @ 8s/call)")
    t0 = time.time()

    def _run_one(args):
        o, code = args
        scores_list = []
        for _ in range(JUDGE_N_REPEATS):
            s = call_judge(
                query=o["query"], gold=o["gold"], source=o["source"],
                predicted=o["pred"], retrieved=o["retrieved"],
                method_code=code,
            )
            if s:
                scores_list.append(s)
        if not scores_list:
            return None
        # Average across multiple repetitions
        avg = {k: sum(s[k] for s in scores_list) / len(scores_list)
               for k in ["factual_correctness", "source_grounding",
                         "constraint_accuracy", "overall_quality"]}
        return {
            "item_id": o["item_id"],
            "method": o["method"],
            "method_code": code,
            "constraint_type": o["constraint_type"],
            "scores": avg,
            "reasons": [s.get("reason", "") for s in scores_list],
            "n_repeats": len(scores_list),
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)
            completed += 1
            if completed % 20 == 0:
                logger.info(f"  judged {completed}/{len(tasks)} "
                            f"({time.time()-t0:.0f}s elapsed)")

    # Save
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)}/{len(tasks)} judgments to {out_file}")
    return results


# ── Aggregate analysis ───────────────────────────────────────────────────────

def aggregate_scores(judgments: list[dict]) -> dict:
    """Aggregate by method: mean + bootstrap 95% CI."""
    by_method = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        for dim, score in j["scores"].items():
            by_method[j["method"]][dim].append(score)

    summary = {}
    dimensions = ["factual_correctness", "source_grounding",
                  "constraint_accuracy", "overall_quality"]
    for method, dims in by_method.items():
        summary[method] = {}
        for dim in dimensions:
            scores = dims.get(dim, [])
            if not scores:
                continue
            mean_s = sum(scores) / len(scores)
            try:
                ci = bootstrap_ci(scores, n_bootstrap=5000)
                lower, upper = ci["ci_lower"], ci["ci_upper"]
            except Exception:
                lower, upper = mean_s, mean_s
            summary[method][dim] = {
                "mean": mean_s,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(scores),
            }
        # Composite score: mean of 4 dimensions
        all_dim_means = [summary[method][d]["mean"] for d in dimensions if d in summary[method]]
        summary[method]["composite"] = sum(all_dim_means) / len(all_dim_means) if all_dim_means else 0.0
    return summary


def main(n_sample: int | None = None, skip_generation: bool = False):
    out_dir = RESULTS_DIR / "s1"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_file = out_dir / "per_item_outputs.jsonl"
    judgments_file = out_dir / "llm_judge_detailed.jsonl"
    summary_file = out_dir / "llm_judge_scores.json"

    # 1. Generate per-item outputs
    if skip_generation and outputs_file.exists():
        logger.info(f"Loading existing outputs from {outputs_file}")
        outputs = [json.loads(l) for l in open(outputs_file, encoding="utf-8") if l.strip()]
    else:
        docs, items = load_test_data(n_sample=n_sample)
        pipelines = {
            "Naive RAG": NaiveRAGPipeline(),
            "Dense RAG": DenseRAGPipeline(),
            "StructStdRAG": StructStdRAGPipeline(),
        }
        outputs = generate_outputs(pipelines, docs, items, outputs_file)

    # 2. Judge
    judgments = blind_judge(outputs, judgments_file)

    # 3. Aggregate
    summary = aggregate_scores(judgments)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 4. Print table
    print("\n" + "=" * 90)
    print(f"{'Method':<15} {'Factual':>12} {'Grounding':>12} {'Constraint':>12} "
          f"{'Overall':>12} {'Composite':>12}")
    print("-" * 90)
    for method in sorted(summary.keys()):
        s = summary[method]
        row = f"{method:<15}"
        for dim in ["factual_correctness", "source_grounding",
                    "constraint_accuracy", "overall_quality"]:
            if dim in s:
                row += f" {s[dim]['mean']:>12.3f}"
            else:
                row += f" {'N/A':>12}"
        row += f" {s.get('composite', 0):>12.3f}"
        print(row)
    print("=" * 90)
    print(f"\nResults saved: {out_dir}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="sample size (default: all 88)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="skip pipeline run, use existing per_item_outputs.jsonl")
    args = parser.parse_args()
    main(n_sample=args.n, skip_generation=args.skip_generation)
