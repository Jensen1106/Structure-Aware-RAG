"""Evaluation Metrics: Automatic Metric Calculation

Includes:
- EM (Exact Match)
- Partial Match
- Tuple-F1
- Recall@K, MRR, NDCG@K (retrieval metrics)
- Hallucination Rate, Numeric Error Rate, Unsupported Answer Rate
- Source Traceability
- Type Match Accuracy
- Context Completeness
"""

import math
import re
from collections import Counter

NUMERIC_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|m|kg|N|h|%|°|步|小时)"
)


# ============================================================
# Generation Quality Metrics
# ============================================================

def exact_match(prediction: str, gold: str) -> float:
    """Exact Match: whether prediction matches gold standard exactly"""
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def partial_match(prediction: str, gold: str) -> float:
    """Partial Match: partial matching based on token overlap"""
    pred_tokens = set(_tokenize(prediction))
    gold_tokens = set(_tokenize(gold))
    if not gold_tokens:
        return 0.0
    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tuple_f1(pred_tuples: list[tuple], gold_tuples: list[tuple]) -> float:
    """Tuple-F1: F1 score for constraint tuples (parameter, value, unit, source)

    Args:
        pred_tuples: list of predicted constraint tuples
        gold_tuples: list of gold standard constraint tuples
    """
    if not gold_tuples:
        return 1.0 if not pred_tuples else 0.0

    matched_gold = set()
    tp = 0.0

    for pred in pred_tuples:
        best_idx = None
        best_score = 0.0
        for idx, gold in enumerate(gold_tuples):
            if idx in matched_gold:
                continue
            score = _tuple_match_score(pred, gold)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.5:
            matched_gold.add(best_idx)
            tp += best_score

    precision = tp / len(pred_tuples) if pred_tuples else 0.0
    recall = tp / len(gold_tuples)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hallucination_rate(predictions: list[dict]) -> float:
    """Hallucination Rate: proportion of predictions containing unsourced
    or out-of-range values

    Args:
        predictions: list of {"answer": str, "has_hallucination": bool}
    """
    if not predictions:
        return 0.0
    hallucinated = sum(1 for p in predictions if p.get("has_hallucination", False))
    return hallucinated / len(predictions)


def numeric_error_rate(pred_values: list[float], gold_values: list[float], tolerance: float = 0.1) -> float:
    """Numeric Error Rate: proportion of numeric deviations exceeding tolerance"""
    if not gold_values:
        return 0.0
    errors = 0
    for pred, gold in zip(pred_values, gold_values):
        if gold == 0:
            if pred != 0:
                errors += 1
        elif abs(pred - gold) / abs(gold) > tolerance:
            errors += 1
    return errors / len(gold_values)


def unsupported_answer_rate(predictions: list[dict]) -> float:
    """Unsupported Answer Rate: proportion of answers contradicting
    retrieval sources"""
    if not predictions:
        return 0.0
    unsupported = sum(1 for p in predictions if p.get("is_unsupported", False))
    return unsupported / len(predictions)


def source_traceability(predictions: list[dict]) -> float:
    """Source Traceability: proportion of answers correctly citing sources"""
    if not predictions:
        return 0.0
    traced = sum(1 for p in predictions if p.get("has_source", False))
    return traced / len(predictions)


# ============================================================
# Retrieval Quality Metrics
# ============================================================

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Recall@K: proportion of relevant documents in top-K retrieval results"""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR (Mean Reciprocal Rank): reciprocal rank of the first relevant document"""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain"""
    relevant_set = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)

    # IDCG (ideal ranking)
    ideal_rels = sorted([1.0] * min(len(relevant_set), k) + [0.0] * max(0, k - len(relevant_set)), reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def type_match_accuracy(retrieved_types: list[list[str]], query_types: list[str]) -> float:
    """Type Match Accuracy: degree of match between constraint types in
    retrieval results and query constraint types"""
    if not query_types or not retrieved_types:
        return 0.0
    query_set = set(query_types)
    matched = 0
    for chunk_types in retrieved_types:
        if set(chunk_types) & query_set:
            matched += 1
    return matched / len(retrieved_types)


def context_completeness(chunk_content: str, gold_content: str) -> float:
    """Context Completeness: how much gold standard content is retained
    in the chunk"""
    gold_tokens = set(_tokenize(gold_content))
    chunk_tokens = set(_tokenize(chunk_content))
    if not gold_tokens:
        return 0.0
    return len(chunk_tokens & gold_tokens) / len(gold_tokens)


# ============================================================
# Generation Evaluation Helpers
# ============================================================

def extract_numeric_values(text: str) -> list[tuple[float, str]]:
    """Extract numeric values+units from text"""
    values = []
    for raw_value, unit in NUMERIC_PATTERN.findall(text or ""):
        try:
            values.append((float(raw_value), unit))
        except ValueError:
            continue
    return values


def extract_answer_tuples(text: str, parameter: str = "", source: str = "") -> list[tuple]:
    """Normalize answer into (parameter, value, unit, source) tuples"""
    values = extract_numeric_values(text)
    if values:
        return [
            (
                parameter or "value",
                _normalize_text(str(value)),
                unit,
                source,
            )
            for value, unit in values
        ]

    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [(parameter or "text", normalized, "", source)]


def estimate_hallucination(
    answer: str,
    supporting_text: str = "",
    confidence: float | None = None,
    metadata: dict | None = None,
) -> bool:
    """Heuristically determine whether the answer contains hallucinations"""
    metadata = metadata or {}

    constraint_values = metadata.get("constraint_values", [])
    for item in constraint_values:
        if item.get("is_hallucination"):
            return True
        if any(flag.startswith("low_confidence") for flag in item.get("flags", [])):
            return True

    if confidence is not None and confidence < 0.6:
        return True

    supporting_text = supporting_text or ""
    for value, unit in extract_numeric_values(answer):
        value_str = str(int(value)) if float(value).is_integer() else str(value)
        if value_str not in supporting_text:
            return True
        if unit and unit not in supporting_text:
            return True

    return False


def estimate_unsupported(answer: str, supporting_text: str = "", has_source: bool = False) -> bool:
    """Heuristically determine whether the answer lacks source support"""
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return True

    supporting_text = supporting_text or ""
    if not supporting_text:
        return True

    answer_values = extract_numeric_values(answer)
    if answer_values:
        for value, _unit in answer_values:
            value_str = str(int(value)) if float(value).is_integer() else str(value)
            if value_str not in supporting_text:
                return True
        return False

    answer_tokens = set(_tokenize(answer))
    support_tokens = set(_tokenize(supporting_text))
    if not answer_tokens:
        return True

    support_ratio = len(answer_tokens & support_tokens) / len(answer_tokens)
    threshold = 0.2 if has_source else 0.3
    return support_ratio < threshold


# ============================================================
# Batch Computation
# ============================================================

def compute_retrieval_metrics(
    all_retrieved: list[list[str]],
    all_relevant: list[list[str]],
    k: int = 5,
) -> dict:
    """Batch compute retrieval metrics"""
    n = len(all_retrieved)
    return {
        f"Recall@{k}": sum(recall_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)) / n,
        "MRR": sum(mrr(r, g) for r, g in zip(all_retrieved, all_relevant)) / n,
        f"NDCG@{k}": sum(ndcg_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)) / n,
    }


def compute_generation_metrics(
    predictions: list[str],
    golds: list[str],
) -> dict:
    """Batch compute generation metrics"""
    n = len(predictions)
    return {
        "EM": sum(exact_match(p, g) for p, g in zip(predictions, golds)) / n,
        "Partial Match": sum(partial_match(p, g) for p, g in zip(predictions, golds)) / n,
    }


# ============================================================
# Helpers
# ============================================================

def _tokenize(text: str) -> list[str]:
    """Simple tokenization"""
    return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())


def _normalize_text(text: str) -> str:
    """Normalize text for soft matching"""
    return " ".join(_tokenize(text))


def _tuple_match_score(pred: tuple, gold: tuple) -> float:
    """Soft matching tuple score, compatible with text-type and numeric-type
    constraints"""
    pred_param, pred_value, pred_unit = _unpack_tuple(pred)
    gold_param, gold_value, gold_unit = _unpack_tuple(gold)

    param_score = partial_match(pred_param, gold_param) if pred_param or gold_param else 1.0

    pred_num = _safe_float(pred_value)
    gold_num = _safe_float(gold_value)
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            value_score = 1.0 if pred_num == 0 else 0.0
        else:
            rel_err = abs(pred_num - gold_num) / abs(gold_num)
            value_score = 1.0 if rel_err <= 0.1 else 0.0
        unit_score = 1.0 if pred_unit == gold_unit else 0.0
        return (param_score + value_score + unit_score) / 3

    value_score = partial_match(pred_value, gold_value)
    if pred_unit or gold_unit:
        unit_score = 1.0 if pred_unit == gold_unit else 0.0
        return (param_score + value_score + unit_score) / 3
    return (param_score + value_score) / 2


def _unpack_tuple(item: tuple) -> tuple[str, str, str]:
    values = list(item) + ["", "", ""]
    return str(values[0]), str(values[1]), str(values[2])


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
