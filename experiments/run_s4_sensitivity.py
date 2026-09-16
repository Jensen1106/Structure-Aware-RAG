"""S4: Hyperparameter Sensitivity Analysis

Sensitivity analysis for 3 key hyperparameters of StructStdRAG:
1. chunk.maint_paragraph_max (Layer 2 paragraph chunk size)
2. retriever.top_k (number of retrieved results returned)
3. retriever.alpha (Dense vs BM25 weight)

Output line charts to demonstrate the robustness of the method.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, ChunkConfig, RetrieverConfig
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.data.loader import Document
from src.evaluation.metrics import (
    tuple_f1,
    hallucination_rate,
    recall_at_k,
    extract_answer_tuples,
    estimate_hallucination,
)
from src.utils.logger import get_logger

logger = get_logger("S4")


def load_test_data(test_file: str, limit: int | None = None):
    """Load test set"""
    test_path = Path(test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test set does not exist: {test_file}")

    items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break

    logger.info(f"Loaded test set: {len(items)} items")
    return items


def load_documents(doc_file: str):
    """Load documents"""
    doc_path = Path(doc_file)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document does not exist: {doc_file}")

    documents = []
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            doc_data = json.loads(line)
            # Create Document object
            doc = Document(
                doc_id=doc_data.get("page_id", ""),
                title=doc_data.get("section", ""),
                content=doc_data.get("content", ""),
                source_path=doc_file,
                doc_type="jsonl",
                metadata={
                    "page": doc_data.get("page", 0),
                    "section": doc_data.get("section", ""),
                }
            )
            documents.append(doc)

    logger.info(f"Loaded documents: {len(documents)} pages")
    return documents


def evaluate_pipeline(pipeline, test_items, documents):
    """Evaluate pipeline performance"""
    # Ingest documents
    logger.info("Ingesting documents...")
    pipeline.ingest(documents)

    # Batch evaluation
    tuple_f1_scores = []
    hallucination_flags = []
    recall_scores = []

    for idx, item in enumerate(test_items):
        if (idx + 1) % 20 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(test_items)}")

        query = item["query"]
        gold_answer = item["answer"]

        # Run pipeline
        result = pipeline.run(query)
        pred_answer = result.generation.answer

        # 1. Tuple-F1
        gold_tuples = extract_answer_tuples(gold_answer, parameter="constraint")
        pred_tuples = extract_answer_tuples(pred_answer, parameter="constraint")
        tuple_f1_scores.append(tuple_f1(pred_tuples, gold_tuples))

        # 2. Hallucination Rate
        supporting_text = " ".join([r.chunk.content for r in result.retrieved])
        has_hallucination = estimate_hallucination(
            pred_answer,
            supporting_text=supporting_text,
            confidence=result.generation.metadata.get("confidence"),
            metadata=result.generation.metadata,
        )
        hallucination_flags.append(has_hallucination)

        # 3. Recall@5
        retrieved_ids = [r.chunk.chunk_id for r in result.retrieved]
        # Find chunk IDs containing the answer (simplified: text-match based)
        relevant_ids = []
        for doc in documents:
            if gold_answer in doc.content:
                relevant_ids.append(doc.doc_id)

        if relevant_ids:
            recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=5))
        else:
            recall_scores.append(0.0)

    # Compute average metrics
    metrics = {
        "Tuple-F1": sum(tuple_f1_scores) / len(tuple_f1_scores) if tuple_f1_scores else 0.0,
        "Hallucination Rate": sum(hallucination_flags) / len(hallucination_flags) if hallucination_flags else 0.0,
        "Recall@5": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }

    return metrics


def run_sensitivity_analysis(
    test_file: str,
    doc_file: str,
    output_dir: str,
    test_limit: int = 88,
):
    """Run hyperparameter sensitivity analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    test_items = load_test_data(test_file, limit=test_limit)
    documents = load_documents(doc_file)

    results = {
        "test_size": len(test_items),
        "doc_size": len(documents),
        "experiments": {},
    }

    # ============================================================
    # Experiment 1: chunk.maint_paragraph_max (Layer 2 paragraph chunk size)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1: chunk.maint_paragraph_max Sensitivity Analysis")
    logger.info("=" * 60)

    param_values = [256, 384, 512, 768, 1024]
    exp1_results = {}

    for value in param_values:
        logger.info(f"\nTesting maint_paragraph_max = {value}")

        # Create config
        config = ExperimentConfig()
        config.chunk.maint_paragraph_max = value

        # Create pipeline (with config)
        pipeline = StructStdRAGPipeline()
        pipeline.chunker.paragraph_max = value

        # Evaluate
        metrics = evaluate_pipeline(pipeline, test_items, documents)
        exp1_results[value] = metrics

        logger.info(f"  Results: {metrics}")

    results["experiments"]["maint_paragraph_max"] = exp1_results

    # ============================================================
    # Experiment 2: retriever.top_k (number of retrieved results)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2: retriever.top_k Sensitivity Analysis")
    logger.info("=" * 60)

    param_values = [3, 5, 7, 10, 15]
    exp2_results = {}

    for value in param_values:
        logger.info(f"\nTesting top_k = {value}")

        # Create config
        config = ExperimentConfig()
        config.retriever.top_k = value

        # Create pipeline
        pipeline = StructStdRAGPipeline()
        pipeline.retriever.top_k = value

        # Evaluate
        metrics = evaluate_pipeline(pipeline, test_items, documents)
        exp2_results[value] = metrics

        logger.info(f"  Results: {metrics}")

    results["experiments"]["top_k"] = exp2_results

    # ============================================================
    # Experiment 3: retriever.alpha (Dense vs BM25 weight)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 3: retriever.alpha Sensitivity Analysis")
    logger.info("=" * 60)

    param_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    exp3_results = {}

    for value in param_values:
        logger.info(f"\nTesting alpha = {value}")

        # Create config
        config = ExperimentConfig()
        config.retriever.alpha = value

        # Create pipeline
        pipeline = StructStdRAGPipeline()
        pipeline.retriever.alpha = value
        pipeline.retriever.beta = 1.0 - value

        # Evaluate
        metrics = evaluate_pipeline(pipeline, test_items, documents)
        exp3_results[value] = metrics

        logger.info(f"  Results: {metrics}")

    results["experiments"]["alpha"] = exp3_results

    # ============================================================
    # Save results
    # ============================================================
    result_file = output_path / "s4_sensitivity.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to: {result_file}")

    # ============================================================
    # Plot line charts
    # ============================================================
    logger.info("\nPlotting line charts...")
    plot_sensitivity_curves(results, output_path)

    # ============================================================
    # Output optimal values
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Optimal Hyperparameter Values")
    logger.info("=" * 60)

    for param_name, param_results in results["experiments"].items():
        best_value = None
        best_f1 = -1

        for value, metrics in param_results.items():
            if metrics["Tuple-F1"] > best_f1:
                best_f1 = metrics["Tuple-F1"]
                best_value = value

        logger.info(f"{param_name}: {best_value} (Tuple-F1 = {best_f1:.4f})")


def plot_sensitivity_curves(results: dict, output_dir: Path):
    """Plot sensitivity analysis line charts"""
    import matplotlib.pyplot as plt
    import matplotlib

    # Set CJK font support
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    experiments = results["experiments"]

    # Plot one chart per hyperparameter
    for param_name, param_results in experiments.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        values = sorted(param_results.keys(), key=lambda x: float(x))
        tuple_f1_scores = [param_results[v]["Tuple-F1"] for v in values]
        hallucination_rates = [param_results[v]["Hallucination Rate"] for v in values]
        recall_scores = [param_results[v]["Recall@5"] for v in values]

        # Plot lines
        ax.plot(values, tuple_f1_scores, marker='o', label='Tuple-F1', linewidth=2)
        ax.plot(values, hallucination_rates, marker='s', label='Hallucination Rate', linewidth=2)
        ax.plot(values, recall_scores, marker='^', label='Recall@5', linewidth=2)

        # Set labels
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Sensitivity Analysis: {param_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save figure
        fig_file = output_dir / f"s4_param_{param_name}.png"
        plt.tight_layout()
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Chart saved: {fig_file}")


if __name__ == "__main__":
    run_sensitivity_analysis(
        test_file="data/maintqa/test_mil470a_88.jsonl",
        doc_file="data/mil470a/mil470a_pages_clean.jsonl",
        output_dir="results/s4",
        test_limit=88,  # Use all 88 test items
    )
