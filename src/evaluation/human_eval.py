"""Human evaluation toolkit: generate evaluation forms, compute inter-rater agreement"""

import json
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HumanEvalItem:
    """Human evaluation item"""
    item_id: str
    query: str
    method: str
    answer: str
    # Scoring dimensions (1–5 Likert)
    correctness: int | None = None
    numeric_accuracy: int | None = None
    type_match: int | None = None
    readability: int | None = None
    annotator: str = ""


class HumanEvalManager:
    """Human evaluation management"""

    DIMENSIONS = ["correctness", "numeric_accuracy", "type_match", "readability"]

    def generate_eval_form(
        self,
        queries: list[str],
        method_answers: dict[str, list[str]],
        output_path: Path,
        shuffle_seed: int = 42,
    ):
        """Generate blind evaluation form (CSV)

        Args:
            queries: List of queries
            method_answers: {method_name: [list of answers]}
            output_path: Output path
            shuffle_seed: Random seed (shuffles method order to achieve blind review)
        """
        import random
        rng = random.Random(shuffle_seed)

        rows = []
        item_id = 0
        methods = list(method_answers.keys())

        for i, query in enumerate(queries):
            # Shuffle method order
            shuffled = list(methods)
            rng.shuffle(shuffled)

            for method in shuffled:
                answer = method_answers[method][i]
                rows.append({
                    "item_id": f"eval_{item_id:04d}",
                    "query": query,
                    "method_code": f"M{shuffled.index(method) + 1}",  # blind the method name
                    "answer": answer,
                    "correctness": "",
                    "numeric_accuracy": "",
                    "type_match": "",
                    "readability": "",
                    "annotator": "",
                })
                item_id += 1

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Write method mapping (sealed, opened after evaluation is complete)
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping = {f"M{i+1}": m for i, m in enumerate(methods)}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        return len(rows)

    def load_annotations(self, csv_path: Path) -> list[HumanEvalItem]:
        """Load annotation results"""
        items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(HumanEvalItem(
                    item_id=row["item_id"],
                    query=row["query"],
                    method=row["method_code"],
                    answer=row["answer"],
                    correctness=int(row["correctness"]) if row["correctness"] else None,
                    numeric_accuracy=int(row["numeric_accuracy"]) if row["numeric_accuracy"] else None,
                    type_match=int(row["type_match"]) if row["type_match"] else None,
                    readability=int(row["readability"]) if row["readability"] else None,
                    annotator=row.get("annotator", ""),
                ))
        return items

    def compute_agreement(self, annotations: dict[str, list[HumanEvalItem]]) -> dict:
        """Compute inter-annotator agreement (Fleiss' Kappa)

        Args:
            annotations: {annotator_id: [evaluation results]}
        """
        # Align by item_id
        annotators = list(annotations.keys())
        if len(annotators) < 2:
            return {"error": "At least two annotators are required."}

        results = {}
        for dim in self.DIMENSIONS:
            scores_by_item: dict[str, list[int]] = {}
            for ann_id, items in annotations.items():
                for item in items:
                    score = getattr(item, dim)
                    if score is not None:
                        scores_by_item.setdefault(item.item_id, []).append(score)

            # Compute Fleiss' Kappa
            kappa = self._fleiss_kappa(scores_by_item, n_categories=5)
            results[dim] = kappa

        return results

    def _fleiss_kappa(self, scores_by_item: dict[str, list[int]], n_categories: int = 5) -> float:
        """Fleiss' Kappa calculation"""
        import numpy as np

        items = [v for v in scores_by_item.values() if len(v) >= 2]
        if not items:
            return 0.0

        n = len(items)
        N = len(items[0])  # number of annotators

        # Build count matrix
        matrix = np.zeros((n, n_categories))
        for i, scores in enumerate(items):
            for s in scores:
                if 1 <= s <= n_categories:
                    matrix[i][s - 1] += 1

        # P_i
        P_i = (np.sum(matrix ** 2, axis=1) - N) / (N * (N - 1))
        P_bar = np.mean(P_i)

        # P_e
        p_j = np.sum(matrix, axis=0) / (n * N)
        P_e = np.sum(p_j ** 2)

        if P_e == 1:
            return 1.0
        return float((P_bar - P_e) / (1 - P_e))
