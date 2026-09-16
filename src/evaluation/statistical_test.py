"""Statistical tests: paired t-test, bootstrap confidence interval"""

import numpy as np
from typing import Optional


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired t-test

    Args:
        scores_a: Scores of method A on each sample
        scores_b: Scores of method B on each sample

    Returns:
        {"t_statistic": float, "p_value": float, "significant": bool}
    """
    from scipy import stats

    a = np.array(scores_a)
    b = np.array(scores_b)
    t_stat, p_val = stats.ttest_rel(a, b)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "mean_diff": float(np.mean(a - b)),
        "std_diff": float(np.std(a - b, ddof=1)),
    }


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float, "std": float}
    """
    rng = np.random.RandomState(seed)
    scores_arr = np.array(scores)
    n = len(scores_arr)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "mean": float(np.mean(scores_arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(scores_arr, ddof=1)),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def bootstrap_paired_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap paired test (non-parametric)

    Returns:
        {"mean_diff": float, "p_value": float, "significant": bool, "ci_lower": float, "ci_upper": float}
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    observed_diff = np.mean(diff)
    n = len(diff)

    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diff, size=n, replace=True)
        if np.mean(sample) <= 0:
            count += 1

    p_val = count / n_bootstrap

    # 95% CI of the difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diff, size=n, replace=True)
        boot_diffs.append(np.mean(sample))
    boot_diffs = np.array(boot_diffs)

    return {
        "mean_diff": float(observed_diff),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
    }
