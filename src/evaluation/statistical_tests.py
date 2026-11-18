"""Statistical significance tests for comparing retrieval methods."""

import numpy as np
from scipy import stats


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap hypothesis test."""
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert n == len(b), "Score lists must have same length"

    observed_diff = np.mean(a) - np.mean(b)
    diffs = a - b

    boot_diffs = []
    for _ in range(n_resamples):
        sample_idx = rng.randint(0, n, size=n)
        boot_diffs.append(np.mean(diffs[sample_idx]))

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    if observed_diff > 0:
        p_value = float(2 * np.mean(boot_diffs <= 0))
    else:
        p_value = float(2 * np.mean(boot_diffs >= 0))

    return {
        "p_value": p_value,
        "mean_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_005": p_value < 0.05,
    }


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict[str, float]:
    """Paired t-test for two systems."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    diff = np.array(scores_a) - np.array(scores_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "significant_005": p_value < 0.05,
    }


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni correction to multiple p-values."""
    m = len(p_values)
    adjusted_alpha = alpha / m
    return [
        {
            "original_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
        }
        for p in p_values
    ]


def significance_matrix(
    method_scores: dict[str, list[float]],
    test: str = "bootstrap",
    n_resamples: int = 10000,
) -> dict[str, dict[str, dict]]:
    """Compute pairwise significance tests between all methods."""
    methods = list(method_scores.keys())
    results = {}

    for i, a in enumerate(methods):
        results[a] = {}
        for j, b in enumerate(methods):
            if i >= j:
                continue
            if test == "bootstrap":
                r = paired_bootstrap_test(method_scores[a], method_scores[b], n_resamples)
            else:
                r = paired_t_test(method_scores[a], method_scores[b])
            results[a][b] = r

    return results
