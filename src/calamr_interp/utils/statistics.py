"""Statistical testing utilities: bootstrap CIs, permutation tests, effect sizes."""

from typing import Tuple, Optional, Dict, List

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of values.
        statistic: Function to compute (default: np.mean).
        n_resamples: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))

    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(statistic(data)), float(lower), float(upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def mann_whitney_test(
    group1: np.ndarray, group2: np.ndarray
) -> Dict[str, float]:
    """Perform Mann-Whitney U test.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Dict with statistic, p_value, and effect_size (rank-biserial r).
    """
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    n1, n2 = len(group1), len(group2)
    # Rank-biserial correlation as effect size
    r = 1 - (2 * stat) / (n1 * n2)
    return {"statistic": float(stat), "p_value": float(p), "effect_size_r": float(r)}


def wilcoxon_signed_rank(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test for paired data.

    Args:
        x: First paired values.
        y: Second paired values.

    Returns:
        Dict with statistic and p_value.
    """
    stat, p = stats.wilcoxon(x, y, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction to multiple p-values.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of corrected p-values (capped at 1.0).
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Estimate mutual information between continuous feature and binary label.

    Args:
        x: Continuous feature values.
        y: Binary labels (0/1).
        n_bins: Number of bins for discretization.

    Returns:
        Estimated mutual information in nats.
    """
    from sklearn.metrics import mutual_info_score

    x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins))
    return float(mutual_info_score(x_binned, y))


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic=lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Perform a permutation test.

    Args:
        group1: First group values.
        group2: Second group values.
        statistic: Test statistic function.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with observed_statistic and p_value.
    """
    rng = np.random.RandomState(seed)
    observed = statistic(group1, group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic(combined[:n1], combined[n1:])
        if abs(perm_stat) >= abs(observed):
            count += 1

    return {
        "observed_statistic": float(observed),
        "p_value": float((count + 1) / (n_permutations + 1)),
    }
