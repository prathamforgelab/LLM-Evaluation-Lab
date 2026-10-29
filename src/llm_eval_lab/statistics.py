"""
Statistical estimation for experiment comparisons.

Method: Paired bootstrap resampling
------------------------------------
Given two experiments run on the same set of test cases, we pair each test
case's scores and compute the distribution of the aggregate-score difference
by resampling with replacement.

Why paired bootstrap?
  * Pairing by case ID controls for natural variability between test cases.
  * Bootstrap makes no distributional assumptions about the scores.
  * It is straightforward to explain and audit.

Limitations and assumptions
----------------------------
1. The bootstrap estimate assumes that test cases are exchangeable (i.e.
   drawn independently from some population).  If your dataset has strong
   internal structure (e.g. all cases from one domain), the interval may
   underestimate true uncertainty.
2. Bootstrap confidence intervals are approximate for small samples.
   Treat estimates with n < 30 as indicative, not definitive.
3. A confidence interval that does not cross zero is not a guarantee that
   the difference is practically meaningful.
4. Two identical experiments may still show a non-zero CI due to randomness
   in the resampling.  Fix the seed for reproducible comparisons.
5. Bootstrap does NOT correct for multiple comparisons.  Comparing many
   metrics increases the probability of a false positive.

References
----------
Efron, B. and Hastie, T. (2016). Computer Age Statistical Inference.
Dror et al. (2018). "The Hitchhiker's Guide to Testing Statistical Significance
  in Natural Language Processing." ACL 2018.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Minimum sample size before a warning is issued
_MIN_SAMPLES_WARN = 10


def paired_scores(
    baseline_scores: dict[str, float | None],
    candidate_scores: dict[str, float | None],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Extract aligned paired score arrays from two case-keyed dicts.

    Parameters
    ----------
    baseline_scores:
        Mapping from case_id to score for the baseline experiment.
    candidate_scores:
        Mapping from case_id to score for the candidate experiment.

    Returns
    -------
    base_arr, cand_arr:
        Paired arrays (same length, same order) of scores for cases that
        appear in both experiments with non-None values.
    missing_base, missing_cand:
        Case IDs present in one experiment but not the other.
    failed:
        Case IDs present in both but with None scores on either side.
    """
    all_ids = set(baseline_scores.keys()) | set(candidate_scores.keys())
    missing_base = [i for i in all_ids if i not in baseline_scores]
    missing_cand = [i for i in all_ids if i not in candidate_scores]
    common = sorted(all_ids - set(missing_base) - set(missing_cand))

    base_vals = []
    cand_vals = []
    failed = []

    for cid in common:
        b = baseline_scores[cid]
        c = candidate_scores[cid]
        if b is None or c is None:
            failed.append(cid)
        else:
            base_vals.append(b)
            cand_vals.append(c)

    return (
        np.array(base_vals, dtype=float),
        np.array(cand_vals, dtype=float),
        missing_base,
        missing_cand,
        failed,
    )


def bootstrap_mean_diff_ci(
    base_arr: np.ndarray,
    cand_arr: np.ndarray,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a paired bootstrap CI for the mean difference (candidate − baseline).

    Parameters
    ----------
    base_arr, cand_arr:
        Paired score arrays of equal length.
    n_resamples:
        Number of bootstrap resamples.
    confidence_level:
        Confidence level in (0, 1).  Typical: 0.95.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    observed_diff, ci_lower, ci_upper
        The observed mean difference and the percentile bootstrap interval.

    Notes
    -----
    Uses the percentile method (not BCa) for simplicity and transparency.
    BCa may be more accurate for strongly skewed distributions.
    """
    if len(base_arr) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = len(base_arr)
    diffs = cand_arr - base_arr
    observed_diff = float(np.mean(diffs))

    if len(base_arr) < _MIN_SAMPLES_WARN:
        logger.warning(
            "Bootstrap CI computed with only %d paired samples. "
            "Interpret with caution.",
            n,
        )

    # Resample differences (paired bootstrap)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return observed_diff, ci_lower, ci_upper


def safe_relative_change(baseline_mean: float, absolute_change: float) -> float | None:
    """Compute percentage relative change, returning None for zero baselines.

    Percentage relative change is undefined (and potentially misleading) when
    the baseline mean is zero.
    """
    if baseline_mean == 0.0:
        return None
    return absolute_change / abs(baseline_mean) * 100.0
