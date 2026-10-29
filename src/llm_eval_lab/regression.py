"""
Regression testing engine.

Decision classification
-----------------------
Given a confidence interval [lo, hi] for (candidate_mean − baseline_mean):

IMPROVED:
    The entire CI lies above the practical significance threshold
    (for higher-is-better metrics) or below −threshold (for lower-is-better).

REGRESSED:
    The entire CI lies below −threshold (higher-is-better) or above
    +threshold (lower-is-better).

INCONCLUSIVE:
    The CI crosses zero or the practical threshold.

INSUFFICIENT_DATA:
    Fewer paired samples than the configured minimum, or any critical
    data-quality problem.

Design notes
------------
* The exact decision rules are explicit and configurable per metric.
* A CI that does not cross zero does not automatically imply practical
  significance.  The caller must configure practical_significance to require
  a minimum meaningful effect.
* We never convert an INCONCLUSIVE result to IMPROVED or REGRESSED based
  on the point estimate alone.
* Multiple comparison correction is NOT applied automatically.  Users
  comparing many metrics should be aware of the increased false-positive rate.
"""

from __future__ import annotations

import logging

from llm_eval_lab.schemas import (
    CaseResult,
    ComparisonReport,
    ConfidenceInterval,
    Experiment,
    MetricComparison,
    MetricDirection,
    RegressionConfig,
    RegressionDecision,
)
from llm_eval_lab.statistics import (
    bootstrap_mean_diff_ci,
    paired_scores,
    safe_relative_change,
)

logger = logging.getLogger(__name__)


def _extract_scores(results: list[CaseResult], metric_name: str) -> dict[str, float | None]:
    """Build a case_id → score mapping for one metric."""
    return {
        r.case_id: next(
            (m.score for m in r.metrics if m.metric_name == metric_name),
            None,
        )
        for r in results
    }


def _metric_direction(results: list[CaseResult], metric_name: str) -> MetricDirection:
    """Return the direction of *metric_name* from the first available result."""
    for r in results:
        for m in r.metrics:
            if m.metric_name == metric_name:
                return m.direction
    return MetricDirection.HIGHER_IS_BETTER


def classify_decision(
    ci_lower: float,
    ci_upper: float,
    practical_significance: float,
    direction: MetricDirection,
    n_paired: int,
    min_samples: int,
) -> tuple[RegressionDecision, list[str]]:
    """Apply the configured decision rules.

    Parameters
    ----------
    ci_lower, ci_upper:
        Bootstrap confidence interval for (candidate − baseline).
        Positive = candidate is higher.
    practical_significance:
        Minimum absolute effect the CI must clear (in both directions) to
        trigger IMPROVED or REGRESSED.
    direction:
        MetricDirection of this metric.
    n_paired:
        Number of paired samples used.
    min_samples:
        Minimum required for any decision other than INSUFFICIENT_DATA.

    Returns
    -------
    (decision, notes)
    """
    notes: list[str] = []

    if n_paired < min_samples:
        notes.append(
            f"Only {n_paired} paired samples available; minimum required is {min_samples}."
        )
        return RegressionDecision.INSUFFICIENT_DATA, notes

    if direction == MetricDirection.LOWER_IS_BETTER:
        # Flip the sign convention: positive CI is bad for lower-is-better
        ci_lower, ci_upper = -ci_upper, -ci_lower
        notes.append(
            "Direction is lower-is-better; CI has been sign-flipped for decision logic."
        )

    threshold = abs(practical_significance)

    if ci_lower > threshold:
        return RegressionDecision.IMPROVED, notes
    if ci_upper < -threshold:
        return RegressionDecision.REGRESSED, notes

    if practical_significance > 0 and ci_lower > 0 and ci_upper <= threshold:
        notes.append(
            f"Positive CI but improvement is smaller than practical threshold "
            f"({practical_significance}). Classified as INCONCLUSIVE."
        )
    elif ci_lower <= 0 <= ci_upper:
        notes.append("CI crosses zero; evidence is insufficient to determine direction.")
    else:
        notes.append("CI does not meet improvement or regression thresholds.")

    return RegressionDecision.INCONCLUSIVE, notes


def compare_experiments(
    baseline: Experiment,
    candidate: Experiment,
    regression_config: RegressionConfig | None = None,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> ComparisonReport:
    """Compare two experiments and return a full statistical comparison report.

    Parameters
    ----------
    baseline, candidate:
        Completed experiment objects.
    regression_config:
        Per-metric decision thresholds.  Uses defaults if None.
    bootstrap_n, bootstrap_seed, confidence_level:
        Statistical parameters.

    Returns
    -------
    ComparisonReport
    """
    if regression_config is None:
        regression_config = RegressionConfig()

    report = ComparisonReport(
        baseline_id=baseline.experiment_id,
        candidate_id=candidate.experiment_id,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    # Collect all metric names from both experiments
    metric_names: set[str] = set()
    for r in baseline.results + candidate.results:
        for m in r.metrics:
            metric_names.add(m.metric_name)

    metric_comparisons: list[MetricComparison] = []

    for metric_name in sorted(metric_names):
        direction = _metric_direction(baseline.results + candidate.results, metric_name)
        threshold_cfg = regression_config.get_threshold(metric_name)

        base_scores = _extract_scores(baseline.results, metric_name)
        cand_scores = _extract_scores(candidate.results, metric_name)

        base_arr, cand_arr, missing_base, missing_cand, failed = paired_scores(
            base_scores, cand_scores
        )

        n_paired = len(base_arr)

        # Aggregate means
        base_mean = float(base_arr.mean()) if n_paired > 0 else None
        cand_mean = float(cand_arr.mean()) if n_paired > 0 else None

        if n_paired == 0:
            mc = MetricComparison(
                metric_name=metric_name,
                direction=direction,
                baseline_mean=base_mean,
                candidate_mean=cand_mean,
                n_paired=0,
                n_missing_baseline=len(missing_base),
                n_missing_candidate=len(missing_cand),
                n_failed=len(failed),
                decision=RegressionDecision.INSUFFICIENT_DATA,
                notes=["No paired samples available."],
            )
            metric_comparisons.append(mc)
            continue

        # Bootstrap CI
        obs_diff, ci_lo, ci_hi = bootstrap_mean_diff_ci(
            base_arr,
            cand_arr,
            n_resamples=bootstrap_n,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )

        ci = ConfidenceInterval(
            lower=round(ci_lo, 6),
            upper=round(ci_hi, 6),
            level=confidence_level,
            method="paired_bootstrap_percentile",
        )

        rel_change = safe_relative_change(base_mean, obs_diff) if base_mean is not None else None

        decision, notes = classify_decision(
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            practical_significance=threshold_cfg.practical_significance,
            direction=direction,
            n_paired=n_paired,
            min_samples=threshold_cfg.min_samples,
        )

        mc = MetricComparison(
            metric_name=metric_name,
            direction=direction,
            baseline_mean=round(base_mean, 6) if base_mean is not None else None,
            candidate_mean=round(cand_mean, 6) if cand_mean is not None else None,
            absolute_change=round(obs_diff, 6),
            relative_change_pct=round(rel_change, 3) if rel_change is not None else None,
            ci=ci,
            decision=decision,
            n_paired=n_paired,
            n_missing_baseline=len(missing_base),
            n_missing_candidate=len(missing_cand),
            n_failed=len(failed),
            notes=notes,
        )
        metric_comparisons.append(mc)

    report.metric_comparisons = metric_comparisons

    # Overall decision: any regression → REGRESSED; all insufficient → INSUFFICIENT_DATA
    decisions = [mc.decision for mc in metric_comparisons]
    quality_metrics = [
        mc for mc in metric_comparisons
        if mc.metric_name not in ("latency_ms",)  # latency is informational
    ]
    q_decisions = [mc.decision for mc in quality_metrics]

    if not q_decisions:
        report.overall_decision = RegressionDecision.INSUFFICIENT_DATA
    elif all(d == RegressionDecision.INSUFFICIENT_DATA for d in q_decisions):
        report.overall_decision = RegressionDecision.INSUFFICIENT_DATA
    elif any(d == RegressionDecision.REGRESSED for d in q_decisions):
        report.overall_decision = RegressionDecision.REGRESSED
    elif all(d == RegressionDecision.IMPROVED for d in q_decisions):
        report.overall_decision = RegressionDecision.IMPROVED
    else:
        report.overall_decision = RegressionDecision.INCONCLUSIVE

    # Case-level change tracking (using first quality metric with paired data)
    if quality_metrics:
        primary = quality_metrics[0].metric_name
        base_scores_primary = _extract_scores(baseline.results, primary)
        cand_scores_primary = _extract_scores(candidate.results, primary)

        for cid in set(base_scores_primary) & set(cand_scores_primary):
            bv = base_scores_primary.get(cid)
            cv = cand_scores_primary.get(cid)
            if bv is None or cv is None:
                continue
            if cv > bv:
                report.cases_improved.append(cid)
            elif cv < bv:
                report.cases_regressed.append(cid)

    report.cases_missing_baseline = [
        mc.metric_name for mc in metric_comparisons if mc.n_missing_baseline > 0
    ]
    report.cases_missing_candidate = [
        mc.metric_name for mc in metric_comparisons if mc.n_missing_candidate > 0
    ]

    logger.info(
        "Comparison %s → %s: overall decision = %s",
        baseline.experiment_id[:8],
        candidate.experiment_id[:8],
        report.overall_decision.value,
    )

    return report
