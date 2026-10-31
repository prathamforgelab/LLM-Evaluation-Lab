"""
Aggregate metric calculations.

This module computes summary statistics across all CaseResults in an
experiment.  Each function operates on lists of numeric scores extracted from
the per-case MetricResult objects.

Execution metrics (latency, token counts, cost) are computed separately from
quality metrics to keep the distinction explicit.
"""

from __future__ import annotations

import numpy as np

from llm_eval_lab.schemas import (
    CaseResult,
    ExperimentConfig,
    ExperimentSummary,
    TokenUsage,
)


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile using linear interpolation."""
    return float(np.percentile(values, pct))


def compute_latency_stats(results: list[CaseResult]) -> dict[str, float | None]:
    """Compute latency statistics across all results.

    Returns a dict with: mean, median, p90, p95.  p90/p95 are None when
    fewer than 10 / 20 samples are available respectively (guidance only —
    the caller may override).
    """
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    if not latencies:
        return {"mean": None, "median": None, "p90": None, "p95": None}

    n = len(latencies)
    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p90": _percentile(latencies, 90) if n >= 10 else None,
        "p95": _percentile(latencies, 95) if n >= 20 else None,
    }


def aggregate_metric_scores(results: list[CaseResult], metric_name: str) -> float | None:
    """Return the mean score for *metric_name* across all results.

    Returns None when no valid scores exist.
    """
    scores = [
        m.score
        for r in results
        for m in r.metrics
        if m.metric_name == metric_name and m.score is not None
    ]
    return float(np.mean(scores)) if scores else None


def sum_token_usage(results: list[CaseResult]) -> TokenUsage | None:
    """Sum token usage across all results.  Returns None if none reported."""
    prompt_total = 0
    completion_total = 0
    has_any = False

    for r in results:
        if r.token_usage is not None:
            has_any = True
            prompt_total += r.token_usage.prompt_tokens or 0
            completion_total += r.token_usage.completion_tokens or 0

    if not has_any:
        return None
    return TokenUsage(
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=prompt_total + completion_total,
    )


def estimate_cost(
    usage: TokenUsage | None,
    cost_per_prompt_token: float | None,
    cost_per_completion_token: float | None,
) -> float | None:
    """Estimate USD cost given token counts and per-token prices.

    Returns None if pricing inputs are not configured or token counts are
    unavailable.  Never invents a cost estimate.
    """
    if usage is None:
        return None
    if cost_per_prompt_token is None or cost_per_completion_token is None:
        return None
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    return prompt_tokens * cost_per_prompt_token + completion_tokens * cost_per_completion_token


def build_summary(
    results: list[CaseResult],
    config: ExperimentConfig,
) -> ExperimentSummary:
    """Build an :class:`ExperimentSummary` from completed results."""
    total = len(results)
    failed = sum(1 for r in results if r.error is not None)
    successful = total - failed
    success_rate = successful / total if total > 0 else 0.0

    latency = compute_latency_stats(results)
    usage = sum_token_usage(results)
    cost = estimate_cost(
        usage,
        config.cost_per_prompt_token,
        config.cost_per_completion_token,
    )

    # Collect all metric names
    metric_names: set[str] = set()
    for r in results:
        for m in r.metrics:
            if m.score is not None:
                metric_names.add(m.metric_name)

    aggregate_metrics = {
        name: agg
        for name in sorted(metric_names)
        if (agg := aggregate_metric_scores(results, name)) is not None
    }

    return ExperimentSummary(
        total_cases=total,
        successful_cases=successful,
        failed_cases=failed,
        success_rate=round(success_rate, 6),
        mean_latency_ms=latency["mean"],
        median_latency_ms=latency["median"],
        p90_latency_ms=latency["p90"],
        p95_latency_ms=latency["p95"],
        total_prompt_tokens=usage.prompt_tokens if usage else None,
        total_completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        estimated_cost_usd=cost,
        aggregate_metrics=aggregate_metrics,
    )
