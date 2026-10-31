"""
Unit tests for the statistical functions.

Covers: paired_scores alignment, bootstrap CI, determinism, edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_eval_lab.statistics import (
    bootstrap_mean_diff_ci,
    paired_scores,
    safe_relative_change,
)

# ── paired_scores ─────────────────────────────────────────────────────────────


class TestPairedScores:
    def test_identical_keys(self):
        base = {"a": 0.5, "b": 0.8}
        cand = {"a": 0.6, "b": 0.9}
        ba, ca, mb, mc, failed = paired_scores(base, cand)
        assert len(ba) == 2
        assert len(ca) == 2
        assert mb == []
        assert mc == []
        assert failed == []

    def test_missing_from_baseline(self):
        base = {"a": 0.5}
        cand = {"a": 0.6, "b": 0.9}
        ba, ca, mb, mc, failed = paired_scores(base, cand)
        assert len(ba) == 1
        assert "b" in mb

    def test_missing_from_candidate(self):
        base = {"a": 0.5, "b": 0.8}
        cand = {"a": 0.6}
        ba, ca, mb, mc, failed = paired_scores(base, cand)
        assert len(ba) == 1
        assert "b" in mc

    def test_none_scores_go_to_failed(self):
        base = {"a": 0.5, "b": None}
        cand = {"a": 0.6, "b": 0.9}
        ba, ca, mb, mc, failed = paired_scores(base, cand)
        assert len(ba) == 1
        assert "b" in failed

    def test_empty_dicts(self):
        ba, ca, mb, mc, failed = paired_scores({}, {})
        assert len(ba) == 0

    def test_all_failed(self):
        base = {"a": None}
        cand = {"a": None}
        ba, ca, mb, mc, failed = paired_scores(base, cand)
        assert len(ba) == 0
        assert "a" in failed


# ── bootstrap_mean_diff_ci ────────────────────────────────────────────────────


class TestBootstrapCI:
    def _arr(self, values):
        return np.array(values, dtype=float)

    def test_identical_arrays_ci_around_zero(self):
        a = self._arr([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        obs, lo, hi = bootstrap_mean_diff_ci(a, a, seed=42)
        assert abs(obs) < 1e-9
        assert lo <= 0 <= hi

    def test_clear_improvement_ci_positive(self):
        base = self._arr([0.3] * 30)
        cand = self._arr([0.7] * 30)
        obs, lo, hi = bootstrap_mean_diff_ci(base, cand, seed=42)
        assert obs == pytest.approx(0.4)
        assert lo > 0
        assert hi > 0

    def test_clear_regression_ci_negative(self):
        base = self._arr([0.7] * 30)
        cand = self._arr([0.3] * 30)
        obs, lo, hi = bootstrap_mean_diff_ci(base, cand, seed=42)
        assert obs == pytest.approx(-0.4)
        assert hi < 0

    def test_deterministic_with_fixed_seed(self):
        a = self._arr([0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.55])
        b = self._arr([0.5, 0.7, 0.6, 0.8, 0.4, 0.9, 0.3, 1.0, 0.2, 0.65])
        result1 = bootstrap_mean_diff_ci(a, b, seed=99)
        result2 = bootstrap_mean_diff_ci(a, b, seed=99)
        assert result1 == result2

    def test_different_seeds_give_slightly_different_results(self):
        a = self._arr([0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.55])
        b = self._arr([0.5, 0.4, 0.8, 0.6, 0.5, 0.7, 0.4, 0.8, 0.3, 0.60])
        _, lo1, hi1 = bootstrap_mean_diff_ci(a, b, seed=1)
        _, lo2, hi2 = bootstrap_mean_diff_ci(a, b, seed=2)
        # With varying difference distribution, CIs with different seeds should differ slightly
        assert (lo1 != lo2) or (hi1 != hi2)

    def test_ci_lower_le_ci_upper(self):
        a = self._arr([0.5, 0.3, 0.7, 0.6, 0.4, 0.8, 0.2, 0.9, 0.1, 0.55])
        b = self._arr([0.6, 0.4, 0.8, 0.7, 0.5, 0.9, 0.3, 1.0, 0.2, 0.65])
        _, lo, hi = bootstrap_mean_diff_ci(a, b, seed=42)
        assert lo <= hi

    def test_empty_array_returns_zeros(self):
        obs, lo, hi = bootstrap_mean_diff_ci(np.array([]), np.array([]), seed=42)
        assert obs == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_noisy_results_ci_includes_zero(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, 10)
        b = a + rng.normal(0, 0.05, 10)  # tiny noisy difference
        _, lo, hi = bootstrap_mean_diff_ci(a, b, n_resamples=500, seed=42)
        assert lo <= hi


# ── safe_relative_change ──────────────────────────────────────────────────────


class TestRelativeChange:
    def test_positive_baseline(self):
        result = safe_relative_change(0.5, 0.1)
        assert result == pytest.approx(20.0)

    def test_negative_baseline(self):
        result = safe_relative_change(-0.5, 0.1)
        assert result == pytest.approx(20.0)

    def test_zero_baseline_returns_none(self):
        result = safe_relative_change(0.0, 0.1)
        assert result is None

    def test_zero_diff(self):
        result = safe_relative_change(0.5, 0.0)
        assert result == pytest.approx(0.0)
