"""
Deterministic string-based evaluators.

All metrics defined here use transparent, deterministic rules.
No embedding models or external APIs are required.

Metric definitions
------------------
exact_match
    Predicted text equals reference text after optional normalization.
    Score: 1.0 (match) or 0.0 (no match).

normalized_exact_match
    Same as exact_match with whitespace normalization and case folding applied.
    Score: 1.0 or 0.0.

token_overlap_f1
    Unigram token F1 between prediction and reference.
    Measures word-level overlap, NOT semantic equivalence.
    Tokens are lower-cased and split on whitespace after stripping punctuation.
    Score: F1 in [0, 1].

keyword_coverage
    Fraction of expected_keywords present in the output (case-insensitive).
    Score: coverage fraction in [0, 1].

latency_ms
    Informational metric recording wall-clock latency per test case.
    Direction: lower_is_better.
"""

from __future__ import annotations

import re
import string

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.schemas import MetricDirection, MetricResult, ModelResponse, TestCase


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().split())


def _tokenize(text: str) -> list[str]:
    """Split on whitespace after removing punctuation."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower().split()


class ExactMatchEvaluator(BaseEvaluator):
    """Exact string match between prediction and reference.

    Normalization is NOT applied.  Use NormalizedExactMatchEvaluator for a
    whitespace-insensitive comparison.
    """

    metric_name = "exact_match"

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if case.reference is None:
            return self._unavailable("No reference answer available for this test case.")
        match = response.text == case.reference
        return MetricResult(
            metric_name=self.metric_name,
            score=1.0 if match else 0.0,
            passed=match,
            direction=self.direction,
        )


class NormalizedExactMatchEvaluator(BaseEvaluator):
    """Exact match after whitespace normalization and case folding.

    Normalization steps applied (in order):
    1. Strip leading/trailing whitespace.
    2. Collapse internal whitespace runs to a single space.
    3. Fold to lowercase.

    These steps are documented so results are reproducible.
    """

    metric_name = "normalized_exact_match"

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if case.reference is None:
            return self._unavailable("No reference answer available.")
        norm_pred = _normalize(response.text)
        norm_ref = _normalize(case.reference)
        match = norm_pred == norm_ref
        return MetricResult(
            metric_name=self.metric_name,
            score=1.0 if match else 0.0,
            passed=match,
            details={"normalized_prediction": norm_pred, "normalized_reference": norm_ref},
            direction=self.direction,
        )


class TokenOverlapF1Evaluator(BaseEvaluator):
    """Unigram token F1 between prediction and reference.

    Algorithm
    ---------
    1. Tokenize both strings: lowercase, strip punctuation, split on whitespace.
    2. Compute precision = |common_tokens| / |prediction_tokens|.
    3. Compute recall = |common_tokens| / |reference_tokens|.
    4. F1 = harmonic mean of precision and recall.

    Important limitation: token overlap does not measure semantic correctness.
    A response with high overlap may still be factually wrong or misleading.
    """

    metric_name = "token_overlap_f1"

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if case.reference is None:
            return self._unavailable("No reference answer available.")

        pred_tokens = _tokenize(response.text)
        ref_tokens = _tokenize(case.reference)

        if not pred_tokens and not ref_tokens:
            return MetricResult(
                metric_name=self.metric_name,
                score=1.0,
                passed=True,
                details={"precision": 1.0, "recall": 1.0, "f1": 1.0},
                direction=self.direction,
            )
        if not pred_tokens or not ref_tokens:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
                direction=self.direction,
            )

        pred_set = _bag_of_tokens(pred_tokens)
        ref_set = _bag_of_tokens(ref_tokens)

        common = {tok: min(pred_set[tok], ref_set[tok]) for tok in pred_set if tok in ref_set}
        n_common = sum(common.values())

        precision = n_common / sum(pred_set.values())
        recall = n_common / sum(ref_set.values())

        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return MetricResult(
            metric_name=self.metric_name,
            score=round(f1, 6),
            passed=f1 > 0,
            details={
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            },
            direction=self.direction,
        )


def _bag_of_tokens(tokens: list[str]) -> dict[str, int]:
    bag: dict[str, int] = {}
    for t in tokens:
        bag[t] = bag.get(t, 0) + 1
    return bag


class KeywordCoverageEvaluator(BaseEvaluator):
    """Fraction of expected keywords found in the model output.

    Keywords are matched case-insensitively as whole-word or substring matches
    depending on ``whole_word`` parameter (default True).

    Score = found_count / total_keywords.  Score is None when no keywords are
    configured.
    """

    metric_name = "keyword_coverage"

    def __init__(self, whole_word: bool = True) -> None:
        self.whole_word = whole_word

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if not case.expected_keywords:
            return self._unavailable("No expected_keywords defined for this test case.")

        output_lower = response.text.lower()
        found = []
        missing = []

        for kw in case.expected_keywords:
            kw_lower = kw.lower()
            if self.whole_word:
                pattern = r"\b" + re.escape(kw_lower) + r"\b"
                present = bool(re.search(pattern, output_lower))
            else:
                present = kw_lower in output_lower

            if present:
                found.append(kw)
            else:
                missing.append(kw)

        total = len(case.expected_keywords)
        score = len(found) / total if total > 0 else 0.0

        return MetricResult(
            metric_name=self.metric_name,
            score=round(score, 6),
            passed=score == 1.0,
            details={"found": found, "missing": missing, "total": total},
            direction=self.direction,
        )


class LatencyEvaluator(BaseEvaluator):
    """Records wall-clock latency as an informational metric.

    This evaluator never fails — it always returns the latency value.
    Direction: lower_is_better.
    """

    metric_name = "latency_ms"
    direction = MetricDirection.LOWER_IS_BETTER

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:  # noqa: ARG002
        return MetricResult(
            metric_name=self.metric_name,
            score=round(response.latency_ms, 3),
            passed=None,
            direction=self.direction,
        )
