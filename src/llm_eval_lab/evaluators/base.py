"""
Abstract base class for evaluators.

An evaluator takes a :class:`TestCase` and the model's :class:`ModelResponse`
and returns a :class:`MetricResult`.

Design notes
------------
* Evaluators must be stateless across cases.  They may hold read-only
  configuration set in ``__init__``.
* Each evaluator declares a ``metric_name`` class attribute.
* Errors inside ``evaluate`` are caught by the caller (runner) and stored as
  a failed MetricResult, not raised as exceptions, so one failing metric does
  not abort the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_eval_lab.schemas import MetricDirection, MetricResult, ModelResponse, TestCase


class BaseEvaluator(ABC):
    """Abstract evaluator interface."""

    metric_name: str = "base"
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER

    @abstractmethod
    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        """Compute a metric for one test case.

        Parameters
        ----------
        case:
            The test case containing input, reference, and expectations.
        response:
            The model's response including text and metadata.

        Returns
        -------
        MetricResult
            Always returns a result object.  Set ``error`` when the metric
            cannot be computed rather than raising.
        """

    def _unavailable(self, reason: str) -> MetricResult:
        """Return a result indicating the metric is not computable."""
        return MetricResult(
            metric_name=self.metric_name,
            score=None,
            passed=None,
            error=reason,
            direction=self.direction,
        )
