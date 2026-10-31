"""
JSON validity and JSON Schema compliance evaluators.

json_validity
    Checks whether the model output is parseable as JSON.
    Score: 1.0 (valid JSON) or 0.0 (invalid JSON).

json_schema_compliance
    Checks whether the model output is valid JSON AND satisfies the JSON Schema
    defined in ``case.expected_json_schema``.

    This evaluator distinguishes between:
    - Output that is not valid JSON (json_valid=False).
    - Output that is valid JSON but violates the schema (schema_valid=False).
    - Output that satisfies both (passed=True).

    Requires the ``jsonschema`` package.  If the package is not installed,
    the metric is reported as unavailable rather than silently failing.
"""

from __future__ import annotations

import json
import logging

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.schemas import MetricResult, ModelResponse, TestCase

logger = logging.getLogger(__name__)


class JSONValidityEvaluator(BaseEvaluator):
    """Check whether the model output is valid JSON."""

    metric_name = "json_validity"

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:  # noqa: ARG002
        if response.error:
            return self._unavailable("Model request failed.")

        try:
            parsed = json.loads(response.text)
            return MetricResult(
                metric_name=self.metric_name,
                score=1.0,
                passed=True,
                details={"parsed_type": type(parsed).__name__},
                direction=self.direction,
            )
        except json.JSONDecodeError as exc:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                details={"json_error": str(exc)},
                direction=self.direction,
            )


class JSONSchemaEvaluator(BaseEvaluator):
    """Validate model output against a JSON Schema.

    The schema must be provided in ``case.expected_json_schema``.
    Requires ``jsonschema>=4.0``.
    """

    metric_name = "json_schema_compliance"

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if not case.expected_json_schema:
            return self._unavailable("No expected_json_schema defined for this test case.")

        # Parse JSON
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as exc:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                details={"json_valid": False, "json_error": str(exc)},
                direction=self.direction,
            )

        # Validate against schema
        try:
            import jsonschema  # optional dependency
        except ImportError:
            return self._unavailable(
                "jsonschema package is not installed. Install it with: pip install jsonschema"
            )

        try:
            jsonschema.validate(instance=parsed, schema=case.expected_json_schema)
            return MetricResult(
                metric_name=self.metric_name,
                score=1.0,
                passed=True,
                details={"json_valid": True, "schema_valid": True},
                direction=self.direction,
            )
        except jsonschema.ValidationError as exc:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                details={
                    "json_valid": True,
                    "schema_valid": False,
                    "schema_error": exc.message,
                    "schema_path": list(exc.absolute_schema_path),
                },
                direction=self.direction,
            )
