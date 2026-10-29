"""
Unit tests for JSON validity and schema evaluators.
"""

from __future__ import annotations

import pytest

from llm_eval_lab.evaluators.schema_validation import JSONSchemaEvaluator, JSONValidityEvaluator
from llm_eval_lab.schemas import ModelResponse, TestCase


def _case(schema: dict | None = None) -> TestCase:
    return TestCase(id="t1", input="test", expected_json_schema=schema)


def _resp(text: str, error: str | None = None) -> ModelResponse:
    return ModelResponse(text=text, latency_ms=10.0, error=error)


class TestJSONValidity:
    ev = JSONValidityEvaluator()

    def test_valid_object(self):
        r = self.ev.evaluate(_case(), _resp('{"key": "value"}'))
        assert r.score == 1.0
        assert r.passed is True

    def test_valid_array(self):
        r = self.ev.evaluate(_case(), _resp('[1, 2, 3]'))
        assert r.score == 1.0

    def test_valid_string(self):
        r = self.ev.evaluate(_case(), _resp('"hello"'))
        assert r.score == 1.0

    def test_invalid_json(self):
        r = self.ev.evaluate(_case(), _resp("not json"))
        assert r.score == 0.0
        assert r.passed is False
        assert "json_error" in r.details

    def test_provider_error(self):
        r = self.ev.evaluate(_case(), _resp("", error="timeout"))
        assert r.score is None

    def test_nested_valid_json(self):
        r = self.ev.evaluate(_case(), _resp('{"a": {"b": [1, 2, {"c": true}]}}'))
        assert r.score == 1.0


class TestJSONSchema:
    ev = JSONSchemaEvaluator()
    _schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    def test_valid_against_schema(self):
        r = self.ev.evaluate(_case(self._schema), _resp('{"name": "Alice", "age": 30}'))
        assert r.score == 1.0
        assert r.passed is True

    def test_valid_json_but_violates_schema(self):
        r = self.ev.evaluate(_case(self._schema), _resp('{"name": "Alice", "age": "thirty"}'))
        assert r.score == 0.0
        assert r.details["json_valid"] is True
        assert r.details["schema_valid"] is False

    def test_invalid_json_not_valid(self):
        r = self.ev.evaluate(_case(self._schema), _resp("not json at all"))
        assert r.score == 0.0
        assert r.details["json_valid"] is False

    def test_missing_required_field(self):
        r = self.ev.evaluate(_case(self._schema), _resp('{"name": "Alice"}'))
        assert r.score == 0.0

    def test_no_schema_configured(self):
        r = self.ev.evaluate(_case(None), _resp('{"key": "value"}'))
        assert r.score is None
        assert r.error is not None

    def test_provider_error(self):
        r = self.ev.evaluate(_case(self._schema), _resp("", error="error"))
        assert r.score is None
