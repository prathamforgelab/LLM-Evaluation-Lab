"""
Unit tests for the mock provider and provider factory.
"""

from __future__ import annotations

import pytest

from llm_eval_lab.providers.base import get_provider
from llm_eval_lab.providers.mock import MockProvider
from llm_eval_lab.schemas import ProviderConfig


def _mock(simulated_latency_ms: float = 0.0) -> MockProvider:
    return MockProvider(ProviderConfig(name="mock", simulated_latency_ms=simulated_latency_ms))


class TestMockProvider:
    def test_returns_response(self):
        p = _mock()
        resp = p.complete("What is the capital of France?")
        assert resp.text != ""
        assert resp.error is None
        assert resp.is_simulated is True

    def test_fixture_matched_by_keyword(self):
        p = _mock()
        resp = p.complete("How do I reset my password?")
        assert "password" in resp.text.lower()

    def test_token_usage_populated(self):
        p = _mock()
        resp = p.complete("What is Python?")
        assert resp.token_usage is not None
        assert resp.token_usage.prompt_tokens > 0
        assert resp.token_usage.completion_tokens > 0

    def test_error_trigger(self):
        p = _mock()
        resp = p.complete("simulate_error this case")
        assert resp.error is not None
        assert resp.text == ""
        assert resp.is_simulated is True

    def test_deterministic_outputs(self):
        p = _mock()
        r1 = p.complete("What is the capital of France?")
        r2 = p.complete("What is the capital of France?")
        assert r1.text == r2.text

    def test_injected_fail_rate(self):
        p = _mock()
        p.set_fail_rate(1.0)  # always fail
        resp = p.complete("any prompt")
        assert resp.error is not None

    def test_zero_fail_rate_never_fails(self):
        p = _mock()
        p.set_fail_rate(0.0)
        for _ in range(20):
            resp = p.complete("What is Python?")
            assert resp.error is None

    def test_provider_name(self):
        p = _mock()
        assert p.name == "mock"


class TestProviderFactory:
    def test_creates_mock(self):
        cfg = ProviderConfig(name="mock")
        p = get_provider(cfg)
        assert isinstance(p, MockProvider)

    def test_unknown_provider_raises(self):
        cfg = ProviderConfig(name="unknown_provider")
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(cfg)
