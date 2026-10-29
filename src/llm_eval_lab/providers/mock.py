"""
Offline mock provider for deterministic testing.

Design rules
------------
* Works without network access.
* Uses fixture-based deterministic responses keyed on the prompt text.
* Never pretends to be a real language model.
* All responses are explicitly marked with ``is_simulated = True``.
* Supports configurable simulated latency and controlled failure injection.
"""

from __future__ import annotations

import hashlib
import time

from llm_eval_lab.providers.base import BaseProvider
from llm_eval_lab.schemas import ModelResponse, ProviderConfig, TokenUsage

# ---------------------------------------------------------------------------
# Fixture bank: deterministic responses keyed on topic keywords.
# These are pre-canned illustrative answers, NOT real model outputs.
# ---------------------------------------------------------------------------
_FIXTURES: dict[str, str] = {
    "password": (
        "To reset your password, visit the sign-in page and click 'Forgot password'. "
        "Follow the email instructions to create a new password."
    ),
    "capital of france": "The capital of France is Paris.",
    "capital of germany": "The capital of Germany is Berlin.",
    "capital of japan": "The capital of Japan is Tokyo.",
    "invoice": '{"invoice_number": "INV-001", "amount": 100.00, "currency": "USD"}',
    "json": '{"status": "ok", "value": 42}',
    "python": "Python is a high-level, interpreted programming language known for its readability.",
    "machine learning": (
        "Machine learning is a subset of artificial intelligence that enables systems to "
        "learn from data and improve over time without being explicitly programmed."
    ),
    "hallucination": (
        "I do not have reliable information about that topic. "
        "Please consult a verified source."
    ),
    "unsupported": (
        "I'm sorry, I cannot assist with that request."
    ),
    "format": "**Title**\n\n- Item 1\n- Item 2\n- Item 3",
    "difficult": (
        "This is a nuanced question. Multiple perspectives exist, and the answer "
        "depends on context. A definitive answer is not possible without more information."
    ),
    "error": None,  # Sentinel: triggers a simulated error
}

_DEFAULT_RESPONSE = (
    "[SIMULATED] This is a deterministic mock response. "
    "No actual language model was queried."
)


def _find_fixture(prompt: str) -> str | None:
    """Return a fixture string matched on keyword, or None for default."""
    lower = prompt.lower()
    # Exact-error trigger
    if "simulate_error" in lower:
        return "ERROR"
    for keyword, response in _FIXTURES.items():
        if keyword in lower:
            return response  # may be None (simulate error for "error" keyword)
    return _DEFAULT_RESPONSE


def _stable_token_count(text: str) -> int:
    """Approximate token count (words / 0.75), deterministic."""
    return max(1, int(len(text.split()) / 0.75))


class MockProvider(BaseProvider):
    """Deterministic offline provider for tests and offline evaluation runs.

    All outputs are clearly labelled as simulated.  The provider never
    contacts external services.
    """

    PROVIDER_NAME = "mock"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._fail_rate: float = 0.0  # injectable for tests

    def set_fail_rate(self, rate: float) -> None:
        """Inject a failure rate (0–1) for reliability testing."""
        self._fail_rate = max(0.0, min(1.0, rate))

    def complete(self, prompt: str, system_prompt: str = "") -> ModelResponse:  # noqa: ARG002
        start = time.perf_counter()

        # Simulated latency
        if self.config.simulated_latency_ms > 0:
            time.sleep(self.config.simulated_latency_ms / 1000.0)

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Deterministic failure based on prompt hash
        if self._fail_rate > 0:
            h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            if (h % 1000) / 1000.0 < self._fail_rate:
                return ModelResponse(
                    text="",
                    latency_ms=latency_ms,
                    provider=self.PROVIDER_NAME,
                    model="mock-v1",
                    error="[SIMULATED] Injected provider failure.",
                    is_simulated=True,
                )

        fixture = _find_fixture(prompt)

        # Simulate a provider error
        if fixture == "ERROR" or fixture is None:
            return ModelResponse(
                text="",
                latency_ms=latency_ms,
                provider=self.PROVIDER_NAME,
                model="mock-v1",
                error="[SIMULATED] Provider returned an error for this input.",
                is_simulated=True,
            )

        prompt_tokens = _stable_token_count(prompt)
        completion_tokens = _stable_token_count(fixture)

        return ModelResponse(
            text=fixture,
            latency_ms=latency_ms,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            provider=self.PROVIDER_NAME,
            model="mock-v1",
            is_simulated=True,
        )
