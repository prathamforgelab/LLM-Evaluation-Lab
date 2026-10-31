"""
OpenAI-compatible chat-completions provider.

Security notes
--------------
* The API key is read from the environment at runtime via ``settings``.
* The key is NEVER logged, stored in the database, or returned in any response.
* Authorization headers are redacted in all log output.
* Timeouts and retries are applied.
* Rate-limit responses (HTTP 429) trigger exponential back-off.

This provider is an optional dependency.  It requires network access and a
valid API key to function.  Core evaluation tests should use the mock provider.
"""

from __future__ import annotations

import logging
import time

import httpx

from llm_eval_lab.config import settings
from llm_eval_lab.providers.base import BaseProvider
from llm_eval_lab.schemas import ModelResponse, ProviderConfig, TokenUsage

logger = logging.getLogger(__name__)

_REDACTED = "***REDACTED***"


def _redact_headers(headers: dict) -> dict:
    """Return headers with Authorization value replaced."""
    return {k: (_REDACTED if k.lower() == "authorization" else v) for k, v in headers.items()}


class OpenAICompatibleProvider(BaseProvider):
    """Provider adapter for any OpenAI chat-completions compatible endpoint.

    Supports OpenAI, Azure OpenAI (with appropriate base_url), and local
    servers such as Ollama or LM Studio.

    API keys are sourced from the environment and never stored.
    """

    PROVIDER_NAME = "openai_compatible"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or settings.openai_base_url
        self._model = config.model or settings.openai_model
        # Retrieve key from env — not from config to avoid accidental storage
        self._api_key = settings.openai_api_key

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(self, prompt: str, system_prompt: str) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self._model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def complete(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        url = self._base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(prompt, system_prompt)
        headers = self._build_headers()

        last_error: str = ""
        for attempt in range(self.config.retries + 1):
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                latency_ms = (time.perf_counter() - start) * 1000.0

                if resp.status_code == 429:
                    wait = 2**attempt
                    logger.warning("Rate limited (429). Waiting %ds before retry.", wait)
                    time.sleep(wait)
                    last_error = "Rate limit exceeded."
                    continue

                resp.raise_for_status()
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                usage_raw = data.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_raw.get("prompt_tokens"),
                    completion_tokens=usage_raw.get("completion_tokens"),
                    total_tokens=usage_raw.get("total_tokens"),
                )
                return ModelResponse(
                    text=text,
                    latency_ms=latency_ms,
                    token_usage=usage,
                    provider=self.PROVIDER_NAME,
                    model=self._model,
                    is_simulated=False,
                )

            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start) * 1000.0
                last_error = f"Request timed out after {self.config.timeout}s."
                logger.warning("Timeout on attempt %d/%d.", attempt + 1, self.config.retries + 1)
            except httpx.HTTPStatusError as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                # Never log response bodies — they may contain sensitive data
                last_error = f"HTTP {exc.response.status_code} error."
                logger.error("HTTP error %d from provider.", exc.response.status_code)
                break
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                last_error = f"Unexpected error: {type(exc).__name__}."
                logger.exception("Unexpected provider error.")
                break

        return ModelResponse(
            text="",
            latency_ms=latency_ms,
            provider=self.PROVIDER_NAME,
            model=self._model,
            error=last_error,
            is_simulated=False,
        )
