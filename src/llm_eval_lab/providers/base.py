"""
Abstract base class for model providers.

All providers must implement :meth:`BaseProvider.complete` which accepts a
rendered prompt string and returns a :class:`ModelResponse`.

Provider errors should be caught internally and returned as a ``ModelResponse``
with ``error`` set, so the runner can record partial results without crashing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_eval_lab.schemas import ModelResponse, ProviderConfig


class BaseProvider(ABC):
    """Abstract model provider interface."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        """Send a prompt and return the model response.

        Parameters
        ----------
        prompt:
            The rendered user prompt.
        system_prompt:
            Optional system-level instruction.

        Returns
        -------
        ModelResponse
            Always returns a response object.  If the request fails, the
            ``error`` field is populated and ``text`` is empty.
        """

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model


def get_provider(config: ProviderConfig) -> BaseProvider:
    """Factory: return the appropriate provider for *config*."""
    if config.name == "mock":
        from llm_eval_lab.providers.mock import MockProvider

        return MockProvider(config)
    if config.name == "openai_compatible":
        from llm_eval_lab.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(config)
    raise ValueError(f"Unknown provider '{config.name}'. Supported: 'mock', 'openai_compatible'.")
