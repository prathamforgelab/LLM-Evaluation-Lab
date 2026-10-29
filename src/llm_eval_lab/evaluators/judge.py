"""
LLM-as-judge evaluator.

Limitations and caveats
-----------------------
* Judge outputs are validated against a defined schema before being stored.
* The judge may be biased, inconsistent, or wrong.  Results should be
  interpreted alongside other metrics.
* Judge provider, model, prompt version, and errors are all stored for
  audit purposes.
* Judge API keys are handled with the same security rules as the main provider.
* Costs are estimated only when valid pricing is configured; otherwise
  they are reported as unavailable.
"""

from __future__ import annotations

import json
import logging
import re

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.providers.base import get_provider
from llm_eval_lab.schemas import MetricResult, ModelResponse, ProviderConfig, TestCase

logger = logging.getLogger(__name__)

_DEFAULT_RUBRIC = """\
You are an impartial evaluator. Score the following model response on a scale
from 0 to 10, where 0 is completely wrong or unhelpful and 10 is perfect.

Question: {input}
{reference_block}
Model Response: {output}

Respond with a JSON object containing exactly two keys:
- "score": an integer from 0 to 10
- "reasoning": a brief explanation of the score

Example: {{"score": 7, "reasoning": "The answer is mostly correct but missing details."}}
"""


class JudgeEvaluator(BaseEvaluator):
    """Model-based rubric evaluator.

    Parameters
    ----------
    judge_config:
        Provider configuration for the judge model.
    rubric:
        Jinja2-style template string for the judge prompt.  Defaults to
        a built-in 0–10 rubric.

    Stored fields
    -------------
    judge_provider, judge_model, judge_prompt_hash, judge_response,
    parsed_score, judge_error, judge_latency_ms.
    """

    metric_name = "judge_score"

    def __init__(
        self,
        judge_config: ProviderConfig,
        rubric: str = "",
        score_min: float = 0.0,
        score_max: float = 10.0,
    ) -> None:
        self._judge_config = judge_config
        self._rubric = rubric or _DEFAULT_RUBRIC
        self._score_min = score_min
        self._score_max = score_max

    def _render_prompt(self, case: TestCase, output: str) -> str:
        ref_block = ""
        if case.reference:
            ref_block = f"Reference Answer: {case.reference}\n"
        return self._rubric.format(
            input=case.input,
            reference_block=ref_block,
            output=output,
        )

    def _parse_score(self, text: str) -> tuple[float | None, str]:
        """Parse judge response, return (score, error_message)."""
        # Try JSON first
        try:
            data = json.loads(text)
            score = data.get("score")
            if score is not None:
                return float(score), ""
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to regex search for a number
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if match:
            return float(match.group(1)), "Score extracted by regex fallback."

        return None, "Could not parse a numeric score from the judge response."

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed; nothing to judge.")

        try:
            provider = get_provider(self._judge_config)
            prompt = self._render_prompt(case, response.text)
            judge_response = provider.complete(prompt)
        except Exception as exc:
            logger.exception("Judge provider error.")
            return MetricResult(
                metric_name=self.metric_name,
                score=None,
                error=f"Judge provider error: {type(exc).__name__}: {exc}",
                details={"judge_provider": self._judge_config.name},
                direction=self.direction,
            )

        if judge_response.error:
            return MetricResult(
                metric_name=self.metric_name,
                score=None,
                error=f"Judge request failed: {judge_response.error}",
                details={
                    "judge_provider": self._judge_config.name,
                    "judge_model": self._judge_config.model,
                    "judge_latency_ms": judge_response.latency_ms,
                },
                direction=self.direction,
            )

        raw_score, parse_error = self._parse_score(judge_response.text)

        if raw_score is not None:
            # Clamp to configured range
            raw_score = max(self._score_min, min(self._score_max, raw_score))
            # Normalize to [0, 1]
            normalized = (raw_score - self._score_min) / max(
                self._score_max - self._score_min, 1e-9
            )
        else:
            normalized = None

        return MetricResult(
            metric_name=self.metric_name,
            score=round(normalized, 6) if normalized is not None else None,
            passed=(normalized is not None and normalized >= 0.5),
            details={
                "judge_provider": self._judge_config.name,
                "judge_model": self._judge_config.model,
                "judge_response": judge_response.text[:500],  # truncate for storage
                "raw_score": raw_score,
                "score_range": [self._score_min, self._score_max],
                "judge_latency_ms": judge_response.latency_ms,
                "parse_note": parse_error or None,
                "caveats": (
                    "Judge scores are subjective estimates, not ground truth. "
                    "The judge model may be biased or inconsistent."
                ),
            },
            error=parse_error if raw_score is None else None,
            direction=self.direction,
        )
