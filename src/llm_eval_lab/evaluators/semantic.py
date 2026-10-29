"""
Semantic similarity evaluator (optional).

This evaluator requires the ``sentence-transformers`` optional dependency.
It will not be activated during default installation or CI runs.

Important limitations
---------------------
* Semantic similarity is NOT equivalent to factual correctness.
* A response can be semantically similar to the reference while being
  factually wrong, or dissimilar while being correct.
* Scores depend on the embedding model chosen.
* The embedding model is downloaded on first use.  Tests must not trigger
  this download without explicit opt-in.

The evaluator is disabled when sentence-transformers is not installed.
It reports the metric as unavailable rather than raising an ImportError.
"""

from __future__ import annotations

import logging

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.schemas import MetricResult, ModelResponse, TestCase

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Cosine similarity between prediction and reference embeddings.

    Parameters
    ----------
    model_name:
        Sentence-Transformers model identifier.  Defaults to
        ``all-MiniLM-L6-v2`` which is small (80 MB) and runs on CPU.
    threshold:
        Minimum cosine similarity to count as ``passed``.

    Notes
    -----
    The embedding model is loaded lazily on the first call to ``evaluate``.
    Set ``LLM_EVAL_SEMANTIC_MODEL`` in the environment to override the default
    model identifier.
    """

    metric_name = "semantic_similarity"

    def __init__(self, model_name: str = _DEFAULT_MODEL, threshold: float = 0.8) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]

                logger.info("Loading embedding model '%s'.", self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                return None
        return self._model

    def evaluate(self, case: TestCase, response: ModelResponse) -> MetricResult:
        if response.error:
            return self._unavailable("Model request failed.")
        if case.reference is None:
            return self._unavailable("No reference answer available.")

        model = self._get_model()
        if model is None:
            return self._unavailable(
                "sentence-transformers is not installed. "
                "Install with: pip install 'llm-eval-lab[semantic]'"
            )

        try:
            import numpy as np

            embeddings = model.encode([response.text, case.reference], normalize_embeddings=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            similarity = max(-1.0, min(1.0, similarity))  # clamp numerical noise
            passed = similarity >= self._threshold

            return MetricResult(
                metric_name=self.metric_name,
                score=round(similarity, 6),
                passed=passed,
                details={
                    "model": self._model_name,
                    "threshold": self._threshold,
                    "similarity": round(similarity, 6),
                    "note": (
                        "Semantic similarity measures embedding distance, "
                        "not factual correctness."
                    ),
                },
                direction=self.direction,
            )
        except Exception as exc:
            logger.exception("SemanticSimilarityEvaluator error.")
            return MetricResult(
                metric_name=self.metric_name,
                score=None,
                error=f"Embedding error: {type(exc).__name__}: {exc}",
                direction=self.direction,
            )
