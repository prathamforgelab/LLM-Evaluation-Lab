"""
Experiment runner.

The runner executes a full evaluation experiment:
1. Loads the dataset.
2. Renders prompts using the configured template.
3. Calls the model provider for each test case.
4. Applies each configured evaluator.
5. Records per-case results.
6. Computes aggregate metrics.

Design decisions
----------------
* Execution is sequential by default to make error handling and rate-limit
  behaviour predictable.
* A failed provider request records the error but does not abort remaining
  cases.
* A failed evaluator records the error for that metric but does not abort
  remaining evaluators.
* The runner never silently re-uses previous successful results when retrying.
* Progress is reported via a callback so the CLI and API can display it
  differently without coupling.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from jinja2 import Environment, StrictUndefined, UndefinedError

from llm_eval_lab.dataset import DatasetParseError, load_dataset
from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.deterministic import (
    ExactMatchEvaluator,
    KeywordCoverageEvaluator,
    LatencyEvaluator,
    NormalizedExactMatchEvaluator,
    TokenOverlapF1Evaluator,
)
from llm_eval_lab.evaluators.schema_validation import JSONSchemaEvaluator, JSONValidityEvaluator
from llm_eval_lab.metrics import build_summary
from llm_eval_lab.providers.base import get_provider
from llm_eval_lab.schemas import (
    CaseResult,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    MetricResult,
    ModelResponse,
    PromptTemplate,
    TestCase,
)

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=StrictUndefined, autoescape=False)


# ── Prompt rendering ──────────────────────────────────────────────────────────


def render_prompt(template: PromptTemplate, case: TestCase) -> str:
    """Render the user prompt template with test-case variables.

    Raises
    ------
    ValueError
        When a required template variable is missing from the test case.
    """
    try:
        tmpl = _jinja_env.from_string(template.user_template)
        return tmpl.render(
            input=case.input,
            reference=case.reference or "",
            context=case.context or "",
            **case.metadata,
        )
    except UndefinedError as exc:
        raise ValueError(f"Template variable error: {exc}") from exc


def compute_template_hash(template: PromptTemplate) -> str:
    """Stable SHA-256 of (system_prompt + user_template)."""
    raw = template.system_prompt + "\n---\n" + template.user_template
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── Evaluator factory ─────────────────────────────────────────────────────────


def build_evaluators(config: ExperimentConfig) -> list[BaseEvaluator]:
    """Instantiate evaluators according to the experiment configuration."""
    ev_cfg = config.evaluators
    evaluators: list[BaseEvaluator] = []

    if ev_cfg.exact_match:
        evaluators.append(ExactMatchEvaluator())
    if ev_cfg.normalized_exact_match:
        evaluators.append(NormalizedExactMatchEvaluator())
    if ev_cfg.token_overlap_f1:
        evaluators.append(TokenOverlapF1Evaluator())
    if ev_cfg.keyword_coverage:
        evaluators.append(KeywordCoverageEvaluator())
    if ev_cfg.json_validity:
        evaluators.append(JSONValidityEvaluator())
    if ev_cfg.json_schema:
        evaluators.append(JSONSchemaEvaluator())
    if ev_cfg.semantic_similarity:
        from llm_eval_lab.evaluators.semantic import SemanticSimilarityEvaluator

        evaluators.append(SemanticSimilarityEvaluator())
    if ev_cfg.judge and ev_cfg.judge_provider:
        from llm_eval_lab.evaluators.judge import JudgeEvaluator

        evaluators.append(
            JudgeEvaluator(
                judge_config=ev_cfg.judge_provider,
                rubric=ev_cfg.judge_rubric,
            )
        )

    # Always collect latency
    evaluators.append(LatencyEvaluator())
    return evaluators


# ── Runner ────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[int, int, str], None]


def _noop_progress(current: int, total: int, case_id: str) -> None:  # noqa: ARG001
    pass


def run_case(
    case: TestCase,
    provider,
    template: PromptTemplate,
    evaluators: list[BaseEvaluator],
) -> CaseResult:
    """Execute one test case and return a CaseResult."""
    # Render prompt
    try:
        prompt = render_prompt(template, case)
    except ValueError as exc:
        return CaseResult(
            case_id=case.id,
            input=case.input,
            reference=case.reference,
            error=f"Prompt rendering failed: {exc}",
        )

    # Call provider
    response: ModelResponse = provider.complete(prompt, system_prompt=template.system_prompt)

    # Run evaluators (failures recorded per-metric, not raised)
    metric_results: list[MetricResult] = []
    for evaluator in evaluators:
        try:
            result = evaluator.evaluate(case, response)
        except Exception as exc:
            logger.exception("Evaluator '%s' raised unexpectedly.", evaluator.metric_name)
            result = MetricResult(
                metric_name=evaluator.metric_name,
                score=None,
                error=f"Evaluator error: {type(exc).__name__}: {exc}",
                direction=evaluator.direction,
            )
        metric_results.append(result)

    return CaseResult(
        case_id=case.id,
        input=case.input,
        reference=case.reference,
        output=response.text if not response.error else None,
        latency_ms=response.latency_ms,
        token_usage=response.token_usage,
        metrics=metric_results,
        error=response.error,
        is_simulated=response.is_simulated,
    )


def run_experiment(
    config: ExperimentConfig,
    progress_callback: ProgressCallback = _noop_progress,
) -> Experiment:
    """Execute a complete evaluation experiment.

    Parameters
    ----------
    config:
        Full experiment configuration including dataset path, provider,
        prompt template, and evaluator settings.
    progress_callback:
        Optional callback called after each case: ``(current, total, case_id)``.

    Returns
    -------
    Experiment
        Completed experiment with status, results, and summary.
    """
    experiment = Experiment(
        config=config,
        status=ExperimentStatus.RUNNING,
        started_at=datetime.now(UTC),
    )

    # Hash the prompt template for reproducibility
    config.prompt.content_hash = compute_template_hash(config.prompt)

    # Load dataset
    try:
        parse_result = load_dataset(config.dataset_path)
    except DatasetParseError as exc:
        experiment.status = ExperimentStatus.FAILED
        experiment.error = str(exc)
        experiment.completed_at = datetime.now(UTC)
        return experiment

    if not parse_result.cases:
        experiment.status = ExperimentStatus.FAILED
        experiment.error = "Dataset contains no valid test cases."
        if parse_result.errors:
            experiment.error += f" Parse errors: {'; '.join(parse_result.errors[:5])}"
        experiment.completed_at = datetime.now(UTC)
        return experiment

    if parse_result.errors:
        logger.warning(
            "Dataset has %d parse errors. Proceeding with %d valid cases.",
            len(parse_result.errors),
            len(parse_result.cases),
        )

    # Build provider and evaluators
    try:
        provider = get_provider(config.provider)
        evaluators = build_evaluators(config)
    except Exception as exc:
        experiment.status = ExperimentStatus.FAILED
        experiment.error = f"Configuration error: {exc}"
        experiment.completed_at = datetime.now(UTC)
        return experiment

    cases = parse_result.cases
    total = len(cases)
    results: list[CaseResult] = []

    for idx, case in enumerate(cases, start=1):
        progress_callback(idx, total, case.id)
        case_result = run_case(case, provider, config.prompt, evaluators)
        results.append(case_result)
        logger.debug("Case %s: error=%s", case.id, case_result.error)

    experiment.results = results
    experiment.summary = build_summary(results, config)
    experiment.status = ExperimentStatus.COMPLETED
    experiment.completed_at = datetime.now(UTC)

    logger.info(
        "Experiment '%s' completed: %d/%d successful.",
        config.experiment_id,
        experiment.summary.successful_cases,
        experiment.summary.total_cases,
    )
    return experiment
