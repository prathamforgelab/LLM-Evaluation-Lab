"""
Pydantic schemas for LLM Evaluation Lab.

All data models are defined here so the rest of the package imports from a
single canonical source.  Models use strict typing and explicit field
documentation so that serialised records are self-describing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Helpers ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Dataset ───────────────────────────────────────────────────────────────────


class TestCase(BaseModel):
    """One evaluation test case read from a JSONL dataset."""

    id: str = Field(..., description="Unique test-case identifier within a dataset.")
    input: str = Field(..., description="Prompt text sent to the model under test.")
    reference: str | None = Field(None, description="Expected / gold-standard answer.")
    context: str | None = Field(None, description="Optional grounding context or evidence.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form metadata.")

    # Optional evaluation-specific expectations
    expected_json_schema: dict[str, Any] | None = Field(
        None, description="JSON Schema the model output must satisfy (if applicable)."
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must appear in the model output.",
    )
    expected_facts: list[str] = Field(
        default_factory=list,
        description="Factual statements the model output should support.",
    )

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Test-case id must not be empty.")
        return v.strip()

    @field_validator("input")
    @classmethod
    def _input_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Test-case input must not be empty.")
        return v


# Prevent pytest from collecting TestCase as a test class
TestCase.__test__ = False  # type: ignore[attr-defined]


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset."""

    path: str
    total_cases: int
    valid_cases: int
    invalid_cases: int
    parse_errors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


# ── Provider / model output ───────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token counts reported by a provider. All fields are optional because
    not every provider reports every counter."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ModelResponse(BaseModel):
    """Raw output from a model provider for one test case."""

    text: str
    latency_ms: float = Field(..., description="Wall-clock latency in milliseconds.")
    token_usage: TokenUsage | None = None
    provider: str = ""
    model: str = ""
    error: str | None = None
    is_simulated: bool = Field(
        False,
        description="True when the response was produced by the mock provider.",
    )


# ── Metric results ────────────────────────────────────────────────────────────


class MetricDirection(StrEnum):
    """Whether a higher or lower score is preferred."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    INFORMATIONAL = "informational"


class MetricResult(BaseModel):
    """Evaluation score for a single metric on a single test case."""

    metric_name: str
    score: float | None = Field(
        None, description="Numeric score; None if the metric could not be computed."
    )
    passed: bool | None = Field(None, description="Boolean pass/fail for threshold-based metrics.")
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER


# ── Per-case result ───────────────────────────────────────────────────────────


class CaseResult(BaseModel):
    """Evaluation outcome for one test case within an experiment."""

    case_id: str
    input: str
    reference: str | None = None
    output: str | None = None
    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    error: str | None = Field(
        None,
        description="Provider error message.  Distinct from metric failures.",
    )
    is_simulated: bool = False
    timestamp: datetime = Field(default_factory=_now)

    def get_metric(self, name: str) -> MetricResult | None:
        for m in self.metrics:
            if m.metric_name == name:
                return m
        return None


# ── Experiment ────────────────────────────────────────────────────────────────


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PromptTemplate(BaseModel):
    """A versioned prompt template with variable substitution."""

    name: str = "default"
    system_prompt: str = ""
    user_template: str = Field(
        "{input}",
        description="Jinja2 template string. Available variables: input, reference, context.",
    )
    version: str = "1.0"
    # SHA256 of (system_prompt + user_template) computed at render time
    content_hash: str = ""


class ProviderConfig(BaseModel):
    """Configuration for the model provider."""

    name: str = Field(..., description="Provider name: 'mock' or 'openai_compatible'.")
    model: str = ""
    base_url: str = ""
    # API key is NOT stored here — it must come from env at runtime
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 30.0
    retries: int = 2
    simulated_latency_ms: float = 0.0  # mock provider only


class EvaluatorConfig(BaseModel):
    """Which evaluators to run and their parameters."""

    exact_match: bool = True
    normalized_exact_match: bool = True
    token_overlap_f1: bool = True
    json_validity: bool = False
    json_schema: bool = False
    keyword_coverage: bool = False
    semantic_similarity: bool = False
    judge: bool = False
    judge_provider: ProviderConfig | None = None
    judge_rubric: str = ""


class ExperimentConfig(BaseModel):
    """Full reproducible configuration for one experiment run."""

    experiment_id: str = Field(default_factory=_new_id)
    name: str = "experiment"
    dataset_path: str = ""
    provider: ProviderConfig
    prompt: PromptTemplate = Field(default_factory=PromptTemplate)
    evaluators: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    # Pricing (optional — cost is reported as unavailable when not configured)
    cost_per_prompt_token: float | None = None
    cost_per_completion_token: float | None = None
    # Statistical
    bootstrap_n: int = 2000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95


class ExperimentSummary(BaseModel):
    """Aggregate statistics computed after an experiment completes."""

    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    success_rate: float = 0.0
    mean_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)


class Experiment(BaseModel):
    """Complete experiment record including configuration, results, and summary."""

    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: list[CaseResult] = Field(default_factory=list)
    summary: ExperimentSummary = Field(default_factory=ExperimentSummary)
    error: str | None = None

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id


# ── Comparison / Regression ───────────────────────────────────────────────────


class RegressionDecision(StrEnum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float
    method: str = "paired_bootstrap"


class MetricComparison(BaseModel):
    """Comparison result for a single metric between baseline and candidate."""

    metric_name: str
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    absolute_change: float | None = None
    relative_change_pct: float | None = None
    ci: ConfidenceInterval | None = None
    decision: RegressionDecision = RegressionDecision.INCONCLUSIVE
    n_paired: int = 0
    n_missing_baseline: int = 0
    n_missing_candidate: int = 0
    n_failed: int = 0
    notes: list[str] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    """Full statistical comparison between a baseline and a candidate experiment."""

    comparison_id: str = Field(default_factory=_new_id)
    baseline_id: str
    candidate_id: str
    created_at: datetime = Field(default_factory=_now)
    metric_comparisons: list[MetricComparison] = Field(default_factory=list)
    overall_decision: RegressionDecision = RegressionDecision.INCONCLUSIVE
    cases_improved: list[str] = Field(default_factory=list)
    cases_regressed: list[str] = Field(default_factory=list)
    cases_missing_baseline: list[str] = Field(default_factory=list)
    cases_missing_candidate: list[str] = Field(default_factory=list)
    bootstrap_n: int = 2000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    notes: list[str] = Field(default_factory=list)


# ── Regression thresholds ─────────────────────────────────────────────────────


class RegressionThreshold(BaseModel):
    """Decision rules for one metric."""

    metric_name: str
    # Minimum absolute change to consider meaningful (0 = any CI beyond zero)
    practical_significance: float = 0.0
    # Minimum paired samples before a decision can be made
    min_samples: int = 5


class RegressionConfig(BaseModel):
    """Complete regression-testing configuration."""

    thresholds: list[RegressionThreshold] = Field(default_factory=list)
    default_min_samples: int = 5
    default_practical_significance: float = 0.0

    def get_threshold(self, metric_name: str) -> RegressionThreshold:
        for t in self.thresholds:
            if t.metric_name == metric_name:
                return t
        return RegressionThreshold(
            metric_name=metric_name,
            practical_significance=self.default_practical_significance,
            min_samples=self.default_min_samples,
        )
