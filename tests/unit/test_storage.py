"""
Unit tests for the SQLite storage layer.

Covers:
- Experiment save, load, list, delete.
- Foreign key cascading deletion of results.
- Comparison report save, load, list.
- Handling of non-existent IDs.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_eval_lab.schemas import (
    CaseResult,
    ComparisonReport,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentSummary,
    ConfidenceInterval,
    MetricComparison,
    MetricResult,
    ModelResponse,
    RegressionDecision,
)
from llm_eval_lab.storage import Storage


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_eval.db"
        yield Storage(db_path=db_path)


def _make_experiment(exp_id: str = "exp-001", name: str = "Test Run") -> Experiment:
    config = ExperimentConfig(
        experiment_id=exp_id,
        name=name,
        dataset_path="examples/sample_dataset.jsonl",
        provider={"name": "mock", "model": "mock-default"},
    )
    res1 = CaseResult(
        case_id="case-1",
        input="Hello",
        output="Hello world",
        latency_ms=45.0,
        metrics=[MetricResult(metric_name="exact_match", score=1.0)],
    )
    res2 = CaseResult(
        case_id="case-2",
        input="Foo",
        output="Bar",
        latency_ms=50.0,
        metrics=[MetricResult(metric_name="exact_match", score=0.0)],
    )
    summary = ExperimentSummary(
        total_cases=2,
        successful_cases=2,
        failed_cases=0,
        mean_scores={"exact_match": 0.5},
        p50_scores={"exact_match": 0.5},
        p95_scores={"exact_match": 0.95},
        mean_latency_ms=47.5,
    )
    return Experiment(
        config=config,
        status=ExperimentStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        results=[res1, res2],
        summary=summary,
    )


def test_save_and_load_experiment(storage: Storage):
    exp = _make_experiment()
    storage.save_experiment(exp)

    loaded = storage.load_experiment("exp-001")
    assert loaded is not None
    assert loaded.config.name == "Test Run"
    assert loaded.status == ExperimentStatus.COMPLETED
    assert len(loaded.results) == 2
    assert loaded.results[0].case_id == "case-1"
    assert loaded.results[0].metrics[0].score == 1.0
    assert loaded.summary is not None
    assert loaded.summary.mean_scores["exact_match"] == 0.5


def test_load_nonexistent_experiment(storage: Storage):
    loaded = storage.load_experiment("does-not-exist")
    assert loaded is None


def test_list_experiments(storage: Storage):
    exp1 = _make_experiment("exp-1", "Run 1")
    exp2 = _make_experiment("exp-2", "Run 2")
    storage.save_experiment(exp1)
    storage.save_experiment(exp2)

    exp_list = storage.list_experiments()
    assert len(exp_list) == 2
    ids = [e["id"] for e in exp_list]
    assert "exp-1" in ids
    assert "exp-2" in ids


def test_delete_experiment(storage: Storage):
    exp = _make_experiment()
    storage.save_experiment(exp)

    deleted = storage.delete_experiment("exp-001")
    assert deleted is True

    # Check experiment is gone
    assert storage.load_experiment("exp-001") is None

    # Check cascading delete in results table
    with storage._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM results WHERE experiment_id = ?", ("exp-001",)).fetchone()[0]
        assert count == 0

    # Deleting again returns False
    assert storage.delete_experiment("exp-001") is False


def test_save_and_load_comparison(storage: Storage):
    report = ComparisonReport(
        comparison_id="comp-123",
        baseline_id="exp-base",
        candidate_id="exp-cand",
        overall_decision=RegressionDecision.IMPROVED,
        metric_comparisons=[
            MetricComparison(
                metric_name="exact_match",
                baseline_mean=0.6,
                candidate_mean=0.8,
                absolute_change=0.2,
                relative_change_pct=33.3,
                ci=ConfidenceInterval(lower=0.1, upper=0.3, level=0.95),
                decision=RegressionDecision.IMPROVED,
                n_paired=20,
            )
        ],
        cases_improved=["case-1"],
        cases_regressed=[],
    )

    storage.save_comparison(report)
    loaded = storage.load_comparison("comp-123")
    assert loaded is not None
    assert loaded.comparison_id == "comp-123"
    assert loaded.overall_decision == RegressionDecision.IMPROVED
    assert len(loaded.metric_comparisons) == 1
    assert loaded.metric_comparisons[0].absolute_change == 0.2

    # Listing comparisons
    comps = storage.list_comparisons()
    assert len(comps) == 1
    assert comps[0]["id"] == "comp-123"

    # Non-existent comparison
    assert storage.load_comparison("none") is None
