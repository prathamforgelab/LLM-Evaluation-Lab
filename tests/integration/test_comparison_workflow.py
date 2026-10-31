"""
Integration tests for the comparison and regression workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval_lab.regression import compare_experiments
from llm_eval_lab.reporting import comparison_to_json, comparison_to_markdown
from llm_eval_lab.runner import run_experiment
from llm_eval_lab.schemas import (
    EvaluatorConfig,
    ExperimentConfig,
    ProviderConfig,
)


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    p = tmp_path / "data.jsonl"
    with p.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    return p


def _run(tmp_path: Path, cases: list[dict], name: str = "exp") -> object:
    dataset = _write_dataset(tmp_path, cases)
    config = ExperimentConfig(
        name=name,
        dataset_path=str(dataset),
        provider=ProviderConfig(name="mock"),
        evaluators=EvaluatorConfig(exact_match=True, token_overlap_f1=True),
    )
    return run_experiment(config)


class TestComparisonWorkflow:
    def test_comparison_report_generated(self, tmp_path):
        cases = [{"id": f"c{i}", "input": f"input {i}", "reference": f"ref {i}"} for i in range(10)]
        baseline = _run(tmp_path, cases, "baseline")
        candidate = _run(tmp_path, cases, "candidate")
        report = compare_experiments(baseline, candidate, bootstrap_n=100, bootstrap_seed=42)
        assert report.baseline_id == baseline.experiment_id
        assert report.candidate_id == candidate.experiment_id
        assert len(report.metric_comparisons) > 0

    def test_comparison_markdown_report(self, tmp_path):
        cases = [{"id": f"c{i}", "input": f"q{i}", "reference": f"a{i}"} for i in range(5)]
        baseline = _run(tmp_path, cases)
        candidate = _run(tmp_path, cases)
        report = compare_experiments(baseline, candidate, bootstrap_n=100, bootstrap_seed=42)
        md = comparison_to_markdown(report)
        assert "# Comparison Report" in md
        assert report.comparison_id in md

    def test_comparison_json_report(self, tmp_path):
        cases = [{"id": f"c{i}", "input": f"q{i}", "reference": f"a{i}"} for i in range(5)]
        baseline = _run(tmp_path, cases)
        candidate = _run(tmp_path, cases)
        report = compare_experiments(baseline, candidate, bootstrap_n=100, bootstrap_seed=42)
        json_str = comparison_to_json(report)
        parsed = json.loads(json_str)
        assert "metric_comparisons" in parsed
        assert "overall_decision" in parsed

    def test_save_and_load_comparison(self, tmp_path):
        from llm_eval_lab.storage import Storage

        cases = [{"id": f"c{i}", "input": f"q{i}"} for i in range(5)]
        baseline = _run(tmp_path, cases)
        candidate = _run(tmp_path, cases)
        storage = Storage(db_path=tmp_path / "test.db")
        storage.save_experiment(baseline)
        storage.save_experiment(candidate)

        report = compare_experiments(baseline, candidate, bootstrap_n=100, bootstrap_seed=42)
        storage.save_comparison(report)

        loaded = storage.load_comparison(report.comparison_id)
        assert loaded is not None
        assert loaded.comparison_id == report.comparison_id
        assert loaded.overall_decision == report.overall_decision
