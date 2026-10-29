"""
Integration tests for complete experiment workflows.

Tests run using the offline mock provider only — no network required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_eval_lab.runner import run_experiment
from llm_eval_lab.reporting import (
    experiment_to_csv,
    experiment_to_json,
    experiment_to_markdown,
)
from llm_eval_lab.schemas import (
    EvaluatorConfig,
    ExperimentConfig,
    ExperimentStatus,
    PromptTemplate,
    ProviderConfig,
)


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    p = tmp_path / "test_data.jsonl"
    with p.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    return p


# ── Core integration ──────────────────────────────────────────────────────────

class TestCompleteExperiment:
    def test_full_offline_experiment(self, tmp_path):
        """An experiment runs end-to-end with mock provider."""
        cases = [
            {"id": "f1", "input": "What is the capital of France?",
             "reference": "The capital of France is Paris."},
            {"id": "f2", "input": "What is Python?",
             "reference": "Python is a programming language."},
        ]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            name="integration-test",
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
            evaluators=EvaluatorConfig(exact_match=True, token_overlap_f1=True),
        )
        exp = run_experiment(config)

        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.summary.total_cases == 2
        assert exp.summary.success_rate > 0
        assert len(exp.results) == 2
        # Every result has metrics
        for r in exp.results:
            assert len(r.metrics) > 0

    def test_all_cases_get_results(self, tmp_path):
        """Each test case produces a CaseResult regardless of match quality."""
        cases = [{"id": f"c{i}", "input": f"input {i}", "reference": f"ref {i}"} for i in range(5)]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        exp = run_experiment(config)
        assert len(exp.results) == 5
        result_ids = {r.case_id for r in exp.results}
        assert result_ids == {f"c{i}" for i in range(5)}

    def test_failed_dataset_returns_failed_status(self, tmp_path):
        config = ExperimentConfig(
            dataset_path=str(tmp_path / "nonexistent.jsonl"),
            provider=ProviderConfig(name="mock"),
        )
        exp = run_experiment(config)
        assert exp.status == ExperimentStatus.FAILED
        assert exp.error is not None

    def test_empty_dataset_returns_failed(self, tmp_path):
        dataset = tmp_path / "empty.jsonl"
        dataset.write_text("")
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        exp = run_experiment(config)
        assert exp.status == ExperimentStatus.FAILED

    def test_progress_callback_called(self, tmp_path):
        calls = []
        def cb(current, total, case_id):
            calls.append((current, total, case_id))

        cases = [{"id": f"c{i}", "input": f"input {i}"} for i in range(3)]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        run_experiment(config, progress_callback=cb)
        assert len(calls) == 3
        assert calls[0][0] == 1
        assert calls[-1][0] == 3

    def test_custom_prompt_template(self, tmp_path):
        cases = [{"id": "t1", "input": "Paris question"}]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
            prompt=PromptTemplate(
                user_template="Please answer: {{ input }}",
                system_prompt="You are helpful.",
            ),
        )
        exp = run_experiment(config)
        assert exp.status == ExperimentStatus.COMPLETED

    def test_summary_contains_aggregate_metrics(self, tmp_path):
        cases = [
            {"id": f"c{i}", "input": "What is the capital of France?",
             "reference": "The capital of France is Paris."}
            for i in range(3)
        ]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
            evaluators=EvaluatorConfig(token_overlap_f1=True, exact_match=True),
        )
        exp = run_experiment(config)
        assert "token_overlap_f1" in exp.summary.aggregate_metrics
        assert "exact_match" in exp.summary.aggregate_metrics


# ── Report generation ─────────────────────────────────────────────────────────

class TestReportGeneration:
    def _run_exp(self, tmp_path: Path) -> object:
        cases = [
            {"id": "r1", "input": "What is Python?", "reference": "A programming language."},
            {"id": "r2", "input": "capital france", "reference": "Paris"},
        ]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            name="report-test",
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        return run_experiment(config)

    def test_markdown_report_generated(self, tmp_path):
        exp = self._run_exp(tmp_path)
        md = experiment_to_markdown(exp)
        assert "# Experiment Report" in md
        assert exp.config.experiment_id in md
        assert "## Summary" in md

    def test_json_report_valid_json(self, tmp_path):
        exp = self._run_exp(tmp_path)
        json_str = experiment_to_json(exp)
        parsed = json.loads(json_str)
        assert "config" in parsed
        assert "results" in parsed

    def test_csv_report_has_header_and_rows(self, tmp_path):
        exp = self._run_exp(tmp_path)
        csv_str = experiment_to_csv(exp)
        lines = csv_str.strip().split("\n")
        assert "case_id" in lines[0]
        assert len(lines) > 1  # header + at least one row


# ── Storage integration ───────────────────────────────────────────────────────

class TestStorageIntegration:
    def test_save_and_load_experiment(self, tmp_path):
        from llm_eval_lab.storage import Storage

        cases = [{"id": "s1", "input": "test input", "reference": "test ref"}]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        exp = run_experiment(config)
        storage = Storage(db_path=tmp_path / "test.db")
        storage.save_experiment(exp)

        loaded = storage.load_experiment(exp.experiment_id)
        assert loaded is not None
        assert loaded.experiment_id == exp.experiment_id
        assert loaded.status == exp.status
        assert len(loaded.results) == len(exp.results)

    def test_list_experiments(self, tmp_path):
        from llm_eval_lab.storage import Storage

        cases = [{"id": "s1", "input": "test"}]
        dataset = _write_dataset(tmp_path, cases)
        storage = Storage(db_path=tmp_path / "test.db")

        for name in ["exp-a", "exp-b"]:
            config = ExperimentConfig(
                name=name,
                dataset_path=str(dataset),
                provider=ProviderConfig(name="mock"),
            )
            exp = run_experiment(config)
            storage.save_experiment(exp)

        rows = storage.list_experiments()
        assert len(rows) == 2

    def test_delete_experiment(self, tmp_path):
        from llm_eval_lab.storage import Storage

        cases = [{"id": "s1", "input": "test"}]
        dataset = _write_dataset(tmp_path, cases)
        config = ExperimentConfig(
            dataset_path=str(dataset),
            provider=ProviderConfig(name="mock"),
        )
        exp = run_experiment(config)
        storage = Storage(db_path=tmp_path / "test.db")
        storage.save_experiment(exp)
        storage.delete_experiment(exp.experiment_id)
        assert storage.load_experiment(exp.experiment_id) is None

    def test_load_nonexistent_returns_none(self, tmp_path):
        from llm_eval_lab.storage import Storage

        storage = Storage(db_path=tmp_path / "test.db")
        result = storage.load_experiment("does-not-exist")
        assert result is None
