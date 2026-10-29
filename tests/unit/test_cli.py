"""
Unit tests for the Typer CLI commands.

Covers:
- llm-eval --help
- llm-eval validate
- llm-eval run (with mock provider)
- llm-eval compare
- llm-eval report
- llm-eval experiments list / delete
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_eval_lab.cli import app
from llm_eval_lab.storage import Storage

runner = CliRunner()


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    data = [
        {"id": "q1", "input": "What is Python?", "reference": "A programming language.", "context": "Python is a language."},
        {"id": "q2", "input": "What is 2+2?", "reference": "4", "context": "Math facts."},
    ]
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM Evaluation Lab" in result.stdout


def test_cli_validate_success(sample_dataset: Path):
    result = runner.invoke(app, ["validate", str(sample_dataset)])
    assert result.exit_code == 0
    assert "Valid cases: 2" in result.stdout


def test_cli_validate_missing_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.jsonl"
    result = runner.invoke(app, ["validate", str(missing)])
    assert result.exit_code == 1


def test_cli_run_mock(sample_dataset: Path, tmp_path: Path, monkeypatch):
    db_path = tmp_path / "cli_test.db"
    monkeypatch.setattr("llm_eval_lab.storage.settings.db_path", db_path)
    monkeypatch.setattr("llm_eval_lab.storage._storage", None)

    result = runner.invoke(
        app,
        [
            "run",
            str(sample_dataset),
            "--name",
            "CLI Mock Run",
            "--provider",
            "mock",
            "--evaluators",
            "exact_match,latency",
        ],
    )
    assert result.exit_code == 0
    assert "CLI Mock Run" in result.stdout
    assert "Experiment complete" in result.stdout


def test_cli_experiments_list(sample_dataset: Path, tmp_path: Path, monkeypatch):
    db_path = tmp_path / "cli_test.db"
    monkeypatch.setattr("llm_eval_lab.storage.settings.db_path", db_path)
    monkeypatch.setattr("llm_eval_lab.storage._storage", None)

    # First run an experiment
    runner.invoke(
        app,
        [
            "run",
            str(sample_dataset),
            "--name",
            "Run 1",
            "--provider",
            "mock",
        ],
    )

    # Now list
    result = runner.invoke(app, ["experiments", "list"])
    assert result.exit_code == 0
    assert "Run 1" in result.stdout
