"""
Unit tests for dataset parsing and validation.

Covers: valid cases, JSON errors, Pydantic validation errors, duplicate IDs,
empty input, unicode, size limits, missing fields, line-number reporting.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from llm_eval_lab.dataset import DatasetParseError, load_dataset, validate_dataset


def _write_jsonl(tmp_path: Path, lines: list[dict | str]) -> Path:
    p = tmp_path / "test.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write((json.dumps(line) if isinstance(line, dict) else line) + "\n")
    return p


# ── Happy path ────────────────────────────────────────────────────────────────

def test_load_valid_dataset(tmp_path):
    cases = [
        {"id": "case-1", "input": "What is Python?"},
        {"id": "case-2", "input": "What is AI?", "reference": "Artificial intelligence."},
    ]
    p = _write_jsonl(tmp_path, cases)
    result = load_dataset(p)
    assert result.is_valid
    assert len(result.cases) == 2
    assert result.cases[0].id == "case-1"
    assert result.cases[1].reference == "Artificial intelligence."


def test_load_with_all_optional_fields(tmp_path):
    row = {
        "id": "full-001",
        "input": "Test input",
        "reference": "Test reference",
        "context": "Some context",
        "metadata": {"category": "test"},
        "expected_keywords": ["test"],
        "expected_facts": ["fact one"],
    }
    p = _write_jsonl(tmp_path, [row])
    result = load_dataset(p)
    assert result.is_valid
    c = result.cases[0]
    assert c.context == "Some context"
    assert c.expected_keywords == ["test"]
    assert c.metadata["category"] == "test"


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"id":"a","input":"hello"}\n\n{"id":"b","input":"world"}\n', encoding="utf-8")
    result = load_dataset(p)
    assert len(result.cases) == 2


def test_utf8_unicode(tmp_path):
    p = _write_jsonl(tmp_path, [{"id": "unicode-1", "input": "日本語テスト 🎉"}])
    result = load_dataset(p)
    assert "🎉" in result.cases[0].input


# ── Error cases ───────────────────────────────────────────────────────────────

def test_invalid_json_line_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"ok","input":"fine"}\nnot valid json\n{"id":"ok2","input":"also fine"}\n')
    result = load_dataset(p)
    assert len(result.cases) == 2
    assert len(result.errors) == 1
    assert "Line 2" in result.errors[0]


def test_missing_required_field_id(tmp_path):
    p = _write_jsonl(tmp_path, [{"input": "no id here"}])
    result = load_dataset(p)
    assert len(result.cases) == 0
    assert len(result.errors) == 1
    assert "Line 1" in result.errors[0]


def test_missing_required_field_input(tmp_path):
    p = _write_jsonl(tmp_path, [{"id": "x1"}])
    result = load_dataset(p)
    assert len(result.cases) == 0
    assert len(result.errors) == 1


def test_empty_id_rejected(tmp_path):
    p = _write_jsonl(tmp_path, [{"id": "  ", "input": "hello"}])
    result = load_dataset(p)
    assert len(result.errors) == 1


def test_empty_input_rejected(tmp_path):
    p = _write_jsonl(tmp_path, [{"id": "x1", "input": "   "}])
    result = load_dataset(p)
    assert len(result.errors) == 1


def test_duplicate_ids_flagged(tmp_path):
    p = _write_jsonl(tmp_path, [
        {"id": "dup", "input": "first"},
        {"id": "dup", "input": "second"},
    ])
    result = load_dataset(p)
    assert len(result.cases) == 1
    assert any("Duplicate" in e for e in result.errors)


def test_file_not_found_raises():
    with pytest.raises(DatasetParseError, match="Cannot open"):
        load_dataset("/nonexistent/path/file.jsonl")


def test_size_limit_enforced(tmp_path):
    cases = [{"id": f"case-{i}", "input": f"input {i}"} for i in range(20)]
    p = _write_jsonl(tmp_path, cases)
    result = load_dataset(p, max_cases=5)
    assert len(result.cases) == 5
    assert any("truncated" in e.lower() for e in result.errors)


def test_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    result = load_dataset(p)
    assert len(result.cases) == 0
    assert result.is_valid  # no errors, just empty


# ── validate_dataset wrapper ──────────────────────────────────────────────────

def test_validate_dataset_returns_true_for_valid(tmp_path):
    p = _write_jsonl(tmp_path, [{"id": "v1", "input": "hello"}])
    ok, errors = validate_dataset(p)
    assert ok
    assert errors == []


def test_validate_dataset_returns_false_for_invalid(tmp_path):
    p = _write_jsonl(tmp_path, [{"input": "no id"}])
    ok, errors = validate_dataset(p)
    assert not ok
    assert len(errors) == 1


def test_dataset_info(tmp_path):
    cases = [
        {"id": "a", "input": "x", "metadata": {"category": "qa"}},
        {"id": "b", "input": "y", "metadata": {"category": "factual"}},
    ]
    p = _write_jsonl(tmp_path, cases)
    result = load_dataset(p)
    info = result.info(str(p))
    assert info.valid_cases == 2
    assert "factual" in info.categories
    assert "qa" in info.categories
