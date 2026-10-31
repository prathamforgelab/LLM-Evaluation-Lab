"""
SQLite persistence layer for experiments and comparisons.

Schema design
-------------
* experiments — one row per experiment configuration + status.
* results — one row per (experiment_id, case_id) result.
* comparisons — one row per comparison report.

All JSON blobs are stored as TEXT.  The schema is intentionally simple to keep
the code auditable.  Secrets are never stored.

Data retention
--------------
Users can delete all history with:  llm-eval experiments delete --all
Individual experiments:             llm-eval experiments delete <id>
The database file is at:            $LLM_EVAL_DATA_DIR/eval_lab.db
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from llm_eval_lab.config import settings
from llm_eval_lab.schemas import (
    CaseResult,
    ComparisonReport,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentSummary,
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT,
    config_json TEXT NOT NULL,
    summary_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    case_id         TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    UNIQUE(experiment_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id);

CREATE TABLE IF NOT EXISTS comparisons (
    id              TEXT PRIMARY KEY,
    baseline_id     TEXT NOT NULL,
    candidate_id    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    report_json     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comparisons_baseline ON comparisons(baseline_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_candidate ON comparisons(candidate_id);
"""


class Storage:
    """SQLite-backed storage for experiments and comparisons."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()
        logger.debug("Database initialised at %s", self._db_path)

    @contextmanager
    def _read_conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Experiments ──────────────────────────────────────────────────────────

    def save_experiment(self, experiment: Experiment) -> None:
        """Insert or replace an experiment record."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                    (id, name, status, started_at, completed_at,
                     config_json, summary_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.config.name,
                    experiment.status.value,
                    experiment.started_at.isoformat() if experiment.started_at else None,
                    experiment.completed_at.isoformat() if experiment.completed_at else None,
                    experiment.config.model_dump_json(),
                    experiment.summary.model_dump_json() if experiment.summary else None,
                    experiment.error,
                ),
            )

            # Upsert results
            for result in experiment.results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO results (experiment_id, case_id, result_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        experiment.experiment_id,
                        result.case_id,
                        result.model_dump_json(),
                    ),
                )
        logger.debug(
            "Saved experiment %s (%d results).", experiment.experiment_id, len(experiment.results)
        )

    def load_experiment(self, experiment_id: str) -> Experiment | None:
        """Load an experiment and its results from storage."""
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                return None

            config = ExperimentConfig.model_validate_json(row["config_json"])
            summary = (
                ExperimentSummary.model_validate_json(row["summary_json"])
                if row["summary_json"]
                else ExperimentSummary()
            )

            result_rows = conn.execute(
                "SELECT result_json FROM results WHERE experiment_id = ? ORDER BY id",
                (experiment_id,),
            ).fetchall()
            results = [CaseResult.model_validate_json(r["result_json"]) for r in result_rows]

            return Experiment(
                config=config,
                status=ExperimentStatus(row["status"]),
                started_at=_parse_dt(row["started_at"]),
                completed_at=_parse_dt(row["completed_at"]),
                results=results,
                summary=summary,
                error=row["error"],
            )

    def list_experiments(self, limit: int = 50) -> list[dict]:
        """Return summary rows for recent experiments."""
        with self._read_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, name, status, started_at, completed_at, error
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its results.  Returns True if found."""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return cur.rowcount > 0

    # ── Comparisons ───────────────────────────────────────────────────────────

    def save_comparison(self, report: ComparisonReport) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO comparisons
                    (id, baseline_id, candidate_id, created_at, report_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.comparison_id,
                    report.baseline_id,
                    report.candidate_id,
                    report.created_at.isoformat(),
                    report.model_dump_json(),
                ),
            )

    def load_comparison(self, comparison_id: str) -> ComparisonReport | None:
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT report_json FROM comparisons WHERE id = ?", (comparison_id,)
            ).fetchone()
            if row is None:
                return None
            return ComparisonReport.model_validate_json(row["report_json"])

    def list_comparisons(self, limit: int = 20) -> list[dict]:
        with self._read_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, baseline_id, candidate_id, created_at
                FROM comparisons
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# Module-level singleton (lazy-initialised)
_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
