"""
Configuration management for LLM Evaluation Lab.

Settings are loaded from environment variables (or a .env file) using
python-dotenv.  Sensitive values are never written to logs or reports.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "")
    try:
        return int(val) if val.strip() else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "")
    try:
        return float(val) if val.strip() else default
    except ValueError:
        return default


def _env_list(key: str, default: list[str]) -> list[str]:
    val = os.environ.get(key, "")
    return [v.strip() for v in val.split(",") if v.strip()] if val.strip() else default


class Settings:
    """Central settings object populated from environment variables."""

    # Data storage
    data_dir: Path = Path(os.environ.get("LLM_EVAL_DATA_DIR", "./data"))
    max_cases: int = _env_int("LLM_EVAL_MAX_CASES", 10_000)

    # HTTP / provider
    request_timeout: float = _env_float("LLM_EVAL_REQUEST_TIMEOUT", 30.0)

    # OpenAI-compatible provider
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Statistics
    bootstrap_n: int = _env_int("LLM_EVAL_BOOTSTRAP_N", 2000)
    bootstrap_seed: int = _env_int("LLM_EVAL_BOOTSTRAP_SEED", 42)
    confidence_level: float = _env_float("LLM_EVAL_CONFIDENCE_LEVEL", 0.95)

    # API server
    api_host: str = os.environ.get("API_HOST", "127.0.0.1")
    api_port: int = _env_int("API_PORT", 8000)
    cors_origins: list[str] = _env_list(
        "API_CORS_ORIGINS",
        ["http://localhost:5173", "http://127.0.0.1:5173"],
    )

    _custom_db_path: Path | None = None

    @property
    def db_path(self) -> Path:
        if self._custom_db_path is not None:
            return self._custom_db_path
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "eval_lab.db"

    @db_path.setter
    def db_path(self, val: Path | None) -> None:
        self._custom_db_path = val

    @property
    def reports_dir(self) -> Path:
        path = self.data_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def redacted_summary(self) -> dict:
        """Return a settings summary safe for logging (no secrets)."""
        return {
            "data_dir": str(self.data_dir),
            "max_cases": self.max_cases,
            "request_timeout": self.request_timeout,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "openai_api_key": "***" if self.openai_api_key else "(not set)",
            "bootstrap_n": self.bootstrap_n,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "api_host": self.api_host,
            "api_port": self.api_port,
        }


# Module-level singleton
settings = Settings()
