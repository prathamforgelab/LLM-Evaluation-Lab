"""Report generation API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from llm_eval_lab.reporting import (
    comparison_to_json,
    comparison_to_markdown,
    experiment_to_csv,
    experiment_to_json,
    experiment_to_markdown,
)
from llm_eval_lab.storage import get_storage

router = APIRouter()


@router.get("/experiment/{experiment_id}")
async def get_experiment_report(experiment_id: str, format: str = "markdown"):
    """Generate a report for an experiment."""
    storage = get_storage()
    exp = storage.load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    if format == "json":
        return PlainTextResponse(experiment_to_json(exp), media_type="application/json")
    elif format == "csv":
        return PlainTextResponse(experiment_to_csv(exp), media_type="text/csv")
    else:
        return PlainTextResponse(experiment_to_markdown(exp), media_type="text/markdown")


@router.get("/comparison/{comparison_id}")
async def get_comparison_report(comparison_id: str, format: str = "markdown"):
    """Generate a report for a comparison."""
    storage = get_storage()
    report = storage.load_comparison(comparison_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")

    if format == "json":
        return PlainTextResponse(comparison_to_json(report), media_type="application/json")
    else:
        return PlainTextResponse(comparison_to_markdown(report), media_type="text/markdown")
