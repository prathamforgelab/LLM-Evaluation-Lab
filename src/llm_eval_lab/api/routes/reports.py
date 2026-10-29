"""Report generation API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
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


def _format_experiment_response(exp, format: str):
    format = format.lower().strip()
    if format == "json":
        return PlainTextResponse(experiment_to_json(exp), media_type="application/json")
    elif format == "csv":
        return PlainTextResponse(experiment_to_csv(exp), media_type="text/csv")
    else:
        return PlainTextResponse(experiment_to_markdown(exp), media_type="text/markdown")


def _format_comparison_response(report, format: str):
    format = format.lower().strip()
    if format == "json":
        return PlainTextResponse(comparison_to_json(report), media_type="application/json")
    else:
        return PlainTextResponse(comparison_to_markdown(report), media_type="text/markdown")


@router.get("/experiment/{experiment_id}")
@router.get("/experiments/{experiment_id}")
async def get_experiment_report(experiment_id: str, format: str = Query("markdown")):
    """Generate a report for an experiment (format: markdown | json | csv)."""
    storage = get_storage()
    exp = storage.load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return _format_experiment_response(exp, format)


@router.get("/experiment/{experiment_id}/{fmt}")
@router.get("/experiments/{experiment_id}/{fmt}")
async def get_experiment_report_formatted(experiment_id: str, fmt: str):
    """Generate a report for an experiment in path format (markdown | json | csv)."""
    storage = get_storage()
    exp = storage.load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return _format_experiment_response(exp, fmt)


@router.get("/comparison/{comparison_id}")
@router.get("/comparisons/{comparison_id}")
async def get_comparison_report(comparison_id: str, format: str = Query("markdown")):
    """Generate a report for a comparison (format: markdown | json)."""
    storage = get_storage()
    report = storage.load_comparison(comparison_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    return _format_comparison_response(report, format)


@router.get("/comparison/{comparison_id}/{fmt}")
@router.get("/comparisons/{comparison_id}/{fmt}")
async def get_comparison_report_formatted(comparison_id: str, fmt: str):
    """Generate a report for a comparison in path format (markdown | json)."""
    storage = get_storage()
    report = storage.load_comparison(comparison_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    return _format_comparison_response(report, fmt)
