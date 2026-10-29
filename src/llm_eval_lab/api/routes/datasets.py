"""Dataset routes: upload, validate, preview."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from llm_eval_lab.dataset import load_dataset, DatasetParseError

router = APIRouter()

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_EXTENSIONS = {".jsonl"}


@router.post("/validate")
async def validate_dataset_upload(file: UploadFile = File(...)):
    """Upload a JSONL dataset and return validation results."""
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .jsonl files are accepted.",
        )

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {_MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    # Write to temp file and parse
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = load_dataset(tmp_path)
    except DatasetParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    info = result.info(file.filename or "upload")
    return {
        "filename": file.filename,
        "valid_cases": info.valid_cases,
        "invalid_cases": info.invalid_cases,
        "total_cases": info.total_cases,
        "categories": info.categories,
        "errors": info.parse_errors[:50],  # cap for response size
        "is_valid": result.is_valid,
    }


@router.post("/preview")
async def preview_dataset(file: UploadFile = File(...), max_rows: int = 10):
    """Return up to max_rows records from an uploaded dataset."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jsonl files accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = load_dataset(tmp_path, max_cases=min(max_rows, 100))
    except DatasetParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "cases": [c.model_dump() for c in result.cases[:max_rows]],
        "errors": result.errors[:10],
    }
