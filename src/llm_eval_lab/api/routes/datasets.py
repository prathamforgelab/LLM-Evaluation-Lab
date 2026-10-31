"""Dataset routes: upload, validate, preview.

Upload security
---------------
* File extension is validated before reading any content.
* File content is read in bounded chunks to avoid loading arbitrarily large
  files into memory before the size limit is applied.
* Temporary files are always cleaned up in a finally block.
* Only .jsonl extensions are accepted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from llm_eval_lab.dataset import DatasetParseError, load_dataset

router = APIRouter()

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_EXTENSIONS = {".jsonl"}
_CHUNK_SIZE = 64 * 1024  # 64 KB read chunks


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file in chunks, raising 413 if it exceeds *max_bytes*.

    This prevents the server from loading an arbitrarily large file into memory
    before the size limit is checked.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is {max_bytes // 1024 // 1024} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_extension(filename: str | None) -> None:
    """Raise 400 if *filename* does not have an allowed extension."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .jsonl files are accepted.",
        )


@router.post("/validate")
async def validate_dataset_upload(file: UploadFile = File(...)):  # noqa: B008
    """Upload a JSONL dataset and return validation results."""
    _validate_extension(file.filename)
    content = await _read_bounded(file, _MAX_UPLOAD_BYTES)

    # Write to temp file and parse
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        result = load_dataset(tmp_path)
    except DatasetParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
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
async def preview_dataset(file: UploadFile = File(...), max_rows: int = 10):  # noqa: B008
    """Return up to max_rows records from an uploaded dataset."""
    _validate_extension(file.filename)
    content = await _read_bounded(file, _MAX_UPLOAD_BYTES)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        result = load_dataset(tmp_path, max_cases=min(max_rows, 100))
    except DatasetParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return {
        "cases": [c.model_dump() for c in result.cases[:max_rows]],
        "errors": result.errors[:10],
    }
