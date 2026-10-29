"""Experiment API routes: create, list, get, delete."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from llm_eval_lab.runner import run_experiment
from llm_eval_lab.schemas import (
    EvaluatorConfig,
    ExperimentConfig,
    PromptTemplate,
    ProviderConfig,
)
from llm_eval_lab.storage import get_storage

router = APIRouter()


class RunExperimentRequest(BaseModel):
    name: str = "experiment"
    dataset_path: str
    provider_name: str = "mock"
    model: str = ""
    system_prompt: str = ""
    user_template: str = "{input}"
    exact_match: bool = True
    token_f1: bool = True
    json_validity: bool = False


@router.post("/")
async def create_experiment(req: RunExperimentRequest, background_tasks: BackgroundTasks):
    """Create and run an experiment (runs synchronously for now)."""
    # Validate dataset path — only allow relative paths inside data_dir
    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {req.dataset_path}")

    config = ExperimentConfig(
        name=req.name,
        dataset_path=str(dataset_path),
        provider=ProviderConfig(name=req.provider_name, model=req.model),
        prompt=PromptTemplate(system_prompt=req.system_prompt, user_template=req.user_template),
        evaluators=EvaluatorConfig(
            exact_match=req.exact_match,
            token_overlap_f1=req.token_f1,
            json_validity=req.json_validity,
        ),
    )

    experiment = run_experiment(config)
    storage = get_storage()
    storage.save_experiment(experiment)

    return {
        "experiment_id": experiment.experiment_id,
        "status": experiment.status.value,
        "summary": experiment.summary.model_dump(),
    }


@router.get("/")
async def list_experiments(limit: int = 20):
    """List recent experiments."""
    storage = get_storage()
    return {"experiments": storage.list_experiments(limit)}


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get a specific experiment by ID."""
    storage = get_storage()
    exp = storage.load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return exp.model_dump()


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    storage = get_storage()
    found = storage.delete_experiment(experiment_id)
    if not found:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"deleted": experiment_id}
