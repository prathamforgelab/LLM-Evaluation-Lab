"""
Example: Running an offline evaluation experiment using Python API.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from llm_eval_lab.runner import run_experiment
from llm_eval_lab.schemas import EvaluatorConfig, ExperimentConfig, PromptTemplate, ProviderConfig
from llm_eval_lab.storage import get_storage


def main():
    dataset_path = Path(__file__).parent / "sample_dataset.jsonl"
    print(f"Loading dataset: {dataset_path}")

    config = ExperimentConfig(
        name="Offline Baseline Run",
        dataset_path=str(dataset_path),
        provider=ProviderConfig(name="mock", model="mock-default"),
        prompt=PromptTemplate(
            system_prompt="You are a helpful and factual technical assistant.",
            user_template="{input}",
        ),
        evaluators=EvaluatorConfig(
            exact_match=True,
            normalized_exact_match=True,
            token_overlap_f1=True,
            json_validity=True,
        ),
    )

    print("Running experiment…")
    experiment = run_experiment(config)

    s = experiment.summary
    print("\n" + "=" * 50)
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Status:        {experiment.status.value}")
    print(f"Total cases:   {s.total_cases}")
    print(f"Success rate:  {s.success_rate * 100:.1f}%")
    print(f"Mean latency:  {s.mean_latency_ms:.1f} ms" if s.mean_latency_ms else "")
    print("\nAggregate Metrics:")
    for metric, score in s.aggregate_metrics.items():
        print(f"  • {metric:<25}: {score:.4f}")
    print("=" * 50)

    # Persist to local SQLite storage
    storage = get_storage()
    storage.save_experiment(experiment)
    print("\nSaved experiment to local SQLite database.")


if __name__ == "__main__":
    main()
