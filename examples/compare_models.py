"""
Example: Comparing two experiments with paired bootstrap resampling.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from llm_eval_lab.regression import compare_experiments
from llm_eval_lab.reporting import comparison_to_markdown
from llm_eval_lab.runner import run_experiment
from llm_eval_lab.schemas import EvaluatorConfig, ExperimentConfig, PromptTemplate, ProviderConfig


def main():
    dataset_path = Path(__file__).parent / "sample_dataset.jsonl"

    # 1. Baseline Experiment (Concise prompt)
    baseline_cfg = ExperimentConfig(
        name="Baseline Prompt",
        dataset_path=str(dataset_path),
        provider=ProviderConfig(name="mock", model="mock-baseline"),
        prompt=PromptTemplate(system_prompt="Answer briefly.", user_template="{input}"),
        evaluators=EvaluatorConfig(exact_match=True, token_overlap_f1=True),
    )
    print("Running baseline experiment…")
    baseline_exp = run_experiment(baseline_cfg)

    # 2. Candidate Experiment (Chain of thought / detailed prompt)
    candidate_cfg = ExperimentConfig(
        name="Candidate Prompt",
        dataset_path=str(dataset_path),
        provider=ProviderConfig(name="mock", model="mock-candidate"),
        prompt=PromptTemplate(
            system_prompt="You are an expert AI. Think step-by-step and answer accurately.",
            user_template="{input}",
        ),
        evaluators=EvaluatorConfig(exact_match=True, token_overlap_f1=True),
    )
    print("Running candidate experiment…")
    candidate_exp = run_experiment(candidate_cfg)

    # 3. Statistical Comparison
    print("\nComputing Paired Bootstrap Resampling (2,000 resamples, 95% CI)…")
    report = compare_experiments(baseline_exp, candidate_exp, bootstrap_n=2000, bootstrap_seed=42)

    print("\n" + "=" * 60)
    print(f"Overall Decision:  {report.overall_decision.value}")
    print(f"Cases Improved:    {len(report.cases_improved)}")
    print(f"Cases Regressed:   {len(report.cases_regressed)}")
    print("=" * 60)

    for m in report.metric_comparisons:
        print(f"\nMetric: {m.metric_name}")
        print(f"  Baseline Mean:  {m.baseline_mean:.4f}")
        print(f"  Candidate Mean: {m.candidate_mean:.4f}")
        rel_str = (
            f" ({m.relative_change_pct:+.2f}%)" if m.relative_change_pct is not None else " (N/A)"
        )
        print(f"  Observed Diff:  {m.absolute_change:+.4f}{rel_str}")
        if m.ci:
            print(f"  95% CI:         [{m.ci.lower:.4f}, {m.ci.upper:.4f}]")
        print(f"  Decision:       {m.decision.value}")

    # Generate Markdown Report
    md = comparison_to_markdown(report)
    print("\nMarkdown Summary Snippet:")
    print("-" * 40)
    print("\n".join(md.split("\n")[:12]))


if __name__ == "__main__":
    main()
