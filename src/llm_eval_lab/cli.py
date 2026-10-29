"""
Command-line interface for LLM Evaluation Lab.

Commands
--------
llm-eval validate <dataset>          Validate a JSONL dataset file.
llm-eval run ...                     Run an evaluation experiment.
llm-eval compare --baseline --candidate  Compare two experiments.
llm-eval report --experiment ...     Generate a report.
llm-eval experiments list            List stored experiments.
llm-eval experiments delete <id>     Delete an experiment.
llm-eval serve                       Start the FastAPI server.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from llm_eval_lab.config import settings
from llm_eval_lab.dataset import load_dataset, validate_dataset
from llm_eval_lab.regression import compare_experiments
from llm_eval_lab.reporting import (
    comparison_to_json,
    comparison_to_markdown,
    experiment_to_csv,
    experiment_to_json,
    experiment_to_markdown,
    save_report,
)
from llm_eval_lab.runner import run_experiment
from llm_eval_lab.schemas import (
    EvaluatorConfig,
    ExperimentConfig,
    PromptTemplate,
    ProviderConfig,
    RegressionConfig,
)
from llm_eval_lab.storage import get_storage

app = typer.Typer(
    name="llm-eval",
    help="LLM Evaluation Lab — Test your prompts. Measure your models. Trust your evidence.",
    add_completion=False,
)
experiments_app = typer.Typer(help="Manage stored experiments.")
app.add_typer(experiments_app, name="experiments")

console = Console()
err_console = Console(stderr=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)


# ── validate ──────────────────────────────────────────────────────────────────

@app.command()
def validate(
    dataset: Path = typer.Argument(..., help="Path to the JSONL dataset file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all valid case IDs."),
) -> None:
    """Validate a JSONL evaluation dataset and report any errors."""
    if not dataset.exists():
        err_console.print(f"[red]File not found:[/red] {dataset}")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task("Validating dataset…", total=None)
        result = load_dataset(dataset)

    if result.errors:
        console.print(Panel(f"[yellow]Found {len(result.errors)} issue(s):[/yellow]", expand=False))
        for err in result.errors:
            console.print(f"  [yellow]•[/yellow] {err}")
    else:
        console.print("[green]✓[/green] No parse errors.")

    console.print(
        f"\n[bold]Valid cases:[/bold] {len(result.cases)}  "
        f"[bold]Invalid:[/bold] {len(result.errors)}"
    )

    if verbose and result.cases:
        console.print("\n[dim]Valid case IDs:[/dim]")
        for c in result.cases:
            console.print(f"  {c.id}")

    raise typer.Exit(0 if result.is_valid else 1)


# ── run ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to JSONL dataset."),
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider name: mock | openai_compatible."),
    model: str = typer.Option("", "--model", "-m", help="Model identifier (optional for mock)."),
    name: str = typer.Option("experiment", "--name", "-n", help="Human-readable experiment name."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to write reports."),
    format: str = typer.Option("markdown", "--format", "-f", help="Report format: markdown | json | csv."),
    evaluators: Optional[str] = typer.Option(None, "--evaluators", help="Comma-separated evaluator names (e.g. exact_match,token_f1)."),
    exact_match: bool = typer.Option(True, "--exact-match/--no-exact-match"),
    token_f1: bool = typer.Option(True, "--token-f1/--no-token-f1"),
    json_valid: bool = typer.Option(False, "--json-validity/--no-json-validity"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist experiment to local database."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run an evaluation experiment against a dataset."""
    if not dataset.exists():
        err_console.print(f"[red]Dataset not found:[/red] {dataset}")
        raise typer.Exit(1)

    provider_cfg = ProviderConfig(name=provider, model=model)
    if evaluators:
        ev_list = [e.strip() for e in evaluators.split(",") if e.strip()]
        eval_cfg = EvaluatorConfig(
            exact_match="exact_match" in ev_list,
            normalized_exact_match="normalized_exact_match" in ev_list,
            token_overlap_f1="token_overlap_f1" in ev_list or "token_f1" in ev_list,
            json_validity="json_validity" in ev_list or "json_valid" in ev_list,
            keyword_coverage="keyword_coverage" in ev_list,
            semantic_similarity="semantic_similarity" in ev_list,
            judge="judge" in ev_list,
        )
    else:
        eval_cfg = EvaluatorConfig(
            exact_match=exact_match,
            normalized_exact_match=True,
            token_overlap_f1=token_f1,
            json_validity=json_valid,
        )
    config = ExperimentConfig(
        name=name,
        dataset_path=str(dataset),
        provider=provider_cfg,
        evaluators=eval_cfg,
    )

    case_count = [0]

    def progress_cb(current: int, total: int, case_id: str) -> None:
        case_count[0] = current
        if verbose:
            console.print(f"  [{current}/{total}] {case_id}")

    console.print(f"[bold]Running experiment[/bold] [cyan]{name}[/cyan] …")
    console.print(f"  Provider: [cyan]{provider}[/cyan]  Model: [cyan]{model or '(default)'}[/cyan]")
    console.print(f"  Dataset: [dim]{dataset}[/dim]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=not verbose) as prog:
        task = prog.add_task("Evaluating test cases…", total=None)
        experiment = run_experiment(config, progress_callback=progress_cb)
        prog.update(task, completed=True)

    s = experiment.summary
    status_color = "green" if experiment.status.value == "completed" else "red"
    console.print(
        f"\n[{status_color}]Status:[/{status_color}] {experiment.status.value}  "
        f"| Success rate: [bold]{s.success_rate * 100:.1f}%[/bold]  "
        f"| Cases: {s.total_cases}  "
        f"| Mean latency: {s.mean_latency_ms:.1f}ms" if s.mean_latency_ms else ""
    )
    console.print(f"  Experiment ID: [dim]{experiment.experiment_id}[/dim]")

    if s.aggregate_metrics:
        tbl = Table(title="Aggregate Metrics", show_header=True)
        tbl.add_column("Metric")
        tbl.add_column("Mean Score", justify="right")
        for mname, score in sorted(s.aggregate_metrics.items()):
            tbl.add_row(mname, f"{score:.4f}")
        console.print(tbl)

    # Save to storage
    if save:
        storage = get_storage()
        storage.save_experiment(experiment)
        console.print(f"[dim]Saved to database.[/dim]")

    # Write report
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        eid = experiment.experiment_id[:8]
        if format == "json":
            content = experiment_to_json(experiment)
            ext = "json"
        elif format == "csv":
            content = experiment_to_csv(experiment)
            ext = "csv"
        else:
            content = experiment_to_markdown(experiment)
            ext = "md"
        report_path = output_dir / f"experiment_{eid}.{ext}"
        save_report(content, report_path)
        console.print(f"Report saved: [cyan]{report_path}[/cyan]")

    raise typer.Exit(0 if experiment.status.value == "completed" else 1)


# ── compare ───────────────────────────────────────────────────────────────────

@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline experiment ID."),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate experiment ID."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown | json"),
    bootstrap_n: int = typer.Option(settings.bootstrap_n, "--bootstrap-n"),
    seed: int = typer.Option(settings.bootstrap_seed, "--seed"),
    confidence: float = typer.Option(settings.confidence_level, "--confidence"),
) -> None:
    """Statistically compare a baseline and a candidate experiment."""
    storage = get_storage()

    baseline_exp = storage.load_experiment(baseline)
    if baseline_exp is None:
        # Try prefix match
        all_exps = storage.list_experiments(200)
        matches = [e for e in all_exps if e["id"].startswith(baseline)]
        if len(matches) == 1:
            baseline_exp = storage.load_experiment(matches[0]["id"])
        else:
            err_console.print(f"[red]Baseline experiment not found:[/red] {baseline}")
            raise typer.Exit(1)

    candidate_exp = storage.load_experiment(candidate)
    if candidate_exp is None:
        all_exps = storage.list_experiments(200)
        matches = [e for e in all_exps if e["id"].startswith(candidate)]
        if len(matches) == 1:
            candidate_exp = storage.load_experiment(matches[0]["id"])
        else:
            err_console.print(f"[red]Candidate experiment not found:[/red] {candidate}")
            raise typer.Exit(1)

    console.print("[bold]Comparing experiments…[/bold]")
    report = compare_experiments(
        baseline_exp,
        candidate_exp,
        regression_config=RegressionConfig(),
        bootstrap_n=bootstrap_n,
        bootstrap_seed=seed,
        confidence_level=confidence,
    )

    storage.save_comparison(report)

    # Print summary
    decision_colors = {
        "IMPROVED": "green",
        "REGRESSED": "red",
        "INCONCLUSIVE": "yellow",
        "INSUFFICIENT_DATA": "dim",
    }
    color = decision_colors.get(report.overall_decision.value, "white")
    console.print(
        f"\nOverall Decision: [{color}][bold]{report.overall_decision.value}[/bold][/{color}]"
    )

    tbl = Table(title="Per-Metric Results", show_header=True)
    for col in ["Metric", "Baseline", "Candidate", "Δ", "CI", "Decision", "n"]:
        tbl.add_column(col)

    for mc in report.metric_comparisons:
        ci_str = (
            f"[{mc.ci.lower:.4f}, {mc.ci.upper:.4f}]"
            if mc.ci else "N/A"
        )
        d_color = decision_colors.get(mc.decision.value, "white")
        tbl.add_row(
            mc.metric_name,
            f"{mc.baseline_mean:.4f}" if mc.baseline_mean is not None else "N/A",
            f"{mc.candidate_mean:.4f}" if mc.candidate_mean is not None else "N/A",
            f"{mc.absolute_change:+.4f}" if mc.absolute_change is not None else "N/A",
            ci_str,
            f"[{d_color}]{mc.decision.value}[/{d_color}]",
            str(mc.n_paired),
        )
    console.print(tbl)
    console.print(f"  Comparison ID: [dim]{report.comparison_id}[/dim]")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if format == "json":
            content = comparison_to_json(report)
            ext = "json"
        else:
            content = comparison_to_markdown(report)
            ext = "md"
        cid = report.comparison_id[:8]
        report_path = output_dir / f"comparison_{cid}.{ext}"
        save_report(content, report_path)
        console.print(f"Report saved: [cyan]{report_path}[/cyan]")

    exit_code = 1 if report.overall_decision.value == "REGRESSED" else 0
    raise typer.Exit(exit_code)


# ── report ────────────────────────────────────────────────────────────────────

@app.command()
def report(
    experiment_id: str = typer.Option(..., "--experiment", "-e", help="Experiment ID."),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown | json | csv"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
) -> None:
    """Generate a report for a stored experiment."""
    storage = get_storage()
    exp = storage.load_experiment(experiment_id)
    if exp is None:
        # prefix match
        matches = [e for e in storage.list_experiments(200) if e["id"].startswith(experiment_id)]
        if len(matches) == 1:
            exp = storage.load_experiment(matches[0]["id"])
        else:
            err_console.print(f"[red]Experiment not found:[/red] {experiment_id}")
            raise typer.Exit(1)

    if format == "json":
        content = experiment_to_json(exp)
    elif format == "csv":
        content = experiment_to_csv(exp)
    else:
        content = experiment_to_markdown(exp)

    if output:
        save_report(content, output)
        console.print(f"Report saved: [cyan]{output}[/cyan]")
    else:
        console.print(content)


# ── experiments subcommand ────────────────────────────────────────────────────

@experiments_app.command("list")
def experiments_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of experiments to show."),
) -> None:
    """List stored experiments."""
    storage = get_storage()
    rows = storage.list_experiments(limit)

    if not rows:
        console.print("[dim]No experiments found. Run 'llm-eval run' to create one.[/dim]")
        return

    tbl = Table(title=f"Experiments (latest {limit})", show_header=True)
    for col in ["ID (prefix)", "Name", "Status", "Completed"]:
        tbl.add_column(col)

    for row in rows:
        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(
            row["status"], "white"
        )
        tbl.add_row(
            row["id"][:8],
            row["name"],
            f"[{status_color}]{row['status']}[/{status_color}]",
            row["completed_at"] or "—",
        )
    console.print(tbl)


@experiments_app.command("delete")
def experiments_delete(
    experiment_id: str = typer.Argument(..., help="Experiment ID or prefix."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete an experiment from local storage."""
    if not yes:
        typer.confirm(f"Delete experiment {experiment_id}?", abort=True)
    storage = get_storage()
    found = storage.delete_experiment(experiment_id)
    if not found:
        err_console.print(f"[red]Experiment not found:[/red] {experiment_id}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted[/green] {experiment_id}")


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host"),
    port: int = typer.Option(settings.api_port, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development only)."),
) -> None:
    """Start the LLM Evaluation Lab web server."""
    try:
        import uvicorn
        from llm_eval_lab.api.main import create_app
    except ImportError as exc:
        err_console.print(f"[red]Cannot start server:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Starting LLM Evaluation Lab[/bold green] at "
        f"[cyan]http://{host}:{port}[/cyan]"
    )
    uvicorn.run(
        "llm_eval_lab.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
