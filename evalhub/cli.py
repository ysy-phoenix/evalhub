"""Command-line interface for EvalHub."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalhub.benchmarks import DATASET_HUB, DATASET_MAP, EVALUATE_DATASETS, THIRD_PARTY_DATASETS
from evalhub.benchmarks.base import Dataset
from evalhub.gen import generate
from evalhub.inference.schemas import GenerationConfig
from evalhub.utils.typer import options
from evalhub.view import view_results

console = Console()

# Create the main Typer app
app = typer.Typer(
    name="evalhub",
    help="EvalHub - All-in-one benchmarking platform for evaluating LLMs.",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
@options(GenerationConfig)
def gen(
    config: GenerationConfig,
    *,
    override_args: Annotated[str | None, typer.Option(help="Override dataset arguments in json string format")] = None,
):
    r"""Run generation on a model with specified dataset."""
    console.print(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for task in config.tasks:
        generate(config=config, task=task, override_args=override_args)


@app.command()
def eval(
    tasks: Annotated[str, typer.Option(help="Tasks to evaluate on, separated by commas")],
    solutions: Annotated[str, typer.Option(help="Solutions to evaluate on, separated by commas")],
    output_dir: Annotated[str, typer.Option(help="Output directory")],
    override_args: Annotated[str | None, typer.Option(help="Override dataset arguments in json string format")] = None,
):
    r"""Evaluate the model on the tasks."""
    tasks = [task.strip().lower() for task in tasks.split(",")]
    solutions = [solution.strip() for solution in solutions.split(",")]
    assert len(tasks) == len(solutions), "Number of tasks and solutions must be the same"
    for task, solution in zip(tasks, solutions, strict=False):
        assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
        dataset: Dataset = DATASET_MAP[task](name=task.lower(), override_args=override_args)
        dataset.evaluate(solution, output_dir)


@app.command()
def view(
    results: Annotated[str, typer.Option(help="Results file path")],
    max_display: Annotated[int, typer.Option(help="Maximum number of samples to display")] = -1,
    false_only: Annotated[bool, typer.Option(help="Only display false samples")] = True,
):
    r"""View and analyze evaluation results with rich formatting.

    Automatically detects the result format:
    - JSONL files: Math evaluation results (GSM8K, etc.)
    - JSON files: LiveCodeBench results
    """
    view_results(
        results_path=Path(results),
        max_display=max_display,
        false_only=false_only,
    )


@app.command(name="tasks")
def list_tasks():
    r"""List all supported tasks and evaluable tasks."""
    # Create a table for all tasks
    task_table = Table(title="EvalHub Supported Tasks")

    task_table.add_column("Task", style="cyan")
    task_table.add_column("Evaluable", style="green")
    task_table.add_column("Huggingface", style="blue")

    # Sort tasks alphabetically for better readability
    sorted_tasks = sorted(DATASET_MAP.keys())

    for task in sorted_tasks:
        evaluable = "✅" if task in EVALUATE_DATASETS else "❌(Third-party)"
        hf_name = DATASET_HUB[task]
        task_table.add_row(task, evaluable, hf_name)

    console.print(task_table)

    # Print usage examples
    console.print("\n[bold yellow]Generation Examples:[/bold yellow]")
    console.print('evalhub gen --model "Qwen2.5-7B-Instruct" --tasks humaneval,mbpp --output-dir ./results')

    console.print("\n[bold yellow]Evaluation Examples:[/bold yellow]")
    for task in list(DATASET_MAP.keys())[:6]:
        if task == "bigcodebench":
            continue
        if task in THIRD_PARTY_DATASETS:
            console.print(f"evalplus.evaluate --dataset {task} --samples ./results/{task}.jsonl")
        else:
            assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
            console.print(f"evalhub eval --tasks {task} --solutions ./results/{task}.jsonl --output-dir ./results")


def main():
    r"""Run the CLI entry point for EvalHub."""
    app()


if __name__ == "__main__":
    main()
