"""Command-line interface for EvalHub."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from evalhub.benchmarks import DATASET_HUB, DATASET_MAP, EVALUATE_DATASETS, THIRD_PARTY_DATASETS
from evalhub.benchmarks.base import Dataset
from evalhub.gen import gen
from evalhub.inference.schemas import GenerationConfig
from evalhub.utils.click import _extract_field_info, options
from evalhub.view import view_results

console = Console()


@click.group()
@click.version_option()
def cli():
    """EvalHub - All-in-one benchmarking platform for evaluating LLMs."""
    pass


@cli.command()
@options(GenerationConfig)
def run(config: GenerationConfig):
    r"""Run evaluation on a model with specified dataset."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for task in config.tasks:
        console.print(f"[bold green]Running evaluation on {task} task[/bold green]")
        gen(config=config, task=task)


@cli.command()
@click.option("--tasks", required=True, help="Tasks to evaluate on, separated by commas")
@click.option("--solutions", required=True, help="Solutions to evaluate on, separated by commas")
@click.option("--output-dir", required=True, help="Output directory")
def eval(tasks: str, solutions: str, output_dir: str):
    r"""Evaluate the model on the tasks."""
    tasks = [task.strip().lower() for task in tasks.split(",")]
    solutions = [solution.strip().lower() for solution in solutions.split(",")]
    assert len(tasks) == len(solutions), "Number of tasks and solutions must be the same"
    for task, solution in zip(tasks, solutions, strict=False):
        console.print(f"[bold green]Running evaluation on {task} task[/bold green]")
        assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
        dataset: Dataset = DATASET_MAP[task](name=task.lower())
        dataset.evaluate(solution, output_dir)


@cli.command()
@click.option("--results", required=True, help="Results file path")
@click.option("--max-display", type=int, default=-1, help="Maximum number of samples to display")
@click.option("--false-only", type=bool, default=True, help="Only display false samples")
def view(results: str, max_display: int, false_only: bool):
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


@cli.command(name="configs")
def list_configs():
    """List all available configuration options with defaults."""
    table = Table(title="EvalHub Configuration Options")

    table.add_column("Parameter", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Required", style="red")
    table.add_column("Description", style="magenta")

    # Extract field information automatically from dataclass metadata
    fields_info = _extract_field_info(GenerationConfig)

    # Sort fields by required status first, then alphabetically
    fields_info.sort(key=lambda x: (not x["required"], x["name"]))

    for field_info in fields_info:
        # Format default value for display
        default_value = field_info["default"]
        match default_value:
            case str() | Path():
                default_str = f'"{default_value}"'
            case list() if not default_value:
                default_str = "[]"
            case _:
                default_str = str(default_value)

        required_str = "✅" if field_info["required"] else "❌"
        table.add_row(field_info["name"], field_info["type"], default_str, required_str, field_info["help"])

    console.print(table)

    console.print("\n[bold yellow]Usage Examples:[/bold yellow]")
    console.print('evalhub run --model "Qwen2.5-7B-Instruct" --tasks humaneval,mbpp --output-dir "./results"')


@cli.command(name="tasks")
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
    console.print('evalhub run --model "Qwen2.5-7B-Instruct" --tasks humaneval,mbpp --output-dir ./results')

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
    console.print("[bold yellow]Welcome to EvalHub - LLM Evaluation Platform[/bold yellow]")
    cli()


if __name__ == "__main__":
    main()
