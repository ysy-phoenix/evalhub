"""Command-line interface for EvalHub."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.benchmarks import DATASET_MAP, EVALUATE_DATASETS, THIRD_PARTY_DATASETS
from src.benchmarks.config import DATASET_HUB
from src.gen import gen, parse_sampling_params
from src.inference.utils import GenerationConfig
from src.view import view_results

console = Console()


@click.group()
@click.version_option()
def cli():
    """EvalHub - All-in-one benchmarking platform for evaluating LLMs."""
    pass


@cli.command()
@click.option("--model", required=True, help="Model to evaluate")
@click.option("--tasks", required=True, help="Tasks to evaluate on, separated by commas")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--system-prompt", help="System prompt for the model")
# Advanced: support for arbitrary model parameters
@click.option(
    "--sampling-param",
    "-p",
    multiple=True,
    help="Additional sampling parameters in format: key=value",
)
def run(model, tasks, output_dir, sampling_param, system_prompt):
    r"""Run evaluation on a model with specified dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sampling_params = parse_sampling_params(sampling_param)
    console.print("[blue]Setting sampling parameters:[/blue]")
    for key, value in sampling_params.items():
        console.print(f"  [cyan]{key}:[/cyan] {value}")

    # Execute generation with model parameters
    for task in tasks.split(","):
        console.print(f"[bold green]Running evaluation on {task} task[/bold green]")
        gen(model, task, output_dir, sampling_params, system_prompt)


@cli.command()
@click.option("--tasks", required=True, help="Tasks to evaluate on, separated by commas")
@click.option("--solutions", required=True, help="Solutions to evaluate on, separated by commas")
@click.option("--output-dir", required=True, help="Output directory")
def eval(tasks, solutions, output_dir):
    r"""Evaluate the model on the tasks."""
    tasks = tasks.split(",")
    solutions = solutions.split(",")
    assert len(tasks) == len(solutions), "Number of tasks and solutions must be the same"
    for task, solution in zip(tasks, solutions):
        console.print(f"[bold green]Running evaluation on {task} task[/bold green]")
        assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
        dataset = DATASET_MAP[task](name=task)
        dataset.evaluate(solution, output_dir)


@cli.command()
@click.option("--results", required=True, help="Results file path")
@click.option("--max-display", type=int, default=None, help="Maximum number of samples to display")
def view(results, max_display):
    r"""View and analyze evaluation results with rich formatting.

    Automatically detects the result format:
    - JSONL files: Math evaluation results (GSM8K, etc.)
    - JSON files: LiveCodeBench results
    """

    view_results(
        results_path=Path(results),
        max_display=max_display,
    )


@cli.command(name="configs")
def list_configs():
    """List all available configuration options with defaults."""
    table = Table(title="EvalHub Configuration Options")

    table.add_column("Parameter", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="magenta")

    config_descriptions = {
        "is_chat": "Whether to use chat format for prompts",
        "model_name": "Name or path of the model to use",
        "temperature": "Sampling temperature (higher = more random)",
        "top_p": "Top-p sampling parameter (nucleus sampling)",
        "max_tokens": "Maximum number of tokens to generate",
        "frequency_penalty": "Penalty for repeating tokens (higher = less repetition)",
        "presence_penalty": "Penalty for new tokens (higher = less new topics)",
        "n_samples": "Number of samples to generate per prompt",
        "num_workers": "Number of parallel workers for generation",
        "timeout": "API request timeout in seconds",
        "stop": "List of sequences where generation should stop",
        "base_url": "Base URL for API endpoint",
        "api_key": "API key for model access",
        "output_dir": "Directory to save generation outputs",
        "baseline": "Whether to run as baseline evaluation",
    }

    # Get default values from GenerationConfig
    default_config = GenerationConfig()
    # Add rows to the table
    for field_name, field_value in vars(default_config).items():
        field_type = type(field_value).__name__
        description = config_descriptions.get(field_name, "No description available")

        # Format default value for display
        if field_type == "NoneType":
            default_str = "None"
        elif field_type == "bool":
            default_str = str(field_value)
        elif field_type == "str":
            default_str = f'"{field_value}"'
        elif field_type == "list" and not field_value:
            default_str = "[]"
        else:
            default_str = str(field_value)

        table.add_row(field_name, field_type, default_str, description)

    console.print(table)

    console.print("\n[bold yellow]Usage Example:[/bold yellow]")
    console.print(
        'evalhub run --model "Qwen2.5-7B-Instruct" --tasks [humaneval,mbpp] --output-dir ./results '
        "-p temperature=0.2 -p top_p=0.95"
    )


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
    console.print(
        'evalhub run --model "Qwen2.5-7B-Instruct" --tasks humaneval,mbpp --output-dir ./results'
    )

    console.print("\n[bold yellow]Evaluation Examples:[/bold yellow]")
    for task in DATASET_MAP.keys():
        if task in THIRD_PARTY_DATASETS:
            console.print(f"evalplus.evaluate --dataset {task} --samples ./results/{task}.jsonl")
        else:
            assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
            console.print(
                f"evalhub eval --tasks {task} --solutions ./results/{task}.jsonl "
                f"--output-dir ./results"
            )


def main():
    r"""Run the CLI entry point for EvalHub."""
    console.print("[bold yellow]Welcome to EvalHub - LLM Evaluation Platform[/bold yellow]")
    cli()


if __name__ == "__main__":
    main()
