"""Command-line interface for EvalHub."""

import click
from rich.console import Console

from src.gen import gen

console = Console()


@click.group()
@click.version_option()
def cli():
    """EvalHub - All-in-one benchmarking platform for evaluating LLMs."""
    pass


@cli.command()
@click.option("--model", default="gpt-3.5-turbo", help="Model to evaluate")
@click.option("--task", default="math::gsm8k", help="Task to evaluate on")
@click.option("--output-dir", default="results", help="Output directory")
def run(model, task, output_dir):
    """Run evaluation on a model with specified dataset."""
    console.print(f"[bold green]Running evaluation on {model} with {task} task[/bold green]")
    gen(model, task, output_dir)


@cli.command()
def results():
    """View evaluation results."""
    console.print("[bold blue]Viewing evaluation results[/bold blue]")


def main():
    """Run the CLI entry point for EvalHub."""
    console.print("[bold yellow]Welcome to EvalHub - LLM Evaluation Platform[/bold yellow]")
    cli()


if __name__ == "__main__":
    main()
