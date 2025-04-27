"""View utilities for EvalHub."""

from pathlib import Path
from typing import Optional

import orjson
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def display_math_results(
    results_path: Path,
    max_display: Optional[int] = -1,
):
    r"""Display math evaluation results."""
    with open(results_path, "rb") as f:
        for i, line in enumerate(f):
            if i >= max_display:
                break
            result = orjson.loads(line)
            rprint(
                Panel.fit(
                    f"[bold]Ground Truth:[/]\n{result['ground_truth']}",
                    title="Reference Answers",
                    border_style="green",
                )
            )

            table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Pass@k", str(result["pass_at_k"]))
            if "majority_vote" in result:
                table.add_row("Majority Vote", str(result["majority_vote"]))
            if "is_correct_majority" in result:
                table.add_row("Is Correct Majority", str(result["is_correct_majority"]))

            rprint(table)

            rprint(
                Panel.fit(
                    f"[bold]Extracted Answers:[/]\n{result['extracted_answers']}",
                    title="Model Answers",
                    border_style="blue",
                )
            )

            rprint("[yellow]" + "─" * 80 + "[/yellow]\n")


def display_livecodebench_results(
    results_path: Path,
    max_display: Optional[int] = -1,
):
    r"""Display LiveCodeBench evaluation results."""
    with open(results_path, "rb") as f:
        results_data = orjson.loads(f.read())
    eval_data = results_data.get("eval", {})
    problems = list(eval_data.values())

    if not problems:
        console.print("[yellow]No results found with current filters.[/yellow]")
        return

    if max_display:
        problems = problems[:max_display]

    stats_table = Table(title="LiveCodeBench Results Summary", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Date", results_data.get("date", "Unknown"))
    stats_table.add_row("Pass@1", f"{results_data.get('pass@1', 0):.2%}")
    stats_table.add_row("Total Problems", str(len(eval_data)))
    stats_table.add_row("Displayed", str(len(problems)))

    detail_pass = results_data.get("detail_pass@1", {})
    for difficulty, rate in detail_pass.items():
        stats_table.add_row(f"{difficulty.title()} Pass@1", f"{rate:.2%}")

    console.print(stats_table)
    console.print("")
    console.print("[bold blue]Problem Results[/bold blue]")

    for i, problem in enumerate(problems, 1):
        title = problem.get("question_title", "Unknown")
        platform = problem.get("platform", "Unknown")
        question_id = problem.get("question_id", "Unknown")
        difficulty = problem.get("difficulty", "Unknown")
        pass_rate = problem.get("pass@1", 0)

        problem_panel = Panel(
            f"[bold]{title}[/bold]\n\n"
            f"Pass@1: {pass_rate:.2%}\n"
            f"Platform: {platform}\n"
            f"Difficulty: {difficulty.title()}\n"
            f"ID: {question_id}\n",
            title=f"Problem #{i}",
            border_style="blue" if pass_rate > 0 else "red",
        )
        console.print(problem_panel)

        if "code_list" in problem:
            code = problem["code_list"][0] if problem["code_list"] else "No code available"
            syntax = Syntax(
                code, lexer="python", theme="monokai", line_numbers=True, indent_guides=True
            )
            code_panel = Panel(syntax, title="Solution Code", border_style="green")
            console.print(code_panel)

        console.print("[dim]" + "─" * 80 + "[/dim]")


def view_results(
    results_path: Path,
    max_display: Optional[int] = -1,
):
    r"""Unified view function that handles different result formats."""
    if results_path.suffix.lower() == ".json":
        try:
            console.print("[blue]Detected LiveCodeBench results format[/blue]")
            display_livecodebench_results(
                results_path=results_path,
                max_display=max_display,
            )
        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            console.print(f"[bold red]Error loading JSON results file:[/bold red] {e}")
            return
    else:
        try:
            console.print("[blue]Detected JSONL format (math evaluation results)[/blue]")
            display_math_results(
                results_path=results_path,
                max_display=max_display,
            )
        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            console.print(f"[bold red]Error loading JSONL results file:[/bold red] {e}")
            return
