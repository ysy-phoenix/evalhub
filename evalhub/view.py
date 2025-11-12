"""View utilities for EvalHub."""

import math
import statistics
from pathlib import Path

import orjson
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from evalhub.utils import console, cprint


def truncate_text(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return "Truncated"


def display_math_results(
    results_path: Path,
    max_display: int | None = -1,
    false_only: bool = True,
):
    r"""Display math evaluation results."""
    if max_display < 0:
        max_display = float("inf")
    cnt = 0

    with open(results_path, "rb") as f:
        for line in f:
            result = orjson.loads(line)
            if false_only and result.get("pass_at_k", {}).get("1", 0) > 0.0:
                continue

            table1 = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
            table1.add_column("Metric", style="cyan")
            table1.add_column("Value", style="green")

            table1.add_row("Pass@k", str(result["pass_at_k"]))
            table1.add_row("Ground Truth", result["ground_truth"])
            if "majority_vote" in result:
                table1.add_row("Majority Vote", str(result["majority_vote"]))
            if "is_correct_majority" in result:
                table1.add_row("Is Correct Majority", str(result["is_correct_majority"]))
            cprint(table1)

            solutions = list(map(truncate_text, result.get("solutions", [])))
            correct = result.get("correct", [])
            median_len = max(int(statistics.median(len(x) for x in solutions)) + 3, 13)
            cols = max(1, min(len(solutions), console.size.width // median_len))

            table2 = Table(title="Solutions", show_header=False, expand=True)
            for _ in range(cols):
                table2.add_column(width=median_len)

            for i in range(math.ceil(len(solutions) / cols)):
                row = []
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(solutions):
                        content = Text(solutions[idx])
                        content.stylize("green") if correct[idx] else content.stylize("red")
                        row.append(content)
                    else:
                        row.append("")
                table2.add_row(*row)
            cprint(table2)

            cprint("[yellow]" + "─" * console.size.width + "[/yellow]\n")
            cnt += 1
            if cnt >= max_display:
                break


def display_livecodebench_results(
    results_path: Path,
    max_display: int | None = -1,
    false_only: bool = True,
):
    r"""Display LiveCodeBench evaluation results."""
    with open(results_path, "rb") as f:
        results_data = orjson.loads(f.read())
    eval_data = results_data.get("eval", {})
    problems = list(eval_data.values())

    if not problems:
        cprint("[yellow]No results found with current filters.[/yellow]")
        return

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

    cprint(stats_table)
    cprint("")
    cprint("[bold blue]Problem Results[/bold blue]")

    cnt = 0
    for i, problem in enumerate(problems, 1):
        title = problem.get("question_title", "Unknown")
        platform = problem.get("platform", "Unknown")
        question_id = problem.get("question_id", "Unknown")
        difficulty = problem.get("difficulty", "Unknown")
        pass_rate = problem.get("pass@1", 0)

        if false_only and pass_rate > 0:
            continue

        problem_panel = Panel(
            f"[bold]{title}[/bold]\n\n"
            f"Pass@1: {pass_rate:.2%}\n"
            f"Platform: {platform}\n"
            f"Difficulty: {difficulty.title()}\n"
            f"ID: {question_id}\n",
            title=f"Problem #{i}",
            border_style="blue" if pass_rate > 0 else "red",
        )
        cprint(problem_panel)

        if "code_list" in problem:
            code = problem["code_list"][0] if problem["code_list"] else "No code available"
            syntax = Syntax(code, lexer="python", theme="monokai", line_numbers=True, indent_guides=True)
            code_panel = Panel(syntax, title="Solution Code", border_style="green")
            cprint(code_panel)

        cprint("[dim]" + "─" * 80 + "[/dim]")
        cnt += 1
        if cnt >= max_display:
            break


def view_results(
    results_path: Path,
    max_display: int | None = -1,
    false_only: bool = True,
):
    r"""Unified view function that handles different result formats."""
    if results_path.suffix.lower() == ".json":
        try:
            cprint("[blue]Detected LiveCodeBench results format[/blue]")
            display_livecodebench_results(
                results_path=results_path,
                max_display=max_display,
                false_only=false_only,
            )
        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            cprint(f"[bold red]Error loading JSON results file:[/bold red] {e}")
            return
    else:
        try:
            cprint("[blue]Detected JSONL format (math evaluation results)[/blue]")
            display_math_results(
                results_path=results_path,
                max_display=max_display,
                false_only=false_only,
            )
        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            cprint(f"[bold red]Error loading JSONL results file:[/bold red] {e}")
            return
