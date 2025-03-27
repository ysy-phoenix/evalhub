"""View utilities for EvalHub."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_math_results(
    samples: List[Dict[str, Any]],
    max_display: Optional[int] = None,
    show_response: bool = False,
    task_ids: Optional[List[str]] = None,
    log_to_file: bool = False,
    results_path: Optional[Path] = None,
):
    r"""Display math evaluation results."""
    if task_ids:
        samples = [s for s in samples if s["task_id"] in task_ids]

    filtered_samples = [sample for sample in samples if not sample.get("correct", False)]

    if not filtered_samples:
        console.print("[yellow]No results found.[/yellow]")
        return

    if max_display:
        filtered_samples = filtered_samples[:max_display]

    total = len(samples)
    correct_count = sum(1 for s in samples if s.get("correct", False))

    stats_table = Table(title="Results Summary", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Total samples", str(total))
    stats_table.add_row("Correct", f"{correct_count} ({correct_count / total:.2%})")
    stats_table.add_row(
        "Incorrect", f"{total - correct_count} ({(total - correct_count) / total:.2%})"
    )
    stats_table.add_row("Displayed", str(len(filtered_samples)))

    if task_ids:
        stats_table.add_row("Task ID filter", ", ".join(task_ids))

    console.print(stats_table)
    console.print("")
    console.print("[bold blue]Incorrect Results[/bold blue]")

    for i, sample in enumerate(filtered_samples, 1):
        if show_response:
            response_text = sample["response"].strip()
            response_panel = Panel(
                response_text,
                title=f"Response #{i} ({sample.get('task_id', 'Unknown task')})",
                border_style="blue",
                width=100,
            )
            console.print(response_panel)

        comparison = Table(box=None, show_header=False, padding=(0, 2))
        comparison.add_column("Label")
        comparison.add_column("Value")

        comparison.add_row("Task ID:", sample["task_id"], style="cyan")
        comparison.add_row("Ground Truth:", sample["ground_truth"], style="green")
        comparison.add_row("Extracted:", sample["extracted_answer"], style="red")

        console.print(comparison)
        console.print("[dim]" + "─" * 80 + "[/dim]")

    if log_to_file and results_path:
        with open(results_path.parent / f"{results_path.stem}_failed.log", "w") as f:
            for sample in samples:
                if not sample.get("correct", False):
                    f.write(f"{sample['task_id']}\n")
                    f.write(f"Ground Truth: {sample['ground_truth']}\n")
                    f.write(f"Extracted: {sample['extracted_answer']}\n")
                    if show_response:
                        f.write(f"Response: {sample['response']}\n")
                    f.write("-" * 80 + "\n")


def display_livecodebench_results(
    results_data: Dict[str, Any],
    max_display: Optional[int] = None,
    show_code: bool = False,
    sort_by: str = "pass@1",
    difficulty_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    log_to_file: bool = False,
    results_path: Optional[Path] = None,
):
    r"""Display LiveCodeBench evaluation results."""
    eval_data = results_data.get("eval", {})
    problems = list(eval_data.values())

    if difficulty_filter:
        problems = [p for p in problems if p.get("difficulty") == difficulty_filter]
    if platform_filter:
        problems = [p for p in problems if p.get("platform") == platform_filter]

    # filter for failed results
    problems = [p for p in problems if not p.get("graded_list", [False])[0]]

    if not problems:
        console.print("[yellow]No results found with current filters.[/yellow]")
        return

    if sort_by == "pass@1":
        problems.sort(key=lambda x: x.get("pass@1", 0), reverse=True)
    elif sort_by == "difficulty":
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        problems.sort(key=lambda x: difficulty_order.get(x.get("difficulty"), 0))
    elif sort_by == "platform":
        problems.sort(key=lambda x: x.get("platform", ""))
    elif sort_by == "date":
        problems.sort(key=lambda x: x.get("contest_date", ""))

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

        if show_code and "code_list" in problem:
            code = problem["code_list"][0] if problem["code_list"] else "No code available"
            code_panel = Panel(code, title="Solution Code", border_style="green")
            console.print(code_panel)

        console.print("[dim]" + "─" * 80 + "[/dim]")

    if log_to_file and results_path:
        with open(results_path.parent / f"{results_path.stem}_summary.log", "w") as f:
            f.write("LiveCodeBench Results Summary\n")
            f.write(f"Date: {results_data.get('date', 'Unknown')}\n")
            f.write(f"Overall Pass@1: {results_data.get('pass@1', 0):.2%}\n\n")

            f.write("Problem-wise results:\n")
            for problem in problems:
                f.write(f"Title: {problem.get('question_title', 'Unknown')}\n")
                f.write(f"Pass@1: {problem.get('pass@1', 0):.2%}\n")
                f.write(f"Platform: {problem.get('platform', 'Unknown')}\n")
                f.write(f"Difficulty: {problem.get('difficulty', 'Unknown')}\n")
                f.write("-" * 80 + "\n")


def view_results(
    results_path: Path,
    max_display: Optional[int] = None,
    show_response: bool = False,
    task_ids: Optional[str] = None,
    sort_by: str = "pass@1",
    difficulty: Optional[str] = None,
    platform: Optional[str] = None,
    log_to_file: bool = False,
):
    r"""Unified view function that handles different result formats."""
    if results_path.suffix.lower() == ".json":
        try:
            with open(results_path) as f:
                results_data = json.load(f)

            if "eval" in results_data and "pass@1" in results_data:
                console.print("[blue]Detected LiveCodeBench results format[/blue]")
                display_livecodebench_results(
                    results_data,
                    max_display=max_display,
                    show_code=show_response,
                    sort_by=sort_by,
                    difficulty_filter=difficulty,
                    platform_filter=platform,
                    log_to_file=log_to_file,
                    results_path=results_path,
                )
            else:
                console.print(
                    "[yellow]Unknown JSON format, trying to parse as general results[/yellow]"
                )
                samples = [results_data]
                display_math_results(
                    samples,
                    max_display=max_display,
                    show_response=show_response,
                    task_ids=task_ids.split(",") if task_ids else None,
                    log_to_file=log_to_file,
                    results_path=results_path,
                )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console.print(f"[bold red]Error loading JSON results file:[/bold red] {e}")
            return
    else:
        try:
            with open(results_path) as f:
                samples = [orjson.loads(line) for line in f]

            console.print("[blue]Detected JSONL format (math evaluation results)[/blue]")
            display_math_results(
                samples,
                max_display=max_display,
                show_response=show_response,
                task_ids=task_ids.split(",") if task_ids else None,
                log_to_file=log_to_file,
                results_path=results_path,
            )
        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            console.print(f"[bold red]Error loading JSONL results file:[/bold red] {e}")
            return
