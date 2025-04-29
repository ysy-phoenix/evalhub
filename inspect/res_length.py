from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from rich.console import Console
from rich.progress import track
from rich.table import Table

CODE_TASKS = ["humaneval", "mbpp", "livecodebench"]
MATH_TASKS = ["aime2024", "math500", "gpqa"]
INCLUDED_TASKS = CODE_TASKS + MATH_TASKS


def get_stats_for_lengths(lengths: list[int]) -> dict[str, Any]:
    lengths = np.array(lengths)
    return {
        "count": len(lengths),
        "min": float(np.min(lengths) / 1024),
        "max": float(np.max(lengths) / 1024),
        "mean": float(np.mean(lengths) / 1024),
        "median": float(np.median(lengths) / 1024),
        "std": float(np.std(lengths) / 1024),
        "percentiles": {
            "25": float(np.percentile(lengths, 25) / 1024),
            "50": float(np.percentile(lengths, 50) / 1024),
            "75": float(np.percentile(lengths, 75) / 1024),
            "95": float(np.percentile(lengths, 95) / 1024),
        },
    }


def analyze_length_distribution(dir_path: Path, show_progress: bool = True):
    console = Console()
    task_lengths = defaultdict(list)
    all_lengths = []

    for task in INCLUDED_TASKS:
        file_path = dir_path / f"{task}_raw.jsonl"
        try:
            with open(file_path, "rb") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        if show_progress:
            iter_lines = track(lines, description=f"[cyan]Analyzing {task} solutions...")
        else:
            iter_lines = lines

        for line in iter_lines:
            data = orjson.loads(line)
            length = len(data["response"].get("content", "")) + len(
                data["response"].get("reasoning_content", "")
            )
            task_lengths[task].append(length)
            all_lengths.append(length)

    # Calculate stats for each task and overall
    stats = {task: get_stats_for_lengths(lengths) for task, lengths in task_lengths.items()}
    stats["overall"] = get_stats_for_lengths(all_lengths)

    # Create table with all tasks
    table = Table(
        title="Solution Length Distribution by Task", show_header=True, header_style="bold magenta"
    )
    table.add_column("Statistic", style="dim")
    for task in list(task_lengths.keys()) + ["overall"]:
        table.add_column(task.upper(), justify="right")

    # Add rows for each statistic
    metrics = [
        ("Total Samples", "count", "integer", ""),
        ("Minimum Length", "min", "float", "K"),
        ("Maximum Length", "max", "float", "K"),
        ("Mean Length", "mean", "float", "K"),
        ("Median Length", "median", "float", "K"),
        ("Std Length", "std", "float", "K"),
        ("25th Percentile", ("percentiles", "25"), "float", "K"),
        ("50th Percentile", ("percentiles", "50"), "float", "K"),
        ("75th Percentile", ("percentiles", "75"), "float", "K"),
        ("95th Percentile", ("percentiles", "95"), "float", "K"),
    ]

    for label, key, value_type, suffix in metrics:
        row = [label]
        for task in list(task_lengths.keys()) + ["overall"]:
            value = stats[task][key] if isinstance(key, str) else stats[task][key[0]][key[1]]
            if value_type == "integer":
                formatted_value = f"{value:,d}"
            else:  # float
                formatted_value = f"{value:,.2f}"
            row.append(f"{formatted_value}{suffix}")
        table.add_row(*row)

    console.print(table)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze the length distribution of solutions in a JSONL file."
    )
    parser.add_argument("--dir", type=str, help="Path to the directory containing the JSONL files.")
    args = parser.parse_args()

    if args.dir:
        stats = analyze_length_distribution(Path(args.dir))
    else:
        print("Please provide a directory path using --dir argument")
