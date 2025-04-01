from pathlib import Path

import numpy as np
import orjson
from rich.console import Console
from rich.progress import track
from rich.table import Table

INCLUDED_TASKS = ["humaneval", "mbpp", "livecodebench"]
INCLUDED_TASKS += [f"{task}-raw" for task in INCLUDED_TASKS]


def get_stats_for_lengths(lengths):
    lengths = np.array(lengths)
    return {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "percentiles": {
            "25": float(np.percentile(lengths, 25)),
            "50": float(np.percentile(lengths, 50)),
            "75": float(np.percentile(lengths, 75)),
            "95": float(np.percentile(lengths, 95)),
        },
    }


def analyze_length_distribution(dir_path: Path, show_progress: bool = True):
    console = Console()
    task_lengths = {task: [] for task in INCLUDED_TASKS}
    all_lengths = []

    for task in INCLUDED_TASKS:
        file_path = dir_path / f"{task}.jsonl"
        with open(file_path, "rb") as f:
            lines = f.readlines()

        if show_progress:
            iter_lines = track(lines, description=f"[cyan]Analyzing {task} solutions...")
        else:
            iter_lines = lines

        for line in iter_lines:
            data = orjson.loads(line)
            length = len(data["solution"])
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
    for task in INCLUDED_TASKS + ["overall"]:
        table.add_column(task.upper(), justify="right")

    # Add rows for each statistic
    metrics = [
        ("Total Samples", "count", ":,"),
        ("Minimum Length", "min", ":,"),
        ("Maximum Length", "max", ":,"),
        ("Mean Length", "mean", ":,.2f"),
        ("Median Length", "median", ":,.2f"),
        ("Standard Deviation", "std", ":,.2f"),
        ("25th Percentile", ("percentiles", "25"), ":,.2f"),
        ("50th Percentile", ("percentiles", "50"), ":,.2f"),
        ("75th Percentile", ("percentiles", "75"), ":,.2f"),
        ("95th Percentile", ("percentiles", "95"), ":,.2f"),
    ]

    for label, key, fmt in metrics:
        row = [label]
        for task in INCLUDED_TASKS + ["overall"]:
            value = stats[task][key] if isinstance(key, str) else stats[task][key[0]][key[1]]
            row.append(f"{value:{fmt[1:]}}")
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
