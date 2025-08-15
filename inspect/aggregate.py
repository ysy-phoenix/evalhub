import argparse
import csv
import json
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table


def save_to_csv(data: list[dict], output_path: Path) -> None:
    fieldnames = ["benchmark", "metric", "value"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            benchmark = entry["file"].stem.removesuffix("_summary")
            is_humaneval = "humaneval" in benchmark or "mbpp" in benchmark

            if is_humaneval:
                for k, v in entry["data"]["base_pass_at_k"].items():
                    writer.writerow({"benchmark": benchmark, "metric": f"base_pass@{k}", "value": v})
                for k, v in entry["data"]["plus_pass_at_k"].items():
                    writer.writerow({"benchmark": benchmark, "metric": f"plus_pass@{k}", "value": v})
            else:
                for k, v in entry["data"]["pass_at_k"].items():
                    writer.writerow({"benchmark": benchmark, "metric": f"pass@{k}", "value": v})
                cons_at_k = entry["data"].get("cons_at_k", None)
                if cons_at_k is not None:
                    writer.writerow({"benchmark": benchmark, "metric": "cons@k", "value": cons_at_k})


def create_table(title: str, data: dict, is_humaneval: bool = False) -> Table:
    table_style = {"title": "[bold green]", "width": 30}

    table = Table(
        title=f"{table_style['title']}{title}[/]",
        show_header=True,
        header_style="bold magenta",
        width=table_style["width"],
        border_style="bright_blue",
        padding=(0, 1),
    )

    if is_humaneval:
        table.add_column("Metric", style="dim", width=10, justify="center")
        table.add_column("Base", justify="center", width=10)
        table.add_column("Plus", justify="center", width=10)

        for k in sorted(data["base_pass_at_k"].keys(), key=int):
            table.add_row(f"pass@{k}", f"{data['base_pass_at_k'][k]:.2%}", f"{data['plus_pass_at_k'][k]:.2%}")
    else:
        table.add_column("Metric", style="dim", width=15, justify="center")
        table.add_column("Value", justify="center", width=15)

        for k, v in sorted(data["pass_at_k"].items(), key=lambda x: int(x[0])):
            table.add_row(f"pass@{k}", f"{v:.2%}")

        if data.get("cons_at_k") is not None:
            table.add_row("cons@k", f"{data['cons_at_k']:.2%}")

    return table


def generate_markdown_table(data: list[dict]) -> str:
    benchmarks = []
    pass1_values = []

    for entry in data:
        benchmark = entry["file"].stem.removesuffix("_summary")
        is_humaneval = "humaneval" in benchmark or "mbpp" in benchmark

        if is_humaneval:
            benchmarks.extend([f"{benchmark} (Base)", f"{benchmark} (Plus)"])
            pass1_values.extend(
                [
                    f"{entry['data']['base_pass_at_k'].get('1', 0):.2%}",
                    f"{entry['data']['plus_pass_at_k'].get('1', 0):.2%}",
                ]
            )
        else:
            benchmarks.append(benchmark)
            pass1_values.append(f"{entry['data']['pass_at_k'].get('1', 0):.2%}")

    # Generate the compact markdown table
    md = "\n## Benchmark Results (pass@1)\n\n"
    md += "| " + " | ".join(benchmarks) + " |\n"
    md += "|" + "|".join(["---"] * len(benchmarks)) + "|\n"
    md += "| " + " | ".join(pass1_values) + " |\n"

    return md


def display_summary_files(directory: Path, csv_output: Path | None = None) -> None:
    console = Console()
    file_patterns = "*_summary.json"

    all_data = []
    for file_path in sorted(directory.glob(file_patterns)):
        with open(file_path) as f:
            data = json.load(f)
        all_data.append({"file": file_path, "data": data})

    if not all_data:
        console.print(f"[red]No evaluation files found in {directory}[/red]")
        return

    if csv_output:
        save_to_csv(all_data, csv_output)
        console.print(f"[green]Results saved to {csv_output}[/green]")

    # Generate and print markdown table
    markdown_table = generate_markdown_table(all_data)
    console.print("\n[bold]Markdown Table:[/bold]")
    console.print(Markdown(markdown_table))
    console.print()

    tables = []
    for entry in all_data:
        benchmark = entry["file"].stem.removesuffix("_summary")
        is_humaneval = "humaneval" in benchmark or "mbpp" in benchmark
        tables.append(create_table(benchmark, entry["data"], is_humaneval))

    unified_width = max(t.width for t in tables) if tables else 30
    terminal_width = console.width
    max_columns = max(1, terminal_width // (unified_width + 2))
    avg_rows = sum(len(t.rows) for t in tables) / len(tables) if tables else 0

    columns_per_row = min(2, max_columns) if avg_rows > 8 else min(3, max_columns) if avg_rows > 4 else max_columns

    for i in range(0, len(tables), columns_per_row):
        group = tables[i : i + columns_per_row]
        columns = Columns(group, equal=False, expand=False, padding=(0, 2), align="left")
        console.print(columns)
        console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display and save evaluation results.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing evaluation files")
    parser.add_argument("--csv", type=str, help="Path to save aggregated results as CSV")
    args = parser.parse_args()

    console = Console()
    directory = Path(args.dir)
    csv_output = Path(args.csv) if args.csv else Path(args.dir) / "summary.csv"

    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        exit(1)

    display_summary_files(directory, csv_output)
