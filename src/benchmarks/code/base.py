import os
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from src.benchmarks.base import Dataset
from src.inference.utils import GenerationResult
from src.utils.pbar import get_progress_bar


class CodeDataset(Dataset):
    r"""Dataset class for code generation problems."""

    def __init__(self, name: str = "code", meta_data: dict[str, Any] = None):
        super().__init__(name, meta_data=meta_data)

    def load_tasks(self):
        r"""Load tasks from math reasoning dataset."""
        raise NotImplementedError

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for math reasoning task."""
        raise NotImplementedError

    def extract_code(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        raise NotImplementedError

    def sanitize_and_save(self, raw_file: PathLike, output_dir: PathLike) -> Path:
        r"""Sanitize and save the results."""
        output_dir = Path(output_dir)
        save_path = output_dir / f"{self.name}.jsonl"
        with open(raw_file, "rb") as f:
            total_samples = sum(1 for _ in f)
        with (
            open(raw_file, "rb") as f,
            open(save_path, "wb") as save_file,
            get_progress_bar() as progress,
        ):
            task = progress.add_task(
                "[bold blue]Sanitizing and saving results", total=total_samples
            )

            for line in f:
                data = orjson.loads(line)
                task_id, solution = data["task_id"], data["solution"]
                solution = solution.split("</think>")[-1].strip()  # FIXME: for long COT
                sanitized_solution = self.extract_code(task_id, solution)
                save_file.write(
                    orjson.dumps({"task_id": task_id, "solution": sanitized_solution}) + b"\n"
                )
                progress.update(task, advance=1)

        return save_path

    def save(self, results: list[GenerationResult], output_dir: PathLike) -> Path:
        r"""Save raw and processed results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        raw_path = output_dir / f"{self.name}_raw.jsonl"
        with open(raw_path, "wb") as f:
            for result in results:
                for response in result.responses:
                    f.write(orjson.dumps({"task_id": result.task_id, "solution": response}) + b"\n")
        save_path = self.sanitize_and_save(raw_path, output_dir)
        return save_path
