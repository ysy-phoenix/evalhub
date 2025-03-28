import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List

import orjson

from src.benchmarks.base import Dataset
from src.inference.utils import GenerationResult
from src.utils.pbar import get_progress_bar


class CodeDataset(Dataset):
    r"""Dataset class for code generation problems."""

    def __init__(self, name: str = "code", meta_data: Dict[str, Any] = None):
        super().__init__(name, meta_data=meta_data)

    def load_tasks(self):
        r"""Load tasks from math reasoning dataset."""
        raise NotImplementedError

    def format_prompt(self, item: Dict[str, Any]) -> str:
        r"""Format the prompt for math reasoning task."""
        raise NotImplementedError

    def extract_code(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        raise NotImplementedError

    def save(self, results: List[GenerationResult], output_dir: PathLike) -> Path:
        """Save raw and processed results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        raw_path = output_dir / f"{self.name}-raw.jsonl"
        save_path = output_dir / f"{self.name}.jsonl"

        total_samples = sum(len(result.responses) for result in results)
        progress = get_progress_bar()

        with progress:
            save_task = progress.add_task("[bold blue]Saving results", total=total_samples)

            with open(save_path, "wb") as save_file, open(raw_path, "wb") as raw_file:
                for sample in results:
                    task_id = sample.task_id
                    for response in sample.responses:
                        save_file.write(
                            orjson.dumps(
                                {
                                    "task_id": task_id,
                                    "solution": self.extract_code(task_id, response),
                                }
                            )
                            + b"\n"
                        )
                        raw_file.write(
                            orjson.dumps({"task_id": task_id, "solution": response}) + b"\n"
                        )
                        progress.update(save_task, advance=1)

        return save_path
