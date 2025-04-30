from typing import Any

from src.benchmarks.base import Dataset


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

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        raise NotImplementedError
