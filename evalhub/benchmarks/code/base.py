from typing import Any

from evalhub.benchmarks.base import Dataset


class CodeDataset(Dataset):
    r"""Dataset class for code generation problems."""

    def __init__(self, name: str = "code", meta_data: dict[str, Any] = None, **kwargs):
        super().__init__(name, meta_data=meta_data, **kwargs)

    def load_tasks(self):
        r"""Load tasks from math reasoning dataset."""
        raise NotImplementedError

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for math reasoning task."""
        raise NotImplementedError
