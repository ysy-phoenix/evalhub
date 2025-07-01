import logging
from io import StringIO
from typing import Any

import jsonlines
import requests

from evalhub.benchmarks.base import Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

IFEVAL = "ifeval"
IFEVAL_HUB = "google/IFEval"
IFEVAL_URL = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"


def read_jsonl(url: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with jsonlines.Reader(StringIO(response.text)) as reader:
            return list(reader)
    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        return []


@register_dataset((IFEVAL, IFEVAL_HUB, False))
class IFEVALDataset(MathDataset):
    r"""Dataset class for IFEVAL problems."""

    def __init__(self, name: str = IFEVAL, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self) -> None:
        r"""Load tasks from IFEVAL dataset."""
        # dataset = load_dataset(IFEVAL_HUB, split="train")
        dataset = read_jsonl(IFEVAL_URL)
        for item in dataset:
            task = Task(
                task_id=f"IFEVAL/{item['key']}",
                prompt=self.format_prompt(item),
            )
            self.add_task(task)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for IFEVAL task."""
        return item["prompt"]

    def extract_solution(self, task_id: str, response: str | None) -> str:
        r"""Extract the answer from the response."""
        return response or ""
