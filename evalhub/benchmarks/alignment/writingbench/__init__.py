import logging
from io import StringIO
from typing import Any

import jsonlines
import requests

from evalhub.benchmarks.base import Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

WRITINGBENCH = "writingbench"
WRITINGBENCH_HUB = "X-PLUG/WritingBench[Placeholder]"
WRITINGBENCH_URL = "https://raw.githubusercontent.com/X-PLUG/WritingBench/main/benchmark_query/benchmark_all.jsonl"


def read_jsonl(url: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with jsonlines.Reader(StringIO(response.text)) as reader:
            return list(reader)
    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        return []


@register_dataset((WRITINGBENCH, WRITINGBENCH_HUB, False))
class WritingBenchDataset(MathDataset):
    r"""Dataset class for WritingBench problems."""

    def __init__(self, name: str = WRITINGBENCH):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from WritingBench dataset."""
        dataset = read_jsonl(WRITINGBENCH_URL)
        for item in dataset:
            task = Task(
                task_id=f"WritingBench/{item['index']}",
                prompt=self.format_prompt(item),
                metadata={
                    "index": item["index"],
                },
            )
            self.add_task(task)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for IFEVAL task."""
        return item["query"]

    def extract_solution(self, task_id: str, response: str | None) -> str:
        r"""Extract the answer from the response."""
        return response or ""
