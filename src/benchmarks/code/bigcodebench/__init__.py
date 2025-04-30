from typing import Any

from datasets import load_dataset

from src.benchmarks.base import Task
from src.benchmarks.code.base import CodeDataset
from src.benchmarks.code.bigcodebench.sanitize import sanitize
from src.benchmarks.config import DATASET_HUB

BIGCODEBENCH_META_DATA = {
    "split": "instruct",
    "subset": "full",
}
BIGCODEBENCH_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}
BIGCODEBENCH_VERSION = "v0.1.4"
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that"
    "solves the following problem in a markdown code block:"
)


class BigCodeBenchDataset(CodeDataset):
    r"""Dataset class for BigCodeBench."""

    def __init__(
        self, name: str = "bigcodebench", meta_data: dict[str, Any] = BIGCODEBENCH_META_DATA
    ):
        super().__init__(name, meta_data=meta_data)
        for key, value in BIGCODEBENCH_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from BigCodeBench dataset."""
        extra = "-" + self.meta_data["subset"] if self.meta_data["subset"] != "full" else ""
        dataset = load_dataset(DATASET_HUB[self.name] + extra, split=BIGCODEBENCH_VERSION)
        for item in dataset:
            task = Task(
                task_id=str(item["task_id"]),
                prompt=self.format_prompt(item, self.meta_data["split"]),
                metadata={"entry_point": item["entry_point"]},
            )
            self.add_task(task)

    def format_prompt(self, item: dict, split: str) -> str:
        r"""Format the prompt for bigcodebench task."""
        prompt = item[f"{split}_prompt"].strip()
        return INSTRUCTION_PREFIX + prompt.strip()

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        return sanitize(response, self.tasks[task_id].metadata["entry_point"])
