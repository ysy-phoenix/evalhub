from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import Task
from evalhub.benchmarks.code.base import CodeDataset
from evalhub.benchmarks.code.bigcodebench.sanitize import sanitize
from evalhub.benchmarks.registry import register_dataset

BIGCODEBENCH = "bigcodebench"
BIGCODEBENCH_HUB = "bigcode/bigcodebench"
BIGCODEBENCH_META_DATA = {
    "split": "instruct",
    "subset": "full",
}
BIGCODEBENCH_VERSION = "v0.1.4"
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script thatsolves the following problem in a markdown code block:"
)


@register_dataset((BIGCODEBENCH, "bigcode/bigcodebench", False))
class BigCodeBenchDataset(CodeDataset):
    r"""Dataset class for BigCodeBench."""

    def __init__(self, name: str = BIGCODEBENCH, meta_data: dict[str, Any] = BIGCODEBENCH_META_DATA, **kwargs):
        super().__init__(name, meta_data=meta_data, **kwargs)

    def load_tasks(self):
        r"""Load tasks from BigCodeBench dataset."""
        extra = "-" + self.meta_data["subset"] if self.meta_data["subset"] != "full" else ""
        dataset = load_dataset(BIGCODEBENCH_HUB + extra, split=BIGCODEBENCH_VERSION)
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
