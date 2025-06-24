from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.config import DATASET_HUB
from evalhub.benchmarks.math.base import MathDataset

MATH500_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class Math500Dataset(MathDataset):
    """Dataset class for Math500 problems."""

    def __init__(self, name: str = "math500"):
        super().__init__(name)
        for key, value in MATH500_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from Math500 dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], split="test")
        for i, item in enumerate(dataset):
            task = Task(
                task_id=f"MATH500/{i}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"MATH500/{i}",
                answer=item["answer"],
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for Math500 task."""
        question = item["problem"].strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question
