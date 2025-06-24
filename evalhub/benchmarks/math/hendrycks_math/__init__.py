from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.config import DATASET_HUB
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.utils import extract_answer

HENDRYCKS_MATH_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class HendrycksMathDataset(MathDataset):
    """Dataset class for Hendrycks Math problems."""

    def __init__(self, name: str = "hendrycks_math"):
        super().__init__(name)
        for key, value in HENDRYCKS_MATH_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from Hendrycks Math dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], "default", split="test")
        for i, item in enumerate(dataset):
            task = Task(
                task_id=f"HENDRYCKS_MATH/{i}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"HENDRYCKS_MATH/{i}",
                answer=extract_answer(item["solution"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for Hendrycks Math task."""
        question = item["problem"].strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question
