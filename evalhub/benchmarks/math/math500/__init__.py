from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.math500.utils import math500_patch
from evalhub.benchmarks.registry import register_dataset

MATH500 = "math500"
MATH500_HUB = "HuggingFaceH4/MATH-500"


@register_dataset((MATH500, MATH500_HUB, True))
class Math500Dataset(MathDataset):
    """Dataset class for Math500 problems."""

    def __init__(self, name: str = MATH500, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from Math500 dataset."""
        dataset = load_dataset(MATH500_HUB, split="test")
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

    def patch(self, extracted_answer: str, ground_truth: str, task_id: str) -> bool:
        r"""Patch the extracted answer."""
        return math500_patch(extracted_answer, ground_truth, task_id)
