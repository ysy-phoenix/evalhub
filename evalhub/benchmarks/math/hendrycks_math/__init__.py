from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.verifier import extract_answer
from evalhub.benchmarks.registry import register_dataset

HENDRYCKS_MATH = "hendrycks_math"
HENDRYCKS_MATH_HUB = "DigitalLearningGmbH/MATH-lighteval"
HENDRYCKS_MATH_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


@register_dataset((HENDRYCKS_MATH, HENDRYCKS_MATH_HUB, True))
class HendrycksMathDataset(MathDataset):
    """Dataset class for Hendrycks Math problems."""

    def __init__(self, name: str = HENDRYCKS_MATH):
        super().__init__(name)
        for key, value in HENDRYCKS_MATH_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from Hendrycks Math dataset."""
        dataset = load_dataset(HENDRYCKS_MATH_HUB, "default", split="test")
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
