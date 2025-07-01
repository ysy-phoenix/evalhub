from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

AIME2024 = "aime2024"
AIME2024_HUB = "HuggingFaceH4/aime_2024"


@register_dataset((AIME2024, AIME2024_HUB, True))
class AIME2024Dataset(MathDataset):
    """Dataset class for AIME2024 problems."""

    def __init__(self, name: str = AIME2024, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from AIME2024 dataset."""
        dataset = load_dataset(AIME2024_HUB, split="train")
        for _, item in enumerate(dataset):
            task = Task(
                task_id=f"AIME2024/{item['id']}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"AIME2024/{item['id']}",
                answer=item["answer"],
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for AIME2024 task."""
        question = item["problem"].strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question
