from typing import Any

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

AIME2025 = "aime2025"
AIME2025_HUB = "opencompass/AIME2025"


@register_dataset((AIME2025, AIME2025_HUB, True))
class AIME2025Dataset(MathDataset):
    """Dataset class for AIME2025 problems."""

    def __init__(self, name: str = AIME2025, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from AIME2025 dataset."""
        configs = get_dataset_config_names(AIME2025_HUB)
        all_datasets = [load_dataset(AIME2025_HUB, name, split="test") for name in configs]
        dataset = concatenate_datasets(all_datasets)
        for i, item in enumerate(dataset):
            task = Task(
                task_id=f"AIME2025/{i}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"AIME2025/{i}",
                answer=item["answer"],
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for AIME2025 task."""
        question = item["question"].strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question
