from typing import Any

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.config import DATASET_HUB
from evalhub.benchmarks.math.base import MathDataset

AIME2025_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class AIME2025Dataset(MathDataset):
    """Dataset class for AIME2025 problems."""

    def __init__(self, name: str = "aime2025"):
        super().__init__(name)
        for key, value in AIME2025_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from AIME2025 dataset."""
        configs = get_dataset_config_names(DATASET_HUB[self.name])
        all_datasets = [
            load_dataset(DATASET_HUB[self.name], name, split="test") for name in configs
        ]
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
        instruction_following = (
            "Let's think step by step and output the final answer within \\boxed{}."
        )

        question += " " + instruction_following
        return question
