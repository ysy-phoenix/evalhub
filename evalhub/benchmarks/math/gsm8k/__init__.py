from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.config import DATASET_HUB
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.gsm8k.utils import extract_ground_truth, gsm8k_patch
from evalhub.benchmarks.math.utils import grade_answer

GSM8K_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class GSM8KDataset(MathDataset):
    """Dataset class for GSM8K math reasoning problems."""

    def __init__(self, name: str = "gsm8k"):
        super().__init__(name)
        for key, value in GSM8K_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from GSM8K dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], "main", split="test")
        for i, item in enumerate(dataset):
            answer = extract_ground_truth(item["answer"])
            task = Task(
                task_id=f"GSM8K/{i}",
                prompt=self.format_prompt(item),
                metadata={
                    "tools": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": answer},
                        },
                    },
                },
            )
            groundtruth = GroundTruth(
                task_id=f"GSM8K/{i}",
                answer=answer,
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for GSM8K task."""
        question = item["question"].strip()
        instruction_following = (
            "Let's think step by step and output the final answer within \\boxed{}."
        )

        question += " " + instruction_following
        return question

    def check_correct(self, extracted_answer: str, ground_truth: str) -> bool:
        r"""Check if the extracted answer is correct."""
        return grade_answer(extracted_answer, ground_truth) or gsm8k_patch(
            extracted_answer, ground_truth
        )
