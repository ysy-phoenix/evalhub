from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.gsm8k.utils import extract_ground_truth, gsm8k_patch
from evalhub.benchmarks.registry import register_dataset

GSM8K = "gsm8k"
GSM8K_HUB = "openai/gsm8k"


@register_dataset((GSM8K, GSM8K_HUB, True))
class GSM8KDataset(MathDataset):
    """Dataset class for GSM8K math reasoning problems."""

    def __init__(self, name: str = GSM8K, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from GSM8K dataset."""
        dataset = load_dataset(GSM8K_HUB, "main", split="test")
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
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question

    def patch(self, extracted_answer: str, ground_truth: str, task_id: str) -> bool:
        r"""Patch the extracted answer."""
        return gsm8k_patch(extracted_answer, ground_truth)
