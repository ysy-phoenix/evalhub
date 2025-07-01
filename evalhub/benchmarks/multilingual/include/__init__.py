import re
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset
from evalhub.utils.logger import logger

INCLUDE = "include"
INCLUDE_HUB = "CohereLabs/include-base-44"
INCLUDE_QUERY_TEMPLATE = (
    "Answer the following multiple choice question.\n"
    "The last line of your response should be of the following format:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n"
    "Think step by step before answering.\n\n"
    "{Question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


@register_dataset((INCLUDE, INCLUDE_HUB, True))
class INCLUDEDataset(MathDataset):
    r"""Dataset class for INCLUDE problems."""

    def __init__(self, name: str = INCLUDE, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self) -> None:
        r"""Load tasks from INCLUDE dataset."""
        configs = get_dataset_config_names(INCLUDE_HUB)
        for name in configs:
            try:
                dataset = load_dataset(INCLUDE_HUB, name, split="test", download_mode="reuse_cache_if_exists")
                for i, item in enumerate(dataset):
                    prompt, answer = self.format_prompt(item)
                    task = Task(
                        task_id=f"INCLUDE/{name}/{i}",
                        prompt=prompt,
                    )
                    groundtruth = GroundTruth(
                        task_id=f"INCLUDE/{name}/{i}",
                        answer=answer,
                    )
                    self.add_task(task)
                    self.add_groundtruth(groundtruth)
            except Exception as e:
                logger.error(f"Error loading dataset {name}: {e}")

    def format_prompt(self, item: dict[str, Any]) -> tuple[str, str]:
        r"""Format the prompt for INCLUDE task."""
        query_prompt = INCLUDE_QUERY_TEMPLATE.format(
            A=item["option_a"], B=item["option_b"], C=item["option_c"], D=item["option_d"], Question=item["question"]
        )
        gold_choice = "ABCD"[item["answer"]]
        return query_prompt, gold_choice

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the answer from the response."""
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        return match.group(1) if match else None

    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str = None) -> bool:
        r"""Check if the extracted answer is correct."""
        if extracted_answer is None:
            return False
        return extracted_answer.lower().strip() == ground_truth.lower().strip()
