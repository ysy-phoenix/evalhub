import re
from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

MMMLU = "mmmlu"
MMMLU_HUB = "openai/MMMLU"
MMMLU_QUERY_TEMPLATE = (
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


@register_dataset((MMMLU, MMMLU_HUB, True))
class MMMLUDataset(MathDataset):
    r"""Dataset class for MMMLU problems."""

    def __init__(self, name: str = MMMLU, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self) -> None:
        r"""Load tasks from MMMLU dataset."""
        dataset = load_dataset(MMMLU_HUB, "default", split="test", download_mode="reuse_cache_if_exists")
        for i, item in enumerate(dataset):
            prompt, answer = self.format_prompt(item)
            task = Task(
                task_id=f"MMMLU/{i}",
                prompt=prompt,
            )
            groundtruth = GroundTruth(
                task_id=f"MMMLU/{i}",
                answer=answer,
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> tuple[str, str]:
        r"""Format the prompt for MMMLU task."""
        query_prompt = MMMLU_QUERY_TEMPLATE.format(
            A=item["A"], B=item["B"], C=item["C"], D=item["D"], Question=item["Question"]
        )
        gold_choice = item["Answer"]
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
