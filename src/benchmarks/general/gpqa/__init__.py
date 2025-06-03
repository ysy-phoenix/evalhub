import random
import re
from typing import Any

from datasets import load_dataset

from src.benchmarks.base import GroundTruth, Task
from src.benchmarks.config import DATASET_HUB
from src.benchmarks.math.base import MathDataset

GPQA_QUERY_TEMPLATE = (
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

DEFAULT_KS = [1, 5, 10]


class GPQADataset(MathDataset):
    r"""Dataset class for GPQA problems."""

    def __init__(self, name: str = "gpqa"):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from GPQA dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], "gpqa_diamond", split="train")
        for i, item in enumerate(dataset):
            prompt, answer = self.format_prompt(item)
            task = Task(
                task_id=f"GPQA/{i}",
                prompt=prompt,
            )
            groundtruth = GroundTruth(
                task_id=f"GPQA/{i}",
                answer=answer,
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> tuple[str, str]:
        r"""Format the prompt for GPQA task."""
        choices = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, item["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=item["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the answer from the response."""
        if "</think>" in response:
            response = response.split("</think>")[-1]
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        return match.group(1) if match else None

    def check_correct(self, extracted_answer: str | None, ground_truth: str) -> bool:
        r"""Check if the extracted answer is correct."""
        if extracted_answer is None:
            return False
        return extracted_answer.lower().strip() == ground_truth.lower().strip()
