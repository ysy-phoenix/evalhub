import re
from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

MMLU_REDUX = "mmlu_redux"
MMLU_REDUX_HUB = "edinburgh-dawg/labelchaos"
MMLU_REDUX_QUERY_TEMPLATE = (
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


@register_dataset((MMLU_REDUX, MMLU_REDUX_HUB, True))
class MMLUReduxDataset(MathDataset):
    r"""Dataset class for MMLU-Redux problems."""

    def __init__(self, name: str = MMLU_REDUX):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from MMLU-Redux dataset."""
        dataset = load_dataset(MMLU_REDUX_HUB, "clean", split="test")
        for i, item in enumerate(dataset):
            prompt, answer = self.format_prompt(item)
            task = Task(
                task_id=f"MMLU_REDUX/{i}",
                prompt=prompt,
            )
            groundtruth = GroundTruth(
                task_id=f"MMLU_REDUX/{i}",
                answer=answer,
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> tuple[str, str]:
        r"""Format the prompt for MMLU-Redux task."""
        choices = item["choices"]
        query_prompt = MMLU_REDUX_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=item["question"]
        )
        gold_choice = "ABCD"[item["answer"]]
        return query_prompt, gold_choice

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the answer from the response."""
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        return match.group(1) if match else None

    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str) -> bool:
        r"""Check if the extracted answer is correct."""
        if extracted_answer is None:
            return False
        return extracted_answer.lower().strip() == ground_truth.lower().strip()
