import re
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

CEVAL = "ceval"
CEVAL_HUB = "ceval/ceval-exam"
CEVAL_QUERY_TEMPLATE = (
    "以下是中国关于 {subject} 考试的单项选择题，请选出其中的正确答案。\n"
    "你回答的最后一行应遵循以下格式：\n"
    "'ANSWER: $LETTER'（不带引号），其中 LETTER 是选项 A、B、C 或 D 中的一个。\n"
    "回答前请一步一步思考。\n\n"
    "问题：{question}\n\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}\n\n"
)
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

DEFAULT_KS = [1, 5, 10]


@register_dataset((CEVAL, CEVAL_HUB, False))
class CEVALDataset(MathDataset):
    r"""Dataset class for CEVAL problems."""

    def __init__(self, name: str = CEVAL):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from CEVAL dataset."""
        configs = get_dataset_config_names(CEVAL_HUB)
        for name in configs:
            dataset = load_dataset(CEVAL_HUB, name, split="test", download_mode="reuse_cache_if_exists")
            for item in dataset:
                prompt, answer = self.format_prompt(item, subject=name)
                task = Task(
                    task_id=f"CEVAL/{name}/{item['id']}",
                    prompt=prompt,
                )
                groundtruth = GroundTruth(
                    task_id=f"CEVAL/{name}/{item['id']}",
                    answer=answer,
                )
                self.add_task(task)
                self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any], subject: str) -> tuple[str, str]:
        r"""Format the prompt for CEVAL task."""
        query_prompt = CEVAL_QUERY_TEMPLATE.format(
            subject=subject, question=item["question"], A=item["A"], B=item["B"], C=item["C"], D=item["D"]
        )
        gold_choice = "A"  # FIXME: no gold choice in the dataset
        return query_prompt, gold_choice

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the answer from the response."""
        if response is None:
            return ""
        if "</think>" in response:
            response = response.split("</think>")[-1]
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        return match.group(1) if match else None

    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str) -> bool:
        r"""Check if the extracted answer is correct."""
        if extracted_answer is None:
            return False
        return extracted_answer.lower().strip() == ground_truth.lower().strip()
