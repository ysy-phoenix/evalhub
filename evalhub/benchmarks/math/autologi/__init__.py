import json
import re
from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset
from evalhub.utils.logger import logger

AUTOLOGI = "autologi"
AUTOLOGI_HUB = "qzhu/AutoLogi"
CODE_TEMPLATE = """
def find_player(role, players):
    return next(player for player in players if player["role"] == role)

{Inputs_Check_code}

{Constraint_List_code}

example = {example}
valid = inputs_check(example)
correct = valid and all(f(example) for f in constraint_list)
result = {{
    'valid': valid,
    'correct': correct
}}
"""
ANSWER_PATTERN = r"```.*?\n(.*?)\n```"


@register_dataset((AUTOLOGI, AUTOLOGI_HUB, True))
class AutoLogiDataset(MathDataset):
    """Dataset class for AutoLogi problems."""

    def __init__(self, name: str = AUTOLOGI, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from AIME2025 dataset."""
        dataset = load_dataset(AUTOLOGI_HUB, split="train")
        for i, item in enumerate(dataset):
            task = Task(
                task_id=f"AutoLogi/{i}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"AutoLogi/{i}",
                answer=json.dumps(item["code"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for AutoLogi task."""
        question = item["prompt"]
        return question

    def rename_constraints(self, code: str) -> str:
        constraint_defs = re.findall(r"def (constraint_\w+)\(", code)

        replacements = {}
        for i, old_name in enumerate(constraint_defs, start=1):
            replacements[old_name] = f"constraint_{i}"

        for old_name, new_name in replacements.items():
            code = code.replace(old_name, new_name)

        return code

    def build_test_code(self, code: dict, example: str):
        return CODE_TEMPLATE.format(
            Inputs_Check_code=code["Inputs_Check_code"],
            Constraint_List_code=self.rename_constraints(code["Constraint_List_code"]),
            example=example,
        )

    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str = None) -> bool:
        code = json.loads(ground_truth)
        code = self.build_test_code(code, extracted_answer)
        namespace = {}
        try:
            exec(code, namespace)
            return namespace["result"].get("correct", False)
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return False

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the answer from the response."""
        match = re.search(ANSWER_PATTERN, response, re.DOTALL)
        return match.group(1).strip() if match else None
