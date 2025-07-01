import json
from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset
from evalhub.utils.logger import logger

ZEBRALOGIC = "zebralogic"
ZEBRALOGIC_HUB = "WildEval/ZebraLogic"

PROMPT_TEMPLATE = """# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street.
Each house is occupied by a different person. Each house has a unique attribute for
each of the following characteristics:
- Each person has a unique name:  `Peter`, `Eric`, `Arnold`.
- Each person has a unique favorite drink: `tea`, `water`, `milk`

# Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

# Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. \
According to Clue 2, Arnold is directly left of the one who only drinks water. \
The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. \
Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. \
Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{puzzle}

# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the JSON format

{json_schema}
"""


@register_dataset((ZEBRALOGIC, ZEBRALOGIC_HUB, True))
class ZebraLogicDataset(MathDataset):
    """Dataset class for ZebraLogic problems."""

    def __init__(self, name: str = ZEBRALOGIC, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from ZebraLogic dataset."""
        dataset = load_dataset(ZEBRALOGIC_HUB, "grid_mode", split="test")
        for _, item in enumerate(dataset):
            task = Task(
                task_id=f"ZEBRALOGIC/{item['id']}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"ZEBRALOGIC/{item['id']}",
                answer=json.dumps(item["solution"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for ZebraLogic task."""
        return PROMPT_TEMPLATE.format(puzzle=item["puzzle"], json_schema=self.build_json_schema(item))

    def build_json_schema(self, item: dict[str, Any]) -> str:
        r"""Build the JSON schema for ZebraLogic task."""
        name, keys = item["solution"]["header"][0], item["solution"]["header"][1:]
        return json.dumps(
            {
                "reasoning": "___",
                "solution": {f"{name} {row[0]}": dict.fromkeys(keys, "___") for row in item["solution"]["rows"]},
            }
        )

    def build_solution(self, solution: dict) -> dict:
        name, keys = solution["header"][0], solution["header"][1:]
        res = {}
        for row in solution["rows"]:
            res[f"{name} {row[0]}"] = {key: row[i + 1] for i, key in enumerate(keys)}
        return res

    def extract_solution(self, task_id: str, response: str | None) -> str:
        return response or ""

    def check_correct(self, extracted_answer: str, ground_truth: str, task_id: str = None) -> bool:
        ground_truth = self.build_solution(json.loads(ground_truth))
        try:
            answer = extracted_answer.removeprefix("```json").removesuffix("```")
            answer = json.loads(answer)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format in answer!")
        return answer.get("solution", {}) == ground_truth
