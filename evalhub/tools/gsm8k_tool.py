from uuid import uuid4

from evalhub.benchmarks.math.utils import extract_answer, grade_answer
from evalhub.tools.base_tool import BaseTool


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    answer = extract_answer(solution_str)
    if answer is None:
        return 0
    else:
        if grade_answer(answer, ground_truth):
            return score
        else:
            return format_score


class Gsm8kTool(BaseTool):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    async def create(
        self, instance_id: str | None = None, ground_truth: str | None = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self.instances[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict) -> tuple[str, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        self.instances[instance_id]["answer"] = answer
        reward = compute_score(
            self.instances[instance_id]["answer"],
            self.instances[instance_id]["ground_truth"],
            format_score=0.0,
            score=1.0,
        )

        return f"Current parsed {answer=} {reward=}", reward, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        del self.instances[instance_id]
