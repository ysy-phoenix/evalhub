from uuid import uuid4

from evalhub.benchmarks.math.utils import extract_answer, grade_answer
from evalhub.tools.base_tool import BaseTool


def compute_score(
    solution_str: str, ground_truth: str, format_score: float = 0.0, score: float = 1.0
):
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
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> str:
        try:
            if isinstance(parameters, int):
                answer = parameters
            elif isinstance(parameters, dict):
                answer = parameters.get("answer", "")
            else:
                return "Invalid parameters"

            if not isinstance(answer, str):
                answer = str(answer)

            self.instances[instance_id]["answer"] = answer
            reward = await self.calc_reward(instance_id)
            self.instances[instance_id]["reward"] = reward
        except Exception as e:
            return f"Error: {e}"

        return f"Current parsed {answer=} {reward=}"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return compute_score(
            self.instances[instance_id]["answer"],
            self.instances[instance_id]["ground_truth"],
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> float:
        reward = self.instances[instance_id].pop("reward", 0.0)
        del self.instances[instance_id]
        return reward
