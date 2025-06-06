from uuid import uuid4

import aiohttp

from evalhub.benchmarks.code.utils import extract_livecodebench_code, process_output
from evalhub.callback.base_callback import BaseCallback

DEFAULT_LANGUAGE = "python"
DEFAULT_TIME_LIMIT = 30
DEFAULT_MEMORY_LIMIT = 4 * 1024
API_BASE_URL = "http://localhost:8000/api/v1/judge"
EMPTY_TEST_CASES = [
    {"input": "", "expected": ""},
]


def extract_solution(response: str) -> str:
    r"""Extract the code from the response."""
    if response.count("```") == 2:
        return extract_livecodebench_code(response)
    else:
        return process_output(response)


class CodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    async def create(self, instance_id: str | None = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self.instances[instance_id] = {
            "reward": 0.0,
        }
        return instance_id

    async def check(self, instance_id: str, response: str, **kwargs) -> bool:
        return True

    async def execute(self, instance_id: str, response: str, **kwargs) -> str:
        try:
            code = extract_solution(response)
            self.instances[instance_id]["code"] = code
            result = await self.submit(instance_id)
            stdout = result.get("test_case_results", [{}])[0].get("actual_output", "")
            stderr = result.get("test_case_results", [{}])[0].get("error_message", "")
            res = f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
            # FIXME: Bonus success tool call
            self.instances[instance_id]["reward"] += 1.0
            return res
        except Exception as e:
            return f"Error: {e}"

    async def submit(self, instance_id: str, **kwargs) -> dict:
        submission = {
            "code": self.instances[instance_id]["code"],
            "language": DEFAULT_LANGUAGE,
            "mode": "execution",
            "test_cases": EMPTY_TEST_CASES,
            "time_limit": DEFAULT_TIME_LIMIT,
            "memory_limit": DEFAULT_MEMORY_LIMIT,
        }
        async with (
            aiohttp.ClientSession() as session,
            session.post(API_BASE_URL, json=submission) as response,
        ):
            result = await response.json()
        return result

    async def release(self, instance_id: str, **kwargs) -> float:
        reward = self.instances[instance_id].pop("reward", 0.0)
        del self.instances[instance_id]
        return reward
