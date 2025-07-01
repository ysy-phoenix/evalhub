import json
from uuid import uuid4

import aiohttp

from evalhub.benchmarks.code.utils import extract_livecodebench_code, process_output
from evalhub.callback.base_callback import BaseCallback

DEFAULT_LANGUAGE = "python"
DEFAULT_TIME_LIMIT = 10
DEFAULT_MEMORY_LIMIT = 4 * 1024
API_BASE_URL = "http://localhost:8000/api/v1/judge"
EMPTY_TEST_CASES = [
    {"input": "", "expected": ""},
]


def truncate_output(output: str | None, max_length: int = 512) -> str | None:
    if output is None:
        return None
    if len(output) <= max_length:
        return output
    else:
        return output[: max_length // 2] + "[... truncated ...]" + output[-max_length // 2 :]


def extract_solution(response: str) -> str:
    r"""Extract the code from the response."""
    if response.count("```") == 2:
        return extract_livecodebench_code(response)
    else:
        return process_output(response)


class CodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    async def create(self, instance_id: str | None = None, test_cases: str | None = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self.instances[instance_id] = {
            "reward": 0.0,
            "status": None,
            "test_cases": json.loads(test_cases),
        }
        return instance_id

    async def check(self, instance_id: str, response: str, **kwargs) -> bool:
        return self.instances[instance_id]["status"] == "accepted"

    async def execute(self, instance_id: str, response: str, **kwargs) -> str:
        try:
            code = extract_solution(response)
            self.instances[instance_id]["code"] = code
            result = await self.submit(instance_id)
            metadata: dict = result.get("metadata", None)
            status: str = result.get("status", None)
            self.instances[instance_id]["status"] = status
            if status == "accepted":
                self.instances[instance_id]["reward"] = 1
                return "All public test cases passed."

            msg = ""
            if metadata is not None:
                passed = metadata.get("passed", 0)
                total = metadata.get("total", 0)
                msg = f"Passed {passed} out of {total} public test cases.\n"
                msg += "Detail of first failed test case:\n"
                test_case_results = result.get("test_case_results", [])
                if test_case_results is not None and len(test_case_results) > 0:
                    msg += f"Input:\n{truncate_output(test_case_results[0].get('input', ''))}\n"
                    msg += f"Expected:\n{truncate_output(test_case_results[0].get('expected', ''))}\n"
                    msg += f"Actual:\n{truncate_output(test_case_results[0].get('actual', ''))}\n"
                msg += f"Error message:\n{truncate_output(result.get('error_message', ''))}\n"
                self.instances[instance_id]["reward"] += passed / total if total > 0 else 0
            else:
                msg = "Code block execution failed.\n"
                msg += f"Error message:\n{truncate_output(result.get('error_message', ''))}\n"
                self.instances[instance_id]["reward"] = 0

            return msg
        except Exception as e:
            import traceback

            traceback.print_exc()
            return f"Error: {e}"

    async def submit(self, instance_id: str, **kwargs) -> dict:
        test_cases = self.instances[instance_id]["test_cases"]
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        entry_point = test_cases.get("fn_name", None)
        if entry_point is not None:
            mode = "leetcode"
            inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in inputs]
            outputs = [json.loads(output) for output in outputs]
        else:
            mode = "acm"
        test_cases = [{"input": inp, "expected": out} for inp, out in zip(inputs, outputs, strict=False)]
        submission = {
            "code": self.instances[instance_id]["code"],
            "language": DEFAULT_LANGUAGE,
            "mode": mode,
            "test_cases": test_cases,
            "entry_point": entry_point,
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
