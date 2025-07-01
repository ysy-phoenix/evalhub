import ast
import re
import traceback

import aiohttp
import numpy as np

from evalhub.utils.logger import logger

API_BASE_URL = "http://localhost:8000/api/v1/judge"
EMPTY_TEST_CASES = [
    {"input": "", "expected": ""},
]


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    r"""Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


async def judge(id: str, submission: dict) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(API_BASE_URL, json=submission) as response:
            result = await response.json()
            return id, result


def process_output(output: str) -> str:
    r"""Extract the code block from the output."""
    if "```" not in output:
        return output
    try:
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        result = re.findall(pattern, output)
        return result[0][1]
    except Exception as e:
        logger.error(f"Error processing output: {e}")
        return output


def extract_livecodebench_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
