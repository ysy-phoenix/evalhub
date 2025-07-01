# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py

import re

from evalhub.utils.logger import logger


def extract_ground_truth(solution_str: str) -> str:
    r"""Extract the ground truth from the solution string."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def extract_solution(solution: str, method: str = "strict") -> str:
    r"""Extract the solution from the solution string."""
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def remove_units(solution: str) -> str:
    r"""Remove units from mathematical solutions.

    Examples:
        "2:00 \\text{ PM}" -> "2:00"

    """
    # remove \text{...}
    solution = re.sub(r"\\text\{\s*[^}]*\}", "", solution)
    # remove $
    solution = re.sub(r"\$\s*", "", solution)
    # remove extra spaces
    solution = re.sub(r"\s+", " ", solution).strip()
    return solution


def gsm8k_patch(solution: str | None, ground_truth: str) -> bool:
    r"""Patch for GSM8K."""
    if solution is None:
        return False
    solution = remove_units(solution)
    if solution == "2:00" and ground_truth == "2":  # FIXME: patch for 2:00 PM
        return True
    if solution == ground_truth:
        logger.info(f"Patching GSM8K: {solution} == {ground_truth}")
        return True
    return False
