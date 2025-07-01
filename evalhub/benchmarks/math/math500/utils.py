from evalhub.utils.logger import logger


def math500_patch(solution: str | None, ground_truth: str, task_id: str) -> bool:
    r"""Patch for Math500."""
    match task_id:
        case "MATH500/25":
            res = solution.strip() == "-2, 1"
        case "MATH500/383":
            res = solution.strip() == "[-2, 7]"
        case "MATH500/422":
            res = "-2" in solution and (
                ("1 - \\sqrt{5}" in solution and "1 + \\sqrt{5}" in solution) or ("1 \\pm \\sqrt{5}" in solution)
            )
        case "MATH500/96":
            res = "1 \\pm \\sqrt{19}" in solution or ("1 - \\sqrt{19}" in solution and "1 + \\sqrt{19}" in solution)
        case _:
            return False
    logger.info(f"Patching {task_id}: {solution} == {ground_truth}, {res}")
    return res
