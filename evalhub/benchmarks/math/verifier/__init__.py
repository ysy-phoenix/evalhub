from .dapo import verify_dapo
from .rllm import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy


def extract_answer(passage: str) -> str:
    r"""Extract the answer from the passage."""
    if "</think>" in passage:
        passage = passage.split("</think>")[-1]
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return passage[:-300]  # Limit solution length for efficiency


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    r"""Grade the answer."""
    return any(
        verifier(given_answer, ground_truth) for verifier in [grade_answer_mathd, grade_answer_sympy, verify_dapo]
    )


__all__ = ["grade_answer"]
