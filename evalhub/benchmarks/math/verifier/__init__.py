import signal
from collections.abc import Callable
from contextlib import contextmanager

from evalhub.utils.logger import logger

from .dapo import verify_dapo
from .rllm import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy


def extract_answer(passage: str) -> str:
    r"""Extract the answer from the passage."""
    if "</think>" in passage:
        passage = passage.split("</think>")[-1]
    if "\\boxed" in passage:
        passage = extract_boxed_answer(passage)
    return passage[-300:] if passage is not None else ""  # Limit solution length for efficiency


@contextmanager
def timeout_handler(timeout_seconds: float):
    r"""Context manager for timeout using signal.alarm."""

    def timeout_signal_handler(signum, frame):
        raise KeyboardInterrupt(f"Function timed out after {timeout_seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)

    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    except AttributeError:
        signal.alarm(max(1, int(timeout_seconds)))

    try:
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except AttributeError:
            signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _verify_with_timeout(verifier: Callable, given_answer: str, ground_truth: str, timeout: float = 1.0) -> bool:
    r"""Execute verifier with timeout protection using signal.alarm."""
    try:
        with timeout_handler(timeout):
            result = verifier(given_answer, ground_truth)
            return result
    except KeyboardInterrupt:
        logger.warning(f"Verifier {verifier.__name__} timed out")
        logger.warning(f"answer: {given_answer}")
        logger.warning(f"ground truth: {ground_truth}")
        return False
    except Exception as e:
        logger.warning(f"Verifier {verifier.__name__} failed")
        logger.warning(f"answer: {given_answer}")
        logger.warning(f"ground truth: {ground_truth}")
        logger.warning(f"error: {e}")
        return False


def grade_answer(given_answer: str, ground_truth: str, timeout: float = 1.0) -> bool:
    r"""Grade the answer with timeout protection."""
    if "!" in given_answer and "!" not in ground_truth:
        return False

    verifiers = [grade_answer_mathd, grade_answer_sympy, verify_dapo]
    for verifier in verifiers:
        if _verify_with_timeout(verifier, given_answer, ground_truth, timeout):
            return True

    return False


__all__ = ["grade_answer"]
