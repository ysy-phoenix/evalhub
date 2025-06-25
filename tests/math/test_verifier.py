import pytest

from evalhub.benchmarks.math.math500.utils import math500_patch
from evalhub.benchmarks.math.verifier import grade_answer


@pytest.mark.parametrize(
    "ground_truth,given_answer,task_id",
    [
        ("4210_{5}", "4210", "MATH500/127"),  # MATH500/127 Express $555_{10}$ in base $5$.
        ("1,-2", "-2, 1", "MATH500/25"),  # MATH500/25
        ("864 \\mbox{ inches}^2", "864", "MATH500/257"),  # MATH500/257
        ("\\frac{11+9a}{20}", "\\dfrac{9a + 11}{20}", "MATH500/318"),  # MATH500/318
        ("4343_6", "4343", "MATH500/338"),  # MATH500/338
        ("x \\in [-2, 7]", "[-2, 7]", "MATH500/383"),  # MATH500/383
        ("\\{1\\pm\\sqrt{5},-2\\}", "\\{-2, 1 - \\sqrt{5}, 1 + \\sqrt{5}\\}", "MATH500/422"),  # MATH500/422
        ("15\\mbox{ cm}^2", "15", "MATH500/467"),  # MATH500/467
        ("2516_8", "2516", "MATH500/71"),  # MATH500/71
        ("1 \\pm \\sqrt{19}", "1 + \\sqrt{19}, 1 - \\sqrt{19}", "MATH500/96"),  # MATH500/96
    ],
)
def test_math500(ground_truth, given_answer, task_id):
    assert grade_answer(given_answer, ground_truth) or math500_patch(given_answer, ground_truth, task_id)
