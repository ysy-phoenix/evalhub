from collections import Counter
from typing import Any

import numpy as np


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    r"""Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def get_majority_vote(predictions: list[Any]) -> Any:
    if not predictions:
        return None

    counter = Counter(predictions)
    return counter.most_common(1)[0][0]
