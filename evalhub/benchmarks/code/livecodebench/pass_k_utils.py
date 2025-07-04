import numpy as np

DEFAULT_K_LIST = [2**i for i in range(11)]


def estimate_pass_at_k(num_samples: int | list[int], num_correct: int | list[int], k: int) -> np.ndarray:
    r"""Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        r"""Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct, strict=False)])


def compute_metrics_from_results(results: dict[str, list[list[int]]], k_list=DEFAULT_K_LIST):
    r"""Compute pass@k metrics from results.

    Args:
        results: A dictionary mapping task IDs to n generations;
        each generation contains testcase results.
        k_list: A list of integers representing the values of k for which to compute pass@k.

    Returns:
        {
            "pass@k": {
                "pass@{1}": value of pass@1,
                "pass@{5}": value of pass@5,
                ...
                "pass@{k}": value of pass@k,
            },
            "detail": {
                "pass@1": {
                    "task_id_1": [pass@1 for generation_1, generation_2, ...],
                    "task_id_2": [pass@1 for generation_1, generation_2, ...],
                    ...
                },
                "pass@5": {
                    "task_id_1": [pass@5 for generation_1, generation_2, ...],
                    "task_id_2": [pass@5 for generation_1, generation_2, ...],
                    ...
                },
                ...
            }
        }
    """
    total: list[int] = []
    correct: list[int] = []
    task_ids: list[str] = []
    for task_id, res in results.items():
        all_correct: list[bool] = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))  # n
        correct.append(sum(all_correct))  # c
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k: dict(zip(task_ids, v, strict=False)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(results: dict[str, list[list[int]]]) -> list[list[int]]:
    r"""Extract instance-wise results from results."""
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            if isinstance(generation, int):
                generation = [generation]
            instance_wise_grades[task_id].append(all(g > 0 for g in generation))

    instance_wise_grades = [v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])]
    return instance_wise_grades
