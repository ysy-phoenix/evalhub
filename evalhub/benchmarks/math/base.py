from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from evalhub.benchmarks.base import Dataset
from evalhub.benchmarks.math.verifier import extract_answer, grade_answer
from evalhub.utils.logger import logger
from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote
from evalhub.utils.pbar import get_progress_bar

DEFAULT_KS = [1, 5, 10]


class MathDataset(Dataset):
    r"""Dataset class for math reasoning problems."""

    def __init__(self, name: str = "math"):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from math reasoning dataset."""
        raise NotImplementedError

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for math reasoning task."""
        raise NotImplementedError

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the solution from the response."""
        return extract_answer(response)

    def check_correct(self, extracted_answer: str, ground_truth: str, task_id: str) -> bool:
        r"""Check if the extracted answer is correct."""
        return grade_answer(extracted_answer, ground_truth) or self.patch(extracted_answer, ground_truth, task_id)

    def patch(self, extracted_answer: str, ground_truth: str, task_id: str) -> bool:
        r"""Patch the extracted answer."""
        return False

    def _load_solutions(self, solution_path: PathLike) -> dict[str, list[str]]:
        r"""Load predictions from solution file."""
        solutions = defaultdict(list)
        solution_path = Path(solution_path)

        with open(solution_path, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                task_id = sample["task_id"]
                solution = sample["solution"]
                solutions[task_id].append(solution)

        return solutions

    def evaluate(self, solution: PathLike, output_dir: PathLike) -> None:
        r"""Evaluate the solution."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        id2solutions = self._load_solutions(solution)
        assert len(id2solutions) == len(self.groundtruth), (
            f"Predictions ({len(id2solutions)}) must match groundtruths ({len(self.groundtruth)})"
        )

        results, correct, total = [], 0, len(id2solutions)
        progress = get_progress_bar()
        with progress:
            eval_task = progress.add_task("[bold blue]Evaluating", total=total)

            for task_id, solutions in id2solutions.items():
                ground_truth = self.groundtruth[task_id].answer
                is_correct = [self.check_correct(solution, ground_truth, task_id) for solution in solutions]

                # Calculate pass@k metrics
                pass_at_k = defaultdict(float)
                for k in DEFAULT_KS:
                    if k > len(solutions):
                        continue
                    pass_at_k[str(k)] = compute_pass_at_k(len(solutions), sum(is_correct), k)

                # Calculate majority vote
                majority_vote = get_majority_vote(solutions)
                is_correct_majority = self.check_correct(majority_vote, ground_truth)

                result = {
                    "task_id": task_id,
                    "solutions": solutions,
                    "ground_truth": self.groundtruth[task_id].answer,
                    "correct": is_correct,
                    "pass_at_k": pass_at_k,
                    "majority_vote": majority_vote,
                    "is_correct_majority": is_correct_majority,
                }

                progress.update(eval_task, advance=1)
                results.append(result)
                correct += int(is_correct_majority)

        # Calculate aggregate metrics
        pass_at_k = {
            k: sum(result["pass_at_k"].get(k, 0) for result in results) / total for k in results[0]["pass_at_k"]
        }
        cons_at_k = correct / total

        # Log metrics
        for k, value in pass_at_k.items():
            logger.info(f"Pass@{k}: {value:.2%}")
        logger.info(f"Cons@{len(results[0]['solutions'])}: {cons_at_k:.2%}")

        # Save detailed results
        result_path = output_dir / f"{self.name}_results.jsonl"
        with open(result_path, "wb") as f:
            for result in results:
                try:
                    f.write(orjson.dumps(result) + b"\n")
                except Exception as e:
                    logger.error(f"Error dumping result: {result.keys()}")
                    logger.error(f"Error: {e}")
                    exit(1)
        logger.info(f"Evaluation results saved to {result_path}")

        # Save summary
        summary_path = output_dir / f"{self.name}_summary.json"
        with open(summary_path, "wb") as f:
            f.write(orjson.dumps({"pass_at_k": pass_at_k, "cons_at_k": cons_at_k}))
        logger.info(f"Evaluation summary saved to {summary_path}")
