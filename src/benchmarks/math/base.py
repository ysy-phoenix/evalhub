from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from src.benchmarks.base import Dataset
from src.benchmarks.math.utils import extract_answer, grade_answer
from src.inference.utils import GenerationResult
from src.utils.logger import logger
from src.utils.metrics import compute_pass_at_k, get_majority_vote
from src.utils.pbar import get_progress_bar

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

    def check_correct(self, extracted_answer: str, ground_truth: str) -> bool:
        r"""Check if the extracted answer is correct."""
        return grade_answer(extracted_answer, ground_truth)

    def save(self, results: list[GenerationResult], output_dir: str | Path) -> Path:
        r"""Save raw results to a JSONL file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / f"{self.name}.jsonl"

        with open(save_path, "wb") as save_file:
            for sample in results:
                task_id = sample.task_id
                for response in sample.responses:
                    save_file.write(
                        orjson.dumps({"task_id": task_id, "response": response}) + b"\n"
                    )

        return save_path

    def _load_solutions(self, solution_path: str | Path) -> dict[str, list[str]]:
        r"""Load predictions from solution file."""
        predictions = defaultdict(list)
        solution_path = Path(solution_path)

        with open(solution_path, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                task_id = sample["task_id"]
                response = sample["response"]
                predictions[task_id].append(response)

        return predictions

    def evaluate(self, solution: str | Path, output_dir: str | Path) -> None:
        r"""Evaluate the solution."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        solutions = self._load_solutions(solution)
        assert len(solutions) == len(self.groundtruth), (
            f"Predictions ({len(solutions)}) must match groundtruths ({len(self.groundtruth)})"
        )

        results, correct, total = [], 0, len(solutions)
        progress = get_progress_bar()
        with progress:
            eval_task = progress.add_task("[bold blue]Evaluating", total=total)

            for task_id, responses in solutions.items():
                extracted_answers = [extract_answer(response) for response in responses]
                ground_truth = self.groundtruth[task_id].answer
                is_correct = [
                    self.check_correct(answer, ground_truth) for answer in extracted_answers
                ]

                # Calculate pass@k metrics
                pass_at_k = defaultdict(float)
                for k in DEFAULT_KS:
                    if k > len(responses):
                        continue
                    pass_at_k[str(k)] = compute_pass_at_k(len(responses), sum(is_correct[:k]), 1)

                # Calculate majority vote
                majority_vote = get_majority_vote(extracted_answers)
                is_correct_majority = self.check_correct(majority_vote, ground_truth)

                result = {
                    "task_id": task_id,
                    "responses": responses,
                    "extracted_answers": extracted_answers,
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
            k: sum(result["pass_at_k"].get(k, 0) for result in results) / total
            for k in results[0]["pass_at_k"]
        }
        cons_at_k = correct / total

        # Log metrics
        for k, value in pass_at_k.items():
            logger.info(f"Pass@{k}: {value:.2%}")
        logger.info(f"Cons@{len(results[0]['responses'])}: {cons_at_k:.2%}")

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
