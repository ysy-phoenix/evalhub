import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson
from datasets import load_dataset
from tqdm import tqdm

from src.benchmarks.base import Dataset, GroundTruth, Task
from src.benchmarks.config import DATASET_HUB
from src.benchmarks.math.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from src.inference.utils import GenerationResult

HENDRYCKS_MATH_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class HendrycksMathDataset(Dataset):
    """Dataset class for Hendrycks Math problems."""

    def __init__(self, name: str = "hendrycks_math"):
        super().__init__(name)
        for key, value in HENDRYCKS_MATH_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from Hendrycks Math dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], "default", split="test")
        for i, item in enumerate(dataset):
            task = Task(
                task_id=f"HENDRYCKS_MATH/{i}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"HENDRYCKS_MATH/{i}",
                answer=extract_answer(item["solution"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        r"""Format the prompt for Hendrycks Math task."""
        question = item["problem"].strip()
        instruction_following = (
            "Let's think step by step and output the final answer within \\boxed{}."
        )

        question += " " + instruction_following
        return question

    def save(self, results: List[GenerationResult], output_dir: PathLike) -> Path:
        r"""Save raw and processed results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        save_path = output_dir / f"{self.name}.jsonl"

        with open(save_path, "wb") as save_file:
            for sample in results:
                task_id = sample.task_id
                for response in sample.responses:
                    save_file.write(
                        orjson.dumps({"task_id": task_id, "response": response}) + b"\n"
                    )

        return save_path

    def evaluate(self, solution: PathLike, output_dir: PathLike) -> Tuple[int, int, float]:
        r"""Evaluate the solution."""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        predictions = []
        with open(solution, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                task_id = sample["task_id"]
                response = sample["response"]
                predictions.append((task_id, response))

        assert len(predictions) == len(self.groundtruth), (
            "Number of predictions and groundtruths must be the same"
        )
        correct, total, results = 0, len(predictions), []
        for task_id, response in tqdm(predictions, desc="Evaluating"):
            extracted_answer = extract_answer(response)
            ground_truth = self.groundtruth[task_id].answer
            is_correct = grade_answer_mathd(extracted_answer, ground_truth) or grade_answer_sympy(
                extracted_answer, ground_truth
            )
            if is_correct:
                correct += 1
            results.append(
                {
                    "task_id": task_id,
                    "response": response,
                    "extracted_answer": extracted_answer,
                    "ground_truth": self.groundtruth[task_id].answer,
                    "correct": is_correct,
                }
            )
        accuracy = correct / total if total > 0 else 0
        result_path = output_dir / f"{self.name}_results.jsonl"
        with open(result_path, "wb") as f:
            for result in results:
                f.write(orjson.dumps(result) + b"\n")

        summary_path = output_dir / f"{self.name}_summary.json"
        summary = {"correct": correct, "total": total, "accuracy": accuracy}
        with open(summary_path, "wb") as f:
            f.write(orjson.dumps(summary))
        return correct, total, accuracy
