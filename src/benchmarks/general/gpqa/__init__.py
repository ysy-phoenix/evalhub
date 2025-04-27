import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson
from datasets import load_dataset

from src.benchmarks.base import Dataset, GroundTruth, Task
from src.benchmarks.config import DATASET_HUB
from src.inference.utils import GenerationResult
from src.utils.logger import logger
from src.utils.metrics import compute_pass_at_k
from src.utils.pbar import get_progress_bar

GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question.\n"
    "The last line of your response should be of the following format:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n"
    "Think step by step before answering.\n\n"
    "{Question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

DEFAULT_KS = [1, 5, 10]


class GPQADataset(Dataset):
    r"""Dataset class for GPQA problems."""

    def __init__(self, name: str = "gpqa"):
        super().__init__(name)

    def load_tasks(self) -> None:
        r"""Load tasks from GPQA dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], "gpqa_diamond", split="train")
        for i, item in enumerate(dataset):
            prompt, answer = self.format_prompt(item)
            task = Task(
                task_id=f"GPQA/{i}",
                prompt=prompt,
            )
            groundtruth = GroundTruth(
                task_id=f"GPQA/{i}",
                answer=answer,
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> tuple[str, str]:
        r"""Format the prompt for GPQA task."""
        choices = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, item["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=item["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    def extract_answer(self, response: str) -> str:
        r"""Extract the answer from the response."""
        if "</think>" in response:
            response = response.split("</think>")[-1]
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        return match.group(1) if match else None

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
                extracted_answers = [self.extract_answer(response) for response in responses]
                ground_truth = self.groundtruth[task_id].answer
                is_correct = [answer == ground_truth for answer in extracted_answers]

                # Calculate pass@k metrics
                pass_at_k = defaultdict(float)
                for k in DEFAULT_KS:
                    if k > len(responses):
                        continue
                    pass_at_k[str(k)] = compute_pass_at_k(len(responses), sum(is_correct), k)

                result = {
                    "task_id": task_id,
                    "responses": responses,
                    "extracted_answers": extracted_answers,
                    "ground_truth": self.groundtruth[task_id].answer,
                    "correct": is_correct,
                    "pass_at_k": pass_at_k,
                }

                progress.update(eval_task, advance=1)
                results.append(result)

        # Calculate aggregate metrics
        pass_at_k = {
            k: sum(result["pass_at_k"].get(k, 0) for result in results) / total
            for k in results[0]["pass_at_k"]
        }
        cons_at_k = correct / total

        # Log metrics
        for k, value in pass_at_k.items():
            logger.info(f"Pass@{k}: {value:.2%}")

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
