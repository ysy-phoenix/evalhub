import json
import os
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

import orjson

from src.benchmarks.base import Dataset, Task
from src.benchmarks.code.livecodebench.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)
from src.benchmarks.code.livecodebench.compute_code_generation_metrics import (
    codegen_metrics,
)
from src.benchmarks.code.livecodebench.pass_k_utils import extract_instance_results
from src.benchmarks.code.utils import extract_livecodebench_code
from src.inference.utils import GenerationResult
from src.utils.logger import logger

LIVECODEBENCH_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 2048,
}

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. "
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests."
)


FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT."
)


class LiveCodeBenchDataset(Dataset):
    r"""Dataset class for LiveCodeBench code generation benchmark."""

    def __init__(self, name: str = "livecodebench"):
        super().__init__(name)
        for key, value in LIVECODEBENCH_CONFIG.items():
            self.config[key] = value
        self.cache_dir = Path(
            os.environ.get("EVALHUB_CACHE_DIR", Path.home() / ".cache" / "evalhub")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def system_prompt(self) -> Optional[str]:
        r"""Get system prompt for the dataset."""
        return SYSTEM_MESSAGE_GENERIC

    def load_tasks(self):
        r"""Load tasks from LiveCodeBench dataset with caching support."""
        dataset = load_code_generation_dataset(release_version="release_latest")
        for problem in dataset:
            task = Task(
                task_id=problem.question_id,
                prompt=self.format_prompt(problem),
            )
            self.add_task(task)

    def format_prompt(self, question: CodeGenerationProblem):
        r"""Format the prompt for LiveCodeBench tasks."""
        prompt = f"### Question:\n{question.question_content}\n\n"
        if question.starter_code:
            prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{question.starter_code}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        return prompt

    def save(self, results: List[GenerationResult], output_dir: str) -> Path:
        """Save raw and processed results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        raw_path = output_dir / f"{self.name}-raw.jsonl"
        save_path = output_dir / f"{self.name}.jsonl"
        with open(raw_path, "wb") as raw_file, open(save_path, "wb") as save_file:
            for sample in results:
                task_id = sample.task_id
                for response in sample.responses:
                    raw_file.write(orjson.dumps({"task_id": task_id, "solution": response}) + b"\n")
                    save_file.write(
                        orjson.dumps(
                            {"task_id": task_id, "solution": extract_livecodebench_code(response)}
                        )
                        + b"\n"
                    )
        return save_path

    # Adapted from https://github.com/wasiahmad/livecodebench/blob/main/livecodebench/evaluate.py
    def evaluate(self, solution: PathLike, output_dir: PathLike) -> Tuple[int, int, float]:
        r"""Evaluate solutions using LiveCodeBench's evaluator."""
        custom_outputs = {}
        with open(solution, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                custom_outputs[sample["task_id"]] = sample

        benchmark = load_code_generation_dataset(release_version="release_latest")
        benchmark = [problem for problem in benchmark if problem.question_id in custom_outputs]
        logger.info(f"Loaded {len(benchmark)} problems")

        assert len(custom_outputs) == len(benchmark), f"{len(custom_outputs)} != {len(benchmark)}"
        assert all(isinstance(custom_output, dict) for custom_output in custom_outputs.values())

        eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
        generations = [
            custom_outputs[instance.question_id]["solution"]
            if isinstance(custom_outputs[instance.question_id]["solution"], list)
            else [custom_outputs[instance.question_id]["solution"]]  # FIXME: for pass@1 only now
            for instance in benchmark
        ]
        metrics, results, metadatas = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=16,
            timeout=6,
        )

        graded = extract_instance_results(results)

        save_eval_results = [
            instance.format_evaluation(
                code_list,
                graded_list,
                metadata=meta,
            )
            for instance, code_list, graded_list, meta in zip(
                benchmark, generations, graded, metadatas
            )
        ]

        # save_eval_results
        output_results = {}
        output_results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        for k in metrics:
            if k.startswith("pass@"):
                logger.info(f"{k}: {metrics[k]}")
                output_results[k] = metrics[k]

        output_results["detail_pass@1"] = {}
        output_results["eval"] = {}
        difficulty_wise_pass_at_1 = {}
        for r in save_eval_results:
            output_results["eval"][r["question_id"]] = r
            if r["difficulty"] not in difficulty_wise_pass_at_1:
                difficulty_wise_pass_at_1[r["difficulty"]] = []
            difficulty_wise_pass_at_1[r["difficulty"]].append(r["pass@1"])

        for tag, v in difficulty_wise_pass_at_1.items():
            pass_at_1 = sum(v) / len(v)
            logger.info(f"{tag} pass@1: {pass_at_1}")
            output_results["detail_pass@1"][tag] = pass_at_1

        with open(Path(output_dir) / f"{self.name}_results.json", "w") as f:
            json.dump(output_results, f, indent=2)

        return None, None, None


"""
illustation of output_results

{
    "date": "2025-03-14 12:00",
    "pass@1": 0.5,
    "detail_pass@1": {"easy": 0.5, "medium": 0.5, "hard": 0.5},
    "eval": {
        "question_id_1": {
            "code_list": ["code1", "code2", "code3"],
            "graded_list": [True, False, True],
        }
    }
}
"""
