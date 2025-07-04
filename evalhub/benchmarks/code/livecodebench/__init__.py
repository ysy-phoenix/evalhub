import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from evalhub.benchmarks.base import Task
from evalhub.benchmarks.code.base import CodeDataset
from evalhub.benchmarks.code.livecodebench.code_generation import (
    CodeGenerationProblem,
    MiniProblem,
    load_code_generation_dataset,
    load_mini_problems,
)
from evalhub.benchmarks.code.livecodebench.compute_code_generation_metrics import (
    codegen_metrics,
)
from evalhub.benchmarks.code.livecodebench.pass_k_utils import extract_instance_results
from evalhub.benchmarks.code.utils import (
    compute_pass_at_k,
    extract_livecodebench_code,
    judge,
    process_output,
)
from evalhub.benchmarks.registry import register_dataset
from evalhub.utils.logger import logger
from evalhub.utils.pbar import get_progress_bar

LIVECODEBENCH = "livecodebench"
LIVECODEBENCH_HUB = "livecodebench/code_generation_lite"
LIVECODEBENCH_META_DATA = {
    "release_version": "v6",
    "start_date": None,  # "2024-08-00T00:00:00",
    "end_date": None,  # "2025-01-00T00:00:00",
}
DEFAULT_TIME_LIMIT = 10
DEFAULT_MEMORY_LIMIT = 4 * 1024
NEW_MODE = False
DEFAULT_KS = [2**i for i in range(11)]

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


@register_dataset((LIVECODEBENCH, LIVECODEBENCH_HUB, True))
class LiveCodeBenchDataset(CodeDataset):
    r"""Dataset class for LiveCodeBench code generation benchmark."""

    def __init__(self, name: str = LIVECODEBENCH, meta_data: dict[str, Any] = LIVECODEBENCH_META_DATA, **kwargs):
        super().__init__(name, meta_data=meta_data, **kwargs)

    @property
    def system_prompt(self) -> str | None:
        r"""Get system prompt for the dataset."""
        return SYSTEM_MESSAGE_GENERIC

    def build_test_cases(self, problem: MiniProblem) -> str:
        r"""Build test cases for the problem."""
        return json.dumps(
            {
                "inputs": [test.input for test in problem.public_test_cases],
                "outputs": [test.output for test in problem.public_test_cases],
                "fn_name": problem.metadata.get("func_name", None),
            }
        )

    def load_tasks(self):
        r"""Load tasks from LiveCodeBench dataset with caching support."""
        problems = load_mini_problems(meta_data=self.meta_data)
        for problem in problems:
            task = Task(
                task_id=problem.question_id,
                prompt=self.format_prompt(problem),
                metadata={
                    "callback": {
                        "create_kwargs": {"test_cases": self.build_test_cases(problem)},
                    },
                },
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

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        if response.count("```") == 2:
            return extract_livecodebench_code(response)
        else:
            return process_output(response)

    async def submit_async(self, eval_samples: dict, model_outputs: dict) -> list:
        r"""Asynchronous submission of code evaluation tasks."""
        submissions = {}
        for id, eval_sample in eval_samples.items():
            assert id in model_outputs, f"{id} not in model_outputs"
            for solution in model_outputs[id]:
                input_output = json.loads(eval_sample["input_output"])
                inputs = input_output["inputs"]
                outputs = input_output["outputs"]
                if input_output.get("fn_name", None) is not None:
                    mode = "leetcode"
                    inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in inputs]
                    outputs = [json.loads(output) for output in outputs]
                else:
                    mode = "acm"
                test_cases = [{"input": inp, "expected": out} for inp, out in zip(inputs, outputs, strict=False)]
                entry_point = input_output.get("fn_name", None)
                submission = {
                    "code": solution,
                    "language": "python",
                    "mode": mode,
                    "test_cases": test_cases,
                    "entry_point": entry_point,
                    "time_limit": DEFAULT_TIME_LIMIT,
                    "memory_limit": DEFAULT_MEMORY_LIMIT,
                }
                submissions[id] = submission

        progress = get_progress_bar()
        results = []
        completed = passed = 0

        with progress:
            eval_task = progress.add_task("[bold blue]Evaluating submissions", total=len(submissions))

            semaphore = asyncio.Semaphore(min(16, os.cpu_count()))

            async def process_submission(id, submission):
                async with semaphore:
                    result = await judge(id, submission)
                    return result

            tasks = [process_submission(id, submission) for id, submission in submissions.items()]

            for future in asyncio.as_completed(tasks):
                id, result = await future
                results.append((id, result))
                completed += 1
                if result.get("status", None) == "accepted":
                    passed += 1

                progress.update(
                    eval_task,
                    completed=completed,
                    description=f"[bold blue]Submissions passed ({passed}/{completed})",
                )

        return results

    def submit(self, eval_samples: dict, model_outputs: dict) -> dict:
        r"""Submit solutions to LiveCodeBench."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.submit_async(eval_samples, model_outputs))

    def new_evaluate(self, solution: PathLike, output_dir: PathLike, ks: list[int] = DEFAULT_KS) -> None:
        r"""Evaluate solutions using LiveCodeBench's evaluator."""
        # Load model outputs
        model_outputs = defaultdict(list)
        with open(solution, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                model_outputs[sample["task_id"]].append(sample["solution"])

        # Load benchmark problems
        logger.info("Loading benchmark problems")
        benchmark = load_code_generation_dataset(meta_data=self.meta_data)
        problems = {instance.question_id: instance for instance in benchmark if instance.question_id in model_outputs}
        logger.info(f"Loaded {len(problems)} problems")

        # Load eval samples
        eval_samples = {instance.question_id: instance.get_evaluation_sample() for instance in problems.values()}
        logger.info(f"Loaded {len(eval_samples)} eval samples")

        # Submit solutions
        responses = self.submit(eval_samples, model_outputs)

        # Aggregate results
        results = defaultdict(list)
        for id, response in responses:
            results[id].append(response)

        with open(Path(output_dir) / f"{self.name}_responses.json", "w") as f:
            json.dump(results, f, indent=2)

        # Compute pass@k
        stats = {
            "overall": defaultdict(list),
            "by_difficulty": defaultdict(lambda: defaultdict(list)),
        }
        for id, responses in results.items():
            difficulty = problems[id].difficulty
            n = len(responses)
            c = sum(response["status"] == "accepted" for response in responses)
            for k in ks:
                if n < k:
                    break
                pass_k = compute_pass_at_k(n, c, k)
                stats["overall"][k].append(pass_k)
                stats["by_difficulty"][difficulty][k].append(pass_k)
        output_results = {}
        output_results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        for k, v in stats["overall"].items():
            output_results[f"pass@{k}"] = np.mean(v)

        output_results["detail_pass@1"] = {}
        for diff, vals in stats["by_difficulty"].items():
            for k, v in vals.items():
                if k == 1:
                    output_results["detail_pass@1"][diff.value] = np.mean(v)

        output_results["eval"] = {}
        passed = total = 0
        for id, instance in problems.items():
            if id not in results:
                continue
            code_list = [model_outputs[id][i] for i in range(len(model_outputs[id]))]
            graded_list = [response["status"] == "accepted" for response in results[id]]
            try:
                passed += sum(response.get("metadata", {}).get("passed", 0) for response in results[id])
                total += sum(response.get("metadata", {}).get("total", 0) for response in results[id])
            except Exception as e:
                logger.error(f"Error processing results for {id}: {e}")
                passed += 0
                total += 0
            eval_result = instance.format_evaluation(
                code_list=code_list,
                graded_list=graded_list,
                metadata=list(results[id]),
            )
            output_results["eval"][id] = eval_result

        logger.info(f"Passed test cases: {passed}/{total}")
        logger.info("Overall pass@k:")
        for k in [k for k in stats["overall"].keys() if k in ks]:
            logger.info(f"pass@{k}: {output_results.get(f'pass@{k}', 0)}")

        logger.info("Difficulty-wise pass@1:")
        for diff in output_results["detail_pass@1"]:
            logger.info(f"{diff} pass@1: {output_results['detail_pass@1'][diff]}")

        with open(Path(output_dir) / f"{self.name}_results.json", "w") as f:
            json.dump(output_results, f, indent=2)

    # Adapted from https://github.com/wasiahmad/livecodebench/blob/main/livecodebench/evaluate.py
    def evaluate(self, solution: PathLike, output_dir: PathLike) -> None:
        r"""Evaluate solutions using LiveCodeBench's evaluator."""
        if NEW_MODE:
            self.new_evaluate(solution, output_dir)
            return

        custom_outputs = defaultdict(list)
        with open(solution, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                custom_outputs[sample["task_id"]].append(sample)
        logger.info(f"Loaded {sum(len(res) for res in custom_outputs.values())} responses")

        benchmark = load_code_generation_dataset(meta_data=self.meta_data)
        benchmark = [problem for problem in benchmark if problem.question_id in custom_outputs]
        logger.info(f"Loaded {len(benchmark)} problems")

        assert len(custom_outputs) == len(benchmark), f"{len(custom_outputs)} != {len(benchmark)}"
        lengths = {len(res) for res in custom_outputs.values()}
        assert len(lengths) == 1, f"Number of responses for each task should be the same: {lengths}"

        eval_samples: list[dict] = [instance.get_evaluation_sample() for instance in benchmark]
        generations: list[list[str]] = [
            [output["solution"] for output in custom_outputs[instance.question_id]] for instance in benchmark
        ]
        metrics, results, metadatas = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=min(64, os.cpu_count()),
            timeout=10,
        )

        graded = extract_instance_results(results)

        save_eval_results = [
            instance.format_evaluation(
                code_list,
                graded_list,
                metadata=meta,
            )
            for instance, code_list, graded_list, meta in zip(benchmark, generations, graded, metadatas, strict=False)
        ]

        # save_eval_results
        output_results = {}
        output_results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        for k in metrics:
            if k.startswith("pass@"):
                logger.info(f"{k}: {metrics[k]}")
                output_results[k] = metrics[k]

        output_results["detail_pass"] = {}
        output_results["eval"] = {}
        difficulty_wise_pass_at_k = {}

        k_list = [k for k in metrics if k.startswith("pass@")]

        for r in save_eval_results:
            output_results["eval"][r["question_id"]] = r
            difficulty = r["difficulty"]

            if difficulty not in difficulty_wise_pass_at_k:
                difficulty_wise_pass_at_k[difficulty] = {k: [] for k in k_list}

            for k in k_list:
                if k in r:
                    difficulty_wise_pass_at_k[difficulty][k].append(r[k])

        for difficulty, k_dict in difficulty_wise_pass_at_k.items():
            output_results["detail_pass"][difficulty] = {}
            for k, values in k_dict.items():
                if values:
                    avg = sum(values) / len(values)
                    logger.info(f"{difficulty} {k}: {avg}")
                    output_results["detail_pass"][difficulty][k] = avg
                else:
                    output_results["detail_pass"][difficulty][k] = 0.0

        with open(Path(output_dir) / f"{self.name}_results.json", "w") as f:
            json.dump(output_results, f, indent=2)


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
