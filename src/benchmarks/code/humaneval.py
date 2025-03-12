import dataclasses
import os
from pathlib import Path
from typing import List

import orjson
from datasets import load_dataset

from src.benchmarks.base import Dataset, Task
from src.benchmarks.config import DATASET_HUB
from src.inference.utils import GenerationResult

HUMANEVAL_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class HumanEvalDataset(Dataset):
    r"""Dataset class for HumanEval/MBPP."""

    def __init__(self, name: str):
        super().__init__(name)
        for key, value in HUMANEVAL_CONFIG.items():
            self.set(key, value)

    def load_tasks(self):
        r"""Load tasks from HumanEval dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], split="test")
        for item in dataset:
            task = Task(
                task_id=str(item["task_id"]),
                prompt=self.format_prompt(item, "python"),  # TODO: add more languages
            )
            self.add_task(task)

    # Copied from https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/HumanEval/eval_instruct.py
    def format_prompt(self, item: dict, lang: str) -> str:
        r"""Format the prompt for humaneval/mbpp task."""
        if self.name == "humaneval":
            prompt = item["prompt"].strip()
        elif self.name == "mbpp":
            prompt = f'"""\n{item["prompt"]}\n{item["test_list"][0]}\n"""'
        return (
            "Please continue to complete the function. "
            + "You are not allowed to modify the given code and do the completion only. "
            + "Please return all completed function in a codeblock. "
            + f"Here is the given code to do completion:\n'''{lang.lower()}\n{prompt}'''"
        )

    def save(self, results: List[GenerationResult], output_dir: str) -> Path:
        r"""Save results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        save_path = Path(output_dir) / f"{self.name}.jsonl"
        with open(save_path, "wb") as f:
            for result in results:
                f.write(orjson.dumps(dataclasses.asdict(result)) + b"\n")
        return save_path
