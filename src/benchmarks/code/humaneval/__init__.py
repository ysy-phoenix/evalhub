from importlib.util import module_from_spec, spec_from_loader
from inspect import getmembers, isfunction

from datasets import load_dataset

from src.benchmarks.base import Task
from src.benchmarks.code.base import CodeDataset
from src.benchmarks.code.humaneval.sanitize import sanitize
from src.benchmarks.config import DATASET_HUB
from src.utils.logger import logger

HUMANEVAL_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


def get_mbpp_entry_point(item: dict) -> str:
    module_name = f"dynamic_module_{item['task_id']}"
    spec = spec_from_loader(module_name, loader=None)
    module = module_from_spec(spec)
    assertion = item["test_list"][0]
    exec(item["code"], module.__dict__)
    functions = [name for name, _ in getmembers(module, isfunction)]
    functions = [func for func in functions if func in assertion]
    if len(functions) > 1:
        logger.warning(
            f"More than one matching function found for task {item['task_id']}: {functions}"
        )
    return functions[0] if functions else None


class HumanEvalDataset(CodeDataset):
    r"""Dataset class for HumanEval/MBPP."""

    def __init__(self, name: str):
        super().__init__(name)
        for key, value in HUMANEVAL_CONFIG.items():
            self.config[key] = value

    def load_tasks(self):
        r"""Load tasks from HumanEval dataset."""
        dataset = load_dataset(DATASET_HUB[self.name], split="test")
        for item in dataset:
            if self.name == "mbpp":
                entry_point = get_mbpp_entry_point(item)
            else:
                entry_point = item["entry_point"]
            task = Task(
                task_id=str(item["task_id"])
                if self.name == "humaneval"
                else f"Mbpp/{item['task_id']}",
                prompt=self.format_prompt(item, "python"),  # TODO: add more languages
                metadata={"entry_point": entry_point},
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
            + f"Here is the given code to do completion:\n```{lang.lower()}\n{prompt}\n```"
        )

    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract the code from the response."""
        if self.name == "humaneval":
            return sanitize(response, self.tasks[task_id].metadata["entry_point"])
        elif self.name == "mbpp":
            return sanitize(response, self.tasks[task_id].metadata["entry_point"])
