import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import aiofiles
import orjson

from evalhub.inference.schemas import GenerationConfig
from evalhub.utils.logger import logger


@dataclass
class Task:
    r"""Base class for all tasks."""

    task_id: str
    prompt: str
    sys_prompt: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.sys_prompt:
            self.sys_prompt = None
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GroundTruth:
    r"""Ground truth for a task."""

    task_id: str
    answer: str


def preprocess_response(func):
    r"""Preprocess the response."""

    @wraps(func)
    def wrapper(self, task_id: str, response: str | None, *args, **kwargs):
        if response is None:
            return ""
        if "</think>" in response:
            response = response.split("</think>")[-1]
            response.removeprefix("<answer>").removesuffix("</answer>")
        return func(self, task_id, response, *args, **kwargs)

    return wrapper


class Dataset(ABC):
    r"""Base class for all datasets."""

    name: ClassVar[str] = ""

    def __init__(
        self,
        name: str | None = None,
        meta_data: dict[str, Any] | None = None,
        reload: bool = False,
        config: GenerationConfig | None = None,
    ):
        self.name = name or self.__class__.name
        self.tasks: dict[str, Task] = {}
        self.groundtruth: dict[str, GroundTruth] = {}
        self.config = config or GenerationConfig()
        self.meta_data: dict[str, Any] = meta_data or {}
        self.cache_dir = Path(os.environ.get("EVALHUB_CACHE_DIR", Path.home() / ".cache" / "evalhub"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.load_cache() or reload:
            self.load_tasks()
            self.save_cache()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # auto decorate the extract_solution method
        if "extract_solution" in cls.__dict__:
            cls.extract_solution = preprocess_response(cls.extract_solution)

    @property
    def system_prompt(self) -> str | None:
        r"""Get system prompt for the dataset."""
        return None

    def load_cache(self) -> bool:
        r"""Load cached results for a task."""
        hash_key = hashlib.md5(json.dumps(self.meta_data).encode()).hexdigest()
        tasks_cache = self.cache_dir / f"{self.name}-{hash_key}-tasks.pkl"
        groundtruth_cache = self.cache_dir / f"{self.name}-{hash_key}-groundtruth.pkl"
        if tasks_cache.exists() and groundtruth_cache.exists():
            with open(tasks_cache, "rb") as f:
                self.tasks = pickle.load(f)
            with open(groundtruth_cache, "rb") as f:
                self.groundtruth = pickle.load(f)
            logger.info(f"Loaded cached results for {self.name} from {self.cache_dir}")
            return True
        return False

    def save_cache(self) -> None:
        r"""Save results to cache."""
        hash_key = hashlib.md5(json.dumps(self.meta_data).encode()).hexdigest()
        tasks_cache = self.cache_dir / f"{self.name}-{hash_key}-tasks.pkl"
        groundtruth_cache = self.cache_dir / f"{self.name}-{hash_key}-groundtruth.pkl"
        with open(tasks_cache, "wb") as f:
            pickle.dump(self.tasks, f)
        with open(groundtruth_cache, "wb") as f:
            pickle.dump(self.groundtruth, f)
        logger.info(f"Saved cached results for {self.name} to {self.cache_dir}")

    async def _init_files(self):
        r"""Initialize the files for the dataset."""
        self.raw_file = await aiofiles.open(self.config.output_dir / f"{self.name}_raw.jsonl", "ab")
        self.sanitized_file = await aiofiles.open(self.config.output_dir / f"{self.name}.jsonl", "ab")

    async def _close_files(self):
        r"""Close the files for the dataset."""
        await self.raw_file.close()
        await self.sanitized_file.close()

    @abstractmethod
    def load_tasks(self):
        r"""Load tasks from the dataset.

        This method should be implemented by derived classes to load their specific tasks.
        """
        raise NotImplementedError("Subclass must implement load_tasks method")

    @abstractmethod
    def format_prompt(self, task: dict) -> str:
        r"""Format the prompt for a specific task.

        Args:
            task: The task to format the prompt for

        Returns:
            Formatted prompt string

        This method should be implemented by derived classes to format prompts
        according to their specific requirements.

        """
        raise NotImplementedError("Subclass must implement format_prompt method")

    @abstractmethod
    def evaluate(self, solution: str, output_dir: PathLike) -> None:
        r"""Evaluate the model on the tasks."""
        raise NotImplementedError("Subclass must implement evaluate method")

    @preprocess_response
    def extract_solution(self, task_id: str, response: str | None) -> str:
        r"""Extract solution from the response."""
        return response or ""

    async def save_single_task(self, task_id: str, responses: list[dict]) -> None:
        r"""Save results for a single task (append mode)."""
        for response in responses:
            await self.raw_file.write(orjson.dumps({"task_id": task_id, "response": response}) + b"\n")
            if "content" in response:  # FIXME: multiturn
                content = response.get("content", "")
            else:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            solution = self.extract_solution(task_id, content)
            await self.sanitized_file.write(orjson.dumps({"task_id": task_id, "solution": solution}) + b"\n")

    def __len__(self) -> int:
        r"""Get number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        r"""Get task by index."""
        return list(self.tasks.values())[idx]

    def get_by_task_id(self, task_id: str) -> Task | None:
        r"""Get a task by its task_id with O(1) complexity."""
        return self.tasks.get(task_id)

    @property
    def task_ids(self) -> list[str]:
        r"""Get all task IDs."""
        return list(self.tasks.keys())

    def add_task(self, task: Task):
        r"""Add a task to the dataset."""
        self.tasks[task.task_id] = task

    def add_groundtruth(self, groundtruth: GroundTruth):
        r"""Add a groundtruth to the dataset."""
        self.groundtruth[groundtruth.task_id] = groundtruth
