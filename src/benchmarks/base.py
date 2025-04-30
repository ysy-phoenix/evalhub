import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Optional

import orjson

from src.inference.utils import GenerationConfig, GenerationResult
from src.utils.logger import logger
from src.utils.pbar import get_progress_bar

DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass
class Task:
    r"""Base class for all tasks."""

    task_id: str
    prompt: str
    sys_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        r"""Ensure prompt ends with newline."""
        if not self.prompt.endswith("\n"):
            self.prompt += "\n"
        if not self.sys_prompt:
            self.sys_prompt = None


@dataclass
class GroundTruth:
    r"""Ground truth for a task."""

    task_id: str
    answer: str


class Dataset(ABC):
    r"""Base class for all datasets."""

    name: ClassVar[str] = ""

    def __init__(
        self,
        name: Optional[str] = None,
        meta_data: Optional[dict[str, Any]] = None,
        reload: bool = False,
    ):
        self.name = name or self.__class__.name
        self.tasks: dict[str, Task] = {}
        self.groundtruth: dict[str, GroundTruth] = {}
        self.config = DEFAULT_GENERATION_CONFIG
        self.meta_data: dict[str, Any] = meta_data or {}
        self.cache_dir = Path(
            os.environ.get("EVALHUB_CACHE_DIR", Path.home() / ".cache" / "evalhub")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.load_cache() or reload:
            self.load_tasks()
            self.save_cache()

    @property
    def system_prompt(self) -> Optional[str]:
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
    def extract_solution(self, task_id: str, response: str) -> str:
        r"""Extract solution from the response."""
        raise NotImplementedError("Subclass must implement extract_solution method")

    def sanitize_and_save(self, results: list[GenerationResult], output_dir: PathLike) -> Path:
        r"""Sanitize and save the results."""
        output_dir = Path(output_dir)
        save_path = output_dir / f"{self.name}.jsonl"
        total_samples = sum(len(result.responses) for result in results)

        with open(save_path, "wb") as save_file, get_progress_bar() as progress:
            task = progress.add_task(
                "[bold blue]Sanitizing and saving results", total=total_samples
            )

            for result in results:
                task_id = result.task_id
                for response in result.responses:
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    solution = self.extract_solution(task_id, content)
                    save_file.write(
                        orjson.dumps({"task_id": task_id, "solution": solution}) + b"\n"
                    )
                    progress.update(task, advance=1)

        return save_path

    def save(self, results: list[GenerationResult], output_dir: PathLike) -> tuple[Path, Path]:
        r"""Save raw results to a JSONL file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        raw_path = output_dir / f"{self.name}_raw.jsonl"

        with open(raw_path, "wb") as raw_file:
            for sample in results:
                task_id = sample.task_id
                for response in sample.responses:
                    raw_file.write(orjson.dumps({"task_id": task_id, "response": response}) + b"\n")

        save_path = self.sanitize_and_save(results, output_dir)
        return raw_path, save_path

    def __len__(self) -> int:
        r"""Get number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        r"""Get task by index."""
        return list(self.tasks.values())[idx]

    def get_by_task_id(self, task_id: str) -> Optional[Task]:
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
