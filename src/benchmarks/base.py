import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from src.inference.utils import GenerationConfig, GenerationResult
from src.utils.logger import logger

DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass
class Task:
    r"""Base class for all tasks."""

    task_id: str
    prompt: str
    sys_prompt: Optional[str] = None

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

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.name
        self.tasks: Dict[str, Task] = {}
        self.groundtruth: Dict[str, GroundTruth] = {}
        self.config = DEFAULT_GENERATION_CONFIG
        self.cache_dir = Path(
            os.environ.get("EVALHUB_CACHE_DIR", Path.home() / ".cache" / "evalhub")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.load_cache():
            self.load_tasks()
            self.save_cache()

    def load_cache(self) -> bool:
        r"""Load cached results for a task."""
        tasks_cache = self.cache_dir / f"{self.name}-tasks.pkl"
        groundtruth_cache = self.cache_dir / f"{self.name}-groundtruth.pkl"
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
        tasks_cache = self.cache_dir / f"{self.name}-tasks.pkl"
        groundtruth_cache = self.cache_dir / f"{self.name}-groundtruth.pkl"
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
    def save(self, results: List[GenerationResult]):
        r"""Save results to a file."""
        raise NotImplementedError("Subclass must implement save method")

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
    def task_ids(self) -> List[str]:
        r"""Get all task IDs."""
        return list(self.tasks.keys())

    def add_task(self, task: Task):
        r"""Add a task to the dataset."""
        self.tasks[task.task_id] = task

    def add_groundtruth(self, groundtruth: GroundTruth):
        r"""Add a groundtruth to the dataset."""
        self.groundtruth[groundtruth.task_id] = groundtruth
