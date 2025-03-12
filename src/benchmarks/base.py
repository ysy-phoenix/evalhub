from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from src.inference.utils import GenerationConfig, GenerationResult

DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass
class Task:
    r"""Base class for all tasks."""

    task_id: str
    prompt: str

    def __post_init__(self):
        r"""Ensure prompt ends with newline."""
        if not self.prompt.endswith("\n"):
            self.prompt += "\n"


class Dataset(ABC):
    r"""Base class for all datasets."""

    name: ClassVar[str] = ""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.name
        self.tasks: Dict[str, Task] = {}
        self.config = DEFAULT_GENERATION_CONFIG
        self.load_tasks()

    def set(self, key: str, value: Any):
        r"""Set a configuration value."""
        self.config[key] = value

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
