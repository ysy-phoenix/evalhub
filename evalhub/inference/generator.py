import asyncio

from evalhub.benchmarks.base import Dataset
from evalhub.inference.schemas import (
    GenerationConfig,
    GenerationResult,
    OpenAIClient,
)
from evalhub.utils.logger import logger
from evalhub.utils.pbar import get_progress_bar


class ProgressTracker:
    r"""Optimized progress tracking for generation tasks."""

    def __init__(self, total_samples: int, total_tasks: int):
        self.progress = get_progress_bar()
        self.completed_samples = 0
        self.completed_tasks = set()
        self.sample_progress = None
        self.task_progress = None
        self.total_samples = total_samples
        self.total_tasks = total_tasks

    def __enter__(self):
        self.progress.__enter__()
        self.sample_progress = self.progress.add_task(
            "[bold blue]Generating samples", total=self.total_samples
        )
        self.task_progress = self.progress.add_task(
            "[bold blue]Completing tasks", total=self.total_tasks
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update_sample_progress(self):
        r"""Update sample completion progress."""
        self.completed_samples += 1
        self.progress.update(self.sample_progress, completed=self.completed_samples)

    def update_task_progress(self, task_id: str, current_responses: int):
        r"""Update task completion progress."""
        if task_id not in self.completed_tasks and current_responses >= 1:
            self.completed_tasks.add(task_id)
            self.progress.update(self.task_progress, completed=len(self.completed_tasks))


class LLMGenerator:
    r"""High-performance class for generating responses via OpenAI Compatible APIs."""

    def __init__(self, config: GenerationConfig, system_prompt: str | None = None) -> None:
        self.config = config
        self.client = OpenAIClient(config)
        self.system_prompt = system_prompt

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        r"""Build message list for API call with caching optimization."""
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        return messages

    async def _generate_single_sample(
        self, task_id: str, prompt: str, sample_id: int
    ) -> tuple[str, int, dict[str, str] | None]:
        r"""Generate a single sample with enhanced error handling."""
        messages = self._build_messages(prompt)
        try:
            response = await self.client.complete(messages)
            return (task_id, sample_id, response)
        except Exception as e:
            logger.error(f"Error processing task {task_id} sample {sample_id}: {str(e)}")
            return (task_id, sample_id, None)

    async def generate_async(self, dataset: Dataset) -> list[GenerationResult]:
        r"""Generate responses asynchronously with optimized performance."""
        tasks_list = list(dataset.tasks.values())
        total_tasks = len(tasks_list)
        total_samples = total_tasks * self.config.n_samples
        results: dict[str, list[str]] = {task.task_id: [] for task in tasks_list}

        coroutines = [
            self._generate_single_sample(task.task_id, task.prompt, sample_id)
            for task in tasks_list
            for sample_id in range(self.config.n_samples)
        ]

        optimal_workers = min(len(coroutines), self.config.num_workers, 1024)
        semaphore = asyncio.Semaphore(optimal_workers)

        async def bounded_task(coro):
            r"""Execute with semaphore and timeout protection."""
            async with semaphore:
                try:
                    return await asyncio.wait_for(coro, timeout=self.config.sample_params.timeout)
                except TimeoutError:
                    logger.warning(f"Task timed out after {self.config.sample_params.timeout}s")
                    return (None, None, None)

        with ProgressTracker(total_samples, total_tasks) as tracker:
            tasks = [bounded_task(coro) for coro in coroutines]

            for future in asyncio.as_completed(tasks):
                task_id, sample_id, response = await future

                if task_id is not None:  # Skip timed out tasks
                    tracker.update_sample_progress()

                    if response:
                        results[task_id].append(response)
                        tracker.update_task_progress(task_id, len(results[task_id]))

        return [
            GenerationResult(task_id=task_id, responses=responses)
            for task_id, responses in sorted(results.items())
        ]

    def generate(self, dataset: Dataset) -> list[GenerationResult]:
        r"""Synchronous API."""
        return asyncio.run(self.generate_async(dataset))
