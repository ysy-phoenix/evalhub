import asyncio
from typing import Optional

from src.benchmarks.base import Dataset
from src.inference.utils import (
    GenerationConfig,
    GenerationResult,
    OpenAICompletion,
)
from src.utils.logger import logger
from src.utils.pbar import get_progress_bar


class LLMGenerator:
    r"""Class for generating responses via OpenAI Compatible APIs."""

    def __init__(self, config: GenerationConfig, system_prompt: Optional[str] = None) -> None:
        self.config = config
        self.client = OpenAICompletion(config)
        self.system_prompt = system_prompt

    async def generate_sample(
        self, task_id: str, prompt: str, sample_id: int
    ) -> tuple[str, int, Optional[dict[str, str]]]:
        r"""generate a single sample asynchronously"""
        # TODO: support for non-chat models
        messages = (
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            if self.system_prompt
            else [{"role": "user", "content": prompt}]
        )

        try:
            response = await self.client.completion(messages)
            return (task_id, sample_id, response)
        except Exception as e:
            logger.error(f"Error processing task {task_id} sample {sample_id}: {str(e)}")
            return (task_id, sample_id, None)

    async def generate_async(self, dataset: Dataset) -> list[GenerationResult]:
        r"""High-level asynchronous generation interface."""
        tasks_list = list(dataset.tasks.values())
        total_tasks = len(tasks_list)
        results_dict: dict[str, list[str]] = {task.task_id: [] for task in tasks_list}
        total_samples = total_tasks * self.config.n_samples

        all_coroutines = []
        for task in tasks_list:
            for sample_id in range(self.config.n_samples):
                all_coroutines.append(self.generate_sample(task.task_id, task.prompt, sample_id))

        progress = get_progress_bar()
        completed_samples = 0
        completed_tasks: set[str] = set()

        with progress:
            sample_progress = progress.add_task(
                "[bold blue]Generating samples", total=total_samples
            )
            task_progress = progress.add_task("[bold blue]Completing tasks", total=total_tasks)
            sem = asyncio.Semaphore(min(len(all_coroutines), self.config.num_workers))

            async def bounded_task(coro):
                async with sem:
                    return await coro

            for future in asyncio.as_completed([bounded_task(coro) for coro in all_coroutines]):
                task_id, sample_id, response = await future

                completed_samples += 1
                progress.update(
                    sample_progress,
                    completed=completed_samples,
                )

                if response:
                    results_dict[task_id].append(response)
                    if task_id not in completed_tasks and len(results_dict[task_id]) >= 1:
                        completed_tasks.add(task_id)
                        progress.update(
                            task_progress,
                            completed=len(completed_tasks),
                        )

        results = [
            GenerationResult(task_id=task_id, responses=responses)
            for task_id, responses in results_dict.items()
        ]
        results.sort(key=lambda x: x.task_id)
        return results

    def generate(self, dataset: Dataset) -> list[GenerationResult]:
        r"""Synchronous interface, internally calling the asynchronous implementation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_async(dataset))
