import asyncio
import json
from collections import defaultdict
from dataclasses import asdict
from os import PathLike
from pathlib import Path

import aiohttp
import orjson
from openai.types.chat.chat_completion import ChatCompletion

from evalhub.benchmarks.base import Dataset
from evalhub.inference.schemas import (
    GenerationConfig,
    GenerationResult,
)
from evalhub.utils.logger import logger
from evalhub.utils.pbar import get_progress_bar

MAX_RETRIES = 3


def truncate(text: str, max_tokens: int = 1024) -> str:
    r"""Truncate text to max_tokens."""
    return text[: max_tokens // 2] + "\n...[truncated]...\n" + text[-max_tokens // 2 :]


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
        self.sample_progress = self.progress.add_task("[bold blue]Generating samples", total=self.total_samples)
        self.task_progress = self.progress.add_task("[bold blue]Completing tasks", total=self.total_tasks)
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

    async def _create_session(self):
        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=0)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    async def _close_session(self):
        await self._session.close()

    async def complete(
        self, messages: list[dict[str, str]], tools: list[dict[str, str]] | None = None
    ) -> ChatCompletion | None:
        try:
            params = asdict(self.config.sample_params)
            if tools:
                params["tools"] = tools
            if self.config.chat:
                async with self._session.post(
                    url=f"{self.config.base_url}/chat/completions",
                    json={"messages": messages, **params},
                ) as resp:
                    data = await resp.read()
                    response = ChatCompletion(**json.loads(data))
                if response.choices[0].finish_reason == "length":
                    logger.warning("Max tokens exceeded!")
            return response
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"API call failed: {str(e)}")
            return None

    async def _generate_single_sample(
        self, task_id: str, sample_id: str, prompt: str, metadata: dict | None = None
    ) -> tuple[str, str, dict[str, str] | None]:
        r"""Generate a single sample with enhanced error handling."""
        messages = self._build_messages(prompt)
        try:
            for _ in range(MAX_RETRIES):
                response = await self.complete(messages)
                if response is not None:
                    break
            return (task_id, sample_id, response.model_dump())
        except Exception as e:
            logger.error(f"Error processing task {task_id} sample {sample_id}: {str(e)}")
            return (task_id, sample_id, None)

    async def generate_async(
        self, dataset: Dataset, output_dir: PathLike, resume: bool = False
    ) -> list[GenerationResult]:
        r"""Generate responses asynchronously with optimized performance."""
        await self._create_session()
        task_ids, tasks_list = list(dataset.tasks.keys()), list(dataset.tasks.values())
        results: dict[str, list[dict[str, str]]] = {task_id: [] for task_id in task_ids}

        if resume:
            results = self.load_results(dataset, output_dir)
            resume_tasks = defaultdict(int)
            for task_id in task_ids:
                exist = len(results[task_id]) if task_id in results else 0
                resume_tasks[task_id] = max(self.config.n_samples - exist, 0)
        else:
            resume_tasks = dict.fromkeys(task_ids, self.config.n_samples)

        coroutines = [
            self._generate_single_sample(task.task_id, str(sample_id), task.prompt, task.metadata)
            for task in tasks_list
            for sample_id in range(resume_tasks[task.task_id])
        ]
        total_tasks = sum(1 if resume_tasks[task_id] > 0 else 0 for task_id in task_ids)
        total_samples = sum(resume_tasks.values())

        optimal_workers = min(len(coroutines), self.config.num_workers)
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

        await self._close_session()

        return [
            GenerationResult(task_id=task_id, responses=responses) for task_id, responses in sorted(results.items())
        ]

    def generate(self, dataset: Dataset, output_dir: PathLike, resume: bool = False) -> list[GenerationResult]:
        r"""Synchronous API."""
        return asyncio.run(self.generate_async(dataset, output_dir, resume))

    def load_results(self, dataset: Dataset, output_dir: PathLike) -> dict[str, list[dict[str, str]]]:
        r"""Load results from a file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        raw_path = output_dir / f"{dataset.name}_raw.jsonl"
        results = defaultdict(list)
        with open(raw_path, "rb") as f:
            for line in f:
                data = orjson.loads(line)
                results[data["task_id"]].append(data["response"])
        logger.info(f"Loaded {sum(len(res) for res in results.values())} responses")
        return results
