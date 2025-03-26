from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.benchmarks.base import Dataset, Task
from src.inference.utils import (
    GenerationConfig,
    GenerationResult,
    OpenAICompletion,
)
from src.utils.logger import logger


class LLMGenerator:
    r"""Class for generating responses via OpenAI Compatible APIs."""

    def __init__(self, config: GenerationConfig, system_prompt: Optional[str] = None) -> None:
        self.config = config
        self.client = OpenAICompletion(config.base_url, config.api_key)
        self.system_prompt = system_prompt

    def generate_single_sample(self, task: Task) -> Optional[str]:
        r"""Generate a single sample for a task."""
        messages = (
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task.prompt},
            ]
            if self.system_prompt
            else [{"role": "user", "content": task.prompt}]
        )
        try:
            response = self.client.completion(
                is_chat=self.config.is_chat,
                model_name=self.config.model_name,
                prompt=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stop=self.config.stop,
                timeout=self.config.timeout,
            )
            return response
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {str(e)}")
            return None

    def process_single_task(self, task: Task) -> GenerationResult:
        r"""Process a single task."""
        with ThreadPoolExecutor(max_workers=self.config.n_samples) as executor:
            responses = [
                result
                for result in executor.map(
                    self.generate_single_sample, [task] * self.config.n_samples
                )
                if result is not None
            ]

        return GenerationResult(task_id=task.task_id, responses=responses)

    def generate(self, dataset: Dataset) -> List[GenerationResult]:
        r"""Generate responses for all tasks in parallel using ThreadPoolExecutor.map()."""

        tasks = list(dataset.tasks.values())
        results = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task_id = progress.add_task(
                f"[bold blue]Generating responses for {len(tasks)} tasks", total=len(tasks)
            )

            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = [executor.submit(self.process_single_task, task) for task in tasks]

                for future in futures:
                    result = future.result()
                    results.append(result)
                    progress.advance(task_id)
                    success_rate = sum(1 for r in results if r.responses) / len(results)
                    progress.update(
                        task_id,
                        description=f"[bold blue]Generating responses ({success_rate:.1%} success)",
                    )

        results.sort(key=lambda x: x.task_id)
        return results
