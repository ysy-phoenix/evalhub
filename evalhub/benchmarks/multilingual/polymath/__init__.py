from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

from .instruction import QUERY_DIC

POLYMATH = "polymath"
POLYMATH_HUB = "Qwen/PolyMath"


@register_dataset((POLYMATH, POLYMATH_HUB, True))
class PolyMathDataset(MathDataset):
    """Dataset class for PolyMath problems."""

    def __init__(self, name: str = POLYMATH, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from PolyMath dataset."""
        configs = get_dataset_config_names(POLYMATH_HUB)
        for lang in configs:
            for split in ["top", "high", "medium", "low"]:
                dataset = load_dataset(POLYMATH_HUB, lang, split=split, download_mode="reuse_cache_if_exists")
                for item in dataset:
                    task = Task(
                        task_id=f"PolyMath/{item['id']}",
                        prompt=self.format_prompt(item, lang),
                    )
                    groundtruth = GroundTruth(
                        task_id=f"PolyMath/{item['id']}",
                        answer=item["answer"],
                    )
                    self.add_task(task)
                    self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any], lang: str) -> str:
        r"""Format the prompt for PolyMath task."""
        question = item["question"]
        instruction_following = QUERY_DIC[lang]

        question += "\n" + instruction_following
        return question
