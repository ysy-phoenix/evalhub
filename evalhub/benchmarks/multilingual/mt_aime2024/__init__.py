from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

MT_AIME2024 = "mt_aime2024"
MT_AIME2024_HUB = "amphora/MCLM"


@register_dataset((MT_AIME2024, MT_AIME2024_HUB, True))
class MTAIME2024Dataset(MathDataset):
    """Dataset class for MT-AIME2024 problems."""

    def __init__(self, name: str = MT_AIME2024, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from MT-AIME2024 dataset."""
        dataset = load_dataset(MT_AIME2024_HUB, "MT-AIME2024", split="test")
        languages = list(dataset[0].keys())
        languages.remove("answer")

        for i, item in enumerate(dataset):
            for lang in languages:
                task = Task(
                    task_id=f"MT-AIME2024/{lang}/{i}",
                    prompt=self.format_prompt(item[lang]),
                )
                groundtruth = GroundTruth(
                    task_id=f"MT-AIME2024/{lang}/{i}",
                    answer=str(item["answer"]),
                )
                self.add_task(task)
                self.add_groundtruth(groundtruth)

    def format_prompt(self, question: str) -> str:
        r"""Format the prompt for MT-AIME2024 task."""
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question
