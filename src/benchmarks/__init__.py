from src.benchmarks.code.bigcodebench import BigCodeBenchDataset
from src.benchmarks.code.humaneval import HumanEvalDataset
from src.benchmarks.code.livecodebench import LiveCodeBenchDataset
from src.benchmarks.general.gpqa import GPQADataset
from src.benchmarks.math.aime2024 import AIME2024Dataset
from src.benchmarks.math.gsm8k import GSM8KDataset
from src.benchmarks.math.hendrycks_math import HendrycksMathDataset
from src.benchmarks.math.math500 import Math500Dataset

DATASET_MAP = {
    "humaneval": HumanEvalDataset,
    "mbpp": HumanEvalDataset,
    "gsm8k": GSM8KDataset,
    "hendrycks_math": HendrycksMathDataset,
    "livecodebench": LiveCodeBenchDataset,
    "math500": Math500Dataset,
    "aime2024": AIME2024Dataset,
    "bigcodebench": BigCodeBenchDataset,
    "gpqa": GPQADataset,
}

EVALUATE_DATASETS = {"gsm8k", "hendrycks_math", "livecodebench", "math500", "aime2024", "gpqa"}

THIRD_PARTY_DATASETS = {"humaneval", "mbpp", "bigcodebench"}
