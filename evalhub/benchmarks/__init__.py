from evalhub.benchmarks.code.bigcodebench import BigCodeBenchDataset
from evalhub.benchmarks.code.humaneval import HumanEvalDataset
from evalhub.benchmarks.code.livecodebench import LiveCodeBenchDataset
from evalhub.benchmarks.general.gpqa import GPQADataset
from evalhub.benchmarks.general.mmlu_redux import MMLUReduxDataset
from evalhub.benchmarks.math.aime2024 import AIME2024Dataset
from evalhub.benchmarks.math.aime2025 import AIME2025Dataset
from evalhub.benchmarks.math.gsm8k import GSM8KDataset
from evalhub.benchmarks.math.hendrycks_math import HendrycksMathDataset
from evalhub.benchmarks.math.math500 import Math500Dataset

DATASET_MAP = {
    "humaneval": HumanEvalDataset,
    "mbpp": HumanEvalDataset,
    "gsm8k": GSM8KDataset,
    "hendrycks_math": HendrycksMathDataset,
    "livecodebench": LiveCodeBenchDataset,
    "math500": Math500Dataset,
    "aime2024": AIME2024Dataset,
    "aime2025": AIME2025Dataset,
    "bigcodebench": BigCodeBenchDataset,
    "gpqa": GPQADataset,
    "mmlu_redux": MMLUReduxDataset,
}

EVALUATE_DATASETS = {
    "gsm8k",
    "hendrycks_math",
    "livecodebench",
    "math500",
    "aime2024",
    "aime2025",
    "gpqa",
    "mmlu_redux",
}

THIRD_PARTY_DATASETS = {"humaneval", "mbpp", "bigcodebench"}
