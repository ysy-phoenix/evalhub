from src.benchmarks.code.humaneval import HumanEvalDataset
from src.benchmarks.math.gsm8k import GSM8KDataset
from src.benchmarks.math.hendrycks_math import HendrycksMathDataset

DATASET_MAP = {
    "humaneval": HumanEvalDataset,
    "mbpp": HumanEvalDataset,
    "gsm8k": GSM8KDataset,
    "hendrycks_math": HendrycksMathDataset,
}

EVALUATE_DATASETS = {"gsm8k", "hendrycks_math"}

THIRD_PARTY_DATASETS = {"humaneval", "mbpp"}
