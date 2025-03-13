from src.benchmarks.code.humaneval import HumanEvalDataset
from src.benchmarks.math.gsm8k import GSM8KDataset

DATASET_MAP = {
    "humaneval": HumanEvalDataset,
    "mbpp": HumanEvalDataset,
    "gsm8k": GSM8KDataset,
}

EVALUATE_DATASETS = {"gsm8k"}
