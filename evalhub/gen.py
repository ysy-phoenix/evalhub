from evalhub.benchmarks import DATASET_MAP
from evalhub.benchmarks.base import Dataset
from evalhub.inference.generator import LLMGenerator
from evalhub.inference.multiturn import MultiTurnGenerator
from evalhub.inference.schemas import GenerationConfig
from evalhub.utils.logger import logger


def generate(config: GenerationConfig, task: str, override_args: str | None) -> None:
    r"""Generate results for a given model and dataset."""
    assert task in DATASET_MAP, f"Dataset {task} not supported for generation"
    dataset: Dataset = DATASET_MAP[task](name=task, config=config, override_args=override_args)
    logger.info(f"Successfully loaded {task} dataset, length: {len(dataset)}")

    if config.system_prompt == "":
        system_prompt = None
    else:
        system_prompt = config.system_prompt or dataset.system_prompt

    # NOTE: This will override the system prompt in the dataset
    if system_prompt:
        logger.info(f"Using system prompt:\n{system_prompt}")
    else:
        logger.info("Not using system prompt!")

    if config.enable_multiturn:
        generator = MultiTurnGenerator(config, system_prompt)
    else:
        generator = LLMGenerator(config, system_prompt)

    if config.resume:
        logger.info(f"Resuming generation from {config.output_dir}")

    generator.generate(dataset)
