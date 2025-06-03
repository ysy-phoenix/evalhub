from typing import Optional

from src.benchmarks import DATASET_MAP
from src.inference.generator import LLMGenerator
from src.utils.logger import logger


def parse_sampling_params(sampling_params: dict) -> dict:
    r"""Parse sampling parameters from command line arguments."""
    ret = {}
    for param in sampling_params:
        key, value = param.split("=", 1)

        # Try to convert string values to appropriate types
        try:
            # Try as int
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                value = int(value)
            # Try as float
            elif "." in value and all(
                part.isdigit() or (i == 0 and part.startswith("-") and part[1:].isdigit())
                for i, part in enumerate(value.split("."))
            ):
                value = float(value)
            # Try as bool
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"
        except ValueError:
            # Keep as string if conversion fails
            pass
        ret[key] = value
    return ret


def gen(
    model: str,
    ds_name: str,
    output_dir: str,
    sampling_params: dict,
    system_prompt: Optional[str] = None,
) -> None:
    r"""Generate results for a given model and dataset."""
    assert ds_name in DATASET_MAP, f"Dataset {ds_name} not supported for generation"
    dataset = DATASET_MAP[ds_name](name=ds_name)
    dataset.config["model"] = model
    for key, value in sampling_params.items():
        try:
            dataset.config[key] = value
        except KeyError:
            logger.warning(f"Parameter {key} not supported!")
    logger.info(f"Successfully loaded {ds_name} dataset, length: {len(dataset)}")
    if system_prompt == "":
        system_prompt = None
    else:
        system_prompt = system_prompt or dataset.system_prompt
    # NOTE: This will override the system prompt in the dataset
    if system_prompt:
        logger.info(f"Using system prompt:\n{system_prompt}")
    else:
        logger.info("Not using system prompt!")

    generator = LLMGenerator(dataset.config, system_prompt)
    results = generator.generate(dataset)
    raw_path, save_path = dataset.save(results, output_dir)
    logger.info(f"Raw results saved to {raw_path}")
    logger.info(f"Solutions saved to {save_path}")
    return raw_path, save_path
