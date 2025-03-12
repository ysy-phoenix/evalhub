from src.benchmarks import DATASET_MAP
from src.inference.generator import LLMGenerator
from src.utils.logger import logger


def gen(model: str, ds_name: str, output_dir: str) -> None:
    r"""Generate results for a given model and dataset."""
    dataset = DATASET_MAP[ds_name](name=ds_name)
    logger.info(f"Successfully loaded {ds_name} dataset, length: {len(dataset)}")
    generator = LLMGenerator(dataset.config)
    logger.info(f"Successfully loaded {model} generator")
    results = generator.generate(dataset)
    save_path = dataset.save(results, output_dir)
    logger.info(f"Results saved to {save_path}")
