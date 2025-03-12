import re

from src.utils.logger import logger


def process_output(output: str) -> str:
    r"""Extract the code block from the output."""
    if "```" not in output:
        return output
    try:
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        result = re.findall(pattern, output)
        return result[0][1]
    except Exception as e:
        logger.error(f"Error processing output: {e}")
        return output
