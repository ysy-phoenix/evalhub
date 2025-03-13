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


def extract_livecodebench_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
