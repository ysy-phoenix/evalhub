from dataclasses import asdict, dataclass, field

from openai import AsyncOpenAI

from src.utils.logger import logger

DEFAULT_CHAT_STOP_TOKENS = [
    "<|eot_id|>",
    "<|im_end|>",
    "</s>",
    "<|EOT|>",
    "<|endoftext|>",
    "<|eos|>",
]
DEFAULT_TEXT_STOP_TOKENS = ["</s>", "<|endoftext|>", "<|eos_token|>"]


def truncate(text: str, max_tokens: int = 1024) -> str:
    r"""Truncate text to max_tokens."""
    return text[: max_tokens // 2] + "\n...[truncated]...\n" + text[-max_tokens // 2 :]


@dataclass
class GenerationResult:
    r"""Class for storing generation results."""

    task_id: str
    responses: list[dict[str, str]]


@dataclass
class SampleParams:
    r"""Parameters related to model sampling behavior."""

    model: str = "Qwen/Qwen2.5-Coder7B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: list[str] | None = None
    timeout: int = 1800

    def __post_init__(self):
        r"""Ensure stop tokens are set."""
        if self.stop is None:
            self.stop = DEFAULT_CHAT_STOP_TOKENS


@dataclass
class GenerationConfig:
    r"""Configuration for generation with separated concerns."""

    # Sampling parameters
    sample_params: SampleParams = field(default_factory=SampleParams)

    # Generation parameters
    chat: bool = True
    n_samples: int = 1
    num_workers: int = 1024
    base_url: str = "http://localhost:30000/v1"
    api_key: str = "EMPTY"
    output_dir: str = "outputs"

    def __setitem__(self, key, value):
        r"""Support dictionary-style item assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        elif key in asdict(self.sample_params):
            setattr(self.sample_params, key, value)
        else:
            raise KeyError(f"GenerationConfig has no attribute '{key}'")

    def __getitem__(self, key):
        r"""Support dictionary-style item access."""
        if hasattr(self, key):
            return getattr(self, key)
        elif key in asdict(self.sample_params):
            return getattr(self.sample_params, key)
        raise KeyError(f"GenerationConfig has no attribute '{key}'")


class OpenAIClient:
    r"""High-performance API client for OpenAI compatible APIs."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    async def complete(self, messages: list[dict[str, str]]) -> dict[str, str] | None:
        r"""Execute a completion request with optimized error handling."""
        try:
            params = asdict(self.config.sample_params)

            if self.config.chat:
                response = await self.client.chat.completions.create(messages=messages, **params)
                if response.choices[0].finish_reason == "length":
                    logger.warning(
                        f"Max tokens exceeded:\n{truncate(response.choices[0].message.content)}"
                    )
            else:
                response = await self.client.completions.create(prompt=messages, **params)

            return response.model_dump()

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return None
