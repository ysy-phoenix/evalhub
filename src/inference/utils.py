from dataclasses import dataclass

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
class GenerationConfig:
    r"""Configuration for generation."""

    chat: bool = True
    model_name: str = "Qwen/Qwen2.5-Coder7B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048
    frequency_penalty: float = 0
    presence_penalty: float = 0
    n_samples: int = 1
    num_workers: int = 1024
    timeout: int = 1800
    stop: list[str] = None
    base_url: str = "http://localhost:30000/v1"
    api_key: str = "EMPTY"
    backend: str = "sglang"
    think: bool = False
    output_dir: str = "outputs"

    def __post_init__(self):
        r"""Ensure stop tokens are set."""
        if self.stop is None:
            self.stop = ["</s>", "<|endoftext|>", "<|eos_token|>"]

    def __setitem__(self, key, value):
        r"""Support dictionary-style item assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"GenerationConfig has no attribute '{key}'")

    def __getitem__(self, key):
        r"""Support dictionary-style item access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"GenerationConfig has no attribute '{key}'")

    def update(self, **kwargs):
        r"""Update multiple configuration attributes at once."""
        for key, value in kwargs.items():
            self[key] = value
        return self


class OpenAICompletion:
    r"""Class for completing OpenAI Compatible APIs."""

    def __init__(self, config: GenerationConfig) -> None:
        r"""Initialize the AsyncOpenAI client."""
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    async def completion(self, messages: list[dict[str, str]] | list[str]) -> dict[str, str]:
        r"""Execute a completion request, supporting both chat and text completion."""
        if self.config.stop is None:
            stop = DEFAULT_CHAT_STOP_TOKENS if self.config.chat else DEFAULT_TEXT_STOP_TOKENS
        else:
            stop = self.config.stop

        try:
            extra_body = {}
            if self.config.backend == "sglang" and self.config.think:
                extra_body["separate_reasoning"] = True
            params = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stop": stop,
                "timeout": self.config.timeout,
                **(extra_body or {}),
            }

            if self.config.chat:
                response = await self.client.chat.completions.create(messages=messages, **params)
                if response.choices[0].finish_reason == "length":  # FIXME: maybe hallucination
                    logger.warning(
                        f"Max tokens exceeded:\n{truncate(response.choices[0].message.content)}"
                    )
            else:
                response = await self.client.completions.create(prompt=messages, **params)
            return response.model_dump()

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return ""
