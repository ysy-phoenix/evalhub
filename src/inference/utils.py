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
    responses: list[str]


@dataclass
class GenerationConfig:
    r"""Configuration for generation."""

    is_chat: bool = True
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
    output_dir: str = "outputs"
    baseline: bool = False

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

    def __init__(self, base_url: str, api_key: str) -> None:
        r"""Initialize the AsyncOpenAI client."""
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def completion(
        self,
        is_chat: bool,
        prompt: str | list[dict[str, str]],
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 3072,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: list[str] | None = None,
        timeout: int = 1800,
        extra_body: dict | None = None,
    ) -> str:
        r"""Execute a completion request, supporting both chat and text completion."""
        if stop is None:
            stop = DEFAULT_CHAT_STOP_TOKENS if is_chat else DEFAULT_TEXT_STOP_TOKENS

        try:
            params = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "timeout": timeout,
                **(extra_body or {}),
            }

            if is_chat:
                response = await self.client.chat.completions.create(messages=prompt, **params)
                if response.choices[0].finish_reason == "length":  # FIXME: maybe hallucination
                    logger.warning(
                        f"Max tokens exceeded:\n{truncate(response.choices[0].message.content)}"
                    )
                return response.choices[0].message.content
            else:
                response = await self.client.completions.create(prompt=prompt, **params)
                return response.choices[0].text

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return ""
