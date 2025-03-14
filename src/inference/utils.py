from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from openai import OpenAI

from src.utils.logger import logger


@dataclass
class GenerationResult:
    r"""Class for storing generation results."""

    task_id: str
    responses: List[str]


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
    num_workers: int = 512
    timeout: int = 200
    stop: List[str] = None
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
        r"""Initialize the OpenAI client."""
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def completion(
        self,
        is_chat: bool,
        prompt: Union[str, List[Dict[str, str]]],
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 3072,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Optional[List[str]] = None,
        timeout: int = 200,
        extra_body: Dict = None,
    ) -> str:
        r"""Execute a completion request, supporting both chat and text completion."""
        # Set default stop tokens based on completion type
        if stop is None:
            stop = (
                [
                    "<|eot_id|>",
                    "<|im_end|>",
                    "</s>",
                    "<|EOT|>",
                    "<|endoftext|>",
                    "<|eos|>",
                ]
                if is_chat
                else ["</s>", "<|endoftext|>", "<|eos_token|>"]
            )

        try:
            # Common parameters for both completion types
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
                response = self.client.chat.completions.create(messages=prompt, **params)
                if response.choices[0].finish_reason == "length":
                    return ""
                return response.choices[0].message.content
            else:
                response = self.client.completions.create(prompt=prompt, **params)
                return response.choices[0].text

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return ""
