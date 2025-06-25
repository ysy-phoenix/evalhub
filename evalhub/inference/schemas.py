from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CHAT_STOP_TOKENS = [
    "<|eot_id|>",
    "<|im_end|>",
    "</s>",
    "<|EOT|>",
    "<|endoftext|>",
    "<|eos|>",
]
DEFAULT_TEXT_STOP_TOKENS = ["</s>", "<|endoftext|>", "<|eos_token|>"]


@dataclass
class GenerationResult:
    r"""Class for storing generation results."""

    task_id: str
    responses: list[dict[str, str]]


@dataclass
class SampleParams:
    r"""Parameters related to model sampling behavior."""

    model: str = "Qwen/Qwen2.5-Coder7B-Instruct"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
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
    output_dir: Path = Path("outputs")
    tool_config_path: Path | None = Path("tool_config.json")
    callback: str | None = Path("callback.py")
    max_turns: int = 3

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.tool_config_path:
            self.tool_config_path = Path(self.tool_config_path)

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
