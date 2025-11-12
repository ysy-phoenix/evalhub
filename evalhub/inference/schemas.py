from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CHAT_STOP_TOKENS = [
    "<|im_end|>",
    "<|endoftext|>",
]


@dataclass
class GenerationResult:
    r"""Class for storing generation results."""

    task_id: str
    responses: list[dict[str, str]]


@dataclass
class SamplingParams:
    r"""Parameters related to model sampling behavior."""

    model: str = field(
        default=...,
        metadata={
            "help": "Model name or path to use for generation",
        },
    )
    temperature: float = field(
        default=0.6,
        metadata={
            "help": "Controls randomness in generation",
        },
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "Nucleus sampling parameter",
        },
    )
    max_completion_tokens: int = field(
        default=2048,
        metadata={
            "help": "Maximum number of tokens to generate",
        },
    )
    frequency_penalty: float = field(
        default=0,
        metadata={
            "help": "Frequency penalty for token repetition",
        },
    )
    presence_penalty: float = field(
        default=0,
        metadata={
            "help": "Presence penalty for token repetition",
        },
    )
    stop: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of stop sequences for generation",
        },
    )
    timeout: int = field(
        default=3600,
        metadata={
            "help": "API request timeout in seconds",
        },
    )

    def __post_init__(self):
        r"""Ensure stop tokens are set."""
        if self.stop is None:
            self.stop = DEFAULT_CHAT_STOP_TOKENS


@dataclass
class GenerationConfig:
    r"""Configuration for generation with separated concerns."""

    # Task parameters
    tasks: list[str] = field(
        default=...,
        metadata={
            "help": "Tasks to evaluate on(specify multiple --tasks or comma-separated)",
        },
    )

    # Sampling parameters
    sampling_params: SamplingParams = field(
        default_factory=SamplingParams,
        metadata={
            "help": "Sampling parameters for the model",
        },
    )

    system_prompt: str = field(
        default="",
        metadata={
            "help": "System prompt to use for evaluation",
        },
    )

    enable_multiturn: bool = field(
        default=False,
        metadata={
            "help": "Enable multiturn conversation",
        },
    )

    resume: bool = field(
        default=False,
        metadata={
            "help": "Resume from previous run",
        },
    )

    # Generation parameters
    n_samples: int = field(
        default=1,
        metadata={
            "help": "Number of samples to generate per prompt",
        },
    )
    num_workers: int = field(
        default=1024,
        metadata={
            "help": "Number of parallel workers for generation",
        },
    )
    output_dir: Path = field(
        default=Path("outputs"),
        metadata={
            "help": "Directory to save generation outputs",
        },
    )
    tool_config: Path | None = field(
        default=Path("tool_config.json"),
        metadata={
            "help": "Path to the tool configuration file",
        },
    )
    callback: str | None = field(
        default=Path("callback.py"),
        metadata={
            "help": "Path to the callback function script",
        },
    )
    max_turns: int = field(
        default=3,
        metadata={
            "help": "Maximum number of conversation turns",
        },
    )

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if len(self.tasks) == 1 and "," in self.tasks[0]:
            self.tasks = [task.strip() for task in self.tasks[0].split(",")]
        if self.tool_config:
            self.tool_config = Path(self.tool_config)

    def __setitem__(self, key, value):
        r"""Support dictionary-style item assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        elif key in asdict(self.sampling_params):
            setattr(self.sampling_params, key, value)
        else:
            raise KeyError(f"GenerationConfig has no attribute '{key}'")

    def __getitem__(self, key):
        r"""Support dictionary-style item access."""
        if hasattr(self, key):
            return getattr(self, key)
        elif key in asdict(self.sampling_params):
            return getattr(self.sampling_params, key)
        raise KeyError(f"GenerationConfig has no attribute '{key}'")
