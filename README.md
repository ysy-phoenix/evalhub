# 🔮 EvalHub

<p align="center">
    <a href="https://github.com/yourusername/evalhub"><img src="https://img.shields.io/badge/Eval-Hub-blue.svg"></a>
    <a href="https://github.com/yourusername/evalhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
    <a href="#-about">📖 About</a> •
    <a href="#-features">✨ Features</a> •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-development">🛠 Development</a> •
    <a href="#-roadmap">🛣 Roadmap</a>
</p>

## 📖 About

All-in-one benchmarking platform for evaluating Large Language Models (LLMs) with comprehensive metrics and standardized testing frameworks.

> [!Warning]
> This project is under active development and the API is not stable yet.

## ✨ Features

- 🔄 **OpenAI API Compatible** - Seamless integration with existing workflows
- 💻 **Command Line Interface** - Easy benchmarking through intuitive commands
- 🧩 **Extensible Framework** - Add custom tasks and evaluation metrics

> [!Important]
> This project is **not** for production-grade applications requiring high robustness and generalizability (like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).
>
> Design principles:
>
> - Separation between generation and evaluation processes
> - Minimal viable code implementation
> - Prioritize simplicity and modularity over comprehensive feature sets
> - Easy to expose evaluation details(prompts, answer extraction, etc.)
>
> We now mainly focus on the evaluation of **code** and **math** benchmarks.


## 📦 Installation

> [!Note]
> [uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Create a new environment
conda create -n evalhub python=3.12 -y
conda activate evalhub

# Recommend using uv to install the package
uv venv --python 3.12
source .venv/bin/activate

# Install the package
uv pip install -r requirements.txt
uv pip install -e ".[dev]" # optional, for development

# Recommend cleaning up cache after pulling the latest changes
rm -rf ~/.cache/evalhub/
```

## 🚀 Quick Start

```bash
# list configs / tasks / help
evalhub configs
evalhub tasks
evalhub run --help

# before running the following commands, serve the model locally
# the default run command will assume the model is served on localhost:30000(i.e. sglang port)
# feel free to pass base_url and api_key to the run command, more details can be found via `evalhub configs`
python -m sglang.launch_server --model-path $HOME/models/Qwen2.5-7B-Instruct
python -m sglang_router.launch_server --model-path $HOME/models/Qwen2.5-Coder-7B-Instruct --dp 4

# humaneval && mbpp
evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ -p temperature=0.2 -p top_p=0.95 # -p key=value to override default config
evalhub run --model Qwen2.5-7B-Instruct --tasks mbpp --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-7B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-7B-Instruct/mbpp.jsonl

# gsm8k
evalhub run --model Qwen2.5-7B-Instruct --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks gsm8k --solutions $HOME/metrics/Qwen2.5-7B-Instruct/gsm8k.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/gsm8k_results.jsonl --max-display 20

# hendrycks_math
evalhub run --model Qwen2.5-7B-Instruct --tasks hendrycks_math --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks hendrycks_math --solutions $HOME/metrics/Qwen2.5-7B-Instruct/hendrycks_math.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/hendrycks_math_results.jsonl --max-display 20

# livecodebench
evalhub run --model Qwen2.5-7B-Instruct --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen2.5-7B-Instruct/livecodebench.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/livecodebench_results.json --show-response --max-display 20
```

For more commands, please refer to [docs/cmds.md](docs/cmds.md).

> [!Note]
> `view` is supported for math and livecodebench tasks only now!

We also provide some all in one scripts in `scripts/`, and feel free to adjust for your own use.

```bash
bash ./scripts/eval_code.sh --model Qwen2.5-Coder-7B-Instruct --temperature 0.6 --max-tokens 4096
```

> [!Important]
> Due to the tightly coupled nature of LiveCodeBench's codebase, despite our efforts to integrate it with minimal cost, we inevitably had to incorporate a significant amount of code from the original repository.
>
> We only suopport codegeneration scenario of LiveCodeBench.

03/26/2025 update: We add a new mode for livecodebench, use [mini-judge](https://github.com/ysy-phoenix/mini-judge) as backend.
- The original LiveCodeBench would return upon encountering the first failed test case
- whereas our new evaluation will execute all test cases.
- As a result, there is a significant difference in speed between the two approaches.
- By default, the original evaluation method is used, but you can modify it [here](src/benchmarks/code/livecodebench/__init__.py).

04/29/2025 update: Evaluation results of r1 recipe reproduction can be found in [docs/baseline.md](docs/baseline.md).

## 🛠 Development

### New Dataset

See [docs/tutorial.md](docs/tutorial.md) for more details.

### Code Quality Tools

> [!Note]
> We use [Ruff](https://github.com/astral-sh/ruff) as our Python linter and formatter.

```bash
# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit Hooks

> [!Note]
> Pre-commit hooks automatically check your code before committing.

```bash
# Installation
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

## 🛣 Roadmap

- [x] code: humaneval && mbpp
- [x] math: gsm8k && hendrycks_math
- [x] code: livecodebench
- [x] math: math500 && AIME-2024
- [x] refactor math framework
- [x] clean up livecodebench framework
- [x] organize docs
- [x] integrate with mini-judge
- [x] code: BigCodeBench

## 🌐 Acknowledgements

- [EvalPlus](https://github.com/evalplus/evalplus)
- [deepscaler](https://github.com/agentica-project/deepscaler)
- [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [verl](https://github.com/volcengine/verl)

## 📄 License

This project is licensed under the terms of the MIT license.
