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

## ✨ Features

- 🔄 **OpenAI API Compatible** - Seamless integration with existing workflows
- 💻 **Command Line Interface** - Easy benchmarking through intuitive commands
- 🧩 **Extensible Framework** - Add custom tasks and evaluation metrics

> [!Important]
> This project is not for production-grade applications requiring high robustness and generalizability.
>
> EvalHub focuses on minimal viable code implementation and strictly follows the principle of separation between generation and evaluation processes.
>
> We prioritize simplicity and modularity over comprehensive feature sets.
>
> We now mainly focus on the evaluation of **code** and **math** benchmarks.


## 📦 Installation

> [!Note]
> [uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Create a new environment
conda create -n evalhub python=3.11 -y
conda activate evalhub

# Recommend using uv to install the package
pip install -U pip
pip install uv

# Install the package
uv pip install -r requirements.txt
uv pip install -e ".[dev]" # optional, for development
```

## 🚀 Quick Start

```bash
# list configs / help
evalhub configs
evalhub run --help

# humaneval && mbpp
evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval,mbpp --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/tmp -p temperature=0.2 -p top_p=0.95
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-7B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-7B-Instruct/mbpp.jsonl
```

## 🛠 Development

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
- [ ] math: gsm8k && math
- [ ] code: livecodebench

## 📝 Change Log

<details><summary>[2025-03-12] Initial Release <i>:: click to expand ::</i></summary>
<div>

- [x] Basic CLI implementation
</div>
</details>

## 📄 License

This project is licensed under the terms of the MIT license.
