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
- 📊 **Comprehensive Metrics** - Detailed performance analysis and reporting
- 🧪 **Standardized Testing** - Consistent evaluation across different models

## 📦 Installation

```bash
# Create a new environment
conda create -n evalhub python=3.11 -y
conda activate evalhub

# Install the package
pip install -e ".[dev]"
```

## 🚀 Quick Start

```bash
# Run a basic evaluation
evalhub run --model gpt-3.5-turbo --task math::gsm8k --output-dir results
```

## 🛠 Development

### With uv (recommended)

> [!Note]
> [uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv
uv pip install -e ".[dev]"
```

### Code Quality Tools

> [!Important]
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

- [ ] Add more benchmark datasets
- [ ] Implement parallel evaluation
- [ ] Create web dashboard for results visualization
- [ ] Core evaluation framework
- [ ] Documentation setup

## 📝 Change Log

<details><summary>[2025-03-12] Initial Release <i>:: click to expand ::</i></summary>
<div>

- [x] Basic CLI implementation
</div>
</details>

## 📄 License

This project is licensed under the terms of the MIT license.
