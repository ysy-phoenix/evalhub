# EvalHub Tutorials

## Adding a New Dataset

EvalHub is designed to be extensible, allowing you to easily add new evaluation datasets. This tutorial walks you through the process of adding a new dataset to the system.

### Overview

To add a new dataset, you'll need to:

1. Create a new dataset directory
2. Implement the dataset class with required methods
3. Register the dataset in the configuration files
4. Test the integration

### Step 1: Create Dataset Directory Structure

First, create a new directory for your dataset in `evalhub/benchmarks/type_of_dataset/your_dataset_name`. The name should reflect the dataset (e.g., `gsm8k`, `humaneval`).

### Step 2: Implement the Dataset Class

In `evalhub/benchmarks/type_of_dataset/your_dataset_name/__init__.py`, implement a class that inherits from the base dataset class. Your class must implement several required methods:

- `register_dataset()`: Register the dataset in the configuration files
- `load_tasks()`: Load tasks from the dataset
- `format_prompt(item: dict[str, Any]) -> str`: Format the prompt for the task
- `extract_solution(task_id: str, response: str) -> str`: Extract the solution from the response
- other methods needed

### Step 3: Register the Dataset

In `evalhub/benchmarks/type_of_dataset/__init__.py`, add your dataset to the appropriate dictionaries:

```python
from .your_dataset_name import YourDatasetName

__all__ = [..., "YourDatasetName"]
```
If you add a new type of dataset, you need to add a new file in `evalhub/benchmarks/__init__.py` to register the dataset.

```python
from .type_of_dataset import *  # noqa
```

### Step 4: Testing Your Dataset

Test your dataset integration:

1. Generation: Test if prompts are correctly generated
   ```bash
   evalhub gen --model "your-model" --tasks your_dataset_name --output-dir ./results
   ```

2. Evaluation: Test if evaluation works correctly
   ```bash
   evalhub eval --tasks your_dataset_name --solutions ./results/your_dataset_name.jsonl --output-dir ./results
   ```
