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

First, create a new directory for your dataset in `src/benchmarks/`. The name should reflect the dataset (e.g., `gsm8k`, `humaneval`).

### Step 2: Implement the Dataset Class

In `src/benchmarks/your_dataset_name/__init__.py`, implement a class that inherits from the base dataset class. Your class must implement several required methods:

- `load_tasks()`: Load tasks from the dataset
- `format_prompt(item: Dict[str, Any]) -> str`: Format the prompt for the task
- `save(results: List[GenerationResult], output_dir: PathLike) -> Path`: Save the results to a file
- `evaluate(solution: PathLike, output_dir: PathLike) -> Tuple[int, int, float]`: Evaluate the results

### Step 3: Register the Dataset

In `src/benchmarks/__init__.py`, add your dataset to the appropriate dictionaries:

```python
from src.benchmarks.your_dataset_name.dataset import YourDatasetName

# Add to dataset map
DATASET_MAP = {
    # Existing datasets...
    "your_dataset_name": YourDatasetName,
}

# If the dataset can be evaluated by EvalHub directly
EVALUATE_DATASETS = [
    # Existing datasets...
    "your_dataset_name",
]

# If it requires third-party evaluation
# THIRD_PARTY_DATASETS = [
#     # Existing datasets...
#     "your_dataset_name",
# ]
```

Also, ensure your dataset is added to the Hugging Face dataset mapping:

```python
# Add to dataset hub mapping
DATASET_HUB = {
    # Existing mappings...
    "your_dataset_name": "org/dataset_name_on_hf",
}
```

### Step 4: Testing Your Dataset

Test your dataset integration:

1. Generation: Test if prompts are correctly generated
   ```bash
   evalhub run --model "your-model" --tasks your_dataset_name --output-dir ./results
   ```

2. Evaluation: Test if evaluation works correctly
   ```bash
   evalhub eval --tasks your_dataset_name --solutions ./results/your_dataset_name.jsonl --output-dir ./results
   ```
