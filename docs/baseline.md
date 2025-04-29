## R1 Recipe

Example commands(some helper scripts can be found in [scripts](../scripts)):
```bash
# serve model via vllm or sglang
vllm serve "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" -dp 8 --port 30000
python -m sglang_router.launch_server --model-path "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --router-worker-startup-check-interval 20 --router-balance-abs-threshold 1 --context-length 32768 --dp 8 --port 30000
curl http://127.0.0.1:30000/v1/models

# aime2024
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks aime2024 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=64 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks aime2024 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/aime2024.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/aime2024_results.jsonl" --max-display 10

# math500
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks math500 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks math500 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/math500.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/math500_results.jsonl" --max-display 10

# livecodebench
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks livecodebench --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 --system-prompt ""
evalhub eval --tasks livecodebench --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/livecodebench.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/livecodebench_results.json" --max-display 10

# gpqa
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks gpqa --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 --system-prompt ""
evalhub eval --tasks gpqa --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/gpqa.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/gpqa_results.jsonl" --max-display 10
```

> [!NOTE]
> Adjust `num_workers` for different models.
>
> There seem to be some issues with the data parallel and scheduling implementation of vLLM. It is recommended to use it in conjunction with [sglang_router](https://docs.sglang.ai/router/router.html).
>
> Example: `bash scripts/serve.sh "$HOME/models/DeepSeek-R1-Distill-Qwen-7B"` && `bash scripts/router.sh`

Official Reported results:

| Model                          | AIME 2024 (pass@1) | AIME 2024 (cons@64) | MATH-500 (pass@1) | GPQA iamond (pass@1) | LiveCodeBench pass@1 |
|--------------------------------|--------------------|---------------------|-------------------|-----------------------|---------------------|
| DeepSeek-R1-Distill-Qwen-1.5B  | 28.9               | 52.7                | 83.9              | 33.8                  | 16.9                |
| DeepSeek-R1-Distill-Qwen-7B    | 55.5               | 83.3                | 92.8              | 49.1                  | 37.6                |
| DeepSeek-R1-Distill-Qwen-14B   | 69.7               | 80.0                | 93.9              | 59.1                  | 53.1                |
| DeepSeek-R1-Distill-Qwen-32B   | 72.6               | 83.3                | 94.3              | 62.1                  | 57.2                |

Reproduced results(vllm):

| Model                          | AIME 2024 (pass@1) | AIME 2024 (cons@64) | MATH-500 (pass@1) | GPQA Diamond (pass@1) | LiveCodeBench pass@1 |
|--------------------------------|--------------------|---------------------|-------------------|-----------------------|---------------------|
| DeepSeek-R1-Distill-Qwen-1.5B  | 29.4               | 56.7                | 83.0              | 38.0                  | 17.3                |
| DeepSeek-R1-Distill-Qwen-7B    | 52.0               | 76.7                | 91.7              | 49.5                  | 37.5                |
| DeepSeek-R1-Distill-Qwen-14B   | 69.0               | 80.0                | 93.0              | 61.2                  | 50.3                |
| DeepSeek-R1-Distill-Qwen-32B   | 69.7               | 83.3                | 93.6              | 62.3                  | 56.4                |

Reproduced results(sglang):

| Model                          | AIME 2024 (pass@1) | AIME 2024 (cons@64) | MATH-500 (pass@1) | GPQA Diamond (pass@1) | LiveCodeBench pass@1 |
|--------------------------------|--------------------|---------------------|-------------------|-----------------------|---------------------|
| DeepSeek-R1-Distill-Qwen-1.5B  | 29.3               | 60.0                | 82.0              | 35.7                  | 16.5                |
| DeepSeek-R1-Distill-Qwen-7B    | 53.9               | 76.7                | 91.9              | 50.4                  | 38.3                |
| DeepSeek-R1-Distill-Qwen-14B   | 68.0               | 83.3                | 93.0              | 58.0                  | 51.6                |
| DeepSeek-R1-Distill-Qwen-32B   | 70.2               | 83.3                | 93.2              | 60.5                  | 57.2                |
