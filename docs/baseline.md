## R1 Recipe

```bash
# serve model
vllm serve "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" -dp 8 --port 30000
python -m sglang_router.launch_server --model-path "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --dp-size 8

# aime2024
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks aime2024 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=64 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks aime2024 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/aime2024.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/aime2024_results.jsonl" --max-display 10

# math500
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks math500 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks math500 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/math500.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/math500_results.jsonl" --max-display 10

# livecodebench
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks livecodebench --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1088 --system-prompt ""
evalhub eval --tasks livecodebench --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/livecodebench.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/livecodebench_results.json" --max-display 10

# gpqa
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" --tasks gpqa --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 --system-prompt ""
evalhub eval --tasks gpqa --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/gpqa.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-7B/gpqa_results.jsonl" --max-display 10
```

Reported results:

| Model                          | AIME 2024 (pass@1) | AIME 2024 (cons@64) | MATH-500 (pass@1)  | GOQA Diamond (pass@1) | LiveCodeBench pass@1 |
|--------------------------------|--------------------|---------------------|--------------------|-----------------------|---------------------|
| DeepSeek-R1-Distill-Qwen-1.5B  | 28.9               | 52.7                | 83.9               | 33.8                  | 16.9                |
| DeepSeek-R1-Distill-Qwen-7B    | 55.5               | 83.3                | 92.8               | 49.1                  | 37.6                |
| DeepSeek-R1-Distill-Qwen-14B   | 69.7               | 80.0                | 93.9               | 59.1                  | 53.1                |
| DeepSeek-R1-Distill-Qwen-32B   | 72.6               | 83.3                | 94.3               | 62.1                  | 57.2                |
