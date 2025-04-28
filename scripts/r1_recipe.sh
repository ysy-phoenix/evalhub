#!/bin/bash

# serve model
vllm serve "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" -dp 8 --port 30000 --enable-chunked-prefill --max_num_batched_tokens 16384 --distributed_executor_backend mp
python -m sglang_router.launch_server --model-path "$HOME/models/DeepSeek-R1-Distill-Qwen-32B" --dp-size 8

# aime2024
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-32B" --tasks aime2024 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=64 -p num_workers=320 -p timeout=3600 --system-prompt ""
evalhub eval --tasks aime2024 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/aime2024.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/aime2024_results.jsonl" --max-display 10

# math500
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-32B" --tasks math500 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=256 -p timeout=3600 --system-prompt ""
evalhub eval --tasks math500 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/math500.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/math500_results.jsonl" --max-display 10

# livecodebench
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-32B" --tasks livecodebench --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=256 --system-prompt ""
evalhub eval --tasks livecodebench --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/livecodebench.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/livecodebench_results.json" --max-display 10

# gpqa
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-32B" --tasks gpqa --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=320 --system-prompt ""
evalhub eval --tasks gpqa --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/gpqa.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-32B/gpqa_results.jsonl" --max-display 10
