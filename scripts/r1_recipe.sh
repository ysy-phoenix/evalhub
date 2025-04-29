#!/bin/bash

# serve model
vllm serve "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" -dp 8 --port 30000 --enable-chunked-prefill --max_num_batched_tokens 16384 --distributed_executor_backend mp
python -m sglang_router.launch_server --model-path "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" --router-worker-startup-check-interval 20 --router-balance-abs-threshold 1 --context-length 32768 --dp 8 --port 30000
curl http://127.0.0.1:30000/v1/models

# aime2024
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" --tasks aime2024 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=64 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks aime2024 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/aime2024.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/aime2024_results.jsonl" --max-display 10

# math500
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" --tasks math500 --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/" -p max_tokens=30720 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 -p timeout=3600 --system-prompt ""
evalhub eval --tasks math500 --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/math500.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/math500_results.jsonl" --max-display 10

# livecodebench
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" --tasks livecodebench --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=1024 --system-prompt ""
evalhub eval --tasks livecodebench --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/livecodebench.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/livecodebench_results.json" --max-display 10

# gpqa
evalhub run --model "$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B" --tasks gpqa --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/" -p max_tokens=28672 -p temperature=0.6 -p top_p=0.95 -p n_samples=4 -p num_workers=768 --system-prompt ""
evalhub eval --tasks gpqa --solutions "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/gpqa.jsonl" --output-dir "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/"
evalhub view --results "$HOME/metrics/DeepSeek-R1-Distill-Qwen-1.5B/gpqa_results.jsonl" --max-display 10
