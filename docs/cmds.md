## Commands

### serve model

```bash
# serve model via vllm or sglang
vllm serve "$HOME/models/Qwen2.5-3B-Instruct" --port 30000
python -m sglang.launch_server --model-path "$HOME/models/Qwen2.5-3B-Instruct"
```

### Math

```bash
# gsm8k
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gsm8k --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k_results.jsonl --max-display 20

# hendrycks_math
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks hendrycks_math --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks hendrycks_math --solutions $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math_results.jsonl --max-display 20

# math500
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks math500 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks math500 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/math500.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/math500_results.jsonl --max-display 20

# aime2024
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks aime2024 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks aime2024 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/aime2024.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/aime2024_results.jsonl --max-display 20

# gpqa
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gpqa --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gpqa --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gpqa.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gpqa_results.jsonl --max-display 20
```

### Code

```bash
# humaneval && mbpp
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks humaneval --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/ -p temperature=0.2 -p top_p=0.95 # -p key=value to override default config
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks mbpp --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-3B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-3B-Instruct/mbpp.jsonl

# livecodebench
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench_results.json --max-display 20

# bigcodebench
evalhub run --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks bigcodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
docker pull bigcodebench/bigcodebench-evaluate
docker run -it \
  --name bcb-eval \
  -v $HOME/metrics/:/app/metrics \
  bigcodebench/bigcodebench-evaluate \
  --execution local \
  --split instruct \
  --subset full \
  --samples /app/metrics/Qwen2.5-3B-Instruct/bigcodebench.jsonl

# run eval next time
docker start bcb-eval
docker exec -it bcb-eval bash
python3 -m bigcodebench.evaluate -execution local --split instruct --subset full --samples /app/data/bigcodebench.jsonl
```
