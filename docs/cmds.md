## Commands

### Marh

```bash
# gsm8k
evalhub run --model Qwen2.5-7B-Instruct --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks gsm8k --solutions $HOME/metrics/Qwen2.5-7B-Instruct/gsm8k.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/gsm8k_results.jsonl --max-display 20 --log-to-file

# hendrycks_math
evalhub run --model Qwen2.5-7B-Instruct --tasks hendrycks_math --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks hendrycks_math --solutions $HOME/metrics/Qwen2.5-7B-Instruct/hendrycks_math.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/hendrycks_math_results.jsonl --max-display 20 --log-to-file

# math500
evalhub run --model Qwen2.5-Math-7B-Instruct --tasks math500 --output-dir $HOME/metrics/Qwen2.5-Math-7B-Instruct/
evalhub eval --tasks math500 --solutions $HOME/metrics/Qwen2.5-Math-7B-Instruct/math500.jsonl --output-dir $HOME/metrics/Qwen2.5-Math-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-Math-7B-Instruct/math500_results.jsonl --max-display 20 --log-to-file

# aime2024
evalhub run --model Qwen2.5-Math-7B-Instruct --tasks aime2024 --output-dir $HOME/metrics/Qwen2.5-Math-7B-Instruct/
evalhub eval --tasks aime2024 --solutions $HOME/metrics/Qwen2.5-Math-7B-Instruct/aime2024.jsonl --output-dir $HOME/metrics/Qwen2.5-Math-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-Math-7B-Instruct/aime2024_results.jsonl --max-display 20 --log-to-file
```

### Code

```bash
# humaneval && mbpp
evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval,mbpp --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ -p temperature=0.2 -p top_p=0.95
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-7B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-7B-Instruct/mbpp.jsonl

# livecodebench
evalhub run --model Qwen2.5-7B-Instruct --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen2.5-7B-Instruct/livecodebench.jsonl --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-7B-Instruct/livecodebench_results.json --show-response --max-display 20 --log-to-file

# bigcodebench
evalhub run --model Qwen2.5-Coder-7B-Instruct --tasks bigcodebench --output-dir $HOME/metrics/Qwen2.5-Coder-7B-Instruct/
docker pull bigcodebench/bigcodebench-evaluate
docker run -it \
  --name bcb-eval \
  -v $HOME/metrics/:/app/metrics \
  bigcodebench/bigcodebench-evaluate \
  --execution local \
  --split instruct \
  --subset full \
  --samples /app/metrics/Qwen2.5-Coder-7B-Instruct/bigcodebench.jsonl

# run eval next time
docker start bcb-eval
docker exec -it bcb-eval bash
python3 -m bigcodebench.evaluate -execution local --split instruct --subset full --samples /app/data/bigcodebench.jsonl
```
