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
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gsm8k --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k_results.jsonl --max-display 20

# hendrycks_math
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks hendrycks_math --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks hendrycks_math --solutions $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math_results.jsonl --max-display 20

# math500
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks math500 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks math500 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/math500.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/math500_results.jsonl --max-display 20

# aime2024
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks aime2024 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks aime2024 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/aime2024.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/aime2024_results.jsonl --max-display 20

# gpqa
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gpqa --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gpqa --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gpqa.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gpqa_results.jsonl --max-display 20
```

### Code

```bash
# humaneval && mbpp
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks humaneval --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/ -p temperature=0.2 -p top_p=0.95 # -p key=value to override default config
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks mbpp --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-3B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-3B-Instruct/mbpp.jsonl

# livecodebench
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench_results.json --max-display 20

# bigcodebench
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks bigcodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
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

### multi-turn & tool call or callback

#### gsm8k with tool call
```bash
temperature=0.6
top_p=0.95
max_tokens=4096
tool_config_path="$HOME/projects/evalhub/evalhub/tools/config/gsm8k_tool_config.yaml"
system_prompt="You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the \`calc_gsm8k_reward\` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary."

evalhub gen --model "$HOME/models/Qwen2.5-7B-Instruct" --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ --max-tokens $max_tokens --temperature $temperature --top-p $top_p --tool-config-path $tool_config_path --enable-multiturn --system-prompt "$system_prompt"
```

#### livecodebench with callback
```bash
temperature=0.6
top_p=0.95
max_tokens=4096
system_prompt="You are an expert Python programmer. \
You will be given a question (problem specification) and \
will generate a correct Python program that matches the specification and passes all tests. \
We will provide you with feedback of public test cases results to help you improve your code."

evalhub gen --model "$HOME/models/Qwen2.5-7B-Instruct" --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ --max-tokens $max_tokens --temperature $temperature --top-p $top_p  --enable-multiturn --system-prompt "$system_prompt" --callback "evalhub.callback.code_callback.CodeCallback"
```
