#!/bin/bash

MODEL=""
TEMPERATURE=0.0
MAX_TOKENS=2048

function show_usage {
    echo "Usage: $0 --model <model_name> [--temperature <temp_value>] [--max-tokens <token_count>]"
    echo "Example: $0 --model Qwen2.5-Coder-7B-Instruct --temperature 0.8 --max-tokens 3000"
    echo ""
    echo "Parameters:"
    echo "  --model <model_name>          Required: The name of the model to evaluate"
    echo "  --temperature <temp_value>    Optional: The temperature for generation (default: 0.0)"
    echo "  --max-tokens <token_count>    Optional: The maximum number of tokens to generate (default: 2048)"
    exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --help)
      show_usage
      ;;
    *)
      echo "Error: Unknown parameter: $1"
      show_usage
      ;;
  esac
done

if [ -z "$MODEL" ]; then
    echo "Error: The model name must be specified"
    show_usage
fi

OUTPUT_DIR="$HOME/metrics/$MODEL"

mkdir -p "$OUTPUT_DIR"

echo "Using model: $MODEL"
echo "Using temperature: $TEMPERATURE"
echo "Using max tokens: $MAX_TOKENS"
echo "Output directory: $OUTPUT_DIR"

evalhub run --model "$MODEL" --tasks humaneval --output-dir "$OUTPUT_DIR" -p temperature="$TEMPERATURE" -p max_tokens="$MAX_TOKENS"
evalplus.evaluate --dataset humaneval --samples "$OUTPUT_DIR/humaneval.jsonl"
evalhub run --model "$MODEL" --tasks mbpp --output-dir "$OUTPUT_DIR" -p temperature="$TEMPERATURE" -p max_tokens="$MAX_TOKENS"
evalplus.evaluate --dataset mbpp --samples "$OUTPUT_DIR/mbpp.jsonl"

evalhub run --model "$MODEL" --tasks livecodebench --output-dir "$OUTPUT_DIR" -p temperature="$TEMPERATURE" -p max_tokens="$MAX_TOKENS"
evalhub eval --tasks livecodebench --solutions "$OUTPUT_DIR/livecodebench.jsonl" --output-dir "$OUTPUT_DIR"
