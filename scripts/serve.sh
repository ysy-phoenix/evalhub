#!/bin/bash

args=("$@")
MODEL_NAME=${args[0]}
BASE_PORT=10086

for GPU_ID in {0..7}; do
  PORT=$((BASE_PORT + GPU_ID))
  echo "Starting vLLM serve on GPU $GPU_ID at port $PORT"

  CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_NAME" \
    --enable-chunked-prefill \
    --max_num_batched_tokens 16384 \
    --port $PORT \
    --disable-log-requests \
    &
done

echo "All vLLM serve instances started."
