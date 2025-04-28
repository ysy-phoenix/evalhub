#!/bin/bash

BASE_PORT=10086

worker_urls=()
for i in {0..7}; do
    worker_urls+=("http://0.0.0.0:$((BASE_PORT + i))")
done

/usr/bin/python3 -m sglang_router.launch_router \
    --worker-urls "${worker_urls[@]}" \
    --balance-abs-threshold 1 \
    --port 30000
