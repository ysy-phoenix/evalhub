import argparse
from collections import defaultdict

import numpy as np
import orjson
from rich.console import Console

from evalhub.utils.metrics import compute_pass_at_k

DEFAULT_KS = [1]
console = Console(width=120, soft_wrap=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--ks", "-k", type=list[int], default=DEFAULT_KS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = []
    rewards = defaultdict(list)
    with open(args.file, "rb") as f:
        for line in f:
            item = orjson.loads(line)
            task_id = item["task_id"]
            reward = sum(item["response"]["reward"].values())
            rewards[task_id].append(reward)
    pass_at_k = defaultdict(list)
    for k in args.ks:
        if k > len(next(iter(rewards.values()))):
            break
        for reward in rewards.values():
            pass_at_k[k].append(compute_pass_at_k(len(reward), sum(r > 0 for r in reward), k))
        console.print(f"Pass@{k}: {np.mean(pass_at_k[k])}")
