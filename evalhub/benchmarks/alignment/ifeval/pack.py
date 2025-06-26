import argparse
from collections import defaultdict

import orjson

from evalhub.benchmarks.alignment.ifeval import IFEVALDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset = IFEVALDataset()
    solutions = defaultdict(str)

    with open(args.input, "rb") as f:
        for line in f:
            item = orjson.loads(line)
            solutions[item["task_id"]] = item["solution"]

    with open(args.output, "wb") as f:
        for task_id in dataset.task_ids:
            prompt = dataset.get_by_task_id(task_id).prompt
            response = solutions[task_id]
            f.write(orjson.dumps({"prompt": prompt, "response": response}) + b"\n")


if __name__ == "__main__":
    main()
