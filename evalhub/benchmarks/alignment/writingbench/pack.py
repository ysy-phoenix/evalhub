import argparse
from collections import defaultdict

import orjson

from evalhub.benchmarks.alignment.writingbench import WritingBenchDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset = WritingBenchDataset()
    solutions = defaultdict(str)

    with open(args.input, "rb") as f:
        for line in f:
            item = orjson.loads(line)
            solutions[item["task_id"]] = item["solution"]

    with open(args.output, "wb") as f:
        for task_id in dataset.task_ids:
            index = int(task_id.split("/")[-1])
            response = solutions[task_id]
            f.write(orjson.dumps({"index": index, "response": response}) + b"\n")


if __name__ == "__main__":
    main()
