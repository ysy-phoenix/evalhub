import argparse
import json
from collections import defaultdict

DEFAULT_K_VALUES = [2**i for i in range(11)]


def calculate_pass_at_k(results: list[dict], k_values: list[int] = DEFAULT_K_VALUES) -> dict[int, float]:
    pass_at_k = {}
    total = len(results)

    for k in k_values:
        if k > len(results):
            break
        passed = sum(1 for result in results[:k] if result["base_status"] == "pass")
        pass_at_k[k] = passed / min(k, total)

    return pass_at_k


def main(input_file: str):
    with open(input_file) as f:
        data = json.load(f)

    base_pass_at_k = defaultdict(list)
    plus_pass_at_k = defaultdict(list)

    for result in data["eval"].values():
        base_results = [{"base_status": item["base_status"]} for item in result]
        plus_results = [{"base_status": item["plus_status"]} for item in result]

        for k, v in calculate_pass_at_k(base_results).items():
            base_pass_at_k[k].append(v)
        for k, v in calculate_pass_at_k(plus_results).items():
            plus_pass_at_k[k].append(v)

    output_file = input_file.replace("eval_results.json", "summary.json")
    print(f"Saving summary to {output_file}...")
    with open(output_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "base_pass_at_k": {k: sum(v) / len(v) for k, v in base_pass_at_k.items()},
                    "plus_pass_at_k": {k: sum(v) / len(v) for k, v in plus_pass_at_k.items()},
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    main(args.input)
