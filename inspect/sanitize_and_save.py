from pathlib import Path

from src.benchmarks.code.humaneval import HumanEvalDataset

raw_file = Path("~/metrics/OlympicCoder-7B/humaneval-raw.jsonl").expanduser()
output_dir = Path("~/metrics/OlympicCoder-7B").expanduser()


def main():
    ds = HumanEvalDataset(name="humaneval")
    ds.sanitize_and_save(raw_file, output_dir)


if __name__ == "__main__":
    main()
