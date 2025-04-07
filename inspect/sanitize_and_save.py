import argparse

from src.benchmarks.code.livecodebench import LiveCodeBenchDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    ds = LiveCodeBenchDataset(name="livecodebench")
    ds.sanitize_and_save(args.raw_file, args.output_dir)


if __name__ == "__main__":
    main()
