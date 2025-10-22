import base64
import json
import os
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from multiprocessing import Pool

import orjson
from datasets import Dataset, load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class MiniProblem:
    question_content: str
    question_id: str
    starter_code: str
    public_test_cases: list[Test]
    metadata: dict


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        if isinstance(self.contest_date, str):
            self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = orjson.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = orjson.loads(self.private_test_cases)  # type: ignore
        except Exception:
            self.private_test_cases = orjson.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = orjson.loads(self.metadata)  # type: ignore

    @property
    def summary(self) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
        }

    def format_evaluation(
        self,
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.summary
        output["code_list"] = code_list
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [t.input for t in self.public_test_cases + self.private_test_cases],
                    "outputs": [t.output for t in self.public_test_cases + self.private_test_cases],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_livecodebench(meta_data: dict) -> Dataset:
    version = meta_data.get("release_version").lstrip("v")
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        data_files={"test": f"test{version}.jsonl"},
        split="test",
        revision="refs/pr/6",  # FIXME: remove this when the PR is merged
    )
    if meta_data["start_date"] is not None:
        dataset = dataset.filter(lambda line: line["contest_date"] >= meta_data["start_date"])
    if meta_data["end_date"] is not None:
        dataset = dataset.filter(lambda line: line["contest_date"] < meta_data["end_date"])
    return dataset


def create_problem(p):
    return CodeGenerationProblem(**p)


def load_code_generation_dataset(meta_data: dict) -> list[CodeGenerationProblem]:
    dataset = load_livecodebench(meta_data)
    with Pool(os.cpu_count()) as pool:
        dataset = pool.map(create_problem, dataset)
    return dataset


def load_mini_problems(meta_data: dict) -> list[MiniProblem]:
    dataset = load_livecodebench(meta_data)
    return [
        MiniProblem(
            question_content=p["question_content"],
            question_id=p["question_id"],
            starter_code=p["starter_code"],
            public_test_cases=[Test(**t) for t in orjson.loads(p["public_test_cases"])],
            metadata=orjson.loads(p["metadata"]),
        )
        for p in dataset
    ]


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
