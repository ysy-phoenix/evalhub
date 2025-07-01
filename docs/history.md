> [!Important]
> Due to the tightly coupled nature of LiveCodeBench's codebase, despite our efforts to integrate it with minimal cost, we inevitably had to incorporate a significant amount of code from the original repository.
>
> We only support codegeneration scenario of LiveCodeBench.

03/26/2025 update: We add a new mode for livecodebench, use [mini-judge](https://github.com/ysy-phoenix/mini-judge) as backend.
- The original LiveCodeBench would return upon encountering the first failed test case.
- whereas our new evaluation will execute all test cases.
- As a result, there is a significant difference in speed between the two approaches.
- By default, the original evaluation method is used, but you can modify it [here](../evalhub/benchmarks/code/livecodebench/__init__.py).

04/29/2025 update: Evaluation results of r1 recipe reproduction can be found in [docs/baseline.md](docs/baseline.md).

06/06/2025 update: We have added an experimental feature referencing verl's implementation: integration of multi-turn and tool calls.

06/30/2025 update: We have integrated most of the benchmarks from the Qwen3 technical report (excluding those that already have official implementations).
