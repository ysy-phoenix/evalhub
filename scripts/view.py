import orjson

results = "/home/v-yeshengyu/metrics/Qwen2.5-7B-Instruct/gsm8k_results.jsonl"
samples = []
with open(results, "rb") as f:
    for line in f:
        sample = orjson.loads(line)
        samples.append(sample)

cnt = 0
for sample in samples:
    if not sample["correct"]:
        cnt += 1
        print(sample["response"])
        print(f"answer: {sample['ground_truth']}")
        print(f"extracted: {sample['extracted_answer']}")
        if cnt > 10:
            break
        print("-" * 100)
