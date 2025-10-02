from datasets import concatenate_datasets, load_dataset, Dataset
import json

from data.utils.code import parse_description

TIME_LIMIT = 1


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _parse_testtype(entry_point: str):
    if entry_point.strip() != "":
        return "functional"
    else:
        return "stdin"


def _translate_test_cases(tests, entry_point: str, context: str):
    if isinstance(tests, list):
        out_tests = {
            "inputs": [t["input"] for t in tests],
            "outputs": [t["output"] for t in tests],
            "testtype": _parse_testtype(entry_point),
            "fn_name": entry_point,
            "context": context,
            "time_limit": TIME_LIMIT,
        }
    else:
        raise ValueError(f"Unexpected type for tests: {type(tests)}")
    return json.dumps(out_tests, ensure_ascii=False)


def load_leetcode() -> Dataset:
    ds = load_dataset("newfacade/LeetCodeDataset", split="train")

    def format_prompt(ex):
        problem = ex["problem_description"]  # already includes public test cases
        if ex["prompt"].strip() != "":
            problem += f"\n\nYou may use the following imports and definitions: ```python\n{ex['prompt']}\n```"
        if ex["starter_code"].strip() != "" and "def " in ex["starter_code"]:
            problem += f"\n\nYour solution should have the following signature: ```python\n{_parse_signature(ex['starter_code'])}\n```"

        entry_point = ex["entry_point"].split(".")[-1]

        description = parse_description(problem)

        return {
            "kind": "code",
            "dataset": "leetcode",
            "description": description,
            "problem": problem,
            "tests": _translate_test_cases(ex["input_output"], entry_point, context=ex["prompt"]),
        }

    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)
    return concatenate_datasets(processed_shards)
