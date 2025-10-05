from datasets import concatenate_datasets, load_dataset, Dataset
import json

from data.utils.code import parse_description, float_with_default

TIME_LIMIT = 1


def _translate_test_cases(private_tests, generated_tests, time_limit = None):
    inputs = private_tests["input"] + generated_tests["input"]
    outputs = private_tests["output"] + generated_tests["output"]
    assert len(inputs) == len(outputs)
    out_tests = {
        "inputs": inputs,
        "outputs": outputs,
        "testtype": "stdin",
        "fn_name": "",
        "time_limit": float_with_default(time_limit["seconds"] if time_limit is not None else None, TIME_LIMIT),
    }
    return json.dumps(out_tests, ensure_ascii=False)


def load_code_contests(skip_without_solution=True) -> Dataset:
    ds = load_dataset("deepmind/code_contests", split="train")

    def _filter(ex):
        if len(ex["solutions"]["solution"]) == 0:
            return False
        if ex["input_file"] is not None and ex["input_file"].strip() != "":
            return False
        if ex["output_file"] is not None and ex["output_file"].strip() != "":
            return False
        return True

    def format_prompt(ex):
        problem = ex["description"]  # already includes public test cases
        description = parse_description(problem)
        return {
            "kind": "code",
            "dataset": "code_contests",
            "description": description,
            "problem": problem,
            "tests": _translate_test_cases(ex["private_tests"], ex["generated_tests"], time_limit=ex["time_limit"]),
        }

    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        if skip_without_solution:
            shard = shard.filter(_filter)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)
    return concatenate_datasets(processed_shards)
