from datasets import load_dataset, concatenate_datasets, DatasetDict
import os

from data.train.decontaminate import robust_decontaminate_dapo, robust_decontaminate_code
from data.train.deduplicate import dedup
from data.train.create_code_datasets import load_code_dataset
from data.utils.math import process_gsm8k

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "verifiable-corpus")


def filter_with_stats(ds, fn, name: str):
    before_count = len(ds)
    filtered = ds.filter(fn)
    after_count = len(filtered)
    removed = before_count - after_count
    print(f"{name} filter: removed {removed} rows; {after_count} remain (from {before_count}).")

    # Print per-dataset statistics for removed rows
    removed_rows = ds.filter(lambda ex: not fn(ex))
    if len(removed_rows) > 0:
        from collections import Counter
        counts_by_dataset = Counter(removed_rows["dataset"])
        print("Removed rows by dataset:")
        for name, cnt in sorted(counts_by_dataset.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {name}: {cnt}")
    else:
        print("Removed rows by dataset: none")
    return filtered


if __name__ == "__main__":
    # 1. Load and preprocess DAPO-Math-17k (subset: all)
    dapo = load_dataset(
        "open-r1/DAPO-Math-17k-Processed",
        "all",
        split="train"
    )
    dapo = dapo.map(
        lambda ex: {
            **ex,
            "kind": "math",
            "dataset": "dapo_math",
            "description": ex["prompt"],
            "problem": ex["prompt"],
            "answer": str(ex["solution"]),
        },
        remove_columns=dapo.column_names,
        desc="Preprocess DAPO-Math"
    )

    # Apply robust decontamination to DAPO
    dapo = robust_decontaminate_dapo(dapo)

    # 2. Load and preprocess Hendrycks MATH benchmark
    math_ds = load_dataset(
        "nlile/hendrycks-MATH-benchmark",
        split="train"
    )
    math_ds = math_ds.map(
        lambda ex: {
            **ex,
            "kind": "math",
            "dataset": "math",
            "description": ex["problem"],
            "problem": ex["problem"],
            "answer": str(ex.get("answer", "")),
        },
        remove_columns=math_ds.column_names,
        desc="Preprocess MATH"
    )

    # 3. Load and preprocess GSM8K (subset: main)
    gsm = load_dataset(
        "openai/gsm8k",
        "main",
        split="train"
    )
    gsm = gsm.map(process_gsm8k, desc="GSM8K answer extraction").map(
        lambda ex: {
            "kind": "math",
            "dataset": "gsm8k",
            "description": ex["question"],
            "problem": ex["question"],
            "answer": ex["answer"],
        },
        remove_columns=gsm.column_names,
        desc="Preprocess GSM8K"
    )

    math_dataset = dedup(concatenate_datasets([dapo, math_ds, gsm]), threshold=0.8, column="description", out_path="math_dedup_examples.json")

    # 4. Load and combine code datasets
    code_datasets = []
    for dataset_name in ["livecodebench", "taco", "primeintellect", "codeforces", "code_contests", "leetcode"]:
        ds = load_code_dataset(dataset_name)
        code_datasets.append(ds)
    code_dataset = dedup(concatenate_datasets(code_datasets), threshold=0.95, column="description", out_path="code_dedup_examples.json")

    # Apply robust decontamination to code datasets against code test sets
    code_dataset = robust_decontaminate_code(code_dataset)

    # 5. Load and preprocess WebInstruct-verified
    # Drop WebInstruct rows with answer_type "Multiple Choice" or "Other"
    web = load_dataset(
        "TIGER-Lab/WebInstruct-verified",
        split="train"
    )
    web = web.filter(lambda ex: ex.get("answer_type", "") not in ["Multiple Choice", "Other", ""])
    web = web.map(
        lambda ex: {
            **ex,
            "kind": "verifier",
            "dataset": "webinstruct",
            "description": ex["question"],
            "problem": ex["question"],
            "answer": ex["answer"]
        },
        remove_columns=web.column_names
    )
    web = dedup(web, threshold=1.0, column="description", out_path="webinstruct_dedup_examples.json")

    # 6. Merge all datasets
    merged = concatenate_datasets([
        math_dataset,
        code_dataset,
        web
    ])

    # Final filter: drop rows with empty answers
    def has_valid_answer(example):
        if example["kind"] == "code":
            return True

        val = example.get("answer", None)
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip() != ""
        return True

    filtered = filter_with_stats(merged, has_valid_answer, "Final answer")

    # Final filter: drop rows with empty (or too short) descriptions
    def has_valid_description(example, min_length_for_code: int = 100):
        val = example.get("description", None)
        if val is None:
            return False
        if not isinstance(val, str):
            return False
        if example["kind"] == "code" and len(val) < min_length_for_code:
            return False
        return val.strip() != ""

    filtered = filter_with_stats(filtered, lambda ex: has_valid_description(ex, min_length_for_code=100), "Description")

    # 9. Drop columns
    filtered = filtered.select_columns(["kind", "dataset", "description", "problem", "answer", "tests"])

    # 10. Wrap in a DatasetDict
    merged_deduped = DatasetDict({"train": filtered})

    # 11. Validate GSM8K answer extraction
    print("Sample extracted answers from GSM8K:")

    df = merged_deduped["train"].to_pandas()
    print(df[df["dataset"] == "gsm8k"].head(5)["answer"])

    # 12. Save locally
    print(f"Total number of rows: {len(merged_deduped['train'])}")
    print(f"Column names: {merged_deduped['train'].column_names}")
    merged_deduped.save_to_disk(OUTPUT_PATH)

    print("Merged dataset ready. Load with load_from_disk().")
