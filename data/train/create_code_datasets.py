from datasets import Dataset
import json

from data.utils.livecodebench import LCB_TRAIN_CUTOFF, load_livecodebench
from data.utils.taco import load_taco
from data.utils.primeintellect import load_primeintellect
from data.utils.codeforces import load_codeforces
from data.utils.code_contests import load_code_contests
from data.utils.leetcode import load_leetcode

MIN_TESTS_DEFAULT = 5
MIN_TESTS = {
    "livecodebench": MIN_TESTS_DEFAULT,
    "taco": MIN_TESTS_DEFAULT,
    "primeintellect": MIN_TESTS_DEFAULT,
    "codeforces": MIN_TESTS_DEFAULT,
    "code_contests": MIN_TESTS_DEFAULT,
    "kodcode": MIN_TESTS_DEFAULT,
    "leetcode": MIN_TESTS_DEFAULT,
}
INPUT_LEN_THRESHOLD = 1000


def load_code_dataset(dataset_name: str) -> Dataset:
    if dataset_name == "livecodebench":
        ds = load_livecodebench(dataset_split="train")
    elif dataset_name == "taco":  # contains some of codeforces, leetcode
        ds = load_taco()
    elif dataset_name == "primeintellect":  # contains some of apps, code_contests
        ds = load_primeintellect()
    elif dataset_name == "codeforces":
        ds = load_codeforces(dataset_split="train")
    elif dataset_name == "code_contests":
        ds = load_code_contests()
    # elif dataset_name == "kodcode":
    #     ds = load_kodcode()
    # elif dataset_name == "apps":
    #     ds = load_apps()
    elif dataset_name == "leetcode":
        ds = load_leetcode()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    def _filter(ex):
        tests = json.loads(ex["tests"])
        if len(tests["inputs"]) < MIN_TESTS[dataset_name]:
            if len(tests["inputs"]) > 0 and len(tests["inputs"][0]) >= INPUT_LEN_THRESHOLD:
                return True
            return False
        if ex["description"] is None or ex["description"].strip() == "":
            return False
        if len(ex["description"]) == len(ex["problem"]):
            return False
        return True

    filtered_ds = ds.filter(_filter)
    print(f"Loaded {len(filtered_ds)} rows from {dataset_name} out of {len(ds)}")
    return filtered_ds
