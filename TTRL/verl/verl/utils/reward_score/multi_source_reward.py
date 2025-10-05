from . import math
from . import code
from . import gpqa
from . import mmlu_pro
from . import maj


def compute_score(
    solution: str,
    ground_truth: str,
    reward_style: str,
    extra_info: dict = None,
    sparse_rewards: bool = False,
    max_test_cases: int | None = None,
    maj_threshold: float = 0.25,
) -> dict:
    if reward_style == "code":
        results = code.compute_score(solution, ground_truth, extra_info, sparse_rewards=sparse_rewards, max_test_cases=max_test_cases)
    elif reward_style == "math" or reward_style == "rule":
        results = math.compute_score(solution, ground_truth, extra_info)
    elif reward_style == "gpqa":
        results = gpqa.compute_score(solution, ground_truth)
    elif reward_style == "mmlu_pro":
        results = mmlu_pro.compute_score(solution, ground_truth)
    elif reward_style.startswith("maj_"):
        results = maj.compute_score(reward_style, solution, extra_info, maj_threshold)
    else:
        raise ValueError(f"Reward style {reward_style} not found.")
    return results
