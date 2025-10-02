from collections import Counter

from verl.utils.reward_score import math_dapo, gpqa, mmlu_pro


def _get_majority_answer(answers: list[str], maj_threshold: float = 0.25):
    assert len(answers) > 0, f"list of answers is empty: {answers}"
    filtered_answers = [x for x in answers if x is not None and x != ""]
    if len(filtered_answers) == 0:
        return None, "no answer"
    counter = Counter(filtered_answers)
    majority_answer, count = counter.most_common(1)[0]
    maj_freq = count / len(answers)

    # Keep only votes with maj_freq in [0.5 - delta, 0.5 + delta]
    delta = 0.5 - maj_threshold
    if maj_freq > 0.5 + delta:
        return None, "too high"
    elif maj_freq < 0.5 - delta:
        return None, "too low"
    return majority_answer, ""


def extract_answer(reward_style, solution):
    pred = None
    if reward_style == "maj_math":
        _ok, pred = math_dapo.verify(solution, "")
    elif reward_style == "maj_gpqa":
        pred = gpqa.get_multiple_choice_answer(solution)
    elif reward_style == "maj_mmlu_pro":
        pred = mmlu_pro.extract_answer(solution)
    else:
        raise NotImplementedError(f"Unknown reward style: {reward_style}")
    return pred if pred is not None else ""


def compute_score(reward_style, solution, extra_info, maj_threshold: float = 0.25):
    group_responses = extra_info["group_responses"]
    answers = [extract_answer(reward_style, sol) for sol in group_responses]
    majority_answer, reason = _get_majority_answer(answers, maj_threshold)
    pred = extract_answer(reward_style, solution)

    if majority_answer is None or majority_answer == "":
        if reason == "too high":  # for purpose of nicer reward curves
            reward = 1.0
        else:
            reward = 0.0
    elif pred is None or pred == "":
        reward = -0.5
    else:
        reward = 1.0 if pred == majority_answer else 0.0
    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
    }
