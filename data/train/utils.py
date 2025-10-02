from __future__ import annotations

from typing import List
import re


def normalize_text(text: str) -> str:
    """Normalize text for overlap checking.

    Lowercases and collapses whitespace to reduce trivial mismatches.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip()).lower()
    return text


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer; lowercases and strips punctuation."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def normalize_answer(text: str) -> str:
    """Normalize answers for comparison: lowercase, collapse whitespace, strip \boxed{...}."""
    answer = text
    prev = None
    while prev != answer:
        prev = answer
        answer = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", answer)
    return re.sub(r"\s+", " ", answer.strip()).lower()


def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    if not a_tokens or not b_tokens:
        return 0.0
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    inter = len(set_a & set_b)
    if inter == 0:
        return 0.0
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def try_fast_ratio(a: str, b: str) -> float:
    """Fast similarity ratio using RapidFuzz if available, else difflib.

    Uses token_set_ratio from rapidfuzz.fuzz for speed and robustness.
    """
    try:
        from rapidfuzz import fuzz  # type: ignore

        return fuzz.token_set_ratio(a, b) / 100.0
    except Exception:
        import difflib

        return difflib.SequenceMatcher(None, a, b).ratio()


