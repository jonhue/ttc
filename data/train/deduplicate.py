from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from datasets import Dataset
import os
from tqdm.auto import tqdm
import json
import math

from data.train.utils import (
    normalize_text,
    tokenize,
    normalize_answer,
)


def dedup(ds: Dataset, column: str = "description", threshold: float = 0.9, out_path: str = "fast_dedup_examples.json") -> Dataset:
    """Fast deduplication by token coverage.

    Removes a row j when at least `threshold` fraction of normalized tokens of one
    description are contained in the other's token set (either direction).

    Always keeps the first occurrence; later covered items are removed. Prints a
    per-dataset summary of removals if the `dataset` column exists.
    """

    if column not in ds.column_names:
        print(f"Deduplicate: column '{column}' not found; skipping.")
        return ds

    texts_raw: List[Optional[str]] = ds[column]
    texts_norm: List[str] = [normalize_text(t if isinstance(t, str) else "") for t in texts_raw]
    answers_raw: Optional[List[Optional[str]]] = ds["answer"] if "answer" in ds.column_names else None
    answers_norm: Optional[List[str]] = [normalize_answer(a if isinstance(a, str) else "") for a in answers_raw] if answers_raw is not None else None
    tokens_list: List[List[str]] = [tokenize(t) for t in tqdm(texts_norm, desc="Tokenize items (fast)")]
    token_sets: List[Set[str]] = [set(toks) for toks in tokens_list]

    keep_flags: List[bool] = [True] * len(texts_norm)
    dataset_col: Optional[List[str]] = ds["dataset"] if "dataset" in ds.column_names else None

    # Tunables
    max_postings = int(os.getenv("FAST_DEDUP_MAX_POSTINGS_PER_TOKEN", "2048"))
    min_token_len = int(os.getenv("FAST_DEDUP_MIN_TOKEN_LENGTH", "2"))
    max_tokens_per_doc = int(os.getenv("FAST_DEDUP_MAX_TOKENS_PER_DOC", "0"))  # 0 = unlimited

    # Inverted index from token to indices of kept items containing it
    token_to_indices: Dict[str, Set[int]] = {}

    # Track first seen exact text to ensure first occurrence is kept
    first_by_text: Dict[Tuple[str, str], int] = {}

    # Track whether we have seen an empty-token document
    first_empty_idx: Optional[int] = None

    # Collect duplicate examples for inspection
    dedup_examples: List[Dict[str, object]] = []

    for i, toks in enumerate(tqdm(tokens_list, desc="Deduplicate")):
        text_i = texts_norm[i]
        ans_i = answers_norm[i] if answers_norm is not None else ""
        # Exact text fast path (with identical answer)
        prev_exact = first_by_text.get((text_i, ans_i))
        if prev_exact is not None:
            keep_flags[i] = False
            # Save example (exact)
            dedup_examples.append({
                "reason": "exact_text",
                "removed_index": i,
                "kept_index": prev_exact,
                "removed_text": texts_norm[i],
                "kept_text": texts_norm[prev_exact],
                "removed_answer": (answers_norm[i] if answers_norm is not None else None),
                "kept_answer": (answers_norm[prev_exact] if answers_norm is not None else None),
                "removed_dataset": (dataset_col[i] if dataset_col is not None else None),
                "kept_dataset": (dataset_col[prev_exact] if dataset_col is not None else None),
                "detail": {"kind": "exact"},
                "overlap_frac": 1.0,
            })
            continue

        set_i = token_sets[i]
        if len(set_i) == 0:
            if first_empty_idx is not None:
                keep_flags[i] = False
                # Save example (empty)
                kept_idx = first_empty_idx
                dedup_examples.append({
                    "reason": "empty",
                    "removed_index": i,
                    "kept_index": kept_idx,
                    "removed_text": texts_norm[i],
                    "kept_text": texts_norm[kept_idx],
                    "removed_answer": (answers_norm[i] if answers_norm is not None else None),
                    "kept_answer": (answers_norm[kept_idx] if answers_norm is not None else None),
                    "removed_dataset": (dataset_col[i] if dataset_col is not None else None),
                    "kept_dataset": (dataset_col[kept_idx] if dataset_col is not None else None),
                    "detail": {"kind": "empty"},
                    "overlap_frac": 1.0,
                })
                continue
            first_empty_idx = i
            first_by_text.setdefault((text_i, ans_i), i)
            continue

        # Select candidate-bearing tokens with frequency and length filters
        candidate_tokens = [
            tok for tok in set_i
            if len(tok) >= min_token_len and len(token_to_indices.get(tok, ())) > 0 and len(token_to_indices.get(tok, ())) <= max_postings
        ]
        if max_tokens_per_doc > 0 and len(candidate_tokens) > max_tokens_per_doc:
            # Prefer rare tokens first
            candidate_tokens = sorted(candidate_tokens, key=lambda t: len(token_to_indices.get(t, ())))[:max_tokens_per_doc]
        else:
            candidate_tokens = sorted(candidate_tokens, key=lambda t: len(token_to_indices.get(t, ())))

        if not candidate_tokens:
            # No overlap possible with kept set; keep and index
            for tok in set_i:
                token_to_indices.setdefault(tok, set()).add(i)
            first_by_text.setdefault((text_i, ans_i), i)
            continue

        # Overlap counting via postings
        req_i = int(math.ceil(threshold * len(set_i)))
        overlap_counts: Dict[int, int] = {}
        removed = False
        matched_cand: Optional[int] = None
        matched_detail: Optional[Dict[str, object]] = None
        for tok in candidate_tokens:
            for cand in token_to_indices.get(tok, ()):  # iterate posting list
                # Require identical normalized answer for a valid candidate
                if answers_norm is not None and answers_norm[cand] != ans_i:
                    continue
                new_val = overlap_counts.get(cand, 0) + 1
                # Early exit if coverage_i_in_c threshold is satisfied
                if new_val >= req_i:
                    keep_flags[i] = False
                    removed = True
                    matched_cand = cand
                    matched_detail = {"kind": "coverage", "trigger": "i_in_c", "overlap": new_val, "req_i": req_i}
                    break
                overlap_counts[cand] = new_val
                # Early exit if coverage_c_in_i threshold is satisfied
                req_c = int(math.ceil(threshold * len(token_sets[cand])))
                if new_val >= req_c:
                    keep_flags[i] = False
                    removed = True
                    matched_cand = cand
                    matched_detail = {"kind": "coverage", "trigger": "c_in_i", "overlap": new_val, "req_c": req_c}
                    break
            if removed:
                break

        if removed:
            # Save example (coverage)
            if matched_cand is not None:
                inter_sz = len(token_sets[i] & token_sets[matched_cand])
                cov_i_in_c = inter_sz / max(len(token_sets[i]), 1)
                cov_c_in_i = inter_sz / max(len(token_sets[matched_cand]), 1)
                dedup_examples.append({
                    "reason": "coverage",
                    "removed_index": i,
                    "kept_index": matched_cand,
                    "removed_text": texts_norm[i],
                    "kept_text": texts_norm[matched_cand],
                    "removed_answer": (answers_norm[i] if answers_norm is not None else None),
                    "kept_answer": (answers_norm[matched_cand] if answers_norm is not None else None),
                    "removed_dataset": (dataset_col[i] if dataset_col is not None else None),
                    "kept_dataset": (dataset_col[matched_cand] if dataset_col is not None else None),
                    "detail": matched_detail or {"kind": "coverage"},
                    "overlap_frac": max(cov_i_in_c, cov_c_in_i),
                })
            continue

        # Keep and update index
        for tok in set_i:
            token_to_indices.setdefault(tok, set()).add(i)
        first_by_text.setdefault((text_i, ans_i), i)

    before_count = len(ds)

    # Per-dataset removal counts before selecting
    removed_counts = None
    if dataset_col is not None:
        try:
            from collections import Counter

            removed_indices = [idx for idx, keep in enumerate(keep_flags) if not keep]
            removed_counts = Counter(dataset_col[idx] for idx in removed_indices)
        except Exception:
            removed_counts = None

    keep_indices = [idx for idx, keep in enumerate(keep_flags) if keep]
    ds = ds.select(keep_indices)
    after_count = len(ds)

    num_removed = before_count - after_count
    print(
        f"Deduplicate: removed {num_removed}/{before_count} items by column '{column}' (threshold={threshold:.2f}). Kept {after_count}."
    )
    if removed_counts is not None:
        if sum(removed_counts.values()) > 0:
            print("Removed rows by dataset:")
            for name, cnt in sorted(removed_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {name}: {cnt}")
        else:
            print("Removed rows by dataset: none")

    # Save dedup examples to JSON if any
    if len(dedup_examples) > 0:
        try:
            dedup_examples.sort(key=lambda e: float(e.get("overlap_frac", 0.0)))
        except Exception:
            pass
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(dedup_examples, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(dedup_examples)} deduplication examples to {out_path}")
        except Exception as e:
            print(f"Failed to write deduplication examples to {out_path}: {e}")

    return ds

