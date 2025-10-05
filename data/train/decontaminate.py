from __future__ import annotations

from typing import Optional, List, Dict, Set, Tuple, Any
from datasets import load_dataset, Dataset
import difflib
import os
import re
import json
import argparse
from tqdm.auto import tqdm

from data.train.utils import normalize_text, tokenize, generate_ngrams
from data.format.code import load_code


def normalize_answer(text: str) -> str:
    """Normalize answers for comparison: lowercase and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove TeX \boxed{...} wrappers if present (repeat to handle nesting)
    answer = text
    prev = None
    while prev != answer:
        prev = answer
        answer = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", answer)
    # Normalize whitespace and lowercase
    return re.sub(r"\s+", " ", answer.strip()).lower()


def load_problem_answer_pairs(repo_id: str, config_name: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Load eval problem/answer pairs from a dataset.

    - Considers only 'test', 'validation', 'dev', 'eval' splits
    - Problem column is either 'question' or 'problem'
    - Answer column is 'answer'
    Returns two aligned lists: [normalized_problem], [normalized_answer]
    """
    split_candidates = ["test", "validation", "dev", "eval"]
    problem_columns = ["question", "problem"]

    for split_name in split_candidates:
        try:
            ds = (
                load_dataset(repo_id, config_name, split=split_name)
                if config_name is not None
                else load_dataset(repo_id, split=split_name)
            )

            # Pick problem column
            problem_col: Optional[str] = None
            for cand in problem_columns:
                if cand in ds.column_names:
                    problem_col = cand
                    break
            if problem_col is None:
                return ([], [])

            # Ensure answer column exists
            if "answer" not in ds.column_names:
                return ([], [])

            problems_raw = ds[problem_col]
            answers_raw = ds["answer"]

            seen_local = set()
            problems: List[str] = []
            answers: List[str] = []
            for p_raw, a_raw in zip(problems_raw, answers_raw):
                if not isinstance(p_raw, str) or not isinstance(a_raw, str):
                    continue
                p = normalize_text(p_raw)
                a = normalize_answer(a_raw)
                # Deduplicate by normalized problem text
                if p and p not in seen_local:
                    seen_local.add(p)
                    problems.append(p)
                    answers.append(a)
            return (problems, answers)
        except Exception:
            continue
    return ([], [])


def robust_decontaminate_dapo(
    dapo: Dataset,
    use_repos: List[Tuple[str, Optional[str]]] | None = None,
) -> Dataset:
    """Decontaminate DAPO-Math against external test sets using n-grams and sequence matching.

    Prints a short summary and one caught example to stdout.
    """
    if use_repos is None:
        use_repos = [
            ("openai/gsm8k", "main"),
            ("math-ai/math500", None),
            ("math-ai/amc23", None),
            ("math-ai/aime24", None),
            ("math-ai/aime25", None),
        ]

    ngram_n = int(os.getenv("DECONTAM_NGRAM_N", "32"))
    ratio_threshold = float(os.getenv("DECONTAM_RATIO_THRESHOLD", "0.75"))
    small_ngram_n = int(os.getenv("DECONTAM_CANDIDATE_TRIGRAM_N", "12"))

    # Load test/eval problem/answer pairs
    test_texts: List[str] = []
    test_answers: List[str] = []
    for repo_id, cfg in use_repos:
        probs, answers = load_problem_answer_pairs(repo_id, cfg)
        test_texts.extend(probs)
        test_answers.extend(answers)

    if len(test_texts) == 0:
        print("Decontamination: no external test items loaded; skipped robust filtering.")
        return dapo

    # Build tokenizations and indices
    test_tokens: List[List[str]] = [tokenize(t) for t in tqdm(test_texts, desc="Tokenize test items")]

    ngram_to_test_indices: Dict[str, Set[int]] = {}
    if ngram_n > 0:
        for idx_t, toks in enumerate(tqdm(test_tokens, desc=f"Index {ngram_n}-grams")):
            for ng in generate_ngrams(toks, ngram_n):
                ngram_to_test_indices.setdefault(ng, set()).add(idx_t)

    small_ngram_to_test_indices: Dict[str, Set[int]] = {}
    if small_ngram_n > 0:
        for idx_t, toks in enumerate(tqdm(test_tokens, desc=f"Index {small_ngram_n}-grams")):
            for ng in generate_ngrams(toks, small_ngram_n):
                small_ngram_to_test_indices.setdefault(ng, set()).add(idx_t)

    dapo_problems: List[str] = [normalize_text(p) for p in dapo["problem"]]
    dapo_answers: List[str] = [normalize_answer(a) for a in dapo["answer"]]
    contaminated_flags: List[bool] = [False] * len(dapo_problems)
    contaminated_reason: List[str] = [""] * len(dapo_problems)
    contaminated_detail: List[Tuple[Optional[int], Optional[str]]] = [(None, None)] * len(dapo_problems)
    contaminated_candidates: List[Optional[Set[int]]] = [None] * len(dapo_problems)

    example_idx: Optional[int] = None
    example_reason: Optional[str] = None
    example_detail: Optional[str] = None

    for i, prob_text in enumerate(tqdm(dapo_problems, desc="Decontam DAPO-Math")):
        toks = tokenize(prob_text)

        # 1) Exact n-gram overlap
        if ngram_n > 0 and len(toks) >= ngram_n:
            for ng in generate_ngrams(toks, ngram_n):
                if ng in ngram_to_test_indices:
                    contaminated_flags[i] = True
                    contaminated_reason[i] = f"ngram_{ngram_n}"
                    contaminated_detail[i] = (None, ng)
                    contaminated_candidates[i] = set(ngram_to_test_indices[ng])
                    if example_idx is None:
                        example_idx = i
                        example_reason = contaminated_reason[i]
                        example_detail = f"matched n-gram: '{ng}'"
                    break
            if contaminated_flags[i]:
                continue

        # 2) Sequence matching with candidate pruning via small n-grams
        candidate_indices: Set[int] = set()
        if small_ngram_n > 0 and len(toks) >= small_ngram_n:
            for sng in generate_ngrams(toks, small_ngram_n):
                if sng in small_ngram_to_test_indices:
                    candidate_indices.update(small_ngram_to_test_indices[sng])

        if not candidate_indices:
            continue

        for cand_idx in candidate_indices:
            ratio = difflib.SequenceMatcher(None, prob_text, test_texts[cand_idx]).ratio()
            if ratio >= ratio_threshold:
                contaminated_flags[i] = True
                contaminated_reason[i] = f"ratio>={ratio_threshold:.2f}"
                contaminated_detail[i] = (cand_idx, None)
                contaminated_candidates[i] = {cand_idx}
                if example_idx is None:
                    example_idx = i
                    example_reason = contaminated_reason[i]
                    example_detail = f"ratio={ratio:.2f} vs test item {cand_idx}"
                break

    # Snapshot state before answer verification for reporting
    initial_contaminated_flags = contaminated_flags.copy()
    initial_reasons = contaminated_reason.copy()
    initial_details = contaminated_detail.copy()
    initial_candidates = contaminated_candidates.copy()

    # Verify answers for all marked items; keep only those with matching answers
    mismatched_kept_indices: List[int] = []
    for i, is_cont in enumerate(contaminated_flags):
        if not is_cont:
            continue
        candidates = contaminated_candidates[i]
        if not candidates:
            # No candidates tracked → keep to avoid false positives
            contaminated_flags[i] = False
            contaminated_reason[i] = ""
            contaminated_detail[i] = (None, None)
            continue
        train_ans = dapo_answers[i]
        has_matching_answer = any(train_ans == test_answers[c_idx] for c_idx in candidates)
        if not has_matching_answer:
            contaminated_flags[i] = False
            contaminated_reason[i] = ""
            contaminated_detail[i] = (None, None)
            mismatched_kept_indices.append(i)

    before_count = len(dapo)
    keep_indices = [idx for idx, is_cont in enumerate(contaminated_flags) if not is_cont]
    dapo = dapo.select(keep_indices)
    after_count = len(dapo)

    num_removed = before_count - after_count
    ngram_removed = sum(1 for r, f in zip(contaminated_reason, contaminated_flags) if f and r.startswith("ngram_"))
    ratio_removed = sum(1 for r, f in zip(contaminated_reason, contaminated_flags) if f and r.startswith("ratio"))
    num_initial_marked = sum(1 for f in initial_contaminated_flags if f)
    num_not_removed_due_to_mismatch = len(mismatched_kept_indices)
    print(
        f"Decontamination (robust): removed {num_removed}/{before_count} items from DAPO-Math. "
        f"Reasons: ngram={ngram_removed}, ratio={ratio_removed}. Kept {after_count}. "
        f"Initially marked: {num_initial_marked}, not removed due to answer mismatch: {num_not_removed_due_to_mismatch}."
    )

    if num_removed > 0 and example_idx is not None and contaminated_flags[example_idx]:
        print("Example contaminated item (from DAPO-Math):")
        print("- Reason:", example_reason)
        print("- Training problem:")
        print(dapo_problems[example_idx][:500])
        det_idx, det_ng = contaminated_detail[example_idx]
        if det_ng is not None:
            print("- Matched n-gram:")
            print(det_ng)
        if det_idx is not None:
            print("- Closest test item:")
            print(test_texts[det_idx][:500])
    elif num_removed > 0:
        # Fallback: print the first still-contaminated example
        try:
            any_idx = next(i for i, f in enumerate(contaminated_flags) if f)
            print("Example contaminated item (from DAPO-Math):")
            print("- Reason:", contaminated_reason[any_idx])
            print("- Training problem:")
            print(dapo_problems[any_idx][:500])
        except StopIteration:
            pass

    print("---------------------")
    print("Examples kept due to answer mismatch:")

    # Print one example of an initially marked item that was kept due to answer mismatch
    # if num_not_removed_due_to_mismatch > 0:
    for i in range(num_not_removed_due_to_mismatch):
        idx = mismatched_kept_indices[i]
        # idx = mismatched_kept_indices[0]
        print("- Initial reason:", initial_reasons[idx])
        print("- Training problem:")
        print(dapo_problems[idx][:500])
        print("- Training answer:")
        print(dapo_answers[idx])
        cand_set = initial_candidates[idx] or set()
        try:
            cand_idx = next(iter(cand_set))
            print("- Candidate test problem:")
            print(test_texts[cand_idx][:500])
            print("- Candidate test answer:")
            print(test_answers[cand_idx])
        except StopIteration:
            pass

    return dapo


def _load_json_records(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL records into a list of dicts.

    - .json -> expects an array of objects
    - .jsonl/.jsonlines -> expects one JSON object per line
    - Also supports an alternative format: a JSON object with key "data" that
      is a list of tuples/lists, where the third element (index 2) is the
      description string. These are converted to {"description": <third>} dicts.
    """
    lower = file_path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".jsonlines"):
        records: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue
        return records
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [obj for obj in data if isinstance(obj, dict)]
        if isinstance(data, dict):
            # Alternative format: { "data": [ (.., .., description), ... ] }
            if "data" in data and isinstance(data["data"], list):
                records: List[Dict[str, Any]] = []
                for item in data["data"]:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        desc = item[2]
                        # Ensure string type for downstream normalization
                        desc_str = desc if isinstance(desc, str) else str(desc)
                        records.append({"description": desc_str})
                return records
            return [data]
        return []


def _write_json_records(file_path: str, records: List[Dict[str, Any]]) -> None:
    lower = file_path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".jsonlines"):
        with open(file_path, "w", encoding="utf-8") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def robust_decontaminate_local(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    text_field: str = "description",
    kind: str = "code",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Decontaminate a local train set against a local test set using n-grams and ratio.

    Uses env vars for thresholds (same names as DAPO flow):
      - DECONTAM_NGRAM_N (default 32)
      - DECONTAM_RATIO_THRESHOLD (default 0.75)
      - DECONTAM_CANDIDATE_TRIGRAM_N (default 12)
    Additionally, only records with train_record["kind"] == kind are considered eligible for
    decontamination; all other records are kept. This gating happens up front to save compute.
    """
    ngram_n = int(os.getenv("DECONTAM_NGRAM_N", "32"))
    ratio_threshold = float(os.getenv("DECONTAM_RATIO_THRESHOLD", "0.75"))
    small_ngram_n = int(os.getenv("DECONTAM_CANDIDATE_TRIGRAM_N", "12"))

    # Gate early by kind to save compute. If no eligible items, keep all.
    eligible_mask: List[bool] = [(rec.get("kind") == kind or kind == "") for rec in train_records]
    eligible_indices: List[int] = [i for i, ok in enumerate(eligible_mask) if ok]
    if not eligible_indices:
        print(f"Decontamination: no train items with kind='{kind}'; keeping all.")
        return train_records, []

    # Build normalized test texts and deduplicate to reduce index size
    seen_tests: Set[str] = set()
    test_texts: List[str] = []
    test_raw_texts: List[str] = []
    for obj in test_records:
        raw = obj.get(text_field, "")
        if not isinstance(raw, str):
            continue
        norm = normalize_text(raw)
        if norm and norm not in seen_tests:
            seen_tests.add(norm)
            test_texts.append(norm)
            test_raw_texts.append(raw)

    if not test_texts:
        print("Decontamination: test set is empty; keeping all train items.")
        return train_records, []

    # Tokenize test texts and build indices
    test_tokens: List[List[str]] = [tokenize(t) for t in tqdm(test_texts, desc="Tokenize test items")]

    ngram_to_test_indices: Dict[str, Set[int]] = {}
    if ngram_n > 0:
        for idx_t, toks in enumerate(tqdm(test_tokens, desc=f"Index {ngram_n}-grams")):
            for ng in generate_ngrams(toks, ngram_n):
                ngram_to_test_indices.setdefault(ng, set()).add(idx_t)

    small_ngram_to_test_indices: Dict[str, Set[int]] = {}
    if small_ngram_n > 0:
        for idx_t, toks in enumerate(tqdm(test_tokens, desc=f"Index {small_ngram_n}-grams")):
            for ng in generate_ngrams(toks, small_ngram_n):
                small_ngram_to_test_indices.setdefault(ng, set()).add(idx_t)

    # Prepare train normalized texts only for eligible items
    train_norm_texts: List[str] = [""] * len(train_records)
    for idx in tqdm(eligible_indices, desc=f"Normalize train {text_field}"):
        obj = train_records[idx]
        raw = obj.get(text_field, "")
        norm = normalize_text(raw if isinstance(raw, str) else "")
        train_norm_texts[idx] = norm

    contaminated_flags: List[bool] = [False] * len(train_norm_texts)
    contaminated_reason: List[str] = [""] * len(train_norm_texts)
    contaminated_detail: List[Tuple[Optional[int], Optional[str]]] = [(None, None)] * len(train_norm_texts)
    # Aggregated contaminated mapping: train_idx -> {train_description, train_dataset, test_descriptions:set}
    contaminated_map: Dict[int, Dict[str, Any]] = {}

    def _ensure_group(idx: int) -> Dict[str, Any]:
        if idx not in contaminated_map:
            contaminated_map[idx] = {
                "train_description": str(train_records[idx].get(text_field, "")),
                "train_dataset": train_records[idx].get("dataset", ""),
                "test_descriptions": set(),
            }
        return contaminated_map[idx]

    example_idx: Optional[int] = None
    example_reason: Optional[str] = None
    example_detail: Optional[str] = None

    for i in tqdm(eligible_indices, desc="Decontam local train"):
        prob_text = train_norm_texts[i]
        toks = tokenize(prob_text)

        # 1) Exact n-gram overlap
        if ngram_n > 0 and len(toks) >= ngram_n:
            for ng in generate_ngrams(toks, ngram_n):
                if ng in ngram_to_test_indices:
                    contaminated_flags[i] = True
                    contaminated_reason[i] = f"ngram_{ngram_n}"
                    contaminated_detail[i] = (None, ng)
                    group = _ensure_group(i)
                    for cand_idx in ngram_to_test_indices[ng]:
                        group["test_descriptions"].add(test_raw_texts[cand_idx])
                    if example_idx is None:
                        example_idx = i
                        example_reason = contaminated_reason[i]
                        example_detail = f"matched n-gram: '{ng}'"
                    break
            if contaminated_flags[i]:
                continue

        # 2) Sequence matching with candidate pruning via small n-grams
        candidate_indices: Set[int] = set()
        if small_ngram_n > 0 and len(toks) >= small_ngram_n:
            for sng in generate_ngrams(toks, small_ngram_n):
                if sng in small_ngram_to_test_indices:
                    candidate_indices.update(small_ngram_to_test_indices[sng])

        if not candidate_indices:
            continue

        matched_any = False
        first_ratio: Optional[float] = None
        first_idx: Optional[int] = None
        for cand_idx in candidate_indices:
            ratio = difflib.SequenceMatcher(None, prob_text, test_texts[cand_idx]).ratio()
            if ratio >= ratio_threshold:
                if not matched_any:
                    contaminated_flags[i] = True
                    contaminated_reason[i] = f"ratio>={ratio_threshold:.2f}"
                    matched_any = True
                    first_ratio = ratio
                    first_idx = cand_idx
                group = _ensure_group(i)
                group["test_descriptions"].add(test_raw_texts[cand_idx])
        if matched_any:
            contaminated_detail[i] = (first_idx, None)
            if example_idx is None and first_ratio is not None and first_idx is not None:
                example_idx = i
                example_reason = contaminated_reason[i]
                example_detail = f"ratio={first_ratio:.2f} vs test item {first_idx}"
            continue

    # Apply new rule for local decontamination: if a train example matches multiple test examples, keep it
    for i, is_cont in enumerate(contaminated_flags):
        if not is_cont:
            continue
        if i in contaminated_map and isinstance(contaminated_map[i].get("test_descriptions"), set):
            if len(contaminated_map[i]["test_descriptions"]) > 1:
                contaminated_flags[i] = False
                contaminated_reason[i] = ""
                contaminated_detail[i] = (None, None)

    before_count = len(train_records)
    keep_indices = [idx for idx, is_cont in enumerate(contaminated_flags) if not is_cont]
    kept_records = [train_records[idx] for idx in keep_indices]
    after_count = len(kept_records)

    num_removed = before_count - after_count
    ngram_removed = sum(1 for r, f in zip(contaminated_reason, contaminated_flags) if f and r.startswith("ngram_"))
    ratio_removed = sum(1 for r, f in zip(contaminated_reason, contaminated_flags) if f and r.startswith("ratio"))
    print(
        f"Decontamination (local): removed {num_removed}/{before_count} items. "
        f"Reasons: ngram={ngram_removed}, ratio={ratio_removed}. Kept {after_count}."
    )

    if num_removed > 0 and example_idx is not None and contaminated_flags[example_idx]:
        print("Example contaminated item (from train):")
        print("- Reason:", example_reason)
        print("- Training text:")
        print(train_norm_texts[example_idx][:500])
        det_idx, det_ng = contaminated_detail[example_idx]
        if det_ng is not None:
            print("- Matched n-gram:")
            print(det_ng)
        if det_idx is not None:
            print("- Closest test text:")
            print(test_texts[det_idx][:500])

    # Convert aggregated map to list and sets to sorted lists
    contaminated_pairs: List[Dict[str, Any]] = []
    for idx, data in tqdm(contaminated_map.items(), desc="Assemble contaminated pairs"):
        test_list = sorted(list(data["test_descriptions"]))
        contaminated_pairs.append(
            {
                "train_description": data["train_description"],
                "train_dataset": data["train_dataset"],
                "test_descriptions": test_list,
            }
        )

    return kept_records, contaminated_pairs


def robust_decontaminate_code(
    code_ds: Dataset,
    test_repo_ids: Optional[List[str]] = None,
) -> Dataset:
    """Decontaminate a combined code training dataset against code test sets.

    - Converts the HF Dataset to local records while preserving original indices
    - Uses robust_decontaminate_local for n-gram and ratio-based filtering
    - Prints a concise summary analogous to robust_decontaminate_dapo
    """
    if test_repo_ids is None:
        test_repo_ids = [
            "open-r1/codeforces",
            "Qwen/CodeElo",
            "livecodebench/code_generation_lite-v6",
            "evalplus/humanevalplus",
            "evalplus/mbppplus",
        ]

    # Prepare train records with stable back-reference via _index
    before_count = len(code_ds)
    train_records: List[Dict[str, Any]] = []
    # Pull only once for performance
    desc_list = code_ds["description"] if "description" in code_ds.column_names else [""] * before_count
    dataset_list = code_ds["dataset"] if "dataset" in code_ds.column_names else ["code_train"] * before_count
    kind_list = code_ds["kind"] if "kind" in code_ds.column_names else ["code"] * before_count
    for i in range(before_count):
        train_records.append({
            "description": desc_list[i] if isinstance(desc_list[i], str) else str(desc_list[i]),
            "dataset": dataset_list[i],
            "kind": kind_list[i],
            "_index": i,
        })

    # Load and merge test datasets into local records
    test_records: List[Dict[str, Any]] = []
    for repo in test_repo_ids:
        try:
            tds = load_code(repo)
            if "description" not in tds.column_names:
                continue
            for txt in tds["description"]:
                if isinstance(txt, str) and txt.strip() != "":
                    test_records.append({"description": txt})
        except Exception:
            # If any test dataset fails to load, skip it to avoid blocking
            continue

    if not test_records:
        print("Decontamination (code robust): no test items loaded; skipping.")
        return code_ds

    kept_records, _pairs = robust_decontaminate_local(
        train_records=train_records,
        test_records=test_records,
        text_field="description",
        kind="code",
    )

    kept_indices: List[int] = [rec["_index"] for rec in kept_records if "_index" in rec]
    after_count = len(kept_indices)
    num_removed = before_count - after_count

    # Produce a brief summary analogous to DAPO flow
    print(
        f"Decontamination (code robust): removed {num_removed}/{before_count} items from code datasets. Kept {after_count}."
    )

    # Return the filtered HF Dataset
    if after_count == before_count:
        return code_ds
    return code_ds.select(kept_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decontaminate train.json against test.json using description field")
    parser.add_argument("--test", required=True, help="Path to test JSON/JSONL file")
    parser.add_argument("--train", required=True, help="Path to train JSON/JSONL file")
    parser.add_argument("--output", help="Optional path to write filtered train JSON/JSONL")
    parser.add_argument("--pairs-output", help="Optional path to write contaminated pairs (train/test descriptions)")
    parser.add_argument("--field", default="description", help="Text field to use (default: description)")
    parser.add_argument("--kind", default="code", help="Only decontaminate training items with this kind (default: code)")
    args = parser.parse_args()

    test_records = _load_json_records(args.test)
    train_records = _load_json_records(args.train)

    print(f"Loaded: test={len(test_records)} records, train={len(train_records)} records")
    kept_records, contaminated_pairs = robust_decontaminate_local(
        train_records,
        test_records,
        text_field=args.field,
        kind=args.kind,
    )
    if args.output:
        _write_json_records(args.output, kept_records)
    if args.pairs_output:
        _write_json_records(args.pairs_output, contaminated_pairs)
    if args.output:
        print(f"Wrote filtered train set: {len(kept_records)} records -> {args.output}")
    else:
        print(f"Filtered train set size: {len(kept_records)} (no output path provided, not saved)")
    if args.pairs_output:
        print(f"Wrote contaminated pairs: {len(contaminated_pairs)} -> {args.pairs_output}")


if __name__ == "__main__":
    main()
