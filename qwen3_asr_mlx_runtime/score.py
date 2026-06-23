#!/usr/bin/env python3
"""Score Qwen3-ASR predictions against an expected JSONL manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ASR predictions with WER and CER")
    parser.add_argument("--expected", required=True, help="JSONL file with audio and text fields")
    parser.add_argument(
        "--predictions",
        default="-",
        help="Perftest JSON file, bridge JSONL file, or - for stdin",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Keep case during normalization. Default lowercases text.",
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Keep punctuation during normalization. Default removes punctuation.",
    )
    return parser.parse_args()


def normalize_text(text: str, *, case_sensitive: bool, keep_punctuation: bool) -> str:
    normalized = text if case_sensitive else text.lower()
    if not keep_punctuation:
        normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_value != right_value)
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def key_variants(path_value: str) -> list[str]:
    path = Path(path_value).expanduser()
    variants = [path_value, str(path)]
    try:
        variants.append(str(path.resolve()))
    except OSError:
        pass
    variants.append(path.name)

    seen = set()
    deduped = []
    for variant in variants:
        if variant and variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def read_expected(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record.get("audio"), str):
                raise ValueError(f"{path}:{line_number}: missing audio field")
            if not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_number}: missing text field")
            records.append(record)
    return records


def prediction_items_from_object(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return [item for item in value["items"] if isinstance(item, dict)]
    if isinstance(value, dict) and value.get("type") == "transcript":
        return [value]
    if isinstance(value, dict) and isinstance(value.get("audio"), str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def read_predictions(path: str) -> list[dict[str, Any]]:
    if path == "-":
        content = sys.stdin.read()
    else:
        content = Path(path).expanduser().read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        return []

    try:
        return prediction_items_from_object(json.loads(stripped))
    except json.JSONDecodeError:
        items = []
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            items.extend(prediction_items_from_object(json.loads(line)))
        return items


def predicted_text(record: dict[str, Any]) -> str:
    for field in ("transcript", "text"):
        value = record.get(field)
        if isinstance(value, str):
            return value
    return ""


def score(args: argparse.Namespace) -> dict[str, Any]:
    expected = read_expected(args.expected)
    predictions = read_predictions(args.predictions)

    prediction_by_key: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        audio = prediction.get("audio")
        if not isinstance(audio, str):
            continue
        for key in key_variants(audio):
            prediction_by_key.setdefault(key, prediction)

    per_item = []
    word_errors = 0
    word_count = 0
    char_errors = 0
    char_count = 0
    missing_predictions = 0
    skipped_empty_refs = 0

    for reference in expected:
        reference_text = normalize_text(
            reference["text"],
            case_sensitive=args.case_sensitive,
            keep_punctuation=args.keep_punctuation,
        )
        if not reference_text:
            skipped_empty_refs += 1
            continue

        prediction = None
        for key in key_variants(reference["audio"]):
            prediction = prediction_by_key.get(key)
            if prediction is not None:
                break
        if prediction is None:
            missing_predictions += 1
            continue

        candidate_text = normalize_text(
            predicted_text(prediction),
            case_sensitive=args.case_sensitive,
            keep_punctuation=args.keep_punctuation,
        )
        reference_words = reference_text.split()
        candidate_words = candidate_text.split()
        item_word_errors = edit_distance(reference_words, candidate_words)
        item_char_errors = edit_distance(list(reference_text), list(candidate_text))

        word_errors += item_word_errors
        word_count += len(reference_words)
        char_errors += item_char_errors
        char_count += len(reference_text)
        per_item.append(
            {
                "audio": reference["audio"],
                "wer": item_word_errors / len(reference_words) if reference_words else None,
                "cer": item_char_errors / len(reference_text) if reference_text else None,
                "word_errors": item_word_errors,
                "word_count": len(reference_words),
                "char_errors": item_char_errors,
                "char_count": len(reference_text),
            }
        )

    return {
        "type": "asr_score",
        "expected": args.expected,
        "predictions": args.predictions,
        "matched": len(per_item),
        "missing_predictions": missing_predictions,
        "skipped_empty_refs": skipped_empty_refs,
        "wer": word_errors / word_count if word_count else None,
        "cer": char_errors / char_count if char_count else None,
        "word_errors": word_errors,
        "word_count": word_count,
        "char_errors": char_errors,
        "char_count": char_count,
        "items": per_item,
    }


def print_summary(result: dict[str, Any]) -> None:
    print("Qwen3-ASR score")
    print(f"  matched: {result['matched']}")
    print(f"  missing_predictions: {result['missing_predictions']}")
    print(f"  skipped_empty_refs: {result['skipped_empty_refs']}")
    print(f"  wer: {result['wer']}")
    print(f"  cer: {result['cer']}")
    print(f"  word_errors: {result['word_errors']} / {result['word_count']}")
    print(f"  char_errors: {result['char_errors']} / {result['char_count']}")


def main() -> int:
    args = parse_args()
    result = score(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
