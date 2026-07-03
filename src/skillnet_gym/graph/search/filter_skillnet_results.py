#!/usr/bin/env python3
"""Sort SkillNet results for each query by stars and keep the top entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAR_KEYS = ("stars", "star", "stargazers_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process SkillNet search results by sorting each query's matches by stars."
    )
    parser.add_argument(
        "--input",
        default="skillnet_semantic_results.json",
        help="Input JSON produced by skillnet_semantic_search.py.",
    )
    parser.add_argument(
        "--output",
        default="skillnet_semantic_results_top20_by_stars.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=20,
        help="Number of results to keep for each query.",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=0,
        help="Minimum stars required before top-k truncation.",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("none", "skill_url", "skill_name", "name_url"),
        default="none",
        help="Optional per-query deduplication before truncating.",
    )
    return parser.parse_args()


def star_count(item: Any) -> int:
    if not isinstance(item, dict):
        return 0

    for key in STAR_KEYS:
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return 0


def dedupe_key(item: dict[str, Any], mode: str) -> Any:
    if mode == "skill_url":
        return item.get("skill_url")
    if mode == "skill_name":
        return item.get("skill_name")
    if mode == "name_url":
        return (item.get("skill_name"), item.get("skill_url"))
    return None


def sort_and_trim(data: Any, keep: int, dedupe_by: str, min_stars: int) -> list[Any]:
    if not isinstance(data, list):
        return []

    sorted_items = sorted(
        (item for item in data if star_count(item) >= min_stars),
        key=star_count,
        reverse=True,
    )
    if dedupe_by == "none":
        return sorted_items[:keep]

    seen = set()
    kept: list[Any] = []
    for item in sorted_items:
        if not isinstance(item, dict):
            kept.append(item)
        else:
            key = dedupe_key(item, dedupe_by)
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)

        if len(kept) >= keep:
            break

    return kept


def main() -> int:
    args = parse_args()
    if args.keep < 1:
        raise ValueError("--keep must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{input_path} must contain a list at key 'results'")

    for entry in results:
        if not isinstance(entry, dict):
            continue
        original_count = len(entry.get("data", [])) if isinstance(entry.get("data"), list) else 0
        entry["data"] = sort_and_trim(entry.get("data"), args.keep, args.dedupe_by, args.min_stars)
        entry["postprocess"] = {
            "sort_by": "stars",
            "sort_order": "desc",
            "keep": args.keep,
            "dedupe_by": args.dedupe_by,
            "min_stars": args.min_stars,
            "original_count": original_count,
            "kept_count": len(entry["data"]),
        }

    meta = payload.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["postprocess"] = {
            "sort_by": "stars",
            "sort_order": "desc",
            "keep": args.keep,
            "dedupe_by": args.dedupe_by,
            "min_stars": args.min_stars,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {len(results)} query entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

