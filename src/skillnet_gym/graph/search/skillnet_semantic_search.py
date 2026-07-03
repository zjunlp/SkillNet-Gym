#!/usr/bin/env python3
"""Run SkillNet semantic search for every query in query_seeds.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "http://api-skillnet.openkg.cn/v1/search"


def load_queries(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    queries = payload.get("query_seeds")
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError(f"{path} must contain a string array at key 'query_seeds'")

    return queries


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"results": []}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} is not a compatible results file")

    return payload


def search_skillnet(
    api_url: str,
    query: str,
    *,
    limit: int,
    threshold: float,
    timeout: float,
) -> dict[str, Any]:
    params = {
        "q": query,
        "mode": "vector",
        "limit": str(limit),
        "threshold": str(threshold),
    }
    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset)

    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("SkillNet returned a non-object JSON response")

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call SkillNet semantic search for every query in query_seeds.json."
    )
    parser.add_argument("--input", default="query_seeds.json", help="Input JSON file.")
    parser.add_argument(
        "--output",
        default="skillnet_semantic_results.json",
        help="Output JSON file. Existing compatible files are resumed.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="SkillNet search API URL.")
    parser.add_argument("--limit", type=int, default=10, help="Results per query, max 50.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Semantic similarity threshold for SkillNet vector mode.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per query.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay after each completed query.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent search workers.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-query items already present in the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1 or args.limit > 50:
        raise ValueError("--limit must be between 1 and 50")
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    queries = load_queries(input_path)

    output = load_existing(output_path)
    if args.force:
        output["results"] = []

    completed = {item.get("query") for item in output["results"] if item.get("ok") is True}
    output["meta"] = {
        "api_url": args.api_url,
        "mode": "vector",
        "limit": args.limit,
        "threshold": args.threshold,
        "input": str(input_path),
        "total_queries": len(queries),
        "workers": args.workers,
    }

    pending = [
        (index, query)
        for index, query in enumerate(queries, start=1)
        if query not in completed
    ]
    for index, query in enumerate(queries, start=1):
        if query in completed:
            print(f"[{index}/{len(queries)}] skip: {query}", flush=True)

    def process_query(index: int, query: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"index": index - 1, "query": query, "ok": False}

        for attempt in range(1, args.retries + 1):
            try:
                payload = search_skillnet(
                    args.api_url,
                    query,
                    limit=args.limit,
                    threshold=args.threshold,
                    timeout=args.timeout,
                )
                entry.update(
                    {
                        "ok": bool(payload.get("success", True)),
                        "data": payload.get("data", []),
                        "response_meta": payload.get("meta", {}),
                    }
                )
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 10))
                else:
                    print(f"  failed after {args.retries} attempts: {entry['error']}", file=sys.stderr)
        return entry

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_query = {
            executor.submit(process_query, index, query): (index, query)
            for index, query in pending
        }
        for future in concurrent.futures.as_completed(future_to_query):
            index, query = future_to_query[future]
            completed_count += 1
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "index": index - 1,
                    "query": query,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            status = "ok" if entry.get("ok") else "failed"
            print(
                f"[{completed_count}/{len(pending)}] {status}: {query}",
                flush=True,
            )

            output["results"].append(entry)
            output["results"].sort(key=lambda item: int(item.get("index", 0)))
            write_json(output_path, output)

            if args.sleep > 0:
                time.sleep(args.sleep)

    failures = sum(1 for item in output["results"] if not item.get("ok"))
    print(f"done: {len(output['results'])} entries written to {output_path} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
