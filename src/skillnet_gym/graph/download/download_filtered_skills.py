#!/usr/bin/env python3
"""Download filtered SkillNet skills from their GitHub skill_url values."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


GITHUB_API = "https://api.github.com"
STAR_KEYS = ("stars", "star", "stargazers_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download skills listed in a filtered SkillNet search result JSON."
    )
    parser.add_argument(
        "--input",
        default="skillnet_semantic_results_by_stars.json",
        help="Filtered SkillNet results JSON.",
    )
    parser.add_argument(
        "--target-dir",
        default="downloaded_skills",
        help="Directory where skill folders/files will be downloaded.",
    )
    parser.add_argument(
        "--manifest",
        default="downloaded_skills_manifest.json",
        help="Download manifest JSON path.",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="Optional GitHub token. Defaults to GITHUB_TOKEN env var.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per URL/API call.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay after each completed download task.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent skill download workers.")
    parser.add_argument(
        "--max-skills",
        type=int,
        default=0,
        help="Download at most this many unique skills. 0 means no limit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a skill when its target directory already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the manifest without downloading files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def star_count(item: dict[str, Any]) -> int:
    for key in STAR_KEYS:
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def safe_name(value: str, *, default: str = "skill", max_len: int = 80) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (name or default)[:max_len]


def parse_github_url(url: str) -> dict[str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() != "github.com":
        raise ValueError(f"not a github.com URL: {url}")

    parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 5 or parts[2] not in {"blob", "tree"}:
        raise ValueError(f"unsupported GitHub URL shape: {url}")

    owner, repo, mode, ref = parts[:4]
    path = "/".join(parts[4:])
    if not owner or not repo or not ref or not path:
        raise ValueError(f"incomplete GitHub URL: {url}")

    return {"owner": owner, "repo": repo, "mode": mode, "ref": ref, "path": path}


def request_json(url: str, token: str, timeout: float) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "download-filtered-skills",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def request_bytes(url: str, token: str, timeout: float) -> bytes:
    headers = {"User-Agent": "download-filtered-skills"}
    if token and urllib.parse.urlparse(url).netloc.lower() == "raw.githubusercontent.com":
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def retry(call, retries: int, *args):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call(*args)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise last_error or RuntimeError("request failed")


def contents_api_url(owner: str, repo: str, path: str, ref: str) -> str:
    quoted_path = urllib.parse.quote(path.strip("/"))
    quoted_ref = urllib.parse.quote(ref)
    return f"{GITHUB_API}/repos/{owner}/{repo}/contents/{quoted_path}?ref={quoted_ref}"


def raw_github_url(owner: str, repo: str, ref: str, path: str) -> str:
    return "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
        urllib.parse.quote(owner),
        urllib.parse.quote(repo),
        urllib.parse.quote(ref, safe=""),
        "/".join(urllib.parse.quote(part) for part in path.split("/")),
    )


def download_contents(
    *,
    owner: str,
    repo: str,
    ref: str,
    remote_path: str,
    local_path: Path,
    token: str,
    timeout: float,
    retries: int,
) -> int:
    api_url = contents_api_url(owner, repo, remote_path, ref)
    payload = retry(request_json, retries, api_url, token, timeout)

    if isinstance(payload, list):
        file_count = 0
        local_path.mkdir(parents=True, exist_ok=True)
        for child in payload:
            if not isinstance(child, dict):
                continue
            child_name = child.get("name")
            child_path = child.get("path")
            child_type = child.get("type")
            if not child_name or not child_path:
                continue
            if child_type == "dir":
                file_count += download_contents(
                    owner=owner,
                    repo=repo,
                    ref=ref,
                    remote_path=child_path,
                    local_path=local_path / child_name,
                    token=token,
                    timeout=timeout,
                    retries=retries,
                )
            elif child_type == "file":
                download_url = child.get("download_url") or raw_github_url(owner, repo, ref, child_path)
                data = retry(request_bytes, retries, download_url, token, timeout)
                target_file = local_path / child_name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_bytes(data)
                file_count += 1
        return file_count

    if isinstance(payload, dict) and payload.get("type") == "file":
        download_url = payload.get("download_url") or raw_github_url(owner, repo, ref, remote_path)
        data = retry(request_bytes, retries, download_url, token, timeout)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return 1

    raise ValueError(f"unexpected GitHub contents response for {owner}/{repo}/{remote_path}")


def collect_skills(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("input JSON must contain a list at key 'results'")

    by_url: dict[str, dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query")
        for item in entry.get("data", []):
            if not isinstance(item, dict):
                continue
            skill_url = item.get("skill_url")
            if not isinstance(skill_url, str) or not skill_url:
                continue

            record = by_url.setdefault(
                skill_url,
                {
                    "skill_url": skill_url,
                    "skill_name": item.get("skill_name") or "skill",
                    "author": item.get("author") or "",
                    "stars": star_count(item),
                    "category": item.get("category"),
                    "queries": [],
                },
            )
            if isinstance(query, str) and query not in record["queries"]:
                record["queries"].append(query)
            record["stars"] = max(record["stars"], star_count(item))

    by_name: dict[str, dict[str, Any]] = {}
    for item in sorted(by_url.values(), key=lambda item: (-int(item["stars"]), item["skill_url"])):
        name = safe_name(str(item.get("skill_name") or "skill")).lower()
        if name in by_name:
            continue
        by_name[name] = item

    return list(by_name.values())


def target_path_for_skill(
    target_dir: Path,
    item: dict[str, Any],
) -> Path:
    name = safe_name(str(item.get("skill_name") or "skill"))
    return target_dir / name


def main() -> int:
    args = parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_path = Path(args.input)
    target_dir = Path(args.target_dir)
    manifest_path = Path(args.manifest)
    payload = load_json(input_path)
    skills = collect_skills(payload)
    if args.max_skills:
        if args.max_skills < 1:
            raise ValueError("--max-skills must be positive, or 0 for no limit")
        skills = skills[: args.max_skills]

    manifest: dict[str, Any] = {
        "input": str(input_path),
        "target_dir": str(target_dir),
        "dry_run": args.dry_run,
        "total_unique_skills": len(skills),
        "downloaded": [],
        "failed": [],
        "skipped": [],
    }

    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(skills, start=1):
        skill_url = str(item["skill_url"])
        local_path = target_path_for_skill(target_dir, item)
        base_record = {
            "index": index,
            "skill_name": item.get("skill_name"),
            "author": item.get("author"),
            "stars": item.get("stars"),
            "skill_url": skill_url,
            "local_path": str(local_path),
            "queries": item.get("queries", []),
        }
        tasks.append({"item": item, "skill_url": skill_url, "local_path": local_path, "record": base_record})

    def process_task(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        skill_url = str(task["skill_url"])
        local_path = Path(task["local_path"])
        base_record = dict(task["record"])

        if args.dry_run:
            return "skipped", {**base_record, "reason": "dry-run"}

        if args.skip_existing and local_path.exists():
            return "skipped", {**base_record, "reason": "target exists"}

        try:
            parsed = parse_github_url(skill_url)
            file_count = download_contents(
                owner=parsed["owner"],
                repo=parsed["repo"],
                ref=parsed["ref"],
                remote_path=parsed["path"],
                local_path=local_path,
                token=args.github_token,
                timeout=args.timeout,
                retries=args.retries,
            )
            return "downloaded", {**base_record, "file_count": file_count}
        except Exception as exc:
            return "failed", {**base_record, "error": f"{type(exc).__name__}: {exc}"}

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                status, record = future.result()
            except Exception as exc:
                record = dict(task["record"])
                record["error"] = f"{type(exc).__name__}: {exc}"
                status = "failed"

            manifest[status].append(record)
            print(
                "[{}/{}] {}: {} -> {}".format(
                    completed,
                    len(tasks),
                    status,
                    record.get("skill_name"),
                    record.get("local_path"),
                ),
                flush=True,
            )
            if status == "failed":
                print(f"  failed: {record.get('error')}", file=sys.stderr, flush=True)

            write_json(manifest_path, manifest)
            if args.sleep > 0:
                time.sleep(args.sleep)

    for key in ("downloaded", "skipped", "failed"):
        manifest[key].sort(key=lambda item: int(item.get("index", 0)))
    write_json(manifest_path, manifest)
    print(
        "done: downloaded={}, skipped={}, failed={}, manifest={}".format(
            len(manifest["downloaded"]),
            len(manifest["skipped"]),
            len(manifest["failed"]),
            manifest_path,
        )
    )
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
