#!/usr/bin/env python3
"""Package sampled tasks into runnable task/environment directories.

Directory layout:
  output_dir/
    task-xxxxxxx/
      environment/
        downloaded files...
        skills/
          skill-a/
          skill-b/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-task folders, copy skills, and download reviewed input entities."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["skill_graph_task_sampled_10_parts/skill_graph_task_sampled_part_01.json"],
        help="Task JSON file(s), glob(s), array, or object with field tasks.",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        default=["entity/task_input_entities_part_01.json"],
        help="Entity JSON file(s) or glob(s). Supports raw queried entities and reviewed keep files.",
    )
    parser.add_argument(
        "--skill-index",
        default="skill_quality_keep.json",
        help="JSON containing skill_id, skill_name, skill_path/local_path fields.",
    )
    parser.add_argument("--skills-dir", default="downloaded_skills")
    parser.add_argument("--output-dir", default="packaged_tasks")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent task packaging workers.")
    parser.add_argument(
        "--max-download-mb",
        type=float,
        default=0.0,
        help="Skip a file if Content-Length or streamed bytes exceed this size. 0 means no size limit.",
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download even if target file exists.")
    parser.add_argument("--dry-run", action="store_true", help="Create manifests without downloading files or copying skills.")
    parser.add_argument(
        "--write-task-files",
        action="store_true",
        help="Also write task.json, input_entities.json, and package_manifest.json inside each task folder.",
    )
    parser.add_argument(
        "--require-keep",
        "--only-keep",
        dest="require_keep",
        action="store_true",
        default=False,
        help="Only package reviewed entities with keep=true. Default is to package all queried entities.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            key = normalize_path(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def load_tasks_from_file(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must contain a JSON array or object field 'tasks'")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        tasks.append(row)
    return tasks


def load_tasks(paths: list[Path]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for task in load_tasks_from_file(path):
            task_id = str(task.get("task_id") or "").strip()
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            tasks.append(task)
    return tasks


def load_entity_rows_from_file(path: Path, *, require_keep: bool) -> list[dict[str, Any]]:
    payload = load_json(path)

    def expand_task_entity_rows(task_rows: list[Any]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for task_row in task_rows:
            if not isinstance(task_row, dict):
                continue
            task_id = str(task_row.get("task_id") or "")
            for index, entity in enumerate(task_row.get("upstream_input_entities", []), start=1):
                if isinstance(entity, dict):
                    expanded.append(
                        {
                            "task_id": task_id,
                            "entity_record_id": f"{task_id}::entity-{index:03d}",
                            **entity,
                        }
                    )
        return expanded

    if isinstance(payload, dict):
        if isinstance(payload.get("entities"), list):
            rows = payload["entities"]
        elif isinstance(payload.get("evaluations"), list):
            rows = payload["evaluations"]
        elif isinstance(payload.get("tasks"), list):
            rows = expand_task_entity_rows(payload["tasks"])
        elif isinstance(payload.get("upstream_input_entities"), list):
            rows = []
            task_id = str(payload.get("task_id") or "")
            for index, entity in enumerate(payload["upstream_input_entities"], start=1):
                if isinstance(entity, dict):
                    rows.append({"task_id": task_id, "entity_record_id": f"{task_id}::entity-{index:03d}", **entity})
        elif "task_id" in payload:
            rows = [payload]
        else:
            rows = []
    elif isinstance(payload, list):
        if any(isinstance(row, dict) and isinstance(row.get("upstream_input_entities"), list) for row in payload):
            rows = expand_task_entity_rows(payload)
        else:
            rows = payload
    else:
        rows = []

    entities: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if require_keep and row.get("keep") is False:
            continue
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        url = str(row.get("entity_url") or row.get("url") or "").strip()
        if not url:
            continue
        normalized = dict(row)
        normalized["task_id"] = task_id
        normalized["entity_url"] = url
        normalized["entity_name"] = str(row.get("entity_name") or row.get("name") or "")
        normalized["entity_type"] = str(row.get("entity_type") or row.get("type") or "")
        normalized["entity_record_id"] = str(
            row.get("entity_record_id") or f"{task_id}::{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}"
        )
        entities.append(normalized)
    return entities


def load_entity_rows(paths: list[Path], *, require_keep: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for row in load_entity_rows_from_file(path, require_keep=require_keep):
            key = (
                str(row.get("task_id") or ""),
                str(row.get("entity_url") or ""),
                str(row.get("entity_name") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def candidate_skill_paths(row: dict[str, Any], skills_dir: Path) -> list[Path]:
    paths: list[Path] = []
    skill_path = row.get("skill_path")
    skill_dir = row.get("skill_dir") or row.get("local_path")
    skill_name = str(row.get("skill_name") or "").strip()
    if skill_path:
        paths.append(Path(str(skill_path)))
        paths.append(skills_dir / Path(str(skill_path)).parent.name / "SKILL.md")
    if skill_dir:
        paths.append(Path(str(skill_dir)) / "SKILL.md")
        paths.append(skills_dir / Path(str(skill_dir)).name / "SKILL.md")
    if skill_name:
        paths.append(skills_dir / skill_name / "SKILL.md")
    return paths


def load_skill_index(path: Path, skills_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        payload = load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
            rows = [row for row in payload["skills"] if isinstance(row, dict)]
        elif isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        else:
            raise ValueError(f"{path} must contain a JSON array or object field 'skills'")

    by_id: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            skill_id = int(row.get("skill_id") or 0)
        except (TypeError, ValueError):
            skill_id = 0
        skill_path = next((candidate for candidate in candidate_skill_paths(row, skills_dir) if candidate.exists()), None)
        if skill_path is None:
            continue
        skill_name = str(row.get("skill_name") or skill_path.parent.name)
        record = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_dir": normalize_path(skill_path.parent),
            "skill_path": normalize_path(skill_path),
        }
        if skill_id > 0:
            by_id[skill_id] = record
        by_name.setdefault(skill_name, record)

    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        skill_name = skill_md.parent.name
        by_name.setdefault(
            skill_name,
            {
                "skill_id": 0,
                "skill_name": skill_name,
                "skill_dir": normalize_path(skill_md.parent),
                "skill_path": normalize_path(skill_md),
            },
        )
    return by_id, by_name


def resolve_skill(skill: dict[str, Any], by_id: dict[int, dict[str, Any]], by_name: dict[str, dict[str, Any]], skills_dir: Path) -> dict[str, Any] | None:
    try:
        skill_id = int(skill.get("skill_id") or 0)
    except (TypeError, ValueError):
        skill_id = 0
    skill_name = str(skill.get("skill_name") or "").strip()
    record = by_id.get(skill_id) or by_name.get(skill_name)
    if record and Path(record["skill_dir"]).exists():
        return record
    fallback = skills_dir / skill_name
    if fallback.exists():
        return {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_dir": normalize_path(fallback),
            "skill_path": normalize_path(fallback / "SKILL.md"),
        }
    return None


def sanitize_name(value: str, fallback: str) -> str:
    value = value.strip()
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    return value[:120] or fallback


def guess_extension(url: str, content_type: str, entity_type: str) -> str:
    path = urllib.parse.urlparse(url).path
    suffix = Path(path).suffix
    if suffix and len(suffix) <= 12:
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    if guessed:
        return guessed
    lowered = entity_type.lower()
    for token, ext in [
        ("csv", ".csv"),
        ("xlsx", ".xlsx"),
        ("excel", ".xlsx"),
        ("json", ".json"),
        ("pdf", ".pdf"),
        ("zip", ".zip"),
        ("vcf", ".vcf"),
        ("bam", ".bam"),
        ("fasta", ".fasta"),
        ("fastq", ".fastq"),
        ("image", ".png"),
        ("audio", ".wav"),
        ("video", ".mp4"),
    ]:
        if token in lowered:
            return ext
    return ".dat"


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find unique filename for {stem}{suffix}")


def find_existing_entity_file(directory: Path, stem: str, suffix: str) -> Path | None:
    exact = directory / f"{stem}{suffix}"
    if exact.exists() and exact.is_file():
        return exact
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(f"{stem}__*{suffix}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_previous_task_manifest(task_dir: Path) -> dict[str, Any]:
    manifest_path = task_dir / "package_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def existing_download_maps(environment_dir: Path, previous_manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    url_to_file: dict[str, str] = {}
    hash_to_file: dict[str, str] = {}
    for row in previous_manifest.get("entities", []):
        if not isinstance(row, dict):
            continue
        target = row.get("target") or row.get("duplicate_of")
        url = str(row.get("entity_url") or "")
        sha = str(row.get("sha256") or "")
        if target and Path(str(target)).exists():
            target_norm = normalize_path(str(target))
            if url:
                url_to_file[url] = target_norm
            if sha:
                hash_to_file[sha] = target_norm

    if environment_dir.exists():
        for path in sorted(environment_dir.iterdir()):
            if not path.is_file() or path.name.startswith(".download_tmp_"):
                continue
            try:
                sha = sha256_file(path)
            except OSError:
                continue
            hash_to_file.setdefault(sha, normalize_path(path))
    return url_to_file, hash_to_file


def request_download(url: str, timeout: float, max_bytes: int = 0) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 task-packager",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length")
        if max_bytes > 0 and content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            if size > max_bytes:
                raise RuntimeError(f"file too large: {size} bytes exceeds limit {max_bytes} bytes")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes > 0 and total > max_bytes:
                raise RuntimeError(f"file too large: streamed bytes exceed limit {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks), content_type


def retry_download(url: str, *, timeout: float, retries: int, max_bytes: int = 0) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return request_download(url, timeout, max_bytes=max_bytes)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
        except RuntimeError as exc:
            raise exc
    raise RuntimeError(f"download failed: {last_error}") from last_error


def copy_task_skills(
    *,
    task: dict[str, Any],
    skills_target: Path,
    by_id: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    skills_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for skill in task.get("skills", []):
        if not isinstance(skill, dict):
            continue
        record = resolve_skill(skill, by_id, by_name, skills_dir)
        skill_name = str(skill.get("skill_name") or record.get("skill_name") if record else "").strip()
        if not record:
            copied.append({"skill_name": skill_name, "status": "missing"})
            continue
        source_dir = Path(record["skill_dir"])
        target_name = sanitize_name(skill_name or source_dir.name, "skill")
        if target_name in seen_names:
            copied.append({"skill_name": skill_name, "source": normalize_path(source_dir), "status": "duplicate_skill_name_skipped"})
            continue
        seen_names.add(target_name)
        target_dir = skills_target / target_name
        if not dry_run:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        copied.append(
            {
                "skill_id": skill.get("skill_id"),
                "skill_name": skill_name,
                "source": normalize_path(source_dir),
                "target": normalize_path(target_dir),
                "status": "copied" if not dry_run else "dry_run",
            }
        )
    return copied


def download_task_entities(
    *,
    entities: list[dict[str, Any]],
    environment_dir: Path,
    previous_manifest: dict[str, Any],
    timeout: float,
    retries: int,
    max_download_mb: float,
    force_download: bool,
    dry_run: bool,
    progress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    url_to_file, hash_to_file = existing_download_maps(environment_dir, previous_manifest)
    downloads: list[dict[str, Any]] = []

    def tick(status: str, name: str) -> None:
        if progress is None:
            return
        lock: Lock = progress["lock"]
        with lock:
            progress["done"] += 1
            done = progress["done"]
            total = progress["total"]
            print(f"[downloads {done}/{total}] {status}: {name}", flush=True)

    for index, entity in enumerate(entities, start=1):
        url = str(entity.get("entity_url") or "").strip()
        entity_name = str(entity.get("entity_name") or f"entity_{index:03d}")
        entity_type = str(entity.get("entity_type") or "")
        record = {
            "entity_record_id": entity.get("entity_record_id"),
            "entity_name": entity_name,
            "entity_url": url,
            "entity_type": entity_type,
            "for_skill_id": entity.get("for_skill_id"),
            "for_skill_name": entity.get("for_skill_name"),
            "snippet": entity.get("snippet"),
        }

        if not url:
            downloads.append({**record, "status": "failed", "error": "missing_url"})
            tick("failed", entity_name)
            continue
        if url in url_to_file:
            downloads.append({**record, "status": "duplicate_url_skipped", "duplicate_of": url_to_file[url]})
            tick("skipped-url", entity_name)
            continue
        if dry_run:
            stem = sanitize_name(entity_name, f"entity_{index:03d}")
            downloads.append({**record, "status": "dry_run", "target": normalize_path(environment_dir / stem)})
            tick("dry-run", entity_name)
            continue

        environment_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_name(entity_name, f"entity_{index:03d}")
        guessed_suffix = guess_extension(url, "", entity_type)
        existing_path = find_existing_entity_file(environment_dir, stem, guessed_suffix)
        if existing_path is not None and not force_download:
            existing_hash = sha256_file(existing_path)
            hash_to_file.setdefault(existing_hash, normalize_path(existing_path))
            url_to_file[url] = normalize_path(existing_path)
            downloads.append(
                {
                    **record,
                    "status": "existing_file_skipped",
                    "target": normalize_path(existing_path),
                    "sha256": existing_hash,
                }
            )
            tick("skipped-existing", entity_name)
            continue

        temp_path = environment_dir / f".download_tmp_{index:03d}"
        try:
            max_bytes = int(max_download_mb * 1024 * 1024) if max_download_mb > 0 else 0
            content, content_type = retry_download(url, timeout=timeout, retries=retries, max_bytes=max_bytes)
            temp_path.write_bytes(content)
            content_hash = sha256_file(temp_path)
            if content_hash in hash_to_file:
                temp_path.unlink(missing_ok=True)
                url_to_file[url] = hash_to_file[content_hash]
                downloads.append(
                    {
                        **record,
                        "status": "duplicate_content_skipped",
                        "sha256": content_hash,
                        "duplicate_of": hash_to_file[content_hash],
                    }
                )
                tick("skipped-content", entity_name)
                continue

            suffix = guess_extension(url, content_type, entity_type)
            target_path = unique_path(environment_dir, stem, suffix)
            if target_path.exists() and not force_download:
                existing_hash = sha256_file(target_path)
                temp_path.unlink(missing_ok=True)
                hash_to_file.setdefault(existing_hash, normalize_path(target_path))
                url_to_file[url] = normalize_path(target_path)
                downloads.append(
                    {
                        **record,
                        "status": "existing_file_skipped",
                        "target": normalize_path(target_path),
                        "sha256": existing_hash,
                    }
                )
                tick("skipped-existing", entity_name)
                continue
            temp_path.replace(target_path)
            hash_to_file[content_hash] = normalize_path(target_path)
            url_to_file[url] = normalize_path(target_path)
            downloads.append(
                {
                    **record,
                    "status": "downloaded",
                    "target": normalize_path(target_path),
                    "sha256": content_hash,
                    "bytes": target_path.stat().st_size,
                    "content_type": content_type,
                }
            )
            tick("downloaded", entity_name)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            downloads.append({**record, "status": "failed", "error": str(exc)})
            tick("failed", entity_name)
    return downloads


def package_tasks(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    task_paths = expand_paths(args.tasks)
    entity_paths = expand_paths(args.entities)
    if not task_paths:
        raise FileNotFoundError(f"No task files matched: {args.tasks}")
    if not entity_paths:
        raise FileNotFoundError(f"No entity files matched: {args.entities}")

    tasks = load_tasks(task_paths)
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    entity_rows = load_entity_rows(entity_paths, require_keep=args.require_keep)
    entities_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entity_rows:
        task_id = str(entity.get("task_id") or "")
        if task_id in task_by_id:
            entities_by_task[task_id].append(entity)

    by_id, by_name = load_skill_index(Path(args.skill_index), Path(args.skills_dir))
    output_dir = Path(args.output_dir)
    skills_dir = Path(args.skills_dir)
    progress = {
        "total": sum(len(rows) for rows in entities_by_task.values()),
        "done": 0,
        "lock": Lock(),
    }

    def package_one_task(task_id: str) -> dict[str, Any]:
        task = task_by_id[task_id]
        task_dir = output_dir / sanitize_name(task_id, "task")
        environment_dir = task_dir / "environment"
        skills_target = environment_dir / "skills"
        previous_manifest = load_previous_task_manifest(task_dir)
        if not args.dry_run:
            skills_target.mkdir(parents=True, exist_ok=True)
        copied_skills = copy_task_skills(
            task=task,
            skills_target=skills_target,
            by_id=by_id,
            by_name=by_name,
            skills_dir=skills_dir,
            dry_run=args.dry_run,
        )
        downloads = download_task_entities(
            entities=entities_by_task[task_id],
            environment_dir=environment_dir,
            previous_manifest=previous_manifest,
            timeout=args.timeout,
            retries=args.retries,
            max_download_mb=args.max_download_mb,
            force_download=args.force_download,
            dry_run=args.dry_run,
            progress=progress,
        )
        manifest = {
            "task_id": task_id,
            "task_dir": normalize_path(task_dir),
            "environment_dir": normalize_path(environment_dir),
            "skills_dir": normalize_path(skills_target),
            "skill_count": len([row for row in copied_skills if row.get("status") in {"copied", "dry_run"}]),
            "entity_count": len(entities_by_task[task_id]),
            "downloaded_count": len([row for row in downloads if row.get("status") == "downloaded"]),
            "skipped_existing_count": len([row for row in downloads if "existing" in str(row.get("status"))]),
            "duplicate_skipped_count": len([row for row in downloads if "duplicate" in str(row.get("status"))]),
            "failed_count": len([row for row in downloads if row.get("status") == "failed"]),
            "skills": copied_skills,
            "entities": downloads,
        }
        if not args.dry_run and args.write_task_files:
            write_json(task_dir / "task.json", task)
            write_json(task_dir / "input_entities.json", entities_by_task[task_id])
            write_json(task_dir / "package_manifest.json", manifest)
        return manifest

    manifests: list[dict[str, Any]] = []
    task_ids = sorted(entities_by_task)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task_id = {executor.submit(package_one_task, task_id): task_id for task_id in task_ids}
        for future in concurrent.futures.as_completed(future_to_task_id):
            task_id = future_to_task_id[future]
            try:
                manifest = future.result()
            except Exception as exc:
                manifest = {
                    "task_id": task_id,
                    "task_dir": normalize_path(output_dir / sanitize_name(task_id, "task")),
                    "environment_dir": normalize_path(output_dir / sanitize_name(task_id, "task") / "environment"),
                    "skills_dir": normalize_path(output_dir / sanitize_name(task_id, "task") / "environment" / "skills"),
                    "skill_count": 0,
                    "entity_count": len(entities_by_task.get(task_id, [])),
                    "downloaded_count": 0,
                    "duplicate_skipped_count": 0,
                    "failed_count": len(entities_by_task.get(task_id, [])),
                    "error": str(exc),
                    "skills": [],
                    "entities": [],
                }
            manifests.append(manifest)
            print(
                "{}: skills={}, entities={}, downloaded={}, dup_skipped={}, failed={}".format(
                    task_id,
                    manifest["skill_count"],
                    manifest["entity_count"],
                    manifest["downloaded_count"],
                    manifest["duplicate_skipped_count"],
                    manifest["failed_count"],
                ),
                flush=True,
            )

    manifests.sort(key=lambda item: str(item.get("task_id") or ""))

    summary = {
        "tasks": [normalize_path(path) for path in task_paths],
        "entities": [normalize_path(path) for path in entity_paths],
        "output_dir": args.output_dir,
        "task_count": len(manifests),
        "entity_count": sum(row["entity_count"] for row in manifests),
        "downloaded_count": sum(row["downloaded_count"] for row in manifests),
        "duplicate_skipped_count": sum(row["duplicate_skipped_count"] for row in manifests),
        "failed_count": sum(row["failed_count"] for row in manifests),
        "dry_run": args.dry_run,
        "workers": args.workers,
        "tasks_manifest": manifests,
    }
    if not args.dry_run:
        write_json(output_dir / "package_manifest.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = package_tasks(args)
    print(
        "done: tasks={}, entities={}, downloaded={}, dup_skipped={}, failed={}, output_dir={}".format(
            summary["task_count"],
            summary["entity_count"],
            summary["downloaded_count"],
            summary["duplicate_skipped_count"],
            summary["failed_count"],
            args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
