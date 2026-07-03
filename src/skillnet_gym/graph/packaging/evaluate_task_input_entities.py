#!/usr/bin/env python3
"""Evaluate whether found input entities fit upstream skills in sampled tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a benchmark input-entity fit evaluator.

Given a composed task, one upstream skill's full SKILL.md, and one candidate input entity snippet, judge whether the entity is suitable as input for that upstream skill in this task.

You only evaluate entity-to-upstream-skill fit. Do not re-evaluate the whole task.

You will receive:
1. task_topology: skill nodes and directed skill connections in the task
2. upstream_skill: the upstream skill being evaluated, including skill_id, skill_name, and full SKILL.md
3. entity: candidate input entity, including entity_name, entity_type, url, snippet, and why_it_matches

Evaluate only these three aspects:

1. Topology match
- Judge whether the entity corresponds to an upstream / predecessor skill in the task.
- chain: A -> B -> C, only A's input should be evaluated.
- fan-out: A -> B and A -> C, only A's input should be evaluated.
- fan-in: A -> C and B -> C, evaluate inputs for A and B separately.
- diamond: A -> B, A -> C, B -> D, C -> D, only A's input should be evaluated.
- If the entity is marked for a downstream skill instead of an upstream skill, lower the score.

2. Input format match
- Use upstream_skill.skill_md to infer what input format this skill requires.
- Judge whether entity_type and the snippet's described file/resource type match that requirement.
- For example, if the skill needs CSV and the entity is CSV/tabular data, format matches.
- If the skill needs VCF and the entity is a general PDF, format does not match.

3. Input semantic match
- Use upstream_skill.skill_md and task_topology to infer what semantic content the upstream skill needs in this task.
- Judge whether the snippet's described content satisfies that semantic need.
- If it is merely in a related domain but cannot serve as the skill input, lower the score.

Do NOT evaluate:
- Whether the URL is really downloadable
- Whether the URL is a homepage
- Whether the whole task is valid
- Whether downstream skills can directly use this entity
- Whether this entity alone is enough to run the entire task

fit_score:
5 = format and semantics are highly suitable, and the entity targets the correct upstream skill
4 = mostly suitable, with only minor uncertainty
3 = somewhat related, but format or semantic fit is clearly uncertain
2 = weakly related and unlikely to be a valid input for this skill
1 = not suitable

Return exactly one valid JSON object. Do not include markdown.

Required JSON shape:
{
  "task_id": "",
  "for_skill_id": null,
  "for_skill_name": "",
  "entity_name": "",
  "entity_url": "",
  "fit_score": 1,
  "decision": "reject",
  "reason": "",
  "format_match": "",
  "semantic_match": "",
  "topology_match": "",
  "matched_requirements": [],
  "mismatched_requirements": [],
  "risk_flags": []
}
"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_default(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def parse_args() -> argparse.Namespace:
    load_env_file(Path(".env"))
    parser = argparse.ArgumentParser(
        description="LLM-review whether found input entities fit upstream skills for sampled graph tasks."
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
        default=["task_input_entities_part_01.json"],
        help="Found entity JSON file(s) or glob(s) from ChatGPT.",
    )
    parser.add_argument(
        "--skill-index",
        default=env_default("TASK_ENTITY_EVAL_SKILL_INDEX", "skill_quality_keep.json"),
        help="JSON containing skill_id, skill_name, skill_path/local_path fields.",
    )
    parser.add_argument(
        "--skills-dir",
        default=env_default("DOWNLOADED_SKILLS_DIR", "downloaded_skills"),
        help="Fallback directory containing skill-name/SKILL.md.",
    )
    parser.add_argument(
        "--output",
        default=env_default("TASK_ENTITY_EVAL_OUTPUT", "task_input_entity_evaluation_part_01.json"),
    )
    parser.add_argument(
        "--keep-output",
        default=env_default("TASK_ENTITY_KEEP_OUTPUT", "task_input_entity_keep_part_01.json"),
    )
    parser.add_argument(
        "--reject-output",
        default=env_default("TASK_ENTITY_REJECT_OUTPUT", "task_input_entity_reject_part_01.json"),
    )
    parser.add_argument("--api-key", default=env_default("API_KEY", ""))
    parser.add_argument("--base-url", default=env_default("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=env_default("MODEL", "gpt-4o-mini"))
    parser.add_argument("--workers", type=int, default=int(env_default("TASK_ENTITY_EVAL_WORKERS", "8")))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N pending entities.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate existing successful rows.")
    parser.add_argument("--min-fit-score", type=float, default=3.0)
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send response_format=json_object for providers that do not support it.",
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


def normalize_entity_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            rows = payload["results"]
        elif isinstance(payload.get("tasks"), list):
            rows = payload["tasks"]
        elif "task_id" in payload:
            rows = [payload]
        else:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def load_entity_records(path: Path) -> list[dict[str, Any]]:
    rows = normalize_entity_rows(load_json(path))
    records: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        entities = row.get("upstream_input_entities")
        if isinstance(entities, list):
            for index, entity in enumerate(entities, start=1):
                if not isinstance(entity, dict):
                    continue
                records.append(
                    {
                        "entity_record_id": f"{task_id}::entity-{index:03d}",
                        "task_id": task_id,
                        "source_status": row.get("status"),
                        "source_notes": row.get("notes"),
                        "entity": entity,
                    }
                )
        else:
            entity = row.get("entity") if isinstance(row.get("entity"), dict) else row
            entity_key = str(entity.get("url") or entity.get("entity_name") or len(records))
            records.append(
                {
                    "entity_record_id": f"{task_id}::{entity_key}",
                    "task_id": task_id,
                    "source_status": row.get("status"),
                    "source_notes": row.get("notes"),
                    "entity": entity,
                }
            )
    return [row for row in records if row["task_id"] and isinstance(row["entity"], dict)]


def load_entity_records_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for record in load_entity_records(path):
            entity = record.get("entity") if isinstance(record.get("entity"), dict) else {}
            key = (
                str(record.get("task_id") or ""),
                str(entity.get("entity_url") or entity.get("url") or ""),
                str(entity.get("entity_name") or entity.get("name") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def candidate_paths(row: dict[str, Any], skills_dir: Path) -> list[Path]:
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
        skill_path = next((candidate for candidate in candidate_paths(row, skills_dir) if candidate.exists()), None)
        if skill_path is None:
            continue
        skill_name = str(row.get("skill_name") or skill_path.parent.name)
        record = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_path": normalize_path(skill_path),
        }
        if skill_id > 0:
            by_id[skill_id] = record
        by_name.setdefault(skill_name, record)

    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        skill_name = skill_md.parent.name
        by_name.setdefault(skill_name, {"skill_id": 0, "skill_name": skill_name, "skill_path": normalize_path(skill_md)})
    return by_id, by_name


def task_topology(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "skills": task.get("skills", []),
        "skill_connections": task.get("skill_connections", []),
        "structure_type": task.get("structure_type"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
    }


def parse_connection(connection: Any, name_to_id: dict[str, int]) -> tuple[int, int] | None:
    match = re.search(r"(.+?)\s*->\s*(.+?)(?:\s*\(|$)", str(connection))
    if not match:
        return None
    source_name = match.group(1).strip()
    target_name = match.group(2).strip()
    if source_name not in name_to_id or target_name not in name_to_id:
        return None
    return name_to_id[source_name], name_to_id[target_name]


def upstream_skill_ids(task: dict[str, Any]) -> set[int]:
    skills = [skill for skill in task.get("skills", []) if isinstance(skill, dict)]
    name_to_id = {
        str(skill.get("skill_name") or ""): int(skill.get("skill_id") or 0)
        for skill in skills
        if skill.get("skill_id") is not None and str(skill.get("skill_name") or "")
    }
    ids = {int(skill.get("skill_id") or 0) for skill in skills}
    indegree = {skill_id: 0 for skill_id in ids if skill_id > 0}
    for connection in task.get("skill_connections", []):
        parsed = parse_connection(connection, name_to_id)
        if parsed is None:
            continue
        _source, target = parsed
        if target in indegree:
            indegree[target] += 1
    roots = {skill_id for skill_id, degree in indegree.items() if degree == 0}
    return roots or ids


def infer_skill_from_entity(entity: dict[str, Any], task: dict[str, Any]) -> dict[str, Any] | None:
    for_skill_id = entity.get("for_skill_id")
    try:
        for_skill_id_int = int(for_skill_id) if for_skill_id is not None else 0
    except (TypeError, ValueError):
        for_skill_id_int = 0
    for_skill_name = str(entity.get("for_skill_name") or "").strip()
    skills = [skill for skill in task.get("skills", []) if isinstance(skill, dict)]
    if for_skill_id_int:
        for skill in skills:
            if int(skill.get("skill_id") or 0) == for_skill_id_int:
                return skill
    if for_skill_name:
        for skill in skills:
            if str(skill.get("skill_name") or "") == for_skill_name:
                return skill
    roots = upstream_skill_ids(task)
    root_skills = [skill for skill in skills if int(skill.get("skill_id") or 0) in roots]
    if len(root_skills) == 1:
        return root_skills[0]
    return None


def read_skill_md(skill: dict[str, Any], by_id: dict[int, dict[str, Any]], by_name: dict[str, dict[str, Any]], skills_dir: Path) -> tuple[Path | None, str]:
    try:
        skill_id = int(skill.get("skill_id") or 0)
    except (TypeError, ValueError):
        skill_id = 0
    skill_name = str(skill.get("skill_name") or "").strip()
    record = by_id.get(skill_id) or by_name.get(skill_name)
    if record:
        path = Path(str(record["skill_path"]))
        if path.exists():
            return path, path.read_text(encoding="utf-8", errors="replace")
    fallback = skills_dir / skill_name / "SKILL.md"
    if fallback.exists():
        return fallback, fallback.read_text(encoding="utf-8", errors="replace")
    return None, ""


def build_user_prompt(task: dict[str, Any], skill: dict[str, Any], skill_md: str, entity: dict[str, Any]) -> str:
    compact = {
        "task_id": task.get("task_id"),
        "task_topology": task_topology(task),
        "upstream_skill": {
            "skill_id": skill.get("skill_id"),
            "skill_name": skill.get("skill_name"),
            "skill_md": skill_md,
        },
        "entity": {
            "entity_name": entity.get("entity_name"),
            "entity_type": entity.get("entity_type"),
            "url": entity.get("url"),
            "snippet": entity.get("snippet"),
            "why_it_matches": entity.get("why_it_matches"),
        },
    }
    return (
        "Evaluate whether this candidate input entity fits the specified upstream skill in this task.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    timeout: float,
    use_response_format: bool,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            data = json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response: {data}") from exc


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object")
    return parsed


def clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_result(
    *,
    task: dict[str, Any],
    entity_record: dict[str, Any],
    skill: dict[str, Any],
    skill_path: Path,
    parsed: dict[str, Any],
    min_fit_score: float,
) -> dict[str, Any]:
    entity = entity_record["entity"]
    fit_score = clamp_score(parsed.get("fit_score"))
    keep = fit_score >= min_fit_score
    return {
        "task_id": str(task.get("task_id") or ""),
        "entity_record_id": entity_record.get("entity_record_id"),
        "for_skill_id": int(skill.get("skill_id") or 0),
        "for_skill_name": str(skill.get("skill_name") or ""),
        "entity_name": str(entity.get("entity_name") or parsed.get("entity_name") or ""),
        "entity_url": str(entity.get("url") or parsed.get("entity_url") or ""),
        "entity_type": str(entity.get("entity_type") or ""),
        "snippet": str(entity.get("snippet") or ""),
        "why_it_matches": str(entity.get("why_it_matches") or ""),
        "fit_score": fit_score,
        "keep": keep,
        "decision": "keep" if keep else "reject",
        "llm_decision": str(parsed.get("decision") or ""),
        "reason": str(parsed.get("reason") or ""),
        "format_match": str(parsed.get("format_match") or ""),
        "semantic_match": str(parsed.get("semantic_match") or ""),
        "topology_match": str(parsed.get("topology_match") or ""),
        "matched_requirements": normalize_string_list(parsed.get("matched_requirements")),
        "mismatched_requirements": normalize_string_list(parsed.get("mismatched_requirements")),
        "risk_flags": normalize_string_list(parsed.get("risk_flags")),
        "skill_path": normalize_path(skill_path),
        "task_topology": task_topology(task),
    }


def failed_record(task: dict[str, Any], entity_record: dict[str, Any], reason: str) -> dict[str, Any]:
    entity = entity_record.get("entity", {})
    return {
        "task_id": str(task.get("task_id") or entity_record.get("task_id") or ""),
        "entity_record_id": entity_record.get("entity_record_id"),
        "for_skill_id": entity.get("for_skill_id"),
        "for_skill_name": entity.get("for_skill_name"),
        "entity_name": entity.get("entity_name"),
        "entity_url": entity.get("url"),
        "entity_type": entity.get("entity_type"),
        "snippet": entity.get("snippet"),
        "fit_score": 1,
        "keep": False,
        "decision": "reject",
        "reason": reason,
        "risk_flags": ["evaluation_failed"],
        "error": reason,
    }


def evaluate_one(
    *,
    task: dict[str, Any],
    entity_record: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    skills_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
    min_fit_score: float,
) -> dict[str, Any]:
    entity = entity_record["entity"]
    skill = infer_skill_from_entity(entity, task)
    if skill is None:
        return failed_record(task, entity_record, "Cannot identify matching upstream skill for entity.")
    skill_path, skill_md = read_skill_md(skill, by_id, by_name, skills_dir)
    if skill_path is None:
        return failed_record(task, entity_record, f"Missing SKILL.md for {skill.get('skill_name')}")

    prompt = build_user_prompt(task, skill, skill_md, entity)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            text = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_prompt=prompt,
                timeout=timeout,
                use_response_format=use_response_format,
            )
            return normalize_result(
                task=task,
                entity_record=entity_record,
                skill=skill,
                skill_path=skill_path,
                parsed=parse_model_json(text),
                min_fit_score=min_fit_score,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
    return failed_record(task, entity_record, f"Evaluation failed: {last_error}")


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("evaluations"), list):
        return [row for row in payload["evaluations"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'evaluations'")


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("entity_record_id") or f"{record.get('task_id')}::{record.get('entity_url')}::{record.get('entity_name')}")


def write_outputs(
    *,
    output_path: Path,
    keep_path: Path,
    reject_path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    keep_rows = [row for row in rows if row.get("keep") is True and "error" not in row]
    reject_rows = [row for row in rows if row not in keep_rows]
    meta = {
        "tasks": args.tasks,
        "entities": args.entities,
        "model": args.model,
        "min_fit_score": args.min_fit_score,
        "total": len(rows),
        "keep_count": len(keep_rows),
        "reject_count": len(reject_rows),
        "failed_count": sum(1 for row in rows if "error" in row),
    }
    write_json(output_path, {"meta": meta, "evaluations": rows})
    write_json(keep_path, {"meta": meta, "entities": keep_rows})
    write_json(reject_path, {"meta": meta, "entities": reject_rows})


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required for entity evaluation.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    task_paths = expand_paths(args.tasks)
    entity_paths = expand_paths(args.entities)
    tasks = load_tasks(task_paths)
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    entity_records = [
        record
        for record in load_entity_records_from_paths(entity_paths)
        if record.get("task_id") in task_by_id
    ]
    existing = [] if args.force else read_existing(Path(args.output))
    rows_by_key = {
        record_key(row): row
        for row in existing
        if record_key(row) and "error" not in row and "fit_score" in row
    }
    pending = [record for record in entity_records if record_key(record) not in rows_by_key]
    if args.limit > 0:
        pending = pending[: args.limit]

    by_id, by_name = load_skill_index(Path(args.skill_index), Path(args.skills_dir))
    print(
        "tasks={}, entity_records={}, existing_ok={}, pending={}, skill_docs={}, workers={}".format(
            len(tasks), len(entity_records), len(rows_by_key), len(pending), len(by_name), args.workers
        )
    )

    completed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_record = {
                executor.submit(
                    evaluate_one,
                    task=task_by_id[str(record["task_id"])],
                    entity_record=record,
                    by_id=by_id,
                    by_name=by_name,
                    skills_dir=Path(args.skills_dir),
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    use_response_format=not args.disable_response_format,
                    min_fit_score=args.min_fit_score,
                ): record
                for record in pending
            }
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                row = future.result()
                rows_by_key[record_key(record)] = row
                completed += 1
                status = "keep" if row.get("keep") else "reject"
                if "error" in row:
                    status = "failed"
                print(
                    "[{}/{}] {}: {} -> {} score={}".format(
                        completed,
                        len(pending),
                        status,
                        row.get("task_id"),
                        row.get("entity_name"),
                        row.get("fit_score"),
                    ),
                    flush=True,
                )
                write_outputs(
                    output_path=Path(args.output),
                    keep_path=Path(args.keep_output),
                    reject_path=Path(args.reject_output),
                    args=args,
                    rows=list(rows_by_key.values()),
                )

    write_outputs(
        output_path=Path(args.output),
        keep_path=Path(args.keep_output),
        reject_path=Path(args.reject_output),
        args=args,
        rows=list(rows_by_key.values()),
    )
    keep_count = sum(1 for row in rows_by_key.values() if row.get("keep") is True and "error" not in row)
    failed = sum(1 for row in rows_by_key.values() if "error" in row)
    print(
        f"done: total={len(rows_by_key)}, keep={keep_count}, reject={len(rows_by_key)-keep_count}, failed={failed}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
