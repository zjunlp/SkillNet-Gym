#!/usr/bin/env python3
"""Evaluate skill-graph task candidates with an LLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


WEIGHTS = {
    "composition_score": 0.50,
    "verifiability_score": 0.50,
}


SYSTEM_PROMPT = """You evaluate whether a candidate skill graph substructure can form one real benchmark task.

Return exactly one JSON object. Do not include markdown.

You will receive:
- structure_type and difficulty
- skills in the candidate
- skill_connections: directed edges such as "skill-a -> skill-b"
- edge evidence: scenario-level handoff examples from source post-scenario to target pre-scenario
- the full SKILL.md content for every skill in the candidate task

Judge the original candidate as a whole. Do not remove, prune, or rewrite the skill set.

Dimension 1: composition_score
Whether the given skills can form one task according to the directed structure and order.
This score must consider:
- The task must follow the given direction exactly. For A -> B, A's result must be a meaningful input, artifact, state, or prerequisite for B.
- Every skill must have a distinct role. Repeated, redundant, or alternative tools should score low.
- The whole structure must describe one coherent workflow, not a bag of related skills.
- For fan-in, multiple upstream skills must produce complementary inputs for one downstream skill.
- For fan-out, one upstream result must naturally branch into multiple distinct downstream steps.
- For diamond, the split and merge must both be meaningful.
- Edge scores and retrieval evidence are supporting signals only; SKILL.md is the source of truth.

composition_score:
- 5: The directed structure is clearly necessary and natural; every skill contributes a distinct step; handoffs are explicit.
- 4: The structure is mostly valid; minor ambiguity, but still a plausible composed task.
- 3: The skills are related and could be forced into a task, but the order/handoff or role separation is weak.
- 2: Mostly topical similarity, wrong direction, repeated/redundant tools, or no real artifact/state handoff.
- 1: Duplicate/unrelated skills, impossible direction, or no coherent composition.

Dimension 2: verifiability_score
Whether the final task result can be objectively verified for benchmark use.
This score must consider:
- High scores require objective checks on the core success criterion, not only superficial checks.
- Good verification: exact JSON/CSV/schema, numeric values, labels, deterministic file transformations, code/tests passing, database rows, chart data values, required fields, validated syntax, or rule-checkable evidence.
- Weak verification: only file existence, generic required sections, keyword presence, or manual visual inspection.
- Low verification: creative writing, subjective quality, style improvement, coaching, strategy, advice, recommendations, resume quality, presentation quality, design quality, open-ended summaries, or external service state.

verifiability_score:
- 5: Core result is highly objective and machine-checkable with strict fields, values, schemas, tests, or deterministic artifacts.
- 4: Mostly objective and rule-checkable; minor natural language is allowed if the core result is structured.
- 3: Partly checkable, but important success criteria still require human judgment.
- 2: Mostly subjective or natural-language output; only superficial checks are possible.
- 1: Almost entirely subjective, creative, external-service-dependent, or impossible to verify reliably.

Important:
- Use SKILL.md content as the source of truth.
- Use skill_connections and edge evidence only for structure, direction, and handoff evidence.
- Do not reward a task just because each individual skill is useful.
- If the directed composition is weak, composition_score must be low.
- If the output is subjective, verifiability_score must be low even if the workflow is coherent.
- Do not suggest removing skills. If a skill is redundant, score the original candidate lower.

Required JSON shape:
{
  "composition_score": 1,
  "verifiability_score": 1,
  "decision": "keep",
  "reason": "",
  "suggested_task": {
    "title": "",
    "input": "",
    "workflow": [],
    "expected_output": "",
    "verification": []
  },
  "flags": []
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
        description="LLM-judge task candidates extracted from scenario_skill_graph.json."
    )
    parser.add_argument("--input", default="skill_graph_task_candidates.json")
    parser.add_argument("--skill-index", default=env_default("TASK_EVAL_SKILL_INDEX", "skill_quality_keep.json"))
    parser.add_argument("--skills-dir", default=env_default("DOWNLOADED_SKILLS_DIR", "downloaded_skills"))
    parser.add_argument("--output", default=env_default("SKILL_GRAPH_TASK_EVAL_OUTPUT", "skill_graph_task_evaluation.json"))
    parser.add_argument("--keep-output", default=env_default("SKILL_GRAPH_TASK_KEEP_OUTPUT", "skill_graph_task_keep.json"))
    parser.add_argument("--reject-output", default=env_default("SKILL_GRAPH_TASK_REJECT_OUTPUT", "skill_graph_task_reject.json"))
    parser.add_argument("--api-key", default=env_default("API_KEY", ""))
    parser.add_argument("--base-url", default=env_default("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=env_default("MODEL", "gpt-4o-mini"))
    parser.add_argument("--workers", type=int, default=int(env_default("SKILL_GRAPH_TASK_EVAL_WORKERS", "8")))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N pending tasks.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate existing successful rows.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional category filter, e.g. chain_2 fan_in_3 diamond_4.",
    )
    parser.add_argument(
        "--structures",
        nargs="*",
        default=[],
        choices=["chain", "fan_in", "fan_out", "diamond"],
        help="Optional structure_type filter.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=[],
        choices=["easy", "medium", "hard", "expert"],
        help="Optional difficulty filter.",
    )
    parser.add_argument("--min-composition", type=float, default=4.0)
    parser.add_argument("--min-verifiability", type=float, default=4.0)
    parser.add_argument("--min-overall", type=float, default=4.0)
    parser.add_argument(
        "--max-edge-evidence-per-task",
        type=int,
        default=12,
        help="Maximum scenario evidence records included in the LLM prompt per task.",
    )
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


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("evaluations"), list):
        return [row for row in payload["evaluations"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'evaluations'")


def load_candidate_tasks(
    path: Path,
    *,
    categories: set[str],
    structures: set[str],
    difficulties: set[str],
) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"{path} must contain object field 'tasks'")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            continue
        if categories and str(task.get("category") or "") not in categories:
            continue
        if structures and str(task.get("structure_type") or "") not in structures:
            continue
        if difficulties and str(task.get("difficulty") or "") not in difficulties:
            continue
        seen.add(task_id)
        tasks.append(task)
    return tasks


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


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
        record = {"skill_id": 0, "skill_name": skill_name, "skill_path": normalize_path(skill_md)}
        by_name.setdefault(skill_name, record)
    return by_id, by_name


def resolve_skill_doc(skill: dict[str, Any], by_id: dict[int, dict[str, Any]], by_name: dict[str, dict[str, Any]], skills_dir: Path) -> tuple[Path | None, str]:
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


def limited_edge_evidence(task: dict[str, Any], max_items: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for edge in task.get("edges", []):
        if not isinstance(edge, dict):
            continue
        for item in edge.get("scenario_connections", []):
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "skill_connection": edge.get("skill_connection"),
                    "source_post_scenario": item.get("source_post_scenario"),
                    "target_pre_scenario": item.get("target_pre_scenario"),
                    "alignment_type": item.get("alignment_type"),
                    "confidence": item.get("confidence"),
                    "retrieval_score": item.get("retrieval_score"),
                }
            )
            if len(evidence) >= max_items:
                return evidence
    return evidence


def build_user_prompt(task: dict[str, Any], skill_docs: list[dict[str, str]], max_edge_evidence: int) -> str:
    compact = {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "structure_type": task.get("structure_type"),
        "difficulty": task.get("difficulty"),
        "task_quality_score": task.get("task_quality_score"),
        "mean_edge_score": task.get("mean_edge_score"),
        "skill_connections": task.get("skill_connections", []),
        "edge_evidence": limited_edge_evidence(task, max_edge_evidence),
        "skills": skill_docs,
    }
    return (
        "Evaluate whether the following candidate skill graph composition is suitable "
        "for one benchmark task. Use SKILL.md as the source of truth.\n\n"
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
        raise RuntimeError(f"HTTP {exc.code} from chat API: {body}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response: {data}") from exc


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
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


def normalize_suggested_task(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    workflow = value.get("workflow", [])
    if isinstance(workflow, str):
        workflow = [workflow]
    verification = value.get("verification", [])
    if isinstance(verification, str):
        verification = [verification]
    return {
        "title": str(value.get("title", "")).strip(),
        "input": str(value.get("input", "")).strip(),
        "workflow": [str(item).strip() for item in workflow if str(item).strip()],
        "expected_output": str(value.get("expected_output", "")).strip(),
        "verification": [str(item).strip() for item in verification if str(item).strip()],
    }


def original_skill_names(task: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for skill in task.get("skills", []):
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("skill_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def parse_connection_names(connection: Any) -> tuple[str, str] | None:
    match = re.search(r"(.+?)\s*->\s*(.+?)(?:\s*\(|$)", str(connection))
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def missing_doc_record(task: dict[str, Any], missing_names: list[str]) -> dict[str, Any]:
    row = dict(task)
    row.update(
        {
            "composition_score": 1,
            "verifiability_score": 1,
            "overall_score": 1.0,
            "keep": False,
            "decision": "reject",
            "llm_decision": "",
            "reason": "Missing SKILL.md for: " + ", ".join(missing_names),
            "suggested_task": normalize_suggested_task({}),
            "flags": ["missing_skill_md"],
            "skill_doc_paths": {},
            "error": "missing_skill_md",
        }
    )
    return row


def build_evaluation_record(
    *,
    task: dict[str, Any],
    parsed: dict[str, Any],
    skill_doc_paths: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    composition = clamp_score(parsed.get("composition_score"))
    verifiability = clamp_score(parsed.get("verifiability_score"))
    overall = WEIGHTS["composition_score"] * composition + WEIGHTS["verifiability_score"] * verifiability

    keep = (
        overall >= args.min_overall
        and composition >= args.min_composition
        and verifiability >= args.min_verifiability
    )
    reason = str(parsed.get("reason", "")).strip()

    row = dict(task)
    row.update(
        {
            "composition_score": composition,
            "verifiability_score": verifiability,
            "overall_score": round(overall, 3),
            "keep": keep,
            "decision": "keep" if keep else "reject",
            "llm_decision": str(parsed.get("decision", "")).strip(),
            "reason": reason,
            "suggested_task": normalize_suggested_task(parsed.get("suggested_task")),
            "flags": normalize_string_list(parsed.get("flags")),
            "skill_doc_paths": skill_doc_paths,
        }
    )
    return row


def evaluate_one(
    *,
    task: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    skills_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    skill_docs: list[dict[str, str]] = []
    skill_doc_paths: dict[str, str] = {}
    missing: list[str] = []
    for skill in task.get("skills", []):
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("skill_name") or "")
        path, content = resolve_skill_doc(skill, by_id, by_name, skills_dir)
        if path is None:
            missing.append(name)
            continue
        skill_doc_paths[name] = normalize_path(path)
        skill_docs.append({"skill_name": name, "skill_md": content})
    if missing:
        return missing_doc_record(task, missing)

    prompt = build_user_prompt(task, skill_docs, args.max_edge_evidence_per_task)
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
            return build_evaluation_record(
                task=task,
                parsed=parse_model_json(text),
                skill_doc_paths=skill_doc_paths,
                args=args,
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
    row = dict(task)
    row.update(
        {
            "composition_score": 1,
            "verifiability_score": 1,
            "overall_score": 1.0,
            "keep": False,
            "decision": "reject",
            "reason": f"Evaluation failed: {last_error}",
            "error": str(last_error),
            "flags": ["evaluation_failed"],
        }
    )
    return row


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
        "input": args.input,
        "model": args.model,
        "weights": WEIGHTS,
        "thresholds": {
            "min_composition": args.min_composition,
            "min_verifiability": args.min_verifiability,
            "min_overall": args.min_overall,
        },
        "total": len(rows),
        "keep_count": len(keep_rows),
        "reject_count": len(reject_rows),
        "failed_count": sum(1 for row in rows if "error" in row),
    }
    write_json(output_path, {"meta": meta, "evaluations": rows})
    write_json(keep_path, {"meta": meta, "tasks": keep_rows})
    write_json(reject_path, {"meta": meta, "tasks": reject_rows})


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required for task evaluation.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_edge_evidence_per_task < 0:
        raise ValueError("--max-edge-evidence-per-task must be non-negative")

    input_path = Path(args.input)
    output_path = Path(args.output)
    keep_path = Path(args.keep_output)
    reject_path = Path(args.reject_output)
    skills_dir = Path(args.skills_dir)

    tasks = load_candidate_tasks(
        input_path,
        categories=set(args.categories),
        structures=set(args.structures),
        difficulties=set(args.difficulties),
    )
    existing = [] if args.force else read_existing(output_path)
    rows_by_id = {
        str(row.get("task_id")): row
        for row in existing
        if row.get("task_id") and "error" not in row and "composition_score" in row
    }
    pending = [task for task in tasks if str(task.get("task_id")) not in rows_by_id]
    if args.limit > 0:
        pending = pending[: args.limit]

    by_id, by_name = load_skill_index(Path(args.skill_index), skills_dir)
    print(
        "tasks={}, existing_ok={}, pending={}, skill_docs={}, workers={}".format(
            len(tasks), len(rows_by_id), len(pending), len(by_name), args.workers
        )
    )

    completed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    evaluate_one,
                    task=task,
                    by_id=by_id,
                    by_name=by_name,
                    skills_dir=skills_dir,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    use_response_format=not args.disable_response_format,
                    args=args,
                ): task
                for task in pending
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                row = future.result()
                rows_by_id[str(task.get("task_id"))] = row
                completed += 1
                status = "keep" if row.get("keep") else "reject"
                if "error" in row:
                    status = "failed"
                print(
                    "[{}/{}] {}: {} {} overall={} comp={} ver={}".format(
                        completed,
                        len(pending),
                        status,
                        row.get("task_id"),
                        row.get("category"),
                        row.get("overall_score"),
                        row.get("composition_score"),
                        row.get("verifiability_score"),
                    ),
                    flush=True,
                )
                write_outputs(
                    output_path=output_path,
                    keep_path=keep_path,
                    reject_path=reject_path,
                    args=args,
                    rows=list(rows_by_id.values()),
                )

    write_outputs(
        output_path=output_path,
        keep_path=keep_path,
        reject_path=reject_path,
        args=args,
        rows=list(rows_by_id.values()),
    )
    keep_count = sum(1 for row in rows_by_id.values() if row.get("keep") is True and "error" not in row)
    failed = sum(1 for row in rows_by_id.values() if "error" in row)
    print(
        f"done: total={len(rows_by_id)}, keep={keep_count}, reject={len(rows_by_id)-keep_count}, failed={failed}, output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
