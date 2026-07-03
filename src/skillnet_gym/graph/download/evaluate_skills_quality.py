#!/usr/bin/env python3
"""Evaluate downloaded skills for benchmark suitability with concurrent LLM calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple


WEIGHTS = {
    "environment_cost_score": 0.30,
    "verifiability_score": 0.45,
    "documentation_quality_score": 0.25,
}


SYSTEM_PROMPT = """You evaluate whether a downloaded skill is suitable for large-scale benchmark task construction.

Return exactly one JSON object. Do not include markdown.

Score three dimensions from 1 to 5. Scores must be integers.

Dimension 1: environment_cost_score
This score means how lightweight and reproducible the skill is. Higher is better.
- 5: Very lightweight; Python standard library or very small/common pip packages.
- 4: Lightweight; common libraries such as pandas, numpy, matplotlib, requests.
- 3: Medium; specialized but stable pip/conda libraries.
- 2: Heavy; system dependencies, external tools, complex environment, large downloads, or unstable software.
- 1: Very heavy; GPU, large local model, paid API, browser automation, complex GUI software, Docker-in-Docker, CUDA, or proprietary service.
Look for complex system dependencies, GPU/large model requirements, external API/paid service, network requirement, large downloads, slow install, unstable dependencies, and Docker reproducibility.

Dimension 2: verifiability_score
This score means whether outputs can be automatically tested. Higher is better.
- 5: Highly structured and easy to check: JSON, CSV, Excel, SQLite, numeric metrics, fixed schema, labels, validated files.
- 4: Clear output that can be checked with rules: Markdown report, chart file, table, formatted file.
- 3: Partly verifiable but some subjectivity remains.
- 2: Mostly natural language and difficult to strictly judge.
- 1: Highly subjective or creative output; almost impossible to automatically evaluate.
Prefer skills with explicit files, schemas, fields, values, metrics, deterministic outputs, hidden-test potential, or validation procedures.

Dimension 3: documentation_quality_score
This score means whether the SKILL.md is complete, concrete, and reusable. Higher is better.
- 5: Very complete: clear use cases, inputs, outputs, steps, dependencies, examples, boundary conditions, and error handling.
- 4: Mostly complete; minor details missing but usable.
- 3: Understandable but inputs/outputs or steps are not detailed enough.
- 2: Very short or generic; execution requires much guessing.
- 1: Almost unusable; only a title, one sentence, keyword tags, or vague generic description.
Look for clear applicability, input description, output description, workflow steps, dependencies, edge cases, examples, and non-vague scope.

Important:
- Judge only from the provided SKILL.md content and skill name.
- Do not reward vague broad capabilities.
- Penalize paid APIs, GPU/large model inference, browser automation, LibreOffice rendering, OCR with Tesseract, FFmpeg/video-heavy workflows, and complex system components for environment cost.
- Penalize subjective outputs such as "make it better", "creative suggestions", "beautiful essay", "general advice" for verifiability.
- Return concise reasons grounded in the SKILL.md.

Required JSON shape:
{
  "environment_cost_score": 1,
  "verifiability_score": 1,
  "documentation_quality_score": 1,
  "environment_cost_reason": "",
  "verifiability_reason": "",
  "documentation_quality_reason": "",
  "flags": []
}
"""


class SkillTask(NamedTuple):
    skill_id: int
    skill_name: str
    skill_dir: Path
    skill_md: Path


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
        description="Evaluate downloaded SKILL.md files for benchmark suitability."
    )
    parser.add_argument(
        "--skills-dir",
        default=env_default("SKILLS_DIR", "downloaded_skills"),
        help="Root directory containing downloaded skills.",
    )
    parser.add_argument(
        "--output",
        default=env_default("SKILL_QUALITY_OUTPUT", "skill_quality_evaluation.json"),
        help="Full evaluation output JSON.",
    )
    parser.add_argument(
        "--keep-output",
        default=env_default("SKILL_QUALITY_KEEP_OUTPUT", "skill_quality_keep.json"),
        help="JSON file containing kept skills only.",
    )
    parser.add_argument(
        "--reject-output",
        default=env_default("SKILL_QUALITY_REJECT_OUTPUT", "skill_quality_reject.json"),
        help="JSON file containing rejected skills only.",
    )
    parser.add_argument(
        "--api-key",
        default=env_default("API_KEY", ""),
        help="LLM API key. Defaults to API_KEY in .env/env.",
    )
    parser.add_argument(
        "--base-url",
        default=env_default("BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=env_default("MODEL", "gpt-4o-mini"),
        help="Chat model name.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(env_default("SKILL_QUALITY_WORKERS", "8")),
        help="Concurrent LLM workers.",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per skill.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N skills.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate existing successes.")
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send response_format=json_object for providers that do not support it.",
    )
    parser.add_argument("--min-environment", type=float, default=2.0)
    parser.add_argument("--min-verifiability", type=float, default=3.0)
    parser.add_argument("--min-documentation", type=float, default=3.5)
    parser.add_argument("--min-overall", type=float, default=3.0)
    return parser.parse_args()


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("evaluations"), list):
        return [item for item in payload["evaluations"] if isinstance(item, dict)]
    raise ValueError(f"{path} must contain a JSON array or an object with key 'evaluations'")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def discover_tasks(skills_dir: Path) -> list[SkillTask]:
    skill_files = sorted(path for path in skills_dir.rglob("SKILL.md") if path.is_file())
    return [
        SkillTask(
            skill_id=index,
            skill_name=path.parent.name,
            skill_dir=path.parent,
            skill_md=path,
        )
        for index, path in enumerate(skill_files, start=1)
    ]


def build_user_prompt(task: SkillTask, content: str) -> str:
    return """Evaluate this skill for benchmark suitability.

skill_id: {skill_id}
skill_name: {skill_name}
skill_path: {skill_path}

SKILL.md:
```markdown
{content}
```""".format(
        skill_id=task.skill_id,
        skill_name=task.skill_name,
        skill_path=task.skill_md.as_posix(),
        content=content,
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
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed


def clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def normalize_flags(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_record(
    *,
    task: SkillTask,
    parsed: dict[str, Any],
    min_environment: int,
    min_verifiability: int,
    min_documentation: int,
    min_overall: float,
) -> dict[str, Any]:
    environment = clamp_score(parsed.get("environment_cost_score"))
    verifiability = clamp_score(parsed.get("verifiability_score"))
    documentation = clamp_score(parsed.get("documentation_quality_score"))
    overall = (
        WEIGHTS["environment_cost_score"] * environment
        + WEIGHTS["verifiability_score"] * verifiability
        + WEIGHTS["documentation_quality_score"] * documentation
    )
    keep = (
        environment >= min_environment
        and verifiability >= min_verifiability
        and documentation >= min_documentation
        and overall >= min_overall
    )
    return {
        "skill_id": task.skill_id,
        "skill_name": task.skill_name,
        "skill_dir": task.skill_dir.as_posix(),
        "skill_path": task.skill_md.as_posix(),
        "environment_cost_score": environment,
        "verifiability_score": verifiability,
        "documentation_quality_score": documentation,
        "overall_score": round(overall, 3),
        "keep": keep,
        "reasons": {
            "environment_cost": str(parsed.get("environment_cost_reason", "")).strip(),
            "verifiability": str(parsed.get("verifiability_reason", "")).strip(),
            "documentation_quality": str(parsed.get("documentation_quality_reason", "")).strip(),
        },
        "flags": normalize_flags(parsed.get("flags")),
    }


def evaluate_one(
    *,
    task: SkillTask,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
    min_environment: int,
    min_verifiability: int,
    min_documentation: int,
    min_overall: float,
) -> dict[str, Any]:
    content = task.skill_md.read_text(encoding="utf-8", errors="replace")
    prompt = build_user_prompt(task, content)

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
            parsed = parse_model_json(text)
            return build_record(
                task=task,
                parsed=parsed,
                min_environment=min_environment,
                min_verifiability=min_verifiability,
                min_documentation=min_documentation,
                min_overall=min_overall,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 20))

    return {
        "skill_id": task.skill_id,
        "skill_name": task.skill_name,
        "skill_dir": task.skill_dir.as_posix(),
        "skill_path": task.skill_md.as_posix(),
        "environment_cost_score": 1,
        "verifiability_score": 1,
        "documentation_quality_score": 1,
        "overall_score": 1.0,
        "keep": False,
        "reasons": {
            "environment_cost": "",
            "verifiability": "",
            "documentation_quality": "",
        },
        "flags": [],
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def write_outputs(
    *,
    output_path: Path,
    keep_path: Path,
    reject_path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    rows = sorted(rows, key=lambda item: int(item.get("skill_id", 0)))
    keep_rows = [row for row in rows if row.get("keep") is True and "error" not in row]
    reject_rows = [row for row in rows if row not in keep_rows]
    meta = {
        "skills_dir": args.skills_dir,
        "model": args.model,
        "weights": WEIGHTS,
        "thresholds": {
            "min_environment": args.min_environment,
            "min_verifiability": args.min_verifiability,
            "min_documentation": args.min_documentation,
            "min_overall": args.min_overall,
        },
        "total": len(rows),
        "keep_count": len(keep_rows),
        "reject_count": len(reject_rows),
    }
    write_json(output_path, {"meta": meta, "evaluations": rows})
    write_json(keep_path, {"meta": meta, "skills": keep_rows})
    write_json(reject_path, {"meta": meta, "skills": reject_rows})


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required. Set it in .env or pass --api-key.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")

    skills_dir = Path(args.skills_dir)
    if not skills_dir.exists():
        raise FileNotFoundError(f"skills directory does not exist: {skills_dir}")

    tasks = discover_tasks(skills_dir)
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive, or 0 for no limit")
        tasks = tasks[: args.limit]

    output_path = Path(args.output)
    keep_path = Path(args.keep_output)
    reject_path = Path(args.reject_output)
    existing = [] if args.force else read_existing(output_path)
    rows_by_id = {
        int(item["skill_id"]): item
        for item in existing
        if isinstance(item.get("skill_id"), int) and "error" not in item
    }
    pending = [task for task in tasks if task.skill_id not in rows_by_id]
    print(
        "skills={}, existing_ok={}, pending={}, workers={}".format(
            len(tasks), len(rows_by_id), len(pending), args.workers
        ),
        flush=True,
    )

    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                evaluate_one,
                task=task,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
                use_response_format=not args.disable_response_format,
                min_environment=args.min_environment,
                min_verifiability=args.min_verifiability,
                min_documentation=args.min_documentation,
                min_overall=args.min_overall,
            ): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "skill_id": task.skill_id,
                    "skill_name": task.skill_name,
                    "skill_dir": task.skill_dir.as_posix(),
                    "skill_path": task.skill_md.as_posix(),
                    "environment_cost_score": 1,
                    "verifiability_score": 1,
                    "documentation_quality_score": 1,
                    "overall_score": 1.0,
                    "keep": False,
                    "reasons": {
                        "environment_cost": "",
                        "verifiability": "",
                        "documentation_quality": "",
                    },
                    "flags": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

            rows_by_id[int(row["skill_id"])] = row
            if "error" in row:
                failed += 1
                status = "failed"
            else:
                status = "keep" if row.get("keep") else "reject"
            print(
                "[{}/{}] {}: {} env={} ver={} doc={} overall={}".format(
                    completed,
                    len(pending),
                    status,
                    row["skill_name"],
                    row["environment_cost_score"],
                    row["verifiability_score"],
                    row["documentation_quality_score"],
                    row["overall_score"],
                ),
                flush=True,
            )
            if "error" in row:
                print(f"  failed: {row['error']}", file=sys.stderr, flush=True)

            write_outputs(
                output_path=output_path,
                keep_path=keep_path,
                reject_path=reject_path,
                rows=list(rows_by_id.values()),
                args=args,
            )

    write_outputs(
        output_path=output_path,
        keep_path=keep_path,
        reject_path=reject_path,
        rows=list(rows_by_id.values()),
        args=args,
    )
    rows = list(rows_by_id.values())
    keep_count = sum(1 for row in rows if row.get("keep") is True and "error" not in row)
    reject_count = len(rows) - keep_count
    print(
        f"done: total={len(rows)}, keep={keep_count}, reject={reject_count}, failed={failed}, output={output_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
