#!/usr/bin/env python3
"""Infer precondition and postcondition scenarios for each retained skill."""

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
from typing import Any


SYSTEM_PROMPT = """You infer scenario-mediated transitions from a skill specification.

Return exactly one JSON object. Do not include markdown.

Definitions:
- A pre_scenario is the semantic state BEFORE using the skill: what data, files, environment, user need, or partial task state makes this skill applicable.
- A post_scenario is the semantic state AFTER successfully using the skill: what data, files, environment, artifact, or task state has been produced or changed.
- A scenario is a state, not an action. Do not write imperative verbs such as "extract", "generate", "analyze", "convert" unless they describe an existing state.

Rules:
- Infer from the full SKILL.md, skill name, triggers, examples, commands, dependencies, and expected workflow.
- Extract all distinct pre_scenarios and post_scenarios that are explicitly stated or strongly implied by the skill.
- Do not omit meaningful scenarios just to keep the list short.
- If the skill supports multiple clearly different input states, workflows, modes, or output states, include each distinct state.
- Scenarios must be short, concrete noun phrases under 18 words.
- Prefer executable/data states over vague intentions.
- Avoid generic scenarios like "user has a task" unless the skill is truly generic.
- Do not include file extensions alone as scenarios; include semantic state and artifact when relevant.
- The post_scenarios must be plausible direct results of running the skill.
- Preserve the provided skill_id and skill_name exactly.

Required JSON shape:
{
  "skill_id": 1,
  "skill_name": "",
  "pre_scenarios": [],
  "post_scenarios": []
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
        description="Infer pre/post scenarios from SKILL.md files using an LLM."
    )
    parser.add_argument(
        "--input",
        default=env_default("SKILL_DEDUP_KEEP_OUTPUT", "skill_dedup_keep.json"),
        help="Input JSON containing retained skills, usually skill_dedup_keep.json.",
    )
    parser.add_argument(
        "--output",
        default=env_default("SKILL_SCENARIOS_OUTPUT", "skill_scenarios.json"),
        help="Scenario extraction output JSON.",
    )
    parser.add_argument("--api-key", default=env_default("API_KEY", ""))
    parser.add_argument("--base-url", default=env_default("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=env_default("MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(env_default("SKILL_SCENARIO_WORKERS", "8")),
        help="Concurrent LLM workers.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pending skills.")
    parser.add_argument("--force", action="store_true", help="Re-extract existing successful rows.")
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
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        return [row for row in payload["skills"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'skills'")


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_skills(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        rows = payload["skills"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must contain a JSON array or object field 'skills'")

    skills: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        skill_path = Path(str(row.get("skill_path") or ""))
        if not skill_path.exists():
            skill_dir = Path(str(row.get("local_path") or row.get("skill_dir") or ""))
            skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            continue
        normalized_path = str(skill_path).replace("\\", "/")
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        skills.append(
            {
                "skill_id": int(row.get("skill_id") or len(skills) + 1),
                "skill_name": str(row.get("skill_name") or skill_path.parent.name),
                "skill_path": normalized_path,
                "local_path": str(row.get("local_path") or skill_path.parent).replace("\\", "/"),
                "stars": int(row.get("stars") or 0),
            }
        )
    return sorted(skills, key=lambda item: int(item["skill_id"]))


def build_user_prompt(skill: dict[str, Any], skill_md: str) -> str:
    return """Infer scenario-mediated preconditions and postconditions for this skill.

Return exactly this JSON shape:
{{
  "skill_id": {skill_id},
  "skill_name": "{skill_name}",
  "pre_scenarios": [],
  "post_scenarios": []
}}

skill_id: {skill_id}
skill_name: {skill_name}
stars: {stars}

SKILL.md:
```markdown
{skill_md}
```
""".format(
        skill_id=int(skill["skill_id"]),
        skill_name=str(skill["skill_name"]).replace('"', '\\"'),
        stars=int(skill.get("stars") or 0),
        skill_md=skill_md,
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
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
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
            response_payload = json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Unexpected chat response: {response_payload}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError(f"Missing message content: {response_payload}")
    return content


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object")
    return payload


def normalize_scenarios(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = re.sub(r"\s+", " ", str(item)).strip(" .;:-")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def normalize_row(parsed: dict[str, Any], skill: dict[str, Any]) -> dict[str, Any]:
    pre = normalize_scenarios(parsed.get("pre_scenarios"))
    post = normalize_scenarios(parsed.get("post_scenarios"))
    if not pre:
        pre = ["Applicable task state requiring this skill"]
    if not post:
        post = ["Task state after applying this skill"]
    return {
        "skill_id": int(skill["skill_id"]),
        "skill_name": str(skill["skill_name"]),
        "skill_path": skill["skill_path"],
        "local_path": skill["local_path"],
        "stars": int(skill.get("stars") or 0),
        "pre_scenarios": pre,
        "post_scenarios": post,
    }


def process_skill(
    skill: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
) -> dict[str, Any]:
    skill_md = read_text_lossy(Path(skill["skill_path"]))
    prompt = build_user_prompt(skill, skill_md)
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
            return normalize_row(parsed, skill)
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
    row = normalize_row({}, skill)
    row["error"] = f"{type(last_error).__name__}: {last_error}"
    return row


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required. Set it in .env or pass --api-key.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    skills = load_skills(input_path)
    existing = [] if args.force else read_existing(output_path)
    rows_by_id = {
        int(row["skill_id"]): row
        for row in existing
        if isinstance(row.get("skill_id"), int)
        and "error" not in row
        and row.get("pre_scenarios")
        and row.get("post_scenarios")
    }
    pending = [skill for skill in skills if int(skill["skill_id"]) not in rows_by_id]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        "skills={}, existing_ok={}, pending={}, workers={}".format(
            len(skills), len(rows_by_id), len(pending), args.workers
        )
    )

    completed = 0
    failed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_skill = {
                executor.submit(
                    process_skill,
                    skill,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    use_response_format=not args.disable_response_format,
                ): skill
                for skill in pending
            }
            for future in concurrent.futures.as_completed(future_to_skill):
                skill = future_to_skill[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = normalize_row({}, skill)
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows_by_id[int(row["skill_id"])] = row
                completed += 1
                if "error" in row:
                    failed += 1
                    print(f"[{completed}/{len(pending)}] failed: {row['skill_id']} - {row['error']}", flush=True)
                else:
                    print(
                        "[{}/{}] ok: {} pre={} post={}".format(
                            completed,
                            len(pending),
                            row["skill_name"],
                            len(row["pre_scenarios"]),
                            len(row["post_scenarios"]),
                        ),
                        flush=True,
                    )
                output_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
                write_json(
                    output_path,
                    {
                        "meta": {
                            "input": str(input_path),
                            "model": args.model,
                            "total": len(skills),
                            "completed": len(output_rows),
                            "failed": sum(1 for item in output_rows if "error" in item),
                        },
                        "skills": output_rows,
                    },
                )

    output_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    write_json(
        output_path,
        {
            "meta": {
                "input": str(input_path),
                "model": args.model,
                "total": len(skills),
                "completed": len(output_rows),
                "failed": sum(1 for item in output_rows if "error" in item),
            },
            "skills": output_rows,
        },
    )
    print(f"done: total={len(skills)}, completed={len(output_rows)}, failed={failed}, output={output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
