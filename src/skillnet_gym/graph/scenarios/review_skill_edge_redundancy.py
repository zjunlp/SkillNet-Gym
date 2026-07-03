#!/usr/bin/env python3
"""Review compatible skill edges and drop functionally redundant skill pairs.

This script is intended to run after align_skill_scenarios.py and before
build_scenario_skill_graph.py. It groups scenario alignments by
source_skill -> target_skill, gives both full SKILL.md files to an LLM, and
keeps only edges where the target skill is a distinct next workflow step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You judge whether two connected skills are functionally redundant.

Return exactly one JSON object. Do not include markdown.

You will receive:
- source_skill: the skill that runs first
- target_skill: the skill proposed to run after source_skill
- scenario_connections: why the previous alignment step thought source can connect to target
- source_skill_md: full SKILL.md for the source skill
- target_skill_md: full SKILL.md for the target skill

Your task is NOT to re-judge scenario compatibility. Your task is to detect whether
the two skills largely perform the same function and should not both appear in one
workflow graph edge.

Drop the edge when:
- The target skill mostly repeats the source skill's main capability.
- The two skills are near-duplicate implementations of the same step.
- The target is only a format/name variant of the source with no distinct downstream work.
- The connection is mostly same-state restatement rather than a new operation.
- The target skill would replace the source skill rather than consume its result.

Keep the edge when:
- The target skill performs a clearly distinct next step on the source output.
- The source prepares, extracts, cleans, validates, converts, or enriches something, and the target analyzes, reports, visualizes, tests, trains, deploys, or otherwise advances it.
- The two skills share a domain but have different roles in a realistic workflow.
- Some overlap exists, but the target has a meaningful additional downstream purpose.

Required JSON shape:
{
  "keep_edge": true,
  "redundant": false,
  "overlap_score": 1,
  "redundancy_type": "none",
  "reason": ""
}

overlap_score is an integer from 1 to 5:
1 = clearly distinct steps
2 = small overlap but distinct workflow roles
3 = moderate overlap, still probably usable as a workflow edge
4 = high functional overlap, usually redundant
5 = near-duplicate or replacement skill

redundancy_type must be one of:
- "none"
- "same_capability"
- "near_duplicate"
- "format_variant"
- "same_state_restatement"
- "replacement_not_handoff"
- "unclear"
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
        description="LLM-review connected skill pairs and remove functionally redundant edges."
    )
    parser.add_argument(
        "--input",
        default=env_default("SCENARIO_ALIGNMENT_KEEP_OUTPUT", "scenario_alignment_keep.json"),
        help="Compatible alignment JSON from align_skill_scenarios.py. Can also be graph.json.",
    )
    parser.add_argument(
        "--skill-index",
        default=env_default("SKILL_EDGE_REVIEW_SKILL_INDEX", "skill_quality_keep.json"),
        help="JSON containing skill_id, skill_name, skill_path/local_path fields.",
    )
    parser.add_argument(
        "--skills-dir",
        default=env_default("DOWNLOADED_SKILLS_DIR", "downloaded_skills"),
        help="Fallback directory containing skill-name/SKILL.md.",
    )
    parser.add_argument(
        "--output",
        default=env_default("SKILL_EDGE_REVIEW_OUTPUT", "skill_edge_redundancy_reviews.json"),
        help="Full pair review output.",
    )
    parser.add_argument(
        "--keep-output",
        default=env_default("SKILL_EDGE_REVIEW_KEEP_OUTPUT", "scenario_alignment_nonredundant_keep.json"),
        help="Filtered alignments used by build_scenario_skill_graph.py.",
    )
    parser.add_argument("--api-key", default=env_default("API_KEY", ""))
    parser.add_argument("--base-url", default=env_default("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=env_default("MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(env_default("SKILL_EDGE_REVIEW_WORKERS", "8")),
        help="Concurrent LLM workers.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Review at most N pending skill pairs.")
    parser.add_argument("--force", action="store_true", help="Re-review existing successful pairs.")
    parser.add_argument(
        "--drop-overlap-score",
        type=int,
        default=4,
        help="Drop an edge when overlap_score is at least this value.",
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


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_alignments(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("alignments"), list):
        return dict(payload.get("meta") or {}), [row for row in payload["alignments"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'alignments'")


def is_compatible_alignment(row: dict[str, Any]) -> bool:
    if row.get("compatible") is not True:
        return False
    alignment_type = str(row.get("alignment_type") or "")
    return alignment_type not in {"duplicate_or_alternative", "topical_only", "incompatible"}


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


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
    if not path.exists():
        return {}, {}
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        rows = payload["skills"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must contain a JSON array or object field 'skills'")

    by_id: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            skill_id = int(row.get("skill_id") or 0)
        except (TypeError, ValueError):
            skill_id = 0
        skill_name = str(row.get("skill_name") or "").strip()
        skill_path = next((candidate for candidate in candidate_skill_paths(row, skills_dir) if candidate.exists()), None)
        if skill_path is None:
            continue
        record = {
            "skill_id": skill_id,
            "skill_name": skill_name or skill_path.parent.name,
            "skill_path": normalize_path(skill_path),
            "local_path": normalize_path(skill_path.parent),
        }
        if skill_id > 0:
            by_id[skill_id] = record
        if record["skill_name"]:
            by_name[record["skill_name"]] = record
    return by_id, by_name


def resolve_skill(
    *,
    skill_id: int,
    skill_name: str,
    by_id: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    skills_dir: Path,
) -> dict[str, Any]:
    if skill_id in by_id:
        return by_id[skill_id]
    if skill_name in by_name:
        return by_name[skill_name]
    fallback = skills_dir / skill_name / "SKILL.md"
    if fallback.exists():
        return {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_path": normalize_path(fallback),
            "local_path": normalize_path(fallback.parent),
        }
    raise FileNotFoundError(f"Cannot resolve SKILL.md for skill_id={skill_id}, skill_name={skill_name!r}")


def pair_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("source_skill_id") or 0), int(row.get("target_skill_id") or 0)


def build_pair_jobs(
    alignments: list[dict[str, Any]],
    *,
    by_id: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    skills_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    for row in alignments:
        if not is_compatible_alignment(row):
            continue
        try:
            source_id, target_id = pair_key(row)
        except (TypeError, ValueError):
            skipped.append({**row, "skip_reason": "invalid_skill_ids"})
            continue
        if source_id <= 0 or target_id <= 0 or source_id == target_id:
            skipped.append({**row, "skip_reason": "invalid_or_self_pair"})
            continue
        grouped.setdefault((source_id, target_id), []).append(row)

    jobs: list[dict[str, Any]] = []
    for (source_id, target_id), rows in sorted(grouped.items()):
        source_name = str(rows[0].get("source_skill_name") or "")
        target_name = str(rows[0].get("target_skill_name") or "")
        try:
            source_skill = resolve_skill(
                skill_id=source_id,
                skill_name=source_name,
                by_id=by_id,
                by_name=by_name,
                skills_dir=skills_dir,
            )
            target_skill = resolve_skill(
                skill_id=target_id,
                skill_name=target_name,
                by_id=by_id,
                by_name=by_name,
                skills_dir=skills_dir,
            )
        except FileNotFoundError as exc:
            for row in rows:
                skipped.append({**row, "skip_reason": str(exc)})
            continue
        jobs.append(
            {
                "pair_id": f"{source_id}->{target_id}",
                "source_skill_id": source_id,
                "source_skill_name": source_name or source_skill["skill_name"],
                "source_skill_path": source_skill["skill_path"],
                "target_skill_id": target_id,
                "target_skill_name": target_name or target_skill["skill_name"],
                "target_skill_path": target_skill["skill_path"],
                "alignment_count": len(rows),
                "scenario_connections": [
                    {
                        "alignment_id": row.get("alignment_id"),
                        "source_post_scenario": row.get("source_post_scenario"),
                        "target_pre_scenario": row.get("target_pre_scenario"),
                        "alignment_type": row.get("alignment_type"),
                        "confidence": row.get("confidence"),
                        "retrieval_score": row.get("retrieval_score"),
                    }
                    for row in rows
                ],
            }
        )
    return jobs, skipped


def build_user_prompt(job: dict[str, Any], source_md: str, target_md: str) -> str:
    compact = {
        "source_skill": {
            "skill_id": job["source_skill_id"],
            "skill_name": job["source_skill_name"],
            "skill_path": job["source_skill_path"],
        },
        "target_skill": {
            "skill_id": job["target_skill_id"],
            "skill_name": job["target_skill_name"],
            "skill_path": job["target_skill_path"],
        },
        "scenario_connections": job["scenario_connections"],
        "source_skill_md": source_md,
        "target_skill_md": target_md,
    }
    return (
        "Review whether this connected skill pair is functionally redundant.\n\n"
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
            payload = json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Unexpected chat response: {payload}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError(f"Missing message content: {payload}")
    return content


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        import re

        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object")
    return payload


def clamp_int(value: Any, *, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def normalize_review(job: dict[str, Any], parsed: dict[str, Any], *, drop_overlap_score: int) -> dict[str, Any]:
    overlap_score = clamp_int(parsed.get("overlap_score"), low=1, high=5, default=3)
    redundant = bool(parsed.get("redundant"))
    keep_edge = bool(parsed.get("keep_edge", not redundant and overlap_score < drop_overlap_score))
    if redundant or overlap_score >= drop_overlap_score:
        keep_edge = False

    valid_types = {
        "none",
        "same_capability",
        "near_duplicate",
        "format_variant",
        "same_state_restatement",
        "replacement_not_handoff",
        "unclear",
    }
    redundancy_type = str(parsed.get("redundancy_type") or "").strip() or "none"
    if redundancy_type not in valid_types:
        redundancy_type = "unclear"

    row = dict(job)
    row.update(
        {
            "keep_edge": keep_edge,
            "redundant": redundant,
            "overlap_score": overlap_score,
            "redundancy_type": redundancy_type,
            "reason": str(parsed.get("reason", "")).strip(),
        }
    )
    return row


def evaluate_job(
    job: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
    drop_overlap_score: int,
) -> dict[str, Any]:
    source_md = read_text_lossy(Path(job["source_skill_path"]))
    target_md = read_text_lossy(Path(job["target_skill_path"]))
    prompt = build_user_prompt(job, source_md, target_md)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_prompt=prompt,
                timeout=timeout,
                use_response_format=use_response_format,
            )
            return normalize_review(
                job,
                parse_model_json(response),
                drop_overlap_score=drop_overlap_score,
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
    row = dict(job)
    row.update({"error": str(last_error), "keep_edge": False})
    return row


def read_existing_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        return [row for row in payload["reviews"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'reviews'")


def write_outputs(
    *,
    output_path: Path,
    keep_path: Path,
    input_meta: dict[str, Any],
    args: argparse.Namespace,
    reviews: list[dict[str, Any]],
    skipped_alignments: list[dict[str, Any]],
    original_alignments: list[dict[str, Any]],
) -> None:
    review_by_pair = {str(row.get("pair_id")): row for row in reviews if row.get("pair_id")}
    kept_pairs = {
        pair_id
        for pair_id, row in review_by_pair.items()
        if row.get("keep_edge") is True and "error" not in row
    }
    kept_alignments: list[dict[str, Any]] = []
    dropped_alignments: list[dict[str, Any]] = []
    for row in original_alignments:
        if not is_compatible_alignment(row):
            continue
        try:
            source_id, target_id = pair_key(row)
        except (TypeError, ValueError):
            continue
        pair_id = f"{source_id}->{target_id}"
        review = review_by_pair.get(pair_id)
        if pair_id in kept_pairs:
            kept_alignments.append(
                {
                    **row,
                    "redundancy_review": {
                        "pair_id": pair_id,
                        "keep_edge": review.get("keep_edge") if review else None,
                        "redundant": review.get("redundant") if review else None,
                        "overlap_score": review.get("overlap_score") if review else None,
                        "redundancy_type": review.get("redundancy_type") if review else None,
                        "reason": review.get("reason") if review else None,
                    },
                }
            )
        else:
            dropped_alignments.append(
                {
                    **row,
                    "drop_reason": "redundant_or_unreviewed_skill_pair",
                    "redundancy_review": review,
                }
            )

    meta = {
        "input": args.input,
        "input_meta": input_meta,
        "model": args.model,
        "skill_index": args.skill_index,
        "skills_dir": args.skills_dir,
        "drop_overlap_score": args.drop_overlap_score,
        "pair_count": len(reviews),
        "kept_pair_count": len(kept_pairs),
        "dropped_pair_count": len(reviews) - len(kept_pairs),
        "original_alignment_count": len(original_alignments),
        "kept_alignment_count": len(kept_alignments),
        "dropped_alignment_count": len(dropped_alignments),
        "skipped_alignment_count": len(skipped_alignments),
        "failed_pair_count": sum(1 for row in reviews if "error" in row),
    }
    write_json(
        output_path,
        {
            "meta": meta,
            "reviews": sorted(reviews, key=lambda row: str(row.get("pair_id") or "")),
            "dropped_alignments": dropped_alignments,
            "skipped_alignments": skipped_alignments,
        },
    )
    write_json(keep_path, {"meta": meta, "alignments": kept_alignments})


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required for LLM review.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.drop_overlap_score < 1 or args.drop_overlap_score > 5:
        raise ValueError("--drop-overlap-score must be between 1 and 5")

    input_path = Path(args.input)
    output_path = Path(args.output)
    keep_path = Path(args.keep_output)
    skills_dir = Path(args.skills_dir)

    input_meta, alignments = read_alignments(input_path)
    by_id, by_name = load_skill_index(Path(args.skill_index), skills_dir)
    jobs, skipped_alignments = build_pair_jobs(
        alignments,
        by_id=by_id,
        by_name=by_name,
        skills_dir=skills_dir,
    )

    existing = [] if args.force else read_existing_reviews(output_path)
    reviews_by_pair = {
        str(row.get("pair_id")): row
        for row in existing
        if row.get("pair_id") and "error" not in row and "keep_edge" in row
    }
    pending = [job for job in jobs if job["pair_id"] not in reviews_by_pair]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        "pairs={}, existing_ok={}, pending={}, skipped_alignments={}, workers={}".format(
            len(jobs), len(reviews_by_pair), len(pending), len(skipped_alignments), args.workers
        )
    )

    completed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    evaluate_job,
                    job,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    use_response_format=not args.disable_response_format,
                    drop_overlap_score=args.drop_overlap_score,
                ): job
                for job in pending
            }
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                row = future.result()
                reviews_by_pair[job["pair_id"]] = row
                completed += 1
                status = "keep" if row.get("keep_edge") is True else "drop"
                if "error" in row:
                    status = "failed"
                print(
                    "[{}/{}] {}: {} -> {} overlap={} type={}".format(
                        completed,
                        len(pending),
                        status,
                        row.get("source_skill_name"),
                        row.get("target_skill_name"),
                        row.get("overlap_score"),
                        row.get("redundancy_type"),
                    ),
                    flush=True,
                )
                write_outputs(
                    output_path=output_path,
                    keep_path=keep_path,
                    input_meta=input_meta,
                    args=args,
                    reviews=list(reviews_by_pair.values()),
                    skipped_alignments=skipped_alignments,
                    original_alignments=alignments,
                )

    write_outputs(
        output_path=output_path,
        keep_path=keep_path,
        input_meta=input_meta,
        args=args,
        reviews=list(reviews_by_pair.values()),
        skipped_alignments=skipped_alignments,
        original_alignments=alignments,
    )
    kept = sum(1 for row in reviews_by_pair.values() if row.get("keep_edge") is True and "error" not in row)
    failed = sum(1 for row in reviews_by_pair.values() if "error" in row)
    print(
        "done: pairs={}, kept_pairs={}, failed_pairs={}, output={}, keep_output={}".format(
            len(reviews_by_pair), kept, failed, output_path, keep_path
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
