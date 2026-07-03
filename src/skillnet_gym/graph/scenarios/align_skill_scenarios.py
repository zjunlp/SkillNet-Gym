#!/usr/bin/env python3
"""Align post-scenarios to pre-scenarios across skills with retrieval + LLM verification."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EMBED_INSTRUCTION = "Instruct: Retrieve scenarios that describe compatible real-world task states.\nQuery: "

SYSTEM_PROMPT = """You judge whether one skill's post-scenario can serve as another skill's pre-scenario.

Return exactly one JSON object. Do not include markdown.

You will receive:
- source_skill: the skill that produced the post-scenario
- source_post_scenario: the state after source_skill succeeds
- target_skill: the skill that would run next
- target_pre_scenario: the state required before target_skill can run

Judge whether the source post-scenario can naturally satisfy or instantiate the target pre-scenario in a real workflow.

Keep only real workflow handoffs:
- The source state must provide a meaningful artifact, data state, environment state, evidence, or prerequisite for the target skill.
- The target skill must perform a distinct next step, not repeat the same capability.
- Compatible states may differ in wording or granularity if the produced artifact/state can reasonably be used by the target.

Reject when:
- The two skills are alternatives or near-duplicates for the same step.
- The scenarios are merely topically similar but no artifact/state handoff exists.
- The direction is wrong.
- Required formats, platforms, or environments are incompatible.
- The source state is too vague to satisfy the target precondition.
- The target pre-scenario is simply a restatement of the source post-scenario with no next step.

Required JSON shape:
{
  "compatible": false,
  "alignment_type": "artifact_handoff",
  "confidence": 1,
  "reason": ""
}

alignment_type must be one of:
- "artifact_handoff"
- "data_state_handoff"
- "environment_state_handoff"
- "evidence_or_metadata_handoff"
- "same_state_merge"
- "incompatible"
- "duplicate_or_alternative"
- "topical_only"

confidence is an integer from 1 to 5.
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
        description="Retrieve and LLM-verify cross-skill scenario alignments."
    )
    parser.add_argument(
        "--input",
        default=env_default("SCENARIO_DEDUP_OUTPUT", "scenario_dedup.json"),
        help="Input JSON from deduplicate_scenarios.py.",
    )
    parser.add_argument(
        "--output",
        default=env_default("SCENARIO_ALIGNMENT_OUTPUT", "scenario_alignments.json"),
        help="All alignment evaluations output JSON.",
    )
    parser.add_argument(
        "--keep-output",
        default=env_default("SCENARIO_ALIGNMENT_KEEP_OUTPUT", "scenario_alignment_keep.json"),
        help="Compatible alignment edges only.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=env_default("SCENARIO_ALIGNMENT_EMBEDDING_CACHE", "scenario_alignment_embedding_cache.json"),
        help="Embedding cache JSON.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=env_default("EMBEDDING_API_KEY", env_default("API_KEY", "")),
    )
    parser.add_argument(
        "--embedding-base-url",
        default=env_default("EMBEDDING_BASE_URL", env_default("BASE_URL", "https://api.openai.com/v1")),
    )
    parser.add_argument("--embedding-model", default=env_default("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--api-key", default=env_default("API_KEY", ""))
    parser.add_argument("--base-url", default=env_default("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=env_default("MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(env_default("SCENARIO_ALIGNMENT_TOP_K", "30")),
        help="Top-k pre-scenarios retrieved per post-scenario.",
    )
    parser.add_argument(
        "--min-retrieval-score",
        type=float,
        default=float(env_default("SCENARIO_ALIGNMENT_MIN_SCORE", "0.72")),
        help="Minimum embedding score for an alignment candidate.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=int(env_default("SCENARIO_ALIGNMENT_EMBEDDING_WORKERS", "4")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(env_default("SCENARIO_ALIGNMENT_WORKERS", "8")),
        help="Concurrent LLM workers.",
    )
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N pending candidates.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate existing rows.")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--disable-response-format", action="store_true")
    return parser.parse_args()


def require_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required. Install with `pip install faiss-cpu`.") from exc


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def text_cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def load_cache(path: Path, *, force: bool) -> dict[str, list[float]]:
    if force or not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    embeddings = payload.get("embeddings", payload)
    if not isinstance(embeddings, dict):
        return {}
    return {
        str(key): value
        for key, value in embeddings.items()
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)
    }


def save_cache(path: Path, model: str, embeddings: dict[str, list[float]]) -> None:
    write_json(path, {"model": model, "count": len(embeddings), "embeddings": embeddings})


def request_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
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
        raise RuntimeError(f"HTTP {exc.code} from embeddings API: {body}") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Unexpected embeddings response: {payload}")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Invalid embedding item: {item}")
        vectors.append([float(x) for x in vector])
    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
    return vectors


def retry_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
    retries: int,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return request_embeddings(
                base_url=base_url,
                api_key=api_key,
                model=model,
                texts=texts,
                timeout=timeout,
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
    raise RuntimeError(f"Embedding request failed: {last_error}") from last_error


def get_embeddings(
    *,
    texts: list[str],
    cache: dict[str, list[float]],
    cache_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    embedding_workers: int,
    timeout: float,
    retries: int,
) -> np.ndarray:
    keys = [text_cache_key(model, text) for text in texts]
    missing_indices = [i for i, key in enumerate(keys) if key not in cache]
    print(
        "embeddings: total={}, cached={}, missing={}, workers={}".format(
            len(texts), len(texts) - len(missing_indices), len(missing_indices), embedding_workers
        )
    )
    batches = [
        missing_indices[start : start + batch_size]
        for start in range(0, len(missing_indices), batch_size)
    ]

    def embed_batch(batch_indices: list[int]) -> tuple[list[int], list[list[float]]]:
        vectors = retry_embeddings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            texts=[texts[i] for i in batch_indices],
            timeout=timeout,
            retries=retries,
        )
        return batch_indices, vectors

    completed = 0
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=embedding_workers) as executor:
            future_to_batch = {executor.submit(embed_batch, batch): batch for batch in batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_indices, vectors = future.result()
                for i, vector in zip(batch_indices, vectors):
                    cache[keys[i]] = vector
                completed += 1
                print(f"embedding batch {completed}/{len(batches)} done: {len(batch_indices)} texts", flush=True)
                save_cache(cache_path, model, cache)

    matrix = np.array([cache[key] for key in keys], dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError("Invalid embedding matrix shape")
    return matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_dedup(path: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    scenarios = payload.get("scenarios")
    skill_scenarios = payload.get("skill_scenarios")
    if not isinstance(scenarios, list) or not isinstance(skill_scenarios, list):
        raise ValueError(f"{path} must contain list fields 'scenarios' and 'skill_scenarios'")
    scenario_by_id: dict[int, str] = {}
    for item in scenarios:
        if isinstance(item, dict) and "scenario_id" in item:
            scenario_by_id[int(item["scenario_id"])] = str(item.get("canonical_scenario", ""))
    skills = [item for item in skill_scenarios if isinstance(item, dict)]
    return scenario_by_id, skills


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("alignments"), list):
        return [item for item in payload["alignments"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'alignments'")


def build_candidates(
    *,
    scenario_by_id: dict[int, str],
    skills: list[dict[str, Any]],
    embeddings: np.ndarray,
    scenario_ids: list[int],
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    faiss = require_faiss()
    scenario_index = {scenario_id: idx for idx, scenario_id in enumerate(scenario_ids)}
    normalized = np.ascontiguousarray(l2_normalize(embeddings).astype("float32"))

    pre_ids = sorted({int(sid) for skill in skills for sid in skill.get("pre_scenario_ids", [])})
    post_ids = sorted({int(sid) for skill in skills for sid in skill.get("post_scenario_ids", [])})
    pre_matrix_indices = [scenario_index[sid] for sid in pre_ids if sid in scenario_index]
    pre_ids = [sid for sid in pre_ids if sid in scenario_index]
    post_ids = [sid for sid in post_ids if sid in scenario_index]

    pre_matrix = np.ascontiguousarray(normalized[pre_matrix_indices].astype("float32"))
    index = faiss.IndexFlatIP(pre_matrix.shape[1])
    index.add(pre_matrix)
    search_k = min(len(pre_ids), top_k)
    scores, indices = index.search(
        np.ascontiguousarray(normalized[[scenario_index[sid] for sid in post_ids]].astype("float32")),
        search_k,
    )

    skills_by_pre: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skills_by_post: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for skill in skills:
        for sid in skill.get("pre_scenario_ids", []):
            skills_by_pre[int(sid)].append(skill)
        for sid in skill.get("post_scenario_ids", []):
            skills_by_post[int(sid)].append(skill)

    for sid in skills_by_pre:
        skills_by_pre[sid].sort(key=lambda item: int(item.get("skill_id") or 0))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for post_sid, row_scores, row_indices in zip(post_ids, scores, indices):
        source_skills = skills_by_post.get(post_sid, [])
        for score, pre_position in zip(row_scores, row_indices):
            if int(pre_position) < 0:
                continue
            score = float(score)
            if score < min_score:
                continue
            pre_sid = pre_ids[int(pre_position)]
            target_skills = skills_by_pre.get(pre_sid, [])
            for source_skill in source_skills:
                for target_skill in target_skills:
                    source_skill_id = int(source_skill.get("skill_id") or 0)
                    target_skill_id = int(target_skill.get("skill_id") or 0)
                    if source_skill_id == target_skill_id:
                        continue
                    key = (source_skill_id, post_sid, target_skill_id, pre_sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "alignment_id": f"align-{len(candidates)+1:07d}",
                            "source_skill_id": source_skill_id,
                            "source_skill_name": source_skill.get("skill_name"),
                            "source_post_scenario_id": post_sid,
                            "source_post_scenario": scenario_by_id[post_sid],
                            "target_skill_id": target_skill_id,
                            "target_skill_name": target_skill.get("skill_name"),
                            "target_pre_scenario_id": pre_sid,
                            "target_pre_scenario": scenario_by_id[pre_sid],
                            "retrieval_score": round(score, 6),
                        }
                    )
    candidates.sort(
        key=lambda item: (
            -float(item["retrieval_score"]),
            int(item["source_skill_id"]),
            int(item["target_skill_id"]),
            int(item["source_post_scenario_id"]),
            int(item["target_pre_scenario_id"]),
        )
    )
    for index_number, item in enumerate(candidates, start=1):
        item["alignment_id"] = f"align-{index_number:07d}"
    return candidates


def build_user_prompt(candidate: dict[str, Any]) -> str:
    compact = {
        "source_skill": {
            "skill_id": candidate["source_skill_id"],
            "skill_name": candidate["source_skill_name"],
        },
        "source_post_scenario": {
            "scenario_id": candidate["source_post_scenario_id"],
            "scenario": candidate["source_post_scenario"],
        },
        "target_skill": {
            "skill_id": candidate["target_skill_id"],
            "skill_name": candidate["target_skill_name"],
        },
        "target_pre_scenario": {
            "scenario_id": candidate["target_pre_scenario_id"],
            "scenario": candidate["target_pre_scenario"],
        },
        "retrieval_score": candidate["retrieval_score"],
    }
    return (
        "Evaluate whether this cross-skill scenario alignment is a valid workflow handoff.\n\n"
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


def clamp_confidence(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, number))


def normalize_result(candidate: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    compatible = bool(parsed.get("compatible"))
    alignment_type = str(parsed.get("alignment_type") or "").strip() or "incompatible"
    valid_types = {
        "artifact_handoff",
        "data_state_handoff",
        "environment_state_handoff",
        "evidence_or_metadata_handoff",
        "same_state_merge",
        "incompatible",
        "duplicate_or_alternative",
        "topical_only",
    }
    if alignment_type not in valid_types:
        alignment_type = "incompatible" if not compatible else "data_state_handoff"
    row = dict(candidate)
    row.update(
        {
            "compatible": compatible,
            "alignment_type": alignment_type,
            "confidence": clamp_confidence(parsed.get("confidence")),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    )
    return row


def evaluate_candidate(
    candidate: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
    use_response_format: bool,
) -> dict[str, Any]:
    prompt = build_user_prompt(candidate)
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
            return normalize_result(candidate, parsed)
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
    row = dict(candidate)
    row.update(
        {
            "compatible": False,
            "alignment_type": "incompatible",
            "confidence": 1,
            "reason": "",
            "error": f"{type(last_error).__name__}: {last_error}",
        }
    )
    return row


def write_outputs(
    *,
    output_path: Path,
    keep_path: Path,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    rows = sorted(rows, key=lambda item: str(item.get("alignment_id", "")))
    keep_rows = [
        row
        for row in rows
        if row.get("compatible") is True
        and int(row.get("confidence") or 0) >= 3
        and "error" not in row
    ]
    meta = dict(meta)
    meta.update(
        {
            "evaluated_count": len(rows),
            "compatible_count": len(keep_rows),
            "failed_count": sum(1 for row in rows if "error" in row),
        }
    )
    write_json(output_path, {"meta": meta, "alignments": rows})
    write_json(keep_path, {"meta": meta, "alignments": keep_rows})


def main() -> int:
    args = parse_args()
    if not args.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY or API_KEY is required for retrieval.")
    if not args.api_key:
        raise ValueError("API_KEY is required for LLM verification.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be at least 1")
    if args.embedding_workers < 1 or args.workers < 1:
        raise ValueError("--embedding-workers and --workers must be at least 1")

    require_faiss()
    input_path = Path(args.input)
    output_path = Path(args.output)
    keep_path = Path(args.keep_output)
    scenario_by_id, skills = load_dedup(input_path)
    scenario_ids = sorted(scenario_by_id)
    texts = [EMBED_INSTRUCTION + scenario_by_id[sid] for sid in scenario_ids]
    cache_path = Path(args.embedding_cache)
    cache = load_cache(cache_path, force=args.force_embeddings)
    embeddings = get_embeddings(
        texts=texts,
        cache=cache,
        cache_path=cache_path,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        embedding_workers=args.embedding_workers,
        timeout=args.embedding_timeout,
        retries=args.retries,
    )
    save_cache(cache_path, args.embedding_model, cache)

    candidates = build_candidates(
        scenario_by_id=scenario_by_id,
        skills=skills,
        embeddings=embeddings,
        scenario_ids=scenario_ids,
        top_k=args.top_k,
        min_score=args.min_retrieval_score,
    )
    existing = [] if args.force else read_existing(output_path)
    rows_by_id = {
        str(row.get("alignment_id")): row
        for row in existing
        if row.get("alignment_id") and "error" not in row and "compatible" in row
    }
    pending = [candidate for candidate in candidates if candidate["alignment_id"] not in rows_by_id]
    if args.limit > 0:
        pending = pending[: args.limit]

    meta = {
        "input": str(input_path),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "min_retrieval_score": args.min_retrieval_score,
        "scenario_count": len(scenario_ids),
        "skill_count": len(skills),
        "candidate_count": len(candidates),
    }
    print(
        "scenarios={}, skills={}, candidates={}, existing_ok={}, pending={}, workers={}".format(
            len(scenario_ids), len(skills), len(candidates), len(rows_by_id), len(pending), args.workers
        )
    )

    completed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_candidate = {
                executor.submit(
                    evaluate_candidate,
                    candidate,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    use_response_format=not args.disable_response_format,
                ): candidate
                for candidate in pending
            }
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(candidate)
                    row.update(
                        {
                            "compatible": False,
                            "alignment_type": "incompatible",
                            "confidence": 1,
                            "reason": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                rows_by_id[str(row["alignment_id"])] = row
                completed += 1
                status = "keep" if row.get("compatible") and int(row.get("confidence") or 0) >= 3 else "reject"
                print(
                    "[{}/{}] {}: {} -> {} conf={} type={}".format(
                        completed,
                        len(pending),
                        status,
                        row.get("source_skill_name"),
                        row.get("target_skill_name"),
                        row.get("confidence"),
                        row.get("alignment_type"),
                    ),
                    flush=True,
                )
                write_outputs(
                    output_path=output_path,
                    keep_path=keep_path,
                    meta=meta,
                    rows=list(rows_by_id.values()),
                )

    write_outputs(
        output_path=output_path,
        keep_path=keep_path,
        meta=meta,
        rows=list(rows_by_id.values()),
    )
    keep_count = sum(
        1
        for row in rows_by_id.values()
        if row.get("compatible") is True and int(row.get("confidence") or 0) >= 3 and "error" not in row
    )
    print(
        f"done: candidates={len(candidates)}, evaluated={len(rows_by_id)}, compatible={keep_count}, output={output_path}, keep_output={keep_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
