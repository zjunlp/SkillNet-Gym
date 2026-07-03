#!/usr/bin/env python3
"""Cluster near-duplicate downloaded skills and keep the highest-star representative."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


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
        description="Cluster downloaded SKILL.md files by embedding similarity and keep highest-star skills."
    )
    parser.add_argument(
        "--skills-dir",
        default=env_default("SKILLS_DIR", "downloaded_skills"),
        help="Directory containing downloaded skill folders.",
    )
    parser.add_argument(
        "--quality-keep",
        default=env_default("SKILL_QUALITY_KEEP_OUTPUT", "skill_quality_keep.json"),
        help="Quality keep JSON. By default, cluster only skills listed in this file.",
    )
    parser.add_argument(
        "--scan-all-downloaded",
        action="store_true",
        help="Ignore --quality-keep and cluster every downloaded skill under --skills-dir.",
    )
    parser.add_argument(
        "--use-quality-keep",
        action="store_true",
        help="Deprecated compatibility flag. Clustering --quality-keep is now the default.",
    )
    parser.add_argument(
        "--manifest",
        default=env_default("DOWNLOADED_SKILLS_MANIFEST", "downloaded_skills_manifest.json"),
        help="Download manifest used to read stars, URLs, and query provenance.",
    )
    parser.add_argument(
        "--clusters-output",
        default=env_default("SKILL_DEDUP_CLUSTERS_OUTPUT", "skill_dedup_clusters.json"),
        help="Cluster audit JSON output.",
    )
    parser.add_argument(
        "--keep-output",
        default=env_default("SKILL_DEDUP_KEEP_OUTPUT", "skill_dedup_keep.json"),
        help="Representative skills JSON output.",
    )
    parser.add_argument(
        "--drop-output",
        default=env_default("SKILL_DEDUP_DROP_OUTPUT", "skill_dedup_drop.json"),
        help="Dropped duplicate skills JSON output.",
    )
    parser.add_argument(
        "--cache",
        default=env_default("SKILL_DEDUP_EMBEDDING_CACHE", "skill_dedup_embedding_cache.json"),
        help="Embedding cache JSON.",
    )
    parser.add_argument(
        "--api-key",
        default=env_default("EMBEDDING_API_KEY", env_default("API_KEY", "")),
        help="Embedding API key. Defaults to EMBEDDING_API_KEY, then API_KEY from .env/env.",
    )
    parser.add_argument(
        "--base-url",
        default=env_default("EMBEDDING_BASE_URL", env_default("BASE_URL", "https://api.openai.com/v1")),
        help="OpenAI-compatible embedding API base URL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=env_default("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model name.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(env_default("SKILL_DEDUP_SIM_THRESHOLD", "0.92")),
        help="Cosine similarity threshold for near-duplicate candidates.",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=int(env_default("SKILL_DEDUP_TOP_NEIGHBORS", "50")),
        help="FAISS top-N neighbors searched for each skill.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=int(env_default("SKILL_DEDUP_CHUNK_CHARS", "12000")),
        help="Split long skill texts into chunks before embedding, then average chunk embeddings. 0 disables chunking.",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=int(env_default("SKILL_DEDUP_EMBEDDING_WORKERS", "4")),
        help="Concurrent embedding API batch requests.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per embedding batch.")
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Ignore embedding cache and request all embeddings again.",
    )
    parser.add_argument(
        "--move-dropped-to",
        default="",
        help="Optional directory to move dropped duplicate skill folders into after clustering.",
    )
    return parser.parse_args()


def require_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required. Install it first, e.g. `pip install faiss-cpu`."
        ) from exc


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalized_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").rstrip("/")


def load_manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    records: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        candidate_lists = []
        for key in ("downloaded", "skipped", "failed"):
            value = payload.get(key)
            if isinstance(value, list):
                candidate_lists.append(value)
        for rows in candidate_lists:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                local_path = row.get("local_path")
                if not local_path:
                    continue
                records[normalized_path(str(local_path))] = row
    return records


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def skill_text(skill_name: str, manifest_row: dict[str, Any], skill_md: str) -> str:
    description = str(manifest_row.get("skill_description", "")).strip()
    queries = manifest_row.get("queries", [])
    query_text = ", ".join(str(item) for item in queries) if isinstance(queries, list) else ""
    return (
        f"Skill name: {skill_name}\n"
        f"Description: {description}\n"
        f"Matched queries: {query_text}\n\n"
        f"SKILL.md:\n{skill_md}"
    )


def scan_skills(skills_dir: Path, manifest_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        skill_dir = skill_path.parent
        key = normalized_path(skill_dir)
        manifest_row = manifest_records.get(key, {})
        skill_name = str(manifest_row.get("skill_name") or skill_dir.name)
        stars = int(manifest_row.get("stars") or 0)
        skill_md = read_text_lossy(skill_path)
        rows.append(
            {
                "skill_id": len(rows) + 1,
                "skill_name": skill_name,
                "stars": stars,
                "skill_url": str(manifest_row.get("skill_url", "")),
                "author": str(manifest_row.get("author", "")),
                "category": str(manifest_row.get("category", "")),
                "queries": manifest_row.get("queries", []),
                "local_path": normalized_path(skill_dir),
                "skill_path": normalized_path(skill_path),
                "text": skill_text(skill_name, manifest_row, skill_md),
            }
        )
    return rows


def load_quality_keep_skills(
    path: Path,
    manifest_records: dict[str, dict[str, Any]],
    skills_dir: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Quality keep file not found: {path}")
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        rows_raw = payload["skills"]
    elif isinstance(payload, list):
        rows_raw = payload
    else:
        raise ValueError(f"{path} must contain a JSON array or an object with key 'skills'")

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in rows_raw:
        if not isinstance(row, dict):
            continue
        if row.get("keep") is False:
            continue
        skill_path_value = row.get("skill_path")
        skill_dir_value = row.get("skill_dir")
        skill_name_hint = str(row.get("skill_name") or "").strip()
        candidate_paths: list[Path] = []
        if skill_path_value:
            candidate_paths.append(Path(str(skill_path_value)))
        if skill_dir_value:
            candidate_paths.append(Path(str(skill_dir_value)) / "SKILL.md")
            candidate_paths.append(skills_dir / Path(str(skill_dir_value)).name / "SKILL.md")
        if skill_name_hint:
            candidate_paths.append(skills_dir / skill_name_hint / "SKILL.md")
        if skill_path_value:
            candidate_paths.append(skills_dir / Path(str(skill_path_value)).parent.name / "SKILL.md")

        skill_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
        if skill_path is None:
            continue
        skill_dir = skill_path.parent
        local_key = normalized_path(skill_dir)
        if local_key in seen_paths:
            continue
        seen_paths.add(local_key)
        manifest_row = manifest_records.get(local_key, {})
        skill_name = str(skill_name_hint or manifest_row.get("skill_name") or skill_dir.name)
        stars = int(manifest_row.get("stars") or row.get("stars") or 0)
        skill_md = read_text_lossy(skill_path)
        rows.append(
            {
                "skill_id": int(row.get("skill_id") or len(rows) + 1),
                "skill_name": skill_name,
                "stars": stars,
                "skill_url": str(manifest_row.get("skill_url", row.get("skill_url", ""))),
                "author": str(manifest_row.get("author", row.get("author", ""))),
                "category": str(manifest_row.get("category", row.get("category", ""))),
                "queries": manifest_row.get("queries", row.get("queries", [])),
                "local_path": local_key,
                "skill_path": normalized_path(skill_path),
                "quality": {
                    "environment_cost_score": row.get("environment_cost_score"),
                    "verifiability_score": row.get("verifiability_score"),
                    "documentation_quality_score": row.get("documentation_quality_score"),
                    "overall_score": row.get("overall_score"),
                },
                "text": skill_text(skill_name, manifest_row, skill_md),
            }
        )
    return rows


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
            len(texts),
            len(texts) - len(missing_indices),
            len(missing_indices),
            embedding_workers,
        )
    )

    batches = [
        missing_indices[start : start + batch_size]
        for start in range(0, len(missing_indices), batch_size)
    ]

    def embed_batch(batch_indices: list[int]) -> tuple[list[int], list[list[float]]]:
        batch_texts = [texts[i] for i in batch_indices]
        vectors = retry_embeddings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            texts=batch_texts,
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
                print(
                    f"embedding batch {completed}/{len(batches)} done: {len(batch_indices)} texts",
                    flush=True,
                )
                save_cache(cache_path, model, cache)

    matrix = np.array([cache[key] for key in keys], dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError("Invalid embedding matrix shape")
    return matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def split_text(text: str, chunk_chars: int) -> list[str]:
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start + chunk_chars // 2:
                end = newline + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [text]


def get_skill_embeddings(
    *,
    texts: list[str],
    chunk_chars: int,
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
    chunk_texts: list[str] = []
    chunk_to_skill: list[int] = []
    for skill_index, text in enumerate(texts):
        for chunk in split_text(text, chunk_chars):
            chunk_texts.append(chunk)
            chunk_to_skill.append(skill_index)

    print(
        "embedding chunks: skills={}, chunks={}, chunk_chars={}".format(
            len(texts),
            len(chunk_texts),
            chunk_chars if chunk_chars > 0 else "disabled",
        )
    )
    chunk_embeddings = get_embeddings(
        texts=chunk_texts,
        cache=cache,
        cache_path=cache_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        batch_size=batch_size,
        embedding_workers=embedding_workers,
        timeout=timeout,
        retries=retries,
    )

    dim = chunk_embeddings.shape[1]
    skill_embeddings = np.zeros((len(texts), dim), dtype="float32")
    counts = np.zeros((len(texts), 1), dtype="float32")
    normalized_chunks = l2_normalize(chunk_embeddings.astype("float32"))
    for chunk_index, skill_index in enumerate(chunk_to_skill):
        skill_embeddings[skill_index] += normalized_chunks[chunk_index]
        counts[skill_index] += 1.0
    counts[counts == 0] = 1.0
    return skill_embeddings / counts


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_duplicate_edges(
    embeddings: np.ndarray,
    *,
    top_neighbors: int,
    threshold: float,
) -> list[dict[str, Any]]:
    faiss = require_faiss()
    normalized = np.ascontiguousarray(l2_normalize(embeddings).astype("float32"))
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    search_k = min(normalized.shape[0], top_neighbors + 1)
    scores, indices = index.search(normalized, search_k)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for source_idx, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        for score, target_idx in zip(row_scores, row_indices):
            target_idx = int(target_idx)
            if target_idx < 0 or target_idx == source_idx:
                continue
            if float(score) < threshold:
                continue
            left, right = sorted((source_idx, target_idx))
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({"source_index": left, "target_index": right, "score": round(float(score), 6)})
    return edges


def choose_representative(member_indices: list[int], skills: list[dict[str, Any]]) -> int:
    return sorted(
        member_indices,
        key=lambda i: (
            -int(skills[i]["stars"]),
            str(skills[i]["skill_name"]).lower(),
            str(skills[i]["skill_url"]),
            str(skills[i]["local_path"]),
        ),
    )[0]


def similarity_to_rep(
    embeddings: np.ndarray,
    representative: int,
    member: int,
) -> float:
    normalized = l2_normalize(embeddings[[representative, member]].astype("float32"))
    return float(np.dot(normalized[0], normalized[1]))


def build_clusters(
    skills: list[dict[str, Any]],
    embeddings: np.ndarray,
    duplicate_edges: list[dict[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uf = UnionFind(len(skills))
    for edge in duplicate_edges:
        uf.union(int(edge["source_index"]), int(edge["target_index"]))

    groups: dict[int, list[int]] = {}
    for index in range(len(skills)):
        groups.setdefault(uf.find(index), []).append(index)

    edge_scores: dict[tuple[int, int], float] = {}
    for edge in duplicate_edges:
        pair = tuple(sorted((int(edge["source_index"]), int(edge["target_index"]))))
        edge_scores[pair] = float(edge["score"])

    clusters: list[dict[str, Any]] = []
    keep_rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []

    cluster_id = 0
    for member_indices in sorted(groups.values(), key=lambda group: min(group)):
        representative = choose_representative(member_indices, skills)
        kept = dict(skills[representative])
        kept.pop("text", None)
        kept["cluster_id"] = cluster_id + 1
        kept["cluster_size"] = len(member_indices)
        keep_rows.append(kept)

        members: list[dict[str, Any]] = []
        dropped_in_cluster: list[dict[str, Any]] = []
        for member in sorted(member_indices, key=lambda i: (-int(skills[i]["stars"]), str(skills[i]["skill_name"]).lower())):
            rep_score = 1.0 if member == representative else similarity_to_rep(embeddings, representative, member)
            row = dict(skills[member])
            row.pop("text", None)
            row["is_representative"] = member == representative
            row["similarity_to_representative"] = round(rep_score, 6)
            members.append(row)
            if member != representative and rep_score >= threshold:
                dropped = dict(row)
                dropped["cluster_id"] = cluster_id + 1
                dropped["representative_skill_name"] = skills[representative]["skill_name"]
                dropped["representative_local_path"] = skills[representative]["local_path"]
                dropped["representative_stars"] = skills[representative]["stars"]
                dropped_in_cluster.append(dropped)
                drop_rows.append(dropped)

        cluster_edges = []
        member_set = set(member_indices)
        for left in member_indices:
            for right in member_indices:
                if left >= right:
                    continue
                score = edge_scores.get((left, right))
                if score is not None and left in member_set and right in member_set:
                    cluster_edges.append(
                        {
                            "source_skill_name": skills[left]["skill_name"],
                            "target_skill_name": skills[right]["skill_name"],
                            "score": round(score, 6),
                        }
                    )

        if len(member_indices) > 1:
            cluster_id += 1
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(member_indices),
                    "representative": kept,
                    "members": members,
                    "dropped_count": len(dropped_in_cluster),
                    "duplicate_edges": sorted(cluster_edges, key=lambda item: item["score"], reverse=True),
                }
            )
            keep_rows[-1]["cluster_id"] = cluster_id
        else:
            keep_rows[-1]["cluster_id"] = None

    return clusters, keep_rows, drop_rows


def move_dropped(drop_rows: list[dict[str, Any]], target_dir: Path) -> int:
    moved = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in drop_rows:
        source = Path(str(row["local_path"]))
        if not source.exists():
            continue
        destination = target_dir / source.name
        suffix = 1
        while destination.exists():
            destination = target_dir / f"{source.name}__{suffix}"
            suffix += 1
        shutil.move(str(source), str(destination))
        row["moved_to"] = normalized_path(destination)
        moved += 1
    return moved


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required. Set it in .env or pass --api-key.")
    if args.threshold < -1 or args.threshold > 1:
        raise ValueError("--threshold must be between -1 and 1")
    if args.top_neighbors < 1:
        raise ValueError("--top-neighbors must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.chunk_chars < 0:
        raise ValueError("--chunk-chars must be >= 0")
    if args.embedding_workers < 1:
        raise ValueError("--embedding-workers must be at least 1")

    require_faiss()

    skills_dir = Path(args.skills_dir)
    manifest_records = load_manifest_records(Path(args.manifest))
    if args.scan_all_downloaded:
        skills = scan_skills(skills_dir, manifest_records)
        input_source = str(skills_dir)
    else:
        skills = load_quality_keep_skills(Path(args.quality_keep), manifest_records, skills_dir)
        input_source = args.quality_keep
    if not skills:
        raise ValueError(f"No SKILL.md files found from input source: {input_source}")
    print(f"skills: total={len(skills)}, manifest_matches={sum(1 for s in skills if s.get('skill_url'))}")

    cache_path = Path(args.cache)
    cache = load_cache(cache_path, force=args.force_embeddings)
    embeddings = get_skill_embeddings(
        texts=[str(skill["text"]) for skill in skills],
        chunk_chars=args.chunk_chars,
        cache=cache,
        cache_path=cache_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.embedding_model,
        batch_size=args.batch_size,
        embedding_workers=args.embedding_workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    save_cache(cache_path, args.embedding_model, cache)

    duplicate_edges = build_duplicate_edges(
        embeddings,
        top_neighbors=args.top_neighbors,
        threshold=args.threshold,
    )
    clusters, keep_rows, drop_rows = build_clusters(
        skills,
        embeddings,
        duplicate_edges,
        threshold=args.threshold,
    )

    moved = 0
    if args.move_dropped_to:
        moved = move_dropped(drop_rows, Path(args.move_dropped_to))

    meta = {
        "skills_dir": str(skills_dir),
        "input_source": input_source,
        "quality_keep": args.quality_keep,
        "scan_all_downloaded": args.scan_all_downloaded,
        "manifest": args.manifest,
        "embedding_model": args.embedding_model,
        "threshold": args.threshold,
        "top_neighbors": args.top_neighbors,
        "chunk_chars": args.chunk_chars,
        "total_skills": len(skills),
        "duplicate_edge_count": len(duplicate_edges),
        "cluster_count": len(clusters),
        "keep_count": len(keep_rows),
        "drop_count": len(drop_rows),
        "moved_count": moved,
    }
    write_json(Path(args.clusters_output), {"meta": meta, "clusters": clusters})
    write_json(Path(args.keep_output), {"meta": meta, "skills": keep_rows})
    write_json(Path(args.drop_output), {"meta": meta, "skills": drop_rows})

    print(
        "done: skills={}, clusters={}, keep={}, drop={}, clusters_output={}, keep_output={}, drop_output={}".format(
            len(skills),
            len(clusters),
            len(keep_rows),
            len(drop_rows),
            args.clusters_output,
            args.keep_output,
            args.drop_output,
        )
    )
    if moved:
        print(f"moved dropped directories: {moved} -> {args.move_dropped_to}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
