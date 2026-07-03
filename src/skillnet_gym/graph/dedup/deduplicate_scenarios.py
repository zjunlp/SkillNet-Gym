#!/usr/bin/env python3
"""Deduplicate inferred scenarios with Louvain buckets and complete-linkage clustering."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EMBED_INSTRUCTION = "Instruct: Retrieve scenarios that describe the same real-world condition.\nQuery: "


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
        description="Deduplicate pre/post scenarios using Louvain coarse buckets and complete-linkage clustering."
    )
    parser.add_argument(
        "--input",
        default=env_default("SKILL_SCENARIOS_OUTPUT", "skill_scenarios.json"),
        help="Input JSON from extract_skill_scenarios.py.",
    )
    parser.add_argument(
        "--output",
        default=env_default("SCENARIO_DEDUP_OUTPUT", "scenario_dedup.json"),
        help="Scenario deduplication output JSON.",
    )
    parser.add_argument(
        "--cache",
        default=env_default("SCENARIO_EMBEDDING_CACHE", "scenario_embedding_cache.json"),
        help="Embedding cache JSON.",
    )
    parser.add_argument(
        "--api-key",
        default=env_default("EMBEDDING_API_KEY", env_default("API_KEY", "")),
        help="Embedding API key. Defaults to EMBEDDING_API_KEY, then API_KEY.",
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
        "--top-neighbors",
        type=int,
        default=int(env_default("SCENARIO_TOP_NEIGHBORS", "100")),
        help="FAISS top-N neighbors for sparse similarity graph construction.",
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=float(env_default("SCENARIO_GRAPH_THRESHOLD", "0.82")),
        help="Minimum cosine similarity edge used in the Louvain graph.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=float(env_default("SCENARIO_CLUSTER_THRESHOLD", "0.88")),
        help="Minimum pairwise cosine similarity required by complete-linkage clusters.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=int(env_default("SCENARIO_EMBEDDING_WORKERS", "4")),
        help="Concurrent embedding API batch requests.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def require_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required. Install with `pip install faiss-cpu`.") from exc


def require_networkx():
    try:
        import networkx as nx  # type: ignore

        if not hasattr(nx.community, "louvain_communities"):
            raise RuntimeError("networkx.community.louvain_communities is unavailable")
        return nx
    except ImportError as exc:
        raise RuntimeError("networkx is required. Install with `pip install networkx`.") from exc


def require_sklearn():
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore

        return AgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install with `pip install scikit-learn`.") from exc


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_scenario(text: str) -> str:
    return " ".join(str(text).split()).strip(" .;:-")


def load_scenarios(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        skills = [row for row in payload["skills"] if isinstance(row, dict)]
    elif isinstance(payload, list):
        skills = [row for row in payload if isinstance(row, dict)]
    else:
        raise ValueError(f"{path} must contain a JSON array or object field 'skills'")

    scenario_records: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for skill in skills:
        if "error" in skill:
            continue
        skill_id = int(skill.get("skill_id") or 0)
        skill_name = str(skill.get("skill_name") or "")
        for side in ("pre", "post"):
            field = f"{side}_scenarios"
            values = skill.get(field, [])
            if not isinstance(values, list):
                continue
            for text in values:
                scenario = normalize_scenario(str(text))
                if not scenario:
                    continue
                key = (skill_id, side, scenario.lower())
                if key in seen:
                    continue
                seen.add(key)
                scenario_records.append(
                    {
                        "raw_scenario_id": len(scenario_records) + 1,
                        "skill_id": skill_id,
                        "skill_name": skill_name,
                        "side": side,
                        "scenario": scenario,
                    }
                )
    return skills, scenario_records


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


def build_similarity_graph(
    embeddings: np.ndarray,
    *,
    top_neighbors: int,
    threshold: float,
    seed: int,
):
    faiss = require_faiss()
    nx = require_networkx()
    normalized = np.ascontiguousarray(l2_normalize(embeddings).astype("float32"))
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    search_k = min(normalized.shape[0], top_neighbors + 1)
    scores, indices = index.search(normalized, search_k)

    graph = nx.Graph()
    graph.add_nodes_from(range(normalized.shape[0]))
    edge_count = 0
    for source_idx, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        for score, target_idx in zip(row_scores, row_indices):
            target_idx = int(target_idx)
            if target_idx < 0 or target_idx == source_idx:
                continue
            score = float(score)
            if score < threshold:
                continue
            left, right = sorted((source_idx, target_idx))
            if graph.has_edge(left, right):
                if score > graph[left][right]["weight"]:
                    graph[left][right]["weight"] = score
                continue
            graph.add_edge(left, right, weight=score)
            edge_count += 1

    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=seed))
    communities = [sorted(list(community)) for community in communities if community]
    communities.sort(key=lambda item: (len(item), item[0]), reverse=True)
    return graph, communities


def complete_linkage_labels(distance_matrix: np.ndarray, threshold_distance: float) -> np.ndarray:
    AgglomerativeClustering = require_sklearn()
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=threshold_distance,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=threshold_distance,
        )
    return model.fit_predict(distance_matrix)


def split_community_complete_linkage(
    community: list[int],
    embeddings: np.ndarray,
    *,
    cluster_threshold: float,
) -> list[list[int]]:
    if len(community) <= 1:
        return [community]
    sub = l2_normalize(embeddings[community].astype("float32"))
    similarity = np.clip(sub @ sub.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    labels = complete_linkage_labels(distance, 1.0 - cluster_threshold)
    clusters: dict[int, list[int]] = defaultdict(list)
    for local_idx, label in enumerate(labels):
        clusters[int(label)].append(community[local_idx])
    result = list(clusters.values())
    result.sort(key=lambda item: (len(item), item[0]), reverse=True)
    return result


def choose_canonical(records: list[dict[str, Any]], member_indices: list[int]) -> str:
    texts = [records[i]["scenario"] for i in member_indices]
    counts = Counter(text.lower() for text in texts)
    return sorted(
        texts,
        key=lambda text: (-counts[text.lower()], len(text), text.lower()),
    )[0]


def deduplicate(
    scenario_records: list[dict[str, Any]],
    embeddings: np.ndarray,
    communities: list[list[int]],
    *,
    cluster_threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    clusters: list[dict[str, Any]] = []
    raw_to_cluster: dict[int, int] = {}
    for community_id, community in enumerate(communities, start=1):
        subclusters = split_community_complete_linkage(
            community,
            embeddings,
            cluster_threshold=cluster_threshold,
        )
        for subcluster in subclusters:
            cluster_id = len(clusters) + 1
            canonical = choose_canonical(scenario_records, subcluster)
            members = []
            sides = Counter()
            skill_ids: set[int] = set()
            for index in sorted(subcluster):
                record = scenario_records[index]
                raw_to_cluster[int(record["raw_scenario_id"])] = cluster_id
                sides[str(record["side"])] += 1
                skill_ids.add(int(record["skill_id"]))
                members.append(record)
            clusters.append(
                {
                    "scenario_id": cluster_id,
                    "canonical_scenario": canonical,
                    "size": len(subcluster),
                    "community_id": community_id,
                    "side_counts": dict(sides),
                    "skill_count": len(skill_ids),
                    "members": members,
                }
            )
    return clusters, raw_to_cluster


def build_skill_scenario_map(
    skills: list[dict[str, Any]],
    scenario_records: list[dict[str, Any]],
    raw_to_cluster: dict[int, int],
    clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_by_id = {int(cluster["scenario_id"]): cluster["canonical_scenario"] for cluster in clusters}
    by_skill: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"pre": [], "post": []})
    for record in scenario_records:
        cluster_id = raw_to_cluster.get(int(record["raw_scenario_id"]))
        if cluster_id is None:
            continue
        side = str(record["side"])
        values = by_skill[int(record["skill_id"])][side]
        if cluster_id not in values:
            values.append(cluster_id)

    rows: list[dict[str, Any]] = []
    for skill in skills:
        if "error" in skill:
            continue
        skill_id = int(skill.get("skill_id") or 0)
        pre_ids = by_skill[skill_id]["pre"]
        post_ids = by_skill[skill_id]["post"]
        rows.append(
            {
                "skill_id": skill_id,
                "skill_name": skill.get("skill_name"),
                "pre_scenario_ids": pre_ids,
                "post_scenario_ids": post_ids,
                "pre_scenarios": [canonical_by_id[item] for item in pre_ids],
                "post_scenarios": [canonical_by_id[item] for item in post_ids],
            }
        )
    return sorted(rows, key=lambda item: int(item["skill_id"]))


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API_KEY is required. Set it in .env or pass --api-key.")
    if args.top_neighbors < 1:
        raise ValueError("--top-neighbors must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.embedding_workers < 1:
        raise ValueError("--embedding-workers must be at least 1")
    if not -1 <= args.graph_threshold <= 1:
        raise ValueError("--graph-threshold must be between -1 and 1")
    if not -1 <= args.cluster_threshold <= 1:
        raise ValueError("--cluster-threshold must be between -1 and 1")

    require_faiss()
    require_networkx()
    require_sklearn()

    input_path = Path(args.input)
    skills, scenario_records = load_scenarios(input_path)
    if not scenario_records:
        raise ValueError("No scenarios found")
    print(f"skills={len(skills)}, raw_scenarios={len(scenario_records)}")

    texts = [EMBED_INSTRUCTION + record["scenario"] for record in scenario_records]
    cache_path = Path(args.cache)
    cache = load_cache(cache_path, force=args.force_embeddings)
    embeddings = get_embeddings(
        texts=texts,
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

    graph, communities = build_similarity_graph(
        embeddings,
        top_neighbors=args.top_neighbors,
        threshold=args.graph_threshold,
        seed=args.seed,
    )
    print(f"similarity_graph: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, communities={len(communities)}")

    clusters, raw_to_cluster = deduplicate(
        scenario_records,
        embeddings,
        communities,
        cluster_threshold=args.cluster_threshold,
    )
    skill_scenarios = build_skill_scenario_map(skills, scenario_records, raw_to_cluster, clusters)
    clusters.sort(key=lambda item: (int(item["scenario_id"])))
    meta = {
        "input": str(input_path),
        "embedding_model": args.embedding_model,
        "top_neighbors": args.top_neighbors,
        "graph_threshold": args.graph_threshold,
        "cluster_threshold": args.cluster_threshold,
        "raw_scenario_count": len(scenario_records),
        "canonical_scenario_count": len(clusters),
        "louvain_community_count": len(communities),
        "similarity_edge_count": graph.number_of_edges(),
        "method": "FAISS sparse similarity graph -> Louvain communities -> complete-linkage agglomerative clustering",
    }
    write_json(
        Path(args.output),
        {
            "meta": meta,
            "scenarios": clusters,
            "skill_scenarios": skill_scenarios,
        },
    )
    print(
        "done: raw_scenarios={}, canonical_scenarios={}, output={}".format(
            len(scenario_records), len(clusters), args.output
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
