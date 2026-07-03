#!/usr/bin/env python3
"""Sample structured task candidates from a directed skill graph."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRUCTURE_BONUS = {
    "chain_2": 0.00,
    "chain_3": 0.03,
    "chain_4": 0.06,
    "chain_5": 0.09,
    "fan_out_3": 0.05,
    "fan_out_4": 0.07,
    "fan_out_5": 0.09,
    "fan_in_3": 0.05,
    "fan_in_4": 0.07,
    "fan_in_5": 0.09,
    "diamond_4": 0.08,
    "diamond_5": 0.12,
}


DIFFICULTY_BY_CATEGORY = {
    "chain_2": "easy",
    "chain_3": "medium",
    "fan_out_3": "medium",
    "fan_in_3": "medium",
    "chain_4": "hard",
    "diamond_4": "hard",
    "fan_out_4": "hard",
    "fan_out_5": "hard",
    "fan_in_4": "hard",
    "fan_in_5": "hard",
    "chain_5": "expert",
    "diamond_5": "expert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract chain, fan-in, fan-out, and diamond task candidates from scenario_skill_graph.json."
    )
    parser.add_argument("--input", default="scenario_skill_graph.json", help="Skill graph JSON.")
    parser.add_argument("--output", default="skill_graph_task_candidates.json", help="Task candidate output JSON.")
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=1000,
        help="Keep at most this many tasks per structure category after sorting.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["quality", "coverage"],
        default="quality",
        help="Task selection strategy. quality keeps the highest scores; coverage greedily prioritizes new skills/edges.",
    )
    parser.add_argument(
        "--coverage-target",
        choices=["edge", "skill", "both"],
        default="edge",
        help="Coverage priority when --selection-strategy coverage is enabled.",
    )
    parser.add_argument(
        "--max-raw-per-category",
        type=int,
        default=50000,
        help="Stop generating a category after this many raw candidates.",
    )
    parser.add_argument(
        "--top-out-per-node",
        type=int,
        default=12,
        help="Use at most this many highest-scoring outgoing edges per node.",
    )
    parser.add_argument(
        "--top-in-per-node",
        type=int,
        default=12,
        help="Use at most this many highest-scoring incoming edges per node.",
    )
    parser.add_argument(
        "--min-edge-score",
        type=float,
        default=0.0,
        help="Drop graph edges whose computed edge score is below this value.",
    )
    parser.add_argument(
        "--structures",
        default="chain,fan_out,fan_in,diamond",
        help="Comma-separated structures to extract.",
    )
    parser.add_argument(
        "--exclude-skill-names",
        default="SKILL.md,README.md",
        help="Comma-separated skill names to exclude from task candidates.",
    )
    parser.add_argument(
        "--max-scenario-connections-per-edge",
        type=int,
        default=3,
        help="Keep at most this many scenario evidence records per edge in task output.",
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def edge_quality(edge: dict[str, Any], max_alignment_count: int) -> float:
    avg_confidence = safe_float(edge.get("avg_confidence")) / 5.0
    avg_retrieval = safe_float(edge.get("avg_retrieval_score"))
    alignment_count = max(0, int(edge.get("alignment_count") or 0))
    if max_alignment_count > 0:
        evidence = math.log1p(alignment_count) / math.log1p(max_alignment_count)
    else:
        evidence = 0.0
    return round(0.45 * avg_confidence + 0.35 * avg_retrieval + 0.20 * evidence, 6)


def compact_edge(edge: dict[str, Any], max_scenario_connections: int) -> dict[str, Any]:
    scenario_connections = edge.get("scenario_connections", [])
    scenario_connections = scenario_connections[:max_scenario_connections]
    return {
        "edge_id": edge.get("id"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "source_skill_id": edge.get("source_skill_id"),
        "source_skill_name": edge.get("source_skill_name"),
        "target_skill_id": edge.get("target_skill_id"),
        "target_skill_name": edge.get("target_skill_name"),
        "skill_connection": edge.get("skill_connection"),
        "alignment_count": edge.get("alignment_count"),
        "alignment_types": edge.get("alignment_types"),
        "avg_confidence": edge.get("avg_confidence"),
        "avg_retrieval_score": edge.get("avg_retrieval_score"),
        "edge_score": edge.get("edge_score"),
        "scenario_connections": scenario_connections,
    }


def build_graph(
    payload: dict[str, Any],
    min_edge_score: float,
    top_out: int,
    top_in: int,
    exclude_skill_names: set[str],
):
    nodes = {
        int(node["id"]): {
            "skill_id": int(node["skill_id"]),
            "skill_name": str(node.get("skill_name") or ""),
        }
        for node in payload.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
        and str(node.get("skill_name") or "") not in exclude_skill_names
    }
    raw_edges = [edge for edge in payload.get("edges", []) if isinstance(edge, dict)]
    max_alignment_count = max((int(edge.get("alignment_count") or 0) for edge in raw_edges), default=0)

    edges: list[dict[str, Any]] = []
    for edge in raw_edges:
        try:
            source = int(edge.get("source"))
            target = int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        if source not in nodes or target not in nodes:
            continue
        score = edge_quality(edge, max_alignment_count)
        if score < min_edge_score:
            continue
        row = dict(edge)
        row["source"] = source
        row["target"] = target
        row["edge_score"] = score
        edges.append(row)

    out_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    in_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    edge_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for edge in edges:
        out_edges[int(edge["source"])].append(edge)
        in_edges[int(edge["target"])].append(edge)
        edge_by_pair[(int(edge["source"]), int(edge["target"]))] = edge

    def edge_sort_key(edge: dict[str, Any]) -> tuple[float, float, int]:
        return (
            -safe_float(edge.get("edge_score")),
            -safe_float(edge.get("avg_confidence")),
            int(edge.get("target") or 0),
        )

    for node_id in list(out_edges):
        out_edges[node_id].sort(key=edge_sort_key)
        out_edges[node_id] = out_edges[node_id][:top_out]
    for node_id in list(in_edges):
        in_edges[node_id].sort(key=edge_sort_key)
        in_edges[node_id] = in_edges[node_id][:top_in]

    full_out_degree = Counter(int(edge["source"]) for edge in edges)
    full_in_degree = Counter(int(edge["target"]) for edge in edges)
    max_degree = max(
        (full_out_degree[node_id] + full_in_degree[node_id] for node_id in nodes),
        default=0,
    )
    return nodes, edges, out_edges, in_edges, edge_by_pair, full_out_degree, full_in_degree, max_degree


def node_list(nodes: dict[int, dict[str, Any]], skill_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "skill_id": skill_id,
            "skill_name": nodes.get(skill_id, {}).get("skill_name", ""),
        }
        for skill_id in skill_ids
    ]


def score_task(
    *,
    category: str,
    skill_ids: list[int],
    edges: list[dict[str, Any]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
) -> dict[str, float]:
    mean_edge_score = sum(safe_float(edge.get("edge_score")) for edge in edges) / max(1, len(edges))
    if max_degree > 0:
        hub_penalty = 0.05 * sum(
            math.log1p(full_out_degree[skill_id] + full_in_degree[skill_id]) / math.log1p(max_degree)
            for skill_id in skill_ids
        ) / len(skill_ids)
    else:
        hub_penalty = 0.0
    structure_bonus = STRUCTURE_BONUS.get(category, 0.0)
    task_quality_score = mean_edge_score + structure_bonus - hub_penalty
    return {
        "task_quality_score": round(task_quality_score, 6),
        "mean_edge_score": round(mean_edge_score, 6),
        "structure_bonus": round(structure_bonus, 6),
        "hub_penalty": round(hub_penalty, 6),
    }


def make_task(
    *,
    category: str,
    structure_type: str,
    skill_ids: list[int],
    edges: list[dict[str, Any]],
    nodes: dict[int, dict[str, Any]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
    max_scenario_connections: int,
) -> dict[str, Any]:
    scores = score_task(
        category=category,
        skill_ids=skill_ids,
        edges=edges,
        full_out_degree=full_out_degree,
        full_in_degree=full_in_degree,
        max_degree=max_degree,
    )
    return {
        "category": category,
        "structure_type": structure_type,
        "difficulty": DIFFICULTY_BY_CATEGORY.get(category, "unknown"),
        **scores,
        "skills": node_list(nodes, skill_ids),
        "skill_ids": skill_ids,
        "skill_connections": [edge.get("skill_connection") for edge in edges],
        "edges": [compact_edge(edge, max_scenario_connections) for edge in edges],
    }


def append_candidate(
    buckets: dict[str, list[dict[str, Any]]],
    seen: dict[str, set[tuple[int, ...]]],
    task: dict[str, Any],
    max_raw_per_category: int,
) -> None:
    category = str(task["category"])
    if len(buckets[category]) >= max_raw_per_category:
        return
    key = tuple(int(item) for item in task["skill_ids"])
    if key in seen[category]:
        return
    seen[category].add(key)
    buckets[category].append(task)


def generate_chains(
    *,
    nodes: dict[int, dict[str, Any]],
    out_edges: dict[int, list[dict[str, Any]]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
    max_raw_per_category: int,
    max_scenario_connections: int,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)

    def dfs(start: int, skill_ids: list[int], path_edges: list[dict[str, Any]], max_nodes: int) -> None:
        if len(skill_ids) >= 2:
            category = f"chain_{len(skill_ids)}"
            append_candidate(
                buckets,
                seen,
                make_task(
                    category=category,
                    structure_type="chain",
                    skill_ids=list(skill_ids),
                    edges=list(path_edges),
                    nodes=nodes,
                    full_out_degree=full_out_degree,
                    full_in_degree=full_in_degree,
                    max_degree=max_degree,
                    max_scenario_connections=max_scenario_connections,
                ),
                max_raw_per_category,
            )
        if len(skill_ids) == max_nodes:
            return
        for edge in out_edges.get(start, []):
            target = int(edge["target"])
            if target in skill_ids:
                continue
            next_category = f"chain_{len(skill_ids) + 1}"
            if len(buckets[next_category]) >= max_raw_per_category:
                continue
            dfs(target, skill_ids + [target], path_edges + [edge], max_nodes)

    starts = sorted(
        nodes,
        key=lambda node_id: (
            -max((safe_float(edge.get("edge_score")) for edge in out_edges.get(node_id, [])), default=0.0),
            node_id,
        ),
    )
    for start in starts:
        dfs(start, [start], [], 5)
    return buckets


def generate_fan_out(
    *,
    nodes: dict[int, dict[str, Any]],
    out_edges: dict[int, list[dict[str, Any]]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
    max_raw_per_category: int,
    max_scenario_connections: int,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    sources = sorted(
        nodes,
        key=lambda node_id: (
            -sum(safe_float(edge.get("edge_score")) for edge in out_edges.get(node_id, [])[:4]),
            node_id,
        ),
    )
    for source in sources:
        edges = out_edges.get(source, [])
        for node_count in (3, 4, 5):
            category = f"fan_out_{node_count}"
            if len(edges) < node_count - 1:
                continue
            for combo in itertools.combinations(edges, node_count - 1):
                targets = [int(edge["target"]) for edge in combo]
                if len(set(targets)) != len(targets) or source in targets:
                    continue
                skill_ids = [source] + targets
                append_candidate(
                    buckets,
                    seen,
                    make_task(
                        category=category,
                        structure_type="fan_out",
                        skill_ids=skill_ids,
                        edges=list(combo),
                        nodes=nodes,
                        full_out_degree=full_out_degree,
                        full_in_degree=full_in_degree,
                        max_degree=max_degree,
                        max_scenario_connections=max_scenario_connections,
                    ),
                    max_raw_per_category,
                )
                if len(buckets[category]) >= max_raw_per_category:
                    break
    return buckets


def generate_fan_in(
    *,
    nodes: dict[int, dict[str, Any]],
    in_edges: dict[int, list[dict[str, Any]]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
    max_raw_per_category: int,
    max_scenario_connections: int,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    targets = sorted(
        nodes,
        key=lambda node_id: (
            -sum(safe_float(edge.get("edge_score")) for edge in in_edges.get(node_id, [])[:4]),
            node_id,
        ),
    )
    for target in targets:
        edges = in_edges.get(target, [])
        for node_count in (3, 4, 5):
            category = f"fan_in_{node_count}"
            if len(edges) < node_count - 1:
                continue
            for combo in itertools.combinations(edges, node_count - 1):
                sources = [int(edge["source"]) for edge in combo]
                if len(set(sources)) != len(sources) or target in sources:
                    continue
                skill_ids = sources + [target]
                append_candidate(
                    buckets,
                    seen,
                    make_task(
                        category=category,
                        structure_type="fan_in",
                        skill_ids=skill_ids,
                        edges=list(combo),
                        nodes=nodes,
                        full_out_degree=full_out_degree,
                        full_in_degree=full_in_degree,
                        max_degree=max_degree,
                        max_scenario_connections=max_scenario_connections,
                    ),
                    max_raw_per_category,
                )
                if len(buckets[category]) >= max_raw_per_category:
                    break
    return buckets


def generate_diamonds(
    *,
    nodes: dict[int, dict[str, Any]],
    out_edges: dict[int, list[dict[str, Any]]],
    edge_by_pair: dict[tuple[int, int], dict[str, Any]],
    full_out_degree: Counter,
    full_in_degree: Counter,
    max_degree: int,
    max_raw_per_category: int,
    max_scenario_connections: int,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    sources = sorted(
        nodes,
        key=lambda node_id: (
            -sum(safe_float(edge.get("edge_score")) for edge in out_edges.get(node_id, [])[:4]),
            node_id,
        ),
    )
    for source in sources:
        first_edges = out_edges.get(source, [])
        for edge_ab, edge_ac in itertools.combinations(first_edges, 2):
            b = int(edge_ab["target"])
            c = int(edge_ac["target"])
            if len({source, b, c}) < 3:
                continue
            b_targets = {int(edge["target"]): edge for edge in out_edges.get(b, [])}
            c_targets = {int(edge["target"]): edge for edge in out_edges.get(c, [])}
            for sink in sorted(set(b_targets) & set(c_targets)):
                if sink in {source, b, c}:
                    continue
                category = "diamond_4"
                skill_ids = [source, b, c, sink]
                diamond_edges = [edge_ab, edge_ac, b_targets[sink], c_targets[sink]]
                append_candidate(
                    buckets,
                    seen,
                    make_task(
                        category=category,
                        structure_type="diamond",
                        skill_ids=skill_ids,
                        edges=diamond_edges,
                        nodes=nodes,
                        full_out_degree=full_out_degree,
                        full_in_degree=full_in_degree,
                        max_degree=max_degree,
                        max_scenario_connections=max_scenario_connections,
                    ),
                    max_raw_per_category,
                )

                if len(buckets["diamond_5"]) >= max_raw_per_category:
                    continue
                for tail_edge in out_edges.get(sink, []):
                    tail = int(tail_edge["target"])
                    if tail in {source, b, c, sink}:
                        continue
                    append_candidate(
                        buckets,
                        seen,
                        make_task(
                            category="diamond_5",
                            structure_type="diamond",
                            skill_ids=[source, b, c, sink, tail],
                            edges=diamond_edges + [tail_edge],
                            nodes=nodes,
                            full_out_degree=full_out_degree,
                            full_in_degree=full_in_degree,
                            max_degree=max_degree,
                            max_scenario_connections=max_scenario_connections,
                        ),
                        max_raw_per_category,
                    )
                    break
                if len(buckets["diamond_4"]) >= max_raw_per_category:
                    break
    return buckets


def merge_buckets(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]]) -> None:
    for category, rows in source.items():
        target[category].extend(rows)


def quality_sort_key(row: dict[str, Any]) -> tuple[float, float, tuple[int, ...]]:
    return (
        -safe_float(row.get("task_quality_score")),
        -safe_float(row.get("mean_edge_score")),
        tuple(int(item) for item in row.get("skill_ids", [])),
    )


def task_skill_keys(task: dict[str, Any]) -> set[int]:
    keys: set[int] = set()
    for skill_id in task.get("skill_ids", []):
        try:
            keys.add(int(skill_id))
        except (TypeError, ValueError):
            continue
    return keys


def task_edge_keys(task: dict[str, Any]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for edge in task.get("edges", []):
        if not isinstance(edge, dict):
            continue
        try:
            keys.add((int(edge["source"]), int(edge["target"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def coverage_priority(
    *,
    task: dict[str, Any],
    covered_edges: set[tuple[int, int]],
    covered_skills: set[int],
    coverage_target: str,
    stable_rank: int,
) -> tuple[float, float, float, float, int]:
    edge_keys = task_edge_keys(task)
    skill_keys = task_skill_keys(task)
    new_edges = len(edge_keys - covered_edges)
    new_skills = len(skill_keys - covered_skills)
    if coverage_target == "skill":
        primary = new_skills
        secondary = new_edges
    elif coverage_target == "both":
        primary = new_edges + new_skills
        secondary = new_edges
    else:
        primary = new_edges
        secondary = new_skills
    return (
        float(primary),
        float(secondary),
        safe_float(task.get("task_quality_score")),
        safe_float(task.get("mean_edge_score")),
        -stable_rank,
    )


def select_by_coverage(
    rows: list[dict[str, Any]],
    *,
    max_per_category: int,
    coverage_target: str,
    covered_edges: set[tuple[int, int]],
    covered_skills: set[int],
) -> list[dict[str, Any]]:
    remaining = list(rows)
    kept: list[dict[str, Any]] = []
    while remaining and len(kept) < max_per_category:
        best_index = max(
            range(len(remaining)),
            key=lambda index: coverage_priority(
                task=remaining[index],
                covered_edges=covered_edges,
                covered_skills=covered_skills,
                coverage_target=coverage_target,
                stable_rank=index,
            ),
        )
        task = remaining.pop(best_index)
        kept.append(task)
        covered_edges.update(task_edge_keys(task))
        covered_skills.update(task_skill_keys(task))
    return kept


def finalize_tasks(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    max_per_category: int,
    selection_strategy: str,
    coverage_target: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    category_stats: dict[str, Any] = {}
    global_covered_edges: set[tuple[int, int]] = set()
    global_covered_skills: set[int] = set()
    ordered_categories = [
        "chain_2",
        "chain_3",
        "chain_4",
        "chain_5",
        "fan_out_3",
        "fan_out_4",
        "fan_out_5",
        "fan_in_3",
        "fan_in_4",
        "fan_in_5",
        "diamond_4",
        "diamond_5",
    ]
    for category in ordered_categories:
        rows = buckets.get(category, [])
        rows.sort(key=quality_sort_key)
        before_edges = set(global_covered_edges)
        before_skills = set(global_covered_skills)
        if selection_strategy == "coverage":
            kept = select_by_coverage(
                rows,
                max_per_category=max_per_category,
                coverage_target=coverage_target,
                covered_edges=global_covered_edges,
                covered_skills=global_covered_skills,
            )
        else:
            kept = rows[:max_per_category]
            for task in kept:
                global_covered_edges.update(task_edge_keys(task))
                global_covered_skills.update(task_skill_keys(task))
        available_edges: set[tuple[int, int]] = set()
        available_skills: set[int] = set()
        kept_edges: set[tuple[int, int]] = set()
        kept_skills: set[int] = set()
        for task in rows:
            available_edges.update(task_edge_keys(task))
            available_skills.update(task_skill_keys(task))
        for task in kept:
            kept_edges.update(task_edge_keys(task))
            kept_skills.update(task_skill_keys(task))
        category_stats[category] = {
            "raw_count": len(rows),
            "kept_count": len(kept),
            "best_score": kept[0]["task_quality_score"] if kept else None,
            "available_skill_count": len(available_skills),
            "available_edge_count": len(available_edges),
            "covered_skill_count": len(kept_skills),
            "covered_edge_count": len(kept_edges),
            "new_global_skill_count": len(global_covered_skills - before_skills),
            "new_global_edge_count": len(global_covered_edges - before_edges),
        }
        tasks.extend(kept)

    for index, task in enumerate(tasks, start=1):
        task["task_id"] = f"task-{index:07d}"
    return tasks, category_stats


def main() -> int:
    args = parse_args()
    if args.max_per_category < 1:
        raise ValueError("--max-per-category must be at least 1")
    if args.max_raw_per_category < args.max_per_category:
        raise ValueError("--max-raw-per-category must be >= --max-per-category")
    if args.top_out_per_node < 1 or args.top_in_per_node < 1:
        raise ValueError("--top-out-per-node and --top-in-per-node must be at least 1")
    if args.max_scenario_connections_per_edge < 0:
        raise ValueError("--max-scenario-connections-per-edge must be non-negative")

    payload = load_json(Path(args.input))
    exclude_skill_names = {item.strip() for item in args.exclude_skill_names.split(",") if item.strip()}
    nodes, edges, out_edges, in_edges, edge_by_pair, full_out_degree, full_in_degree, max_degree = build_graph(
        payload,
        args.min_edge_score,
        args.top_out_per_node,
        args.top_in_per_node,
        exclude_skill_names,
    )
    structures = {item.strip() for item in args.structures.split(",") if item.strip()}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if "chain" in structures:
        merge_buckets(
            buckets,
            generate_chains(
                nodes=nodes,
                out_edges=out_edges,
                full_out_degree=full_out_degree,
                full_in_degree=full_in_degree,
                max_degree=max_degree,
                max_raw_per_category=args.max_raw_per_category,
                max_scenario_connections=args.max_scenario_connections_per_edge,
            ),
        )
    if "fan_out" in structures:
        merge_buckets(
            buckets,
            generate_fan_out(
                nodes=nodes,
                out_edges=out_edges,
                full_out_degree=full_out_degree,
                full_in_degree=full_in_degree,
                max_degree=max_degree,
                max_raw_per_category=args.max_raw_per_category,
                max_scenario_connections=args.max_scenario_connections_per_edge,
            ),
        )
    if "fan_in" in structures:
        merge_buckets(
            buckets,
            generate_fan_in(
                nodes=nodes,
                in_edges=in_edges,
                full_out_degree=full_out_degree,
                full_in_degree=full_in_degree,
                max_degree=max_degree,
                max_raw_per_category=args.max_raw_per_category,
                max_scenario_connections=args.max_scenario_connections_per_edge,
            ),
        )
    if "diamond" in structures:
        merge_buckets(
            buckets,
            generate_diamonds(
                nodes=nodes,
                out_edges=out_edges,
                edge_by_pair=edge_by_pair,
                full_out_degree=full_out_degree,
                full_in_degree=full_in_degree,
                max_degree=max_degree,
                max_raw_per_category=args.max_raw_per_category,
                max_scenario_connections=args.max_scenario_connections_per_edge,
            ),
        )

    tasks, category_stats = finalize_tasks(
        buckets,
        max_per_category=args.max_per_category,
        selection_strategy=args.selection_strategy,
        coverage_target=args.coverage_target,
    )
    meta = {
        "input": args.input,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "top_out_per_node": args.top_out_per_node,
        "top_in_per_node": args.top_in_per_node,
        "min_edge_score": args.min_edge_score,
        "max_per_category": args.max_per_category,
        "max_raw_per_category": args.max_raw_per_category,
        "selection_strategy": args.selection_strategy,
        "coverage_target": args.coverage_target,
        "exclude_skill_names": sorted(exclude_skill_names),
        "max_scenario_connections_per_edge": args.max_scenario_connections_per_edge,
        "task_count": len(tasks),
        "category_stats": category_stats,
        "score_formula": {
            "edge_score": "0.45*(avg_confidence/5)+0.35*avg_retrieval_score+0.20*log1p(alignment_count)/log1p(max_alignment_count)",
            "task_quality_score": "mean_edge_score + structure_bonus - hub_penalty",
        },
    }
    write_json(Path(args.output), {"meta": meta, "tasks": tasks})
    print(
        "done: nodes={}, edges={}, tasks={}, output={}".format(
            len(nodes),
            len(edges),
            len(tasks),
            args.output,
        )
    )
    for category, stat in category_stats.items():
        if stat["kept_count"]:
            print(
                "{}: raw={}, kept={}, best={}, covered_skills={}, covered_edges={}, new_global_skills={}, new_global_edges={}".format(
                    category,
                    stat["raw_count"],
                    stat["kept_count"],
                    stat["best_score"],
                    stat["covered_skill_count"],
                    stat["covered_edge_count"],
                    stat["new_global_skill_count"],
                    stat["new_global_edge_count"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
