#!/usr/bin/env python3
"""Build a directed skill graph from scenario alignments.

Nodes are skills. Directed edges are cross-skill compose_with candidates backed by
post-scenario -> pre-scenario alignments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build directed graph where skill nodes are connected by scenario-alignment edges."
    )
    parser.add_argument(
        "--scenario-dedup",
        default="scenario_dedup.json",
        help="Input JSON from deduplicate_scenarios.py.",
    )
    parser.add_argument(
        "--alignments",
        default="scenario_alignment_keep.json",
        help="Compatible alignments JSON from align_skill_scenarios.py.",
    )
    parser.add_argument(
        "--output",
        default="scenario_skill_graph.json",
        help="Skill graph output JSON.",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=3,
        help="Minimum alignment confidence used as an edge evidence.",
    )
    parser.add_argument(
        "--include-node-scenarios",
        action="store_true",
        help="Include each skill node's pre/post scenario ids and texts when scenario_dedup is available.",
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


def load_scenario_dedup(path: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    scenarios_raw = payload.get("scenarios")
    skill_scenarios_raw = payload.get("skill_scenarios")
    if not isinstance(scenarios_raw, list) or not isinstance(skill_scenarios_raw, list):
        raise ValueError(f"{path} must contain list fields 'scenarios' and 'skill_scenarios'")

    scenario_by_id: dict[int, str] = {}
    for scenario in scenarios_raw:
        if not isinstance(scenario, dict) or "scenario_id" not in scenario:
            continue
        scenario_id = int(scenario["scenario_id"])
        scenario_by_id[scenario_id] = str(scenario.get("canonical_scenario") or "")

    skill_scenarios = [row for row in skill_scenarios_raw if isinstance(row, dict)]
    return scenario_by_id, skill_scenarios


def load_alignments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("alignments"), list):
        return [row for row in payload["alignments"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array or object field 'alignments'")


def is_used_alignment(row: dict[str, Any], *, min_confidence: int) -> bool:
    if row.get("compatible") is not True:
        return False
    try:
        confidence = int(row.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    if confidence < min_confidence:
        return False
    alignment_type = str(row.get("alignment_type") or "")
    return alignment_type not in {"duplicate_or_alternative", "topical_only", "incompatible"}


def as_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def build_nodes(
    skill_scenarios: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    scenario_by_id: dict[int, str],
    *,
    include_node_scenarios: bool,
) -> list[dict[str, Any]]:
    nodes_by_id: dict[int, dict[str, Any]] = {}

    for skill in skill_scenarios:
        try:
            skill_id = int(skill.get("skill_id") or 0)
        except (TypeError, ValueError):
            continue
        if skill_id <= 0:
            continue
        pre_ids = as_int_list(skill.get("pre_scenario_ids"))
        post_ids = as_int_list(skill.get("post_scenario_ids"))
        node = {
            "id": skill_id,
            "skill_id": skill_id,
            "skill_name": str(skill.get("skill_name") or ""),
        }
        if include_node_scenarios:
            node.update(
                {
                    "pre_scenario_ids": pre_ids,
                    "post_scenario_ids": post_ids,
                    "pre_scenarios": [scenario_by_id.get(item, "") for item in pre_ids],
                    "post_scenarios": [scenario_by_id.get(item, "") for item in post_ids],
                }
            )
        nodes_by_id[skill_id] = node

    for row in alignments:
        for prefix in ("source", "target"):
            try:
                skill_id = int(row.get(f"{prefix}_skill_id") or 0)
            except (TypeError, ValueError):
                continue
            if skill_id <= 0 or skill_id in nodes_by_id:
                continue
            nodes_by_id[skill_id] = {
                "id": skill_id,
                "skill_id": skill_id,
                "skill_name": str(row.get(f"{prefix}_skill_name") or ""),
            }

    return [nodes_by_id[key] for key in sorted(nodes_by_id)]


def edge_connection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "alignment_id": row.get("alignment_id"),
        "source_post_scenario_id": row.get("source_post_scenario_id"),
        "source_post_scenario": row.get("source_post_scenario"),
        "target_pre_scenario_id": row.get("target_pre_scenario_id"),
        "target_pre_scenario": row.get("target_pre_scenario"),
        "retrieval_score": row.get("retrieval_score"),
        "alignment_type": row.get("alignment_type"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
    }


def average_number(values: list[Any]) -> float:
    numbers: list[float] = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 6)


def build_edges(alignments: list[dict[str, Any]], *, min_confidence: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_alignments = [row for row in alignments if is_used_alignment(row, min_confidence=min_confidence)]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for row in used_alignments:
        try:
            source = int(row.get("source_skill_id") or 0)
            target = int(row.get("target_skill_id") or 0)
        except (TypeError, ValueError):
            continue
        if source <= 0 or target <= 0 or source == target:
            continue
        grouped[(source, target)].append(row)

    edges: list[dict[str, Any]] = []
    for edge_id, ((source, target), rows) in enumerate(sorted(grouped.items()), start=1):
        rows.sort(key=lambda row: str(row.get("alignment_id") or ""))
        confidences = [row.get("confidence") for row in rows]
        retrieval_scores = [row.get("retrieval_score") for row in rows]
        alignment_types = sorted({str(row.get("alignment_type") or "") for row in rows if row.get("alignment_type")})
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "source_skill_id": source,
                "source_skill_name": rows[0].get("source_skill_name"),
                "target_skill_id": target,
                "target_skill_name": rows[0].get("target_skill_name"),
                "relation": "compose_with",
                "alignment_count": len(rows),
                "alignment_types": alignment_types,
                "avg_confidence": average_number(confidences),
                "max_confidence": max((int(value or 0) for value in confidences), default=0),
                "avg_retrieval_score": average_number(retrieval_scores),
                "max_retrieval_score": max(
                    (float(value or 0.0) for value in retrieval_scores),
                    default=0.0,
                ),
                "skill_connection": f"{rows[0].get('source_skill_name')} -> {rows[0].get('target_skill_name')}",
                "scenario_connections": [edge_connection(row) for row in rows],
            }
        )

    return edges, used_alignments


def add_edge_only_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes_by_id = {int(node["id"]): node for node in nodes}
    for edge in edges:
        for prefix in ("source", "target"):
            skill_id = int(edge[f"{prefix}_skill_id"])
            if skill_id in nodes_by_id:
                continue
            nodes_by_id[skill_id] = {
                "id": skill_id,
                "skill_id": skill_id,
                "skill_name": str(edge.get(f"{prefix}_skill_name") or ""),
            }
    return [nodes_by_id[key] for key in sorted(nodes_by_id)]


def graph_stats(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    out_degree = Counter(int(edge["source"]) for edge in edges)
    in_degree = Counter(int(edge["target"]) for edge in edges)
    connected_skill_ids = set(out_degree) | set(in_degree)
    source_only = 0
    sink_only = 0
    bridge = 0
    isolated = 0
    for node in nodes:
        node_id = int(node["id"])
        has_in = in_degree[node_id] > 0
        has_out = out_degree[node_id] > 0
        if has_in and has_out:
            bridge += 1
        elif has_out:
            source_only += 1
        elif has_in:
            sink_only += 1
        else:
            isolated += 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected_skill_count": len(connected_skill_ids),
        "source_only_skills": source_only,
        "sink_only_skills": sink_only,
        "bridge_skills": bridge,
        "isolated_skills": isolated,
        "max_out_degree": max(out_degree.values(), default=0),
        "max_in_degree": max(in_degree.values(), default=0),
    }


def main() -> int:
    args = parse_args()
    if args.min_confidence < 1 or args.min_confidence > 5:
        raise ValueError("--min-confidence must be between 1 and 5")

    scenario_by_id, skill_scenarios = load_scenario_dedup(Path(args.scenario_dedup))
    alignments = load_alignments(Path(args.alignments))
    nodes = build_nodes(
        skill_scenarios,
        alignments,
        scenario_by_id,
        include_node_scenarios=args.include_node_scenarios,
    )
    edges, used_alignments = build_edges(alignments, min_confidence=args.min_confidence)
    nodes = add_edge_only_nodes(nodes, edges)
    stats = graph_stats(nodes, edges)
    meta = {
        "scenario_dedup": args.scenario_dedup,
        "alignments": args.alignments,
        "min_confidence": args.min_confidence,
        "raw_scenario_count": len(scenario_by_id),
        "raw_skill_scenario_count": len(skill_scenarios),
        "alignment_count": len(alignments),
        "used_alignment_count": len(used_alignments),
        "node_type": "skill",
        "edge_type": "compose_with",
        "include_node_scenarios": args.include_node_scenarios,
        **stats,
    }
    payload = {
        "directed": True,
        "multigraph": False,
        "meta": meta,
        "nodes": nodes,
        "edges": edges,
        "used_alignments": used_alignments,
    }
    write_json(Path(args.output), payload)
    print(
        "done: nodes={}, edges={}, connected_skills={}, used_alignments={}, output={}".format(
            len(nodes),
            len(edges),
            stats["connected_skill_count"],
            len(used_alignments),
            args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
