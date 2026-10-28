"""Deterministic, dependency-free Louvain community detection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple


def louvain_communities(
    edges: Iterable[Tuple[str, str, float]],
    resolution: float = 1.0,
    max_passes: int = 10,
) -> Dict[str, int]:
    graph = _build_graph(edges)
    if not graph:
        return {}
    labels = _phase(graph, resolution, max_passes)
    meta = _aggregate(graph, labels)
    if len(meta) > 1:
        meta_labels = _phase(meta, resolution, max_passes)
        labels = {
            node: str(meta_labels.get(community, community))
            for node, community in labels.items()
        }
    groups: Dict[str, List[str]] = {}
    for node, label in labels.items():
        groups.setdefault(str(label), []).append(node)
    ordered = sorted(groups.values(), key=lambda members: min(members))
    result: Dict[str, int] = {}
    for index, members in enumerate(ordered):
        for node in members:
            result[node] = index
    return result


def _build_graph(
    edges: Iterable[Tuple[str, str, float]]
) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = {}
    for left, right, weight in edges:
        if left == right:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        graph.setdefault(left, {})
        graph.setdefault(right, {})
        graph[left][right] = graph[left].get(right, 0.0) + value
        graph[right][left] = graph[right].get(left, 0.0) + value
    return graph


def _phase(
    graph: Mapping[str, Mapping[str, float]],
    resolution: float,
    max_passes: int,
) -> Dict[str, str]:
    nodes = sorted(graph)
    degree = {node: sum(graph[node].values()) for node in nodes}
    total_weight = sum(degree.values())
    if total_weight <= 0:
        return {node: node for node in nodes}
    community = {node: node for node in nodes}
    totals = {node: degree[node] for node in nodes}
    for _ in range(max(1, max_passes)):
        moved = False
        for node in nodes:
            current = community[node]
            totals[current] = totals.get(current, 0.0) - degree[node]
            weights: Dict[str, float] = {}
            for neighbour, weight in graph[node].items():
                label = community[neighbour]
                weights[label] = weights.get(label, 0.0) + weight
            best = current
            best_gain = weights.get(current, 0.0) - (
                resolution * degree[node] * totals.get(current, 0.0) / total_weight
            )
            for label in sorted(weights):
                gain = weights[label] - (
                    resolution
                    * degree[node]
                    * totals.get(label, 0.0)
                    / total_weight
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = label
            community[node] = best
            totals[best] = totals.get(best, 0.0) + degree[node]
            if best != current:
                moved = True
        if not moved:
            break
    return community


def _aggregate(
    graph: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    meta: Dict[str, Dict[str, float]] = {}
    for node in sorted(graph):
        left = labels[node]
        meta.setdefault(left, {})
        for neighbour, weight in graph[node].items():
            right = labels[neighbour]
            if left == right:
                continue
            meta[left][right] = meta[left].get(right, 0.0) + weight
            meta.setdefault(right, {})
    return meta
