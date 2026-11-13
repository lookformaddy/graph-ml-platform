"""Neighbourhood-aggregated node features.

A node's representation is the concatenation of:
- its own feature vector,
- the mean feature vector of its 1-hop neighbours,
- the mean feature vector of its 2-hop neighbours (excluding self and
  1-hop nodes).

Aggregation uses only features and adjacency — never labels — so the
features are safe to compute for train, validation and test nodes alike
in the transductive setting. Isolated nodes get zero vectors for the
missing hops.
"""

from __future__ import annotations

from graphml.graph import Graph


def _mean_rows(rows: list[list[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0] * dim
    return [sum(r[d] for r in rows) / len(rows) for d in range(dim)]


def aggregate_features(graph: Graph, node: int, hops: int = 2) -> list[float]:
    """Aggregated feature vector for one node (own + 1-hop + 2-hop means)."""
    dim = graph.feature_dim
    own = list(graph.features[node])
    if hops < 1:
        return own
    hop1 = graph.neighbors(node)
    mean1 = _mean_rows([graph.features[n] for n in hop1], dim)
    if hops < 2:
        return own + mean1
    seen = {node, *hop1}
    hop2: list[int] = []
    for n in hop1:
        for m in graph.neighbors(n):
            if m not in seen:
                seen.add(m)
                hop2.append(m)
    mean2 = _mean_rows([graph.features[n] for n in hop2], dim)
    return own + mean1 + mean2


def build_feature_matrix(graph: Graph, nodes: list[int],
                         hops: int = 2) -> list[list[float]]:
    """Feature matrix (one row per node) for the given node ids."""
    return [aggregate_features(graph, node, hops=hops) for node in nodes]


def feature_dim(graph: Graph, hops: int = 2) -> int:
    return graph.feature_dim * (hops + 1)
