"""Deterministic synthetic graph generation and quality checks.

The generator builds a community-structured graph (a stochastic
block-model style construction): nodes belong to one of ``K`` latent
communities, edges are dense inside a community and sparse across them,
and each node's feature vector is drawn around its community centre with
Gaussian noise. Labels equal the community id, with a small configurable
label-noise flip so the task is non-trivial. Everything is seeded, so
the same seed always yields the same graph.

All data is synthetic and generic — no real-world graph, no employer
data, no proprietary structure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Graph:
    """An undirected, attributed, labelled graph (pure-Python storage)."""

    n_nodes: int
    adjacency: dict[int, set[int]]
    features: list[list[float]]  # per-node feature vectors
    labels: list[int]            # per-node class label
    communities: list[int]       # latent community id (== label pre-noise)
    feature_dim: int = 8

    def degree(self, node: int) -> int:
        return len(self.adjacency.get(node, ()))

    def neighbors(self, node: int) -> list[int]:
        return sorted(self.adjacency.get(node, ()))

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adjacency.get(u, ())

    def edge_list(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for u in range(self.n_nodes):
            for v in self.adjacency.get(u, ()):
                if u < v:
                    edges.append((u, v))
        return edges

    def n_edges(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2

    def copy(self) -> "Graph":
        return Graph(
            n_nodes=self.n_nodes,
            adjacency={u: set(v) for u, v in self.adjacency.items()},
            features=[list(row) for row in self.features],
            labels=list(self.labels),
            communities=list(self.communities),
            feature_dim=self.feature_dim,
        )

    def remove_edge(self, u: int, v: int) -> None:
        self.adjacency.get(u, set()).discard(v)
        self.adjacency.get(v, set()).discard(u)


def generate_graph(
    n_nodes: int = 240,
    n_communities: int = 4,
    feature_dim: int = 8,
    p_in: float = 0.12,
    p_out: float = 0.008,
    label_noise: float = 0.05,
    feature_noise: float = 0.5,
    seed: int = 42,
) -> Graph:
    """Generate a seeded community-structured attributed graph.

    Communities are balanced (round-robin assignment). Community centres
    are drawn once per community; node features are centre + Gaussian
    noise. Labels are the community id, flipped to a random other class
    with probability ``label_noise``.
    """
    rng = random.Random(seed)
    communities = [i % n_communities for i in range(n_nodes)]
    rng.shuffle(communities)

    centres = [[rng.gauss(0.0, 1.5) for _ in range(feature_dim)]
               for _ in range(n_communities)]
    features: list[list[float]] = []
    labels: list[int] = []
    for node in range(n_nodes):
        comm = communities[node]
        features.append([centres[comm][d] + rng.gauss(0.0, feature_noise)
                         for d in range(feature_dim)])
        label = comm
        if rng.random() < label_noise:
            others = [c for c in range(n_communities) if c != comm]
            label = rng.choice(others)
        labels.append(label)

    adjacency: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            p = p_in if communities[u] == communities[v] else p_out
            if rng.random() < p:
                adjacency[u].add(v)
                adjacency[v].add(u)
    # Guarantee no isolated nodes: attach any straggler to a random
    # node in its own community (keeps the SBM character intact).
    for u in range(n_nodes):
        if not adjacency[u]:
            same = [v for v in range(n_nodes)
                    if v != u and communities[v] == communities[u]]
            v = rng.choice(same)
            adjacency[u].add(v)
            adjacency[v].add(u)
    return Graph(n_nodes=n_nodes, adjacency=adjacency, features=features,
                 labels=labels, communities=communities,
                 feature_dim=feature_dim)


def check_quality(graph: Graph) -> dict:
    """Structural quality checks: isolation, degree skew, label balance.

    Returns a report dict. A graph is *usable* when there are no isolated
    nodes and no community/label class is empty; skew is reported for
    the caller to judge.
    """
    degrees = [graph.degree(i) for i in range(graph.n_nodes)]
    isolated = [i for i, d in enumerate(degrees) if d == 0]
    mean_deg = sum(degrees) / max(len(degrees), 1)
    max_deg = max(degrees) if degrees else 0
    label_counts: dict[int, int] = {}
    for lab in graph.labels:
        label_counts[lab] = label_counts.get(lab, 0) + 1
    community_counts: dict[int, int] = {}
    for comm in graph.communities:
        community_counts[comm] = community_counts.get(comm, 0) + 1
    return {
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges(),
        "mean_degree": round(mean_deg, 3),
        "max_degree": max_deg,
        "degree_skew": round(max_deg / mean_deg, 2) if mean_deg else 0.0,
        "isolated_nodes": isolated,
        "n_isolated": len(isolated),
        "label_counts": label_counts,
        "community_counts": community_counts,
        "usable": len(isolated) == 0 and all(
            c > 0 for c in label_counts.values()),
    }


def leakage_screen(graph: Graph, train_nodes: list[int],
                   test_nodes: list[int]) -> dict:
    """Screen for train/test leakage before any model touches the data.

    Checks:
    - train and test node sets are disjoint;
    - no feature column is (near-)perfectly correlated with the label,
      i.e. no feature column encodes the label directly;
    - every label value present in test also appears in train (so the
      split is learnable at all).

    Features are label-free by construction (they are drawn around
    community centres, never from labels); this screen is the audit
    trail that proves it for a given split.
    """
    train_set, test_set = set(train_nodes), set(test_nodes)
    overlap = sorted(train_set & test_set)

    n = graph.n_nodes
    labels = graph.labels
    mean_lab = sum(labels) / n
    var_lab = sum((x - mean_lab) ** 2 for x in labels)
    max_abs_corr = 0.0
    for d in range(graph.feature_dim):
        col = [row[d] for row in graph.features]
        mean_col = sum(col) / n
        cov = sum((col[i] - mean_col) * (labels[i] - mean_lab)
                  for i in range(n))
        var_col = sum((x - mean_col) ** 2 for x in col)
        if var_col > 0 and var_lab > 0:
            corr = cov / math.sqrt(var_col * var_lab)
            max_abs_corr = max(max_abs_corr, abs(corr))

    train_labels = {graph.labels[i] for i in train_nodes}
    test_labels = {graph.labels[i] for i in test_nodes}
    return {
        "train_test_disjoint": len(overlap) == 0,
        "overlap": overlap,
        "max_feature_label_correlation": round(max_abs_corr, 3),
        "feature_encodes_label": max_abs_corr > 0.95,
        "test_labels_covered_by_train": test_labels <= train_labels,
        "clean": (len(overlap) == 0 and max_abs_corr <= 0.95
                  and test_labels <= train_labels),
    }
