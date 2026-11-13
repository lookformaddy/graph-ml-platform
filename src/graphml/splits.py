"""Transductive-safe splits for node classification and link prediction.

Node classification (transductive setting): the whole graph — including
test nodes and their edges — is visible at training time, but **test
labels are never visible**. The split therefore only partitions *which
labels* are revealed; features are label-free (see
:func:`graphml.graph.leakage_screen`), and label propagation clamps
only training labels. The split functions here enforce the partition
and the screen proves it.

Link prediction: held-out edges are *removed* from the training graph,
so no model input (heuristic or learned) can see a test edge. Negative
test pairs are sampled from non-edges of the full graph.
"""

from __future__ import annotations

import random

from graphml.graph import Graph


def stratified_node_split(graph: Graph, seed: int = 42,
                          train_frac: float = 0.6,
                          val_frac: float = 0.2) -> dict[str, list[int]]:
    """Stratified train/val/test split of node ids (by label)."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for node, label in enumerate(graph.labels):
        by_label.setdefault(label, []).append(node)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(by_label):
        nodes = list(by_label[label])
        rng.shuffle(nodes)
        n = len(nodes)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train.extend(nodes[:n_train])
        val.extend(nodes[n_train:n_train + n_val])
        test.extend(nodes[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def edge_split(graph: Graph, test_frac: float = 0.15,
               seed: int = 0) -> tuple[Graph, list[tuple[int, int]],
                                       list[tuple[int, int]]]:
    """Hold out a fraction of edges for link-prediction evaluation.

    Returns ``(train_graph, test_edges, test_negatives)``. The training
    graph has the test edges removed; an edge is never removed if doing
    so would isolate one of its endpoints. Negatives are non-edges of
    the full graph, sampled deterministically.
    """
    rng = random.Random(seed)
    edges = graph.edge_list()
    rng.shuffle(edges)
    n_test = max(1, int(round(len(edges) * test_frac)))
    train_graph = graph.copy()
    test_edges: list[tuple[int, int]] = []
    for u, v in edges:
        if len(test_edges) >= n_test:
            break
        if train_graph.degree(u) <= 1 or train_graph.degree(v) <= 1:
            continue
        train_graph.remove_edge(u, v)
        test_edges.append((u, v))

    edge_set = {tuple(sorted(e)) for e in graph.edge_list()}
    test_negatives: list[tuple[int, int]] = []
    attempts = 0
    while len(test_negatives) < len(test_edges) and attempts < 20000:
        attempts += 1
        u = rng.randrange(graph.n_nodes)
        v = rng.randrange(graph.n_nodes)
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in edge_set or pair in test_negatives:
            continue
        test_negatives.append(pair)
    return train_graph, test_edges, test_negatives
