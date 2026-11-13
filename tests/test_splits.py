"""Tests for transductive-safe node splits and edge splits."""
from graphml.graph import generate_graph
from graphml.splits import edge_split, stratified_node_split


def test_node_split_covers_all_nodes_disjointly(small_graph):
    split = stratified_node_split(small_graph, seed=1)
    all_nodes = split["train"] + split["val"] + split["test"]
    assert sorted(all_nodes) == list(range(small_graph.n_nodes))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["val"]) & set(split["test"]))


def test_node_split_is_stratified(small_graph):
    split = stratified_node_split(small_graph, seed=1)
    for part in ("train", "val", "test"):
        labels = {small_graph.labels[n] for n in split[part]}
        assert labels == {0, 1, 2, 3}


def test_node_split_fractions_roughly_hold(small_graph):
    split = stratified_node_split(small_graph, seed=1)
    n = small_graph.n_nodes
    assert abs(len(split["train"]) / n - 0.6) < 0.1
    assert abs(len(split["test"]) / n - 0.2) < 0.1


def test_node_split_deterministic():
    g = generate_graph(n_nodes=60, seed=3)
    a = stratified_node_split(g, seed=9)
    b = stratified_node_split(g, seed=9)
    assert a == b


def test_edge_split_removes_test_edges(small_graph):
    train_graph, test_edges, _negs = edge_split(small_graph, seed=2)
    assert len(test_edges) > 0
    for u, v in test_edges:
        assert not train_graph.has_edge(u, v)
        assert small_graph.has_edge(u, v)


def test_edge_split_never_isolates_a_node(small_graph):
    train_graph, _edges, _negs = edge_split(small_graph, seed=2)
    assert all(train_graph.degree(i) >= 1
               for i in range(train_graph.n_nodes))


def test_edge_split_negatives_are_non_edges(small_graph):
    _train, test_edges, negatives = edge_split(small_graph, seed=2)
    assert len(negatives) == len(test_edges)
    for u, v in negatives:
        assert u != v
        assert not small_graph.has_edge(u, v)


def test_edge_split_fraction(small_graph):
    _train, test_edges, _negs = edge_split(small_graph, test_frac=0.15,
                                           seed=2)
    ratio = len(test_edges) / small_graph.n_edges()
    assert 0.10 < ratio < 0.20
