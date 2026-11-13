"""Tests for synthetic graph generation and quality/leakage checks."""
from graphml.graph import (check_quality, generate_graph, leakage_screen)
from graphml.splits import stratified_node_split


def test_generate_is_deterministic():
    a = generate_graph(n_nodes=60, seed=7)
    b = generate_graph(n_nodes=60, seed=7)
    assert a.edge_list() == b.edge_list()
    assert a.features == b.features
    assert a.labels == b.labels


def test_different_seeds_differ():
    a = generate_graph(n_nodes=60, seed=7)
    b = generate_graph(n_nodes=60, seed=8)
    assert a.edge_list() != b.edge_list()


def test_edges_are_symmetric():
    g = generate_graph(n_nodes=60, seed=7)
    for u, v in g.edge_list():
        assert g.has_edge(v, u)
        assert g.has_edge(u, v)


def test_community_structure_is_dense_within():
    g = generate_graph(n_nodes=120, seed=7)
    within = across = 0
    within_pairs = across_pairs = 0
    for u in range(g.n_nodes):
        for v in range(u + 1, g.n_nodes):
            same = g.communities[u] == g.communities[v]
            if same:
                within_pairs += 1
                within += 1 if g.has_edge(u, v) else 0
            else:
                across_pairs += 1
                across += 1 if g.has_edge(u, v) else 0
    assert within / within_pairs > 5 * (across / across_pairs)


def test_labels_match_communities_mostly():
    g = generate_graph(n_nodes=120, seed=7, label_noise=0.05)
    agree = sum(1 for i in range(g.n_nodes)
                if g.labels[i] == g.communities[i])
    assert agree / g.n_nodes > 0.9


def test_feature_dim_and_shape():
    g = generate_graph(n_nodes=40, feature_dim=6, seed=7)
    assert len(g.features) == 40
    assert all(len(row) == 6 for row in g.features)
    assert len(g.labels) == 40


def test_quality_report_usable():
    g = generate_graph(n_nodes=120, seed=7)
    report = check_quality(g)
    assert report["usable"] is True
    assert report["n_isolated"] == 0
    assert report["n_edges"] > 0
    assert report["degree_skew"] > 1.0


def test_quality_flags_isolated_nodes():
    g = generate_graph(n_nodes=40, seed=7)
    g.adjacency[0] = set()
    for nbrs in g.adjacency.values():
        nbrs.discard(0)
    report = check_quality(g)
    assert report["n_isolated"] >= 1
    assert report["usable"] is False


def test_leakage_screen_clean_on_real_split():
    g = generate_graph(n_nodes=120, seed=7)
    split = stratified_node_split(g, seed=7)
    screen = leakage_screen(g, split["train"], split["test"])
    assert screen["clean"] is True
    assert screen["train_test_disjoint"] is True
    assert screen["feature_encodes_label"] is False


def test_leakage_screen_catches_overlap():
    g = generate_graph(n_nodes=60, seed=7)
    screen = leakage_screen(g, [0, 1, 2], [2, 3, 4])
    assert screen["clean"] is False
    assert screen["overlap"] == [2]
