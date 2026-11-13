"""Tests for neighbourhood-aggregated features."""
from graphml.features import (aggregate_features, build_feature_matrix,
                              feature_dim)
from graphml.graph import generate_graph


def test_feature_dim_is_three_hops_worth(small_graph):
    assert feature_dim(small_graph, hops=2) == small_graph.feature_dim * 3
    assert len(aggregate_features(small_graph, 0)) == 24


def test_own_features_preserved_in_first_block(small_graph):
    row = aggregate_features(small_graph, 5)
    assert row[: small_graph.feature_dim] == small_graph.features[5]


def test_matrix_shape_matches_nodes(small_graph):
    nodes = [0, 1, 2, 10]
    matrix = build_feature_matrix(small_graph, nodes)
    assert len(matrix) == len(nodes)
    assert all(len(r) == 24 for r in matrix)


def test_isolated_node_gets_zero_hop_means():
    g = generate_graph(n_nodes=20, seed=5)
    g.adjacency[0] = set()
    for nbrs in g.adjacency.values():
        nbrs.discard(0)
    row = aggregate_features(g, 0)
    assert row[: g.feature_dim] == g.features[0]
    assert row[g.feature_dim:] == [0.0] * (2 * g.feature_dim)


def test_one_hop_mean_matches_manual_average(small_graph):
    node = 3
    nbrs = small_graph.neighbors(node)
    row = aggregate_features(small_graph, node, hops=1)
    dim = small_graph.feature_dim
    for d in range(dim):
        expected = sum(small_graph.features[n][d] for n in nbrs) / len(nbrs)
        assert abs(row[dim + d] - expected) < 1e-9
