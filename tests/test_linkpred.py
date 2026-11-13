"""Tests for heuristic and learned link prediction."""
from graphml.embeddings import SkipGramEmbeddings
from graphml.graph import generate_graph
from graphml.linkpred import (LearnedLinkScorer, adamic_adar,
                              common_neighbors, heuristic_features, jaccard,
                              preferential_attachment)
from graphml.splits import edge_split


def test_common_neighbors_counts_shared(small_graph):
    u, v = 0, 1
    shared = set(small_graph.neighbors(u)) & set(small_graph.neighbors(v))
    assert common_neighbors(small_graph, u, v) == len(shared)


def test_adamic_adar_nonnegative_and_bounded(small_graph):
    import math
    for u, v in small_graph.edge_list()[:20]:
        aa = adamic_adar(small_graph, u, v)
        cn = common_neighbors(small_graph, u, v)
        assert aa >= 0.0
        # Each common neighbour contributes at most 1/log(2).
        assert aa <= cn / math.log(2) + 1e-9


def test_jaccard_in_unit_interval(small_graph):
    for u, v in small_graph.edge_list()[:20]:
        assert 0.0 <= jaccard(small_graph, u, v) <= 1.0
    assert jaccard(small_graph, 0, 0) == 1.0


def test_preferential_attachment_is_degree_product(small_graph):
    u, v = 0, 1
    assert preferential_attachment(small_graph, u, v) == \
        small_graph.degree(u) * small_graph.degree(v)


def test_heuristic_features_length(small_graph):
    feats = heuristic_features(small_graph, 0, 1)
    assert len(feats) == 8
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1).fit(small_graph)
    feats_emb = heuristic_features(small_graph, 0, 1, emb)
    assert len(feats_emb) == 8
    assert feats_emb[6] != 0.0 or feats_emb[7] != 0.0


def test_learned_scorer_separates_edges_from_non_edges(small_graph):
    train_graph, test_edges, test_negs = edge_split(small_graph, seed=4)
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1).fit(train_graph)
    scorer = LearnedLinkScorer(epochs=100, seed=1)
    scorer.fit(train_graph, train_graph.edge_list()[:200],
               test_negs, emb)
    pos_scores = scorer.score_pairs(train_graph, test_edges[:20], emb)
    neg_scores = scorer.score_pairs(train_graph, test_negs[:20], emb)
    assert sum(pos_scores) / len(pos_scores) > \
        sum(neg_scores) / len(neg_scores)


def test_learned_scorer_scores_in_unit_interval(small_graph):
    train_graph, test_edges, test_negs = edge_split(small_graph, seed=4)
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1).fit(train_graph)
    scorer = LearnedLinkScorer(epochs=50, seed=1)
    scorer.fit(train_graph, train_graph.edge_list()[:100],
               test_negs, emb)
    for u, v in test_edges[:10]:
        assert 0.0 <= scorer.score(train_graph, u, v, emb) <= 1.0


def test_scorer_requires_fit(small_graph):
    scorer = LearnedLinkScorer()
    try:
        scorer.score(small_graph, 0, 1)
    except ValueError as exc:
        assert "not been fitted" in str(exc)
    else:
        raise AssertionError("expected ValueError")
