"""Tests for random walks and skip-gram embeddings."""
from graphml.embeddings import SkipGramEmbeddings, random_walks
from graphml.graph import generate_graph


def test_walks_follow_edges(small_graph):
    walks = random_walks(small_graph, walk_length=8, walks_per_node=2,
                         seed=3)
    assert len(walks) == small_graph.n_nodes * 2
    for walk in walks:
        for a, b in zip(walk, walk[1:]):
            assert small_graph.has_edge(a, b)


def test_walks_are_deterministic():
    g = generate_graph(n_nodes=40, seed=3)
    a = random_walks(g, seed=3)
    b = random_walks(g, seed=3)
    assert a == b


def test_embeddings_cover_all_nodes(small_graph):
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1)
    emb.fit(small_graph)
    assert set(emb.vectors) == set(range(small_graph.n_nodes))
    assert all(len(v) == 8 for v in emb.vectors.values())


def test_embeddings_are_deterministic():
    g = generate_graph(n_nodes=40, seed=3)
    a = SkipGramEmbeddings(dim=8, epochs=1, seed=1).fit(g)
    b = SkipGramEmbeddings(dim=8, epochs=1, seed=1).fit(g)
    assert a.all_embeddings() == b.all_embeddings()


def test_same_community_nodes_more_similar_than_cross(small_graph):
    emb = SkipGramEmbeddings(dim=16, epochs=2, seed=1)
    emb.fit(small_graph)
    same_sims = []
    cross_sims = []
    for u in range(0, 40, 4):
        for v in range(u + 1, 40, 4):
            if small_graph.communities[u] == small_graph.communities[v]:
                same_sims.append(emb.cosine(u, v))
            else:
                cross_sims.append(emb.cosine(u, v))
    assert sum(same_sims) / len(same_sims) > \
        sum(cross_sims) / len(cross_sims)


def test_cosine_self_is_one(small_graph):
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1)
    emb.fit(small_graph)
    assert abs(emb.cosine(0, 0) - 1.0) < 1e-9


def test_nearest_returns_k_ranked(small_graph):
    emb = SkipGramEmbeddings(dim=8, epochs=1, seed=1)
    emb.fit(small_graph)
    nearest = emb.nearest(0, k=5)
    assert len(nearest) == 5
    assert all(n != 0 for n, _s in nearest)
    scores = [s for _n, s in nearest]
    assert scores == sorted(scores, reverse=True)
