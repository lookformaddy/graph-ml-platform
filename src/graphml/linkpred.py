"""Link prediction: heuristic scorers + a learned scorer.

Heuristics (all computed on the *training* graph, which has test edges
removed):
- common neighbours, Adamic–Adar, Jaccard, preferential attachment.

The learned scorer is a from-scratch logistic regression over those
heuristics plus embedding similarity (dot product and cosine), trained
on observed edges vs. sampled non-edges. It must beat random — and
approach the best single heuristic — for the registry gates to pass.

Leakage note: embedding similarity is only a valid scorer input when
the scored edge was *absent* from the graph the embeddings were
trained on. The pipeline therefore trains embeddings on a nested
split that excludes the scorer's training positives (see
:func:`graphml.pipeline.train_version`), so training positives and
held-out test edges are scored under identical conditions. Training a
scorer on embeddings that saw the training edges inflates their
similarity and the scorer overfits to it — that failure mode is
documented in ``docs/architecture.md``.
"""

from __future__ import annotations

import math

from graphml.embeddings import SkipGramEmbeddings
from graphml.graph import Graph
from graphml.models import LogisticRegression, Standardizer


def common_neighbors(graph: Graph, u: int, v: int) -> int:
    return len(set(graph.neighbors(u)) & set(graph.neighbors(v)))


def adamic_adar(graph: Graph, u: int, v: int) -> float:
    total = 0.0
    for w in set(graph.neighbors(u)) & set(graph.neighbors(v)):
        deg = graph.degree(w)
        if deg > 1:
            total += 1.0 / math.log(deg)
    return total


def jaccard(graph: Graph, u: int, v: int) -> float:
    nu = set(graph.neighbors(u))
    nv = set(graph.neighbors(v))
    union = nu | nv
    if not union:
        return 0.0
    return len(nu & nv) / len(union)


def preferential_attachment(graph: Graph, u: int, v: int) -> int:
    return graph.degree(u) * graph.degree(v)


def heuristic_features(graph: Graph, u: int, v: int,
                       embeddings: SkipGramEmbeddings | None = None
                       ) -> list[float]:
    """Feature vector for a candidate pair (u, v)."""
    cn = float(common_neighbors(graph, u, v))
    aa = adamic_adar(graph, u, v)
    jac = jaccard(graph, u, v)
    pa = math.log1p(float(preferential_attachment(graph, u, v)))
    feats = [cn, aa, jac, pa, float(graph.degree(u)), float(graph.degree(v))]
    if embeddings is not None and u in embeddings.vectors \
            and v in embeddings.vectors:
        a, b = embeddings.vectors[u], embeddings.vectors[v]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        cos = dot / (na * nb) if na and nb else 0.0
        feats.extend([dot, cos])
    else:
        feats.extend([0.0, 0.0])
    return feats


class LearnedLinkScorer:
    """Logistic regression over heuristic + embedding features."""

    def __init__(self, lr: float = 0.1, epochs: int = 200,
                 seed: int = 0) -> None:
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.model: LogisticRegression | None = None
        self.scaler = Standardizer()

    def fit(self, graph: Graph, pos_edges: list[tuple[int, int]],
            neg_edges: list[tuple[int, int]],
            embeddings: SkipGramEmbeddings | None = None
            ) -> "LearnedLinkScorer":
        X = [heuristic_features(graph, u, v, embeddings)
             for u, v in pos_edges + neg_edges]
        y = [1] * len(pos_edges) + [0] * len(neg_edges)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model = LogisticRegression(
            n_classes=2, lr=self.lr, epochs=self.epochs, seed=self.seed)
        self.model.fit(Xs, y)
        return self

    def score(self, graph: Graph, u: int, v: int,
              embeddings: SkipGramEmbeddings | None = None) -> float:
        if self.model is None:
            raise ValueError("scorer has not been fitted")
        feats = self.scaler.transform(
            [heuristic_features(graph, u, v, embeddings)])
        return self.model.predict_proba(feats)[0][1]

    def score_pairs(self, graph: Graph, pairs: list[tuple[int, int]],
                    embeddings: SkipGramEmbeddings | None = None
                    ) -> list[float]:
        return [self.score(graph, u, v, embeddings) for u, v in pairs]
