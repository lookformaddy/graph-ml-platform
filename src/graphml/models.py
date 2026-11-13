"""From-scratch node-classification models (pure stdlib).

- :class:`Standardizer` — mean/std scaling, fitted on train only.
- :class:`LogisticRegression` — multinomial softmax regression trained
  with full-batch gradient descent, L2 regularisation, stable softmax.
- :class:`LabelPropagation` — semi-supervised propagation over the
  graph; clamps only training labels, so test labels can never leak in.
- :class:`MajorityBaseline` — the trivial classifier every learned
  model must beat; also the serving fallback.
"""

from __future__ import annotations

import math
import random

from graphml.graph import Graph


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Standardizer:
    """Mean/std scaler (fit on train, applied everywhere)."""

    def __init__(self) -> None:
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit(self, X: list[list[float]]) -> "Standardizer":
        n = len(X)
        dim = len(X[0]) if X else 0
        self.mean = [sum(row[d] for row in X) / n for d in range(dim)]
        self.std = []
        for d in range(dim):
            var = sum((row[d] - self.mean[d]) ** 2 for row in X) / n
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(row[d] - self.mean[d]) / self.std[d]
                 for d in range(len(row))] for row in X]

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict) -> "Standardizer":
        obj = cls()
        obj.mean = list(data["mean"])
        obj.std = list(data["std"])
        return obj


class LogisticRegression:
    """Multinomial logistic regression, full-batch gradient descent."""

    def __init__(self, n_classes: int, lr: float = 0.1, epochs: int = 200,
                 l2: float = 1e-3, seed: int = 0) -> None:
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: list[list[float]] = []  # [class][feature]
        self.bias: list[float] = []
        self.loss_history: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        rng = random.Random(self.seed)
        n = len(X)
        dim = len(X[0])
        self.weights = [[rng.gauss(0.0, 0.01) for _ in range(dim)]
                        for _ in range(self.n_classes)]
        self.bias = [0.0] * self.n_classes
        self.loss_history = []
        for _epoch in range(self.epochs):
            grad_w = [[0.0] * dim for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            loss = 0.0
            for i in range(n):
                logits = [sum(self.weights[c][d] * X[i][d]
                              for d in range(dim)) + self.bias[c]
                          for c in range(self.n_classes)]
                probs = _softmax(logits)
                loss -= math.log(max(probs[y[i]], 1e-15))
                for c in range(self.n_classes):
                    err = probs[c] - (1.0 if c == y[i] else 0.0)
                    grad_b[c] += err
                    for d in range(dim):
                        grad_w[c][d] += err * X[i][d]
            for c in range(self.n_classes):
                for d in range(dim):
                    grad = grad_w[c][d] / n + self.l2 * self.weights[c][d]
                    self.weights[c][d] -= self.lr * grad
                self.bias[c] -= self.lr * grad_b[c] / n
            self.loss_history.append(loss / n)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        dim = len(self.weights[0]) if self.weights else 0
        for row in X:
            logits = [sum(self.weights[c][d] * row[d] for d in range(dim))
                      + self.bias[c] for c in range(self.n_classes)]
            out.append(_softmax(logits))
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [max(range(self.n_classes), key=lambda c: p[c])
                for p in self.predict_proba(X)]


class LabelPropagation:
    """Semi-supervised label propagation over the graph.

    Scores ``F`` are initialised one-hot on training nodes and uniform
    elsewhere, then iterated ``F <- alpha * S F + (1 - alpha) * Y`` where
    ``S`` is the row-normalised adjacency and ``Y`` clamps training
    labels. Test labels are never used — they are not even passed in.
    """

    def __init__(self, n_classes: int, alpha: float = 0.8,
                 iterations: int = 30) -> None:
        self.n_classes = n_classes
        self.alpha = alpha
        self.iterations = iterations
        self.scores: list[list[float]] = []

    def fit(self, graph: Graph, train_nodes: list[int],
            train_labels: list[int]) -> "LabelPropagation":
        n = graph.n_nodes
        clamped = [[0.0] * self.n_classes for _ in range(n)]
        for node, label in zip(train_nodes, train_labels):
            clamped[node][label] = 1.0
        uniform = 1.0 / self.n_classes
        F = [[uniform] * self.n_classes for _ in range(n)]
        for node, label in zip(train_nodes, train_labels):
            F[node] = [1.0 if c == label else 0.0
                       for c in range(self.n_classes)]
        for _ in range(self.iterations):
            new_F: list[list[float]] = []
            for node in range(n):
                nbrs = graph.neighbors(node)
                if nbrs:
                    agg = [sum(F[m][c] for m in nbrs) / len(nbrs)
                           for c in range(self.n_classes)]
                else:
                    agg = list(F[node])
                new_F.append([
                    self.alpha * agg[c] + (1.0 - self.alpha) * clamped[node][c]
                    for c in range(self.n_classes)])
            F = new_F
        self.scores = F
        return self

    def predict_proba(self, nodes: list[int]) -> list[list[float]]:
        out = []
        for node in nodes:
            row = self.scores[node]
            total = sum(row)
            out.append([x / total for x in row] if total > 0
                       else [1.0 / self.n_classes] * self.n_classes)
        return out

    def predict(self, nodes: list[int]) -> list[int]:
        return [max(range(self.n_classes), key=lambda c: self.scores[node][c])
                for node in nodes]


class MajorityBaseline:
    """Predicts the most frequent training label for every node."""

    def __init__(self) -> None:
        self.majority: int = 0
        self.distribution: dict[int, float] = {}

    def fit(self, y: list[int]) -> "MajorityBaseline":
        counts: dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self.majority = max(counts, key=lambda k: counts[k])
        total = len(y)
        self.distribution = {k: v / total for k, v in counts.items()}
        return self

    def predict(self, n: int) -> list[int]:
        return [self.majority] * n

    def predict_proba(self, n: int, n_classes: int) -> list[list[float]]:
        return [[self.distribution.get(c, 0.0) for c in range(n_classes)]
                for _ in range(n)]
