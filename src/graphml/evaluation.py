"""From-scratch evaluation metrics (pure stdlib).

Node classification: accuracy, per-class recall, macro F1, ROC-AUC
(one-vs-rest, trapezoidal rule). Link prediction: ROC-AUC, Hits@K and
MRR, where each held-out edge is ranked against sampled negatives.
"""

from __future__ import annotations

import random

from graphml.embeddings import SkipGramEmbeddings
from graphml.graph import Graph


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true: list[int], y_pred: list[int],
                     n_classes: int) -> list[list[int]]:
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for a, b in zip(y_true, y_pred):
        matrix[a][b] += 1
    return matrix


def per_class_recall(y_true: list[int], y_pred: list[int],
                     n_classes: int) -> list[float]:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    recalls = []
    for c in range(n_classes):
        total = sum(matrix[c])
        recalls.append(matrix[c][c] / total if total else 0.0)
    return recalls


def macro_f1(y_true: list[int], y_pred: list[int],
             n_classes: int) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n_classes)) - tp
        fn = sum(matrix[c]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _roc_auc_binary(y_true: list[int], scores: list[float]) -> float:
    """Trapezoidal ROC-AUC for one binary problem."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        for k in range(i, j):
            if y_true[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
        i = j
    return auc


def roc_auc_multiclass(y_true: list[int], proba: list[list[float]],
                       n_classes: int) -> float:
    """Macro one-vs-rest ROC-AUC."""
    aucs = []
    for c in range(n_classes):
        binary = [1 if y == c else 0 for y in y_true]
        aucs.append(_roc_auc_binary(binary, [row[c] for row in proba]))
    return sum(aucs) / len(aucs) if aucs else 0.5


def hits_at_k(ranks: list[int], k: int) -> float:
    """Fraction of true items ranked in the top k (ranks are 0-based)."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r < k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / (r + 1) for r in ranks) / len(ranks)


def evaluate_link_prediction(
        graph: Graph, test_edges: list[tuple[int, int]],
        scorer, embeddings: SkipGramEmbeddings | None = None,
        n_neg_per_edge: int = 50, k: int = 10,
        seed: int = 0) -> dict:
    """Rank each held-out edge against sampled negatives.

    ``scorer`` is any callable ``(graph, u, v, embeddings) -> float``.
    Returns Hits@K, MRR, and ROC-AUC over the pooled true/negative scores.
    """
    rng = random.Random(seed)
    edge_set = {tuple(sorted(e)) for e in graph.edge_list()}
    ranks: list[int] = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    for u, v in test_edges:
        true_score = scorer(graph, u, v, embeddings)
        negatives: list[tuple[int, int]] = []
        attempts = 0
        while len(negatives) < n_neg_per_edge and attempts < 5000:
            attempts += 1
            w = rng.randrange(graph.n_nodes)
            pair = (min(u, w), max(u, w))
            if w == u or pair in edge_set or pair in negatives:
                continue
            negatives.append(pair)
        neg_scores = [scorer(graph, a, b, embeddings) for a, b in negatives]
        rank = sum(1 for s in neg_scores if s > true_score)
        ranks.append(rank)
        all_scores.append(true_score)
        all_labels.append(1)
        all_scores.extend(neg_scores)
        all_labels.extend([0] * len(neg_scores))
    return {
        "hits_at_k": round(hits_at_k(ranks, k), 4),
        "mrr": round(mrr(ranks), 4),
        "roc_auc": round(_roc_auc_binary(all_labels, all_scores), 4),
        "n_test_edges": len(test_edges),
        "random_hits_at_k": round(k / (n_neg_per_edge + 1), 4),
    }
