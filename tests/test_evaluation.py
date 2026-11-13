"""Tests for from-scratch evaluation metrics."""
from graphml.evaluation import (accuracy, evaluate_link_prediction,
                                hits_at_k, macro_f1, mrr,
                                per_class_recall, roc_auc_multiclass)
from graphml.graph import generate_graph
from graphml.linkpred import adamic_adar
from graphml.splits import edge_split


def test_accuracy_perfect_and_zero():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert accuracy([0, 1, 2], [1, 2, 0]) == 0.0
    assert accuracy([0, 1], [0, 0]) == 0.5


def test_per_class_recall():
    recalls = per_class_recall([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert recalls == [0.5, 1.0]


def test_macro_f1_perfect_and_worst():
    assert macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0
    assert macro_f1([0, 0, 1, 1], [1, 1, 0, 0], 2) == 0.0


def test_roc_auc_perfect_and_random():
    perfect = roc_auc_multiclass(
        [0, 0, 1, 1],
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], 2)
    assert perfect == 1.0
    tied = roc_auc_multiclass(
        [0, 0, 1, 1],
        [[0.5, 0.5]] * 4, 2)
    assert tied == 0.5


def test_hits_at_k_and_mrr():
    assert hits_at_k([0, 5, 9, 10], 10) == 0.75
    assert hits_at_k([0, 0], 1) == 1.0
    assert abs(mrr([0, 1, 3]) - (1 + 0.5 + 0.25) / 3) < 1e-9


def test_link_evaluation_beats_random(small_graph):
    train_graph, test_edges, _negs = edge_split(small_graph, seed=6)
    result = evaluate_link_prediction(
        train_graph, test_edges,
        lambda g, u, v, _e: adamic_adar(g, u, v), None, seed=6)
    assert result["hits_at_k"] > result["random_hits_at_k"]
    assert result["roc_auc"] > 0.5
    assert result["n_test_edges"] == len(test_edges)
