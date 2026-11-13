"""Tests for label propagation, logistic regression and the baseline."""
from graphml.evaluation import accuracy
from graphml.features import build_feature_matrix
from graphml.graph import generate_graph
from graphml.models import (LabelPropagation, LogisticRegression,
                            MajorityBaseline, Standardizer)
from graphml.splits import stratified_node_split


def _train_test_xy(seed=11):
    g = generate_graph(n_nodes=120, seed=seed)
    split = stratified_node_split(g, seed=seed)
    X_tr = build_feature_matrix(g, split["train"])
    X_te = build_feature_matrix(g, split["test"])
    y_tr = [g.labels[n] for n in split["train"]]
    y_te = [g.labels[n] for n in split["test"]]
    return g, split, X_tr, X_te, y_tr, y_te


def test_logreg_beats_majority_baseline():
    _g, _s, X_tr, X_te, y_tr, y_te = _train_test_xy()
    scaler = Standardizer().fit(X_tr)
    model = LogisticRegression(n_classes=4, epochs=150, seed=0)
    model.fit(scaler.transform(X_tr), y_tr)
    acc = accuracy(y_te, model.predict(scaler.transform(X_te)))
    base = MajorityBaseline().fit(y_tr)
    base_acc = accuracy(y_te, base.predict(len(y_te)))
    assert acc > base_acc + 0.2
    assert acc > 0.6


def test_logreg_loss_decreases():
    _g, _s, X_tr, _X_te, y_tr, _y_te = _train_test_xy()
    scaler = Standardizer().fit(X_tr)
    model = LogisticRegression(n_classes=4, epochs=100, seed=0)
    model.fit(scaler.transform(X_tr), y_tr)
    assert model.loss_history[-1] < model.loss_history[0]


def test_logreg_probas_sum_to_one():
    _g, _s, X_tr, X_te, y_tr, _y_te = _train_test_xy()
    scaler = Standardizer().fit(X_tr)
    model = LogisticRegression(n_classes=4, epochs=20, seed=0)
    model.fit(scaler.transform(X_tr), y_tr)
    for row in model.predict_proba(scaler.transform(X_te)):
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in row)


def test_standardizer_round_trip():
    import math
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    scaler = Standardizer().fit(X)
    data = scaler.to_dict()
    restored = Standardizer.from_dict(data)
    assert restored.transform(X) == scaler.transform(X)
    # (1 - 3) / sqrt(8/3) for the first element.
    assert abs(scaler.transform(X)[0][0] + 2.0 / math.sqrt(8.0 / 3.0)) < 1e-9


def test_label_propagation_beats_baseline():
    g, split, _X_tr, _X_te, y_tr, y_te = _train_test_xy()
    lp = LabelPropagation(n_classes=4)
    lp.fit(g, split["train"], y_tr)
    acc = accuracy(y_te, lp.predict(split["test"]))
    base = MajorityBaseline().fit(y_tr)
    assert acc > accuracy(y_te, base.predict(len(y_te)))
    assert acc > 0.5


def test_label_propagation_clamps_train_labels():
    g, split, _X_tr, _X_te, y_tr, _y_te = _train_test_xy()
    lp = LabelPropagation(n_classes=4)
    lp.fit(g, split["train"], y_tr)
    train_pred = lp.predict(split["train"])
    assert accuracy(y_tr, train_pred) > 0.95


def test_label_propagation_scores_are_distributions():
    g, split, _X_tr, _X_te, y_tr, _y_te = _train_test_xy()
    lp = LabelPropagation(n_classes=4)
    lp.fit(g, split["train"], y_tr)
    for row in lp.predict_proba(split["test"]):
        assert abs(sum(row) - 1.0) < 1e-6


def test_majority_baseline_predicts_mode():
    base = MajorityBaseline().fit([0, 1, 1, 2, 1])
    assert base.predict(3) == [1, 1, 1]
    assert base.majority == 1
