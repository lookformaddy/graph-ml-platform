"""Tests for the end-to-end training pipeline."""
from graphml.pipeline import ModelVersion, train_version


def test_train_version_returns_report_with_expected_keys(trained_version):
    m = trained_version.metrics
    for key in ("logreg_test_accuracy", "logreg_macro_f1",
                "logreg_roc_auc", "label_prop_test_accuracy",
                "link_hits_at_10", "link_mrr", "seed_accuracy_std"):
        assert key in m
    assert trained_version.baseline_metrics["majority_test_accuracy"] > 0


def test_trained_version_beats_baseline(trained_version):
    m = trained_version.metrics
    b = trained_version.baseline_metrics
    assert m["logreg_test_accuracy"] > b["majority_test_accuracy"] + 0.15
    assert m["logreg_test_accuracy"] > 0.6


def test_link_prediction_beats_random(trained_version):
    m = trained_version.metrics
    assert m["link_hits_at_10"] > m["link_random_hits_at_10"] + 0.1
    assert m["link_hits_at_10"] >= 0.30  # registry link-quality floor
    assert m["link_roc_auc"] > 0.6
    assert m["link_heuristic_hits_at_10"] > m["link_random_hits_at_10"]


def test_version_json_round_trip(trained_version):
    restored = ModelVersion.from_json(trained_version.to_json())
    assert restored.name == trained_version.name
    assert restored.metrics == trained_version.metrics
    assert restored.logreg_weights == trained_version.logreg_weights
    assert restored.embeddings == trained_version.embeddings


def test_quality_and_leakage_recorded(trained_version):
    assert trained_version.quality["usable"] is True
    assert trained_version.leakage["clean"] is True


def test_stability_seeds_recorded(trained_version):
    m = trained_version.metrics
    assert len(m["seed_accuracies"]) == 2
    assert m["seed_accuracy_std"] >= 0.0
