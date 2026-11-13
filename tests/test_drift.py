"""Tests for embedding-drift monitoring."""
from graphml.drift import DriftMonitor
from graphml.embeddings import SkipGramEmbeddings, random_walks
from graphml.graph import generate_graph
from graphml.splits import edge_split


def test_same_embeddings_score_zero_drift(drift_monitor, trained_version):
    report = drift_monitor.score(
        {int(k): list(v) for k, v in trained_version.embeddings.items()})
    assert report["flagged"] is False
    assert report["mean_cosine_distance"] == 0.0
    assert report["centroid_shift"] == 0.0


def test_shifted_graph_is_flagged(drift_monitor, shifted_embeddings):
    report = drift_monitor.score(shifted_embeddings)
    assert report["flagged"] is True
    assert report["mean_cosine_distance"] > 0.15


def test_no_overlap_is_flagged(drift_monitor):
    report = drift_monitor.score({999: [0.1] * 16})
    assert report["flagged"] is True
    assert report["n_common_nodes"] == 0


def test_monitor_counts_observations(drift_monitor, trained_version):
    before = drift_monitor.n_observations
    drift_monitor.score(
        {int(k): list(v) for k, v in trained_version.embeddings.items()})
    assert drift_monitor.n_observations == before + 1
    report = drift_monitor.report()
    assert report["n_observations"] == drift_monitor.n_observations


def test_small_perturbation_not_flagged(trained_version):
    monitor = DriftMonitor(
        {int(k): list(v) for k, v in trained_version.embeddings.items()})
    perturbed = {int(k): [x * 1.001 for x in v]
                 for k, v in trained_version.embeddings.items()}
    report = monitor.score(perturbed)
    assert report["flagged"] is False
