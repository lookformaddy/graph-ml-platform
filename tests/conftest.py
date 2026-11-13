"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphml.drift import DriftMonitor  # noqa: E402
from graphml.embeddings import SkipGramEmbeddings, random_walks  # noqa: E402
from graphml.graph import generate_graph  # noqa: E402
from graphml.pipeline import train_version  # noqa: E402
from graphml.registry import ModelRegistry  # noqa: E402
from graphml.serving import ServingConfig, ServingLayer  # noqa: E402
from graphml.splits import edge_split  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def small_graph():
    return generate_graph(n_nodes=80, seed=SEED)


@pytest.fixture(scope="session")
def trained_version():
    # Denser small graph: link prediction needs enough edges for the
    # nested scorer-positive split to have a real training set.
    version, _report = train_version(
        "v1-test", seed=SEED, n_nodes=120, p_in=0.2,
        logreg_epochs=60, embedding_epochs=1, walks_per_node=3,
        stability_seeds=2)
    return version


@pytest.fixture(scope="session")
def registry(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    return reg


@pytest.fixture(scope="session")
def drift_monitor(trained_version):
    return DriftMonitor(
        {int(k): list(v) for k, v in trained_version.embeddings.items()})


@pytest.fixture(scope="session")
def serving(registry, drift_monitor):
    graph = generate_graph(n_nodes=120, p_in=0.2, seed=SEED)
    return ServingLayer(registry, ServingConfig(canary_pct=0.0),
                        graph=graph, drift_monitor=drift_monitor)


@pytest.fixture(scope="session")
def shifted_embeddings():
    shifted = generate_graph(n_nodes=80, p_in=0.02, p_out=0.06, seed=SEED)
    train_graph, _edges, _negs = edge_split(shifted, seed=SEED)
    emb = SkipGramEmbeddings(epochs=1, seed=SEED)
    emb.fit(train_graph, walks=random_walks(train_graph, walks_per_node=3,
                                            seed=SEED))
    return emb.all_embeddings()
