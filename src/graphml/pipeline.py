"""End-to-end training pipeline: graph -> splits -> models -> version.

``train_version`` is the single entry point. It generates the synthetic
graph, runs quality + leakage screens, splits nodes (transductive-safe)
and edges (test edges removed from the training graph), trains label
propagation and a logistic regression over aggregated neighbourhood
features, learns skip-gram embeddings, trains a learned link scorer,
evaluates everything on held-out data, and returns a JSON-serialisable
:class:`ModelVersion` plus a human-readable report. A stability check
re-trains the classifier across several seeds and records the spread —
the registry's stability gate reads it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from graphml.embeddings import SkipGramEmbeddings, random_walks
from graphml.evaluation import (accuracy, evaluate_link_prediction,
                                macro_f1, per_class_recall,
                                roc_auc_multiclass)
from graphml.features import build_feature_matrix
from graphml.graph import Graph, check_quality, generate_graph, leakage_screen
from graphml.linkpred import LearnedLinkScorer, adamic_adar
from graphml.models import (LabelPropagation, LogisticRegression,
                            MajorityBaseline, Standardizer)
from graphml.splits import edge_split, stratified_node_split


@dataclass
class ModelVersion:
    """A JSON-serialisable snapshot of one trained version."""

    name: str
    seed: int
    n_classes: int
    feature_dim: int
    logreg_weights: list[list[float]]
    logreg_bias: list[float]
    scaler: dict
    embeddings: dict[str, list[float]]
    link_scorer_weights: list[list[float]]
    link_scorer_bias: list[float]
    link_scaler: dict
    metrics: dict
    baseline_metrics: dict
    quality: dict
    leakage: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name, "seed": self.seed,
            "n_classes": self.n_classes, "feature_dim": self.feature_dim,
            "logreg_weights": self.logreg_weights,
            "logreg_bias": self.logreg_bias,
            "scaler": self.scaler,
            "embeddings": self.embeddings,
            "link_scorer_weights": self.link_scorer_weights,
            "link_scorer_bias": self.link_scorer_bias,
            "link_scaler": self.link_scaler,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "quality": self.quality,
            "leakage": self.leakage,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            name=data["name"], seed=data["seed"],
            n_classes=data["n_classes"], feature_dim=data["feature_dim"],
            logreg_weights=data["logreg_weights"],
            logreg_bias=data["logreg_bias"], scaler=data["scaler"],
            embeddings=data["embeddings"],
            link_scorer_weights=data["link_scorer_weights"],
            link_scorer_bias=data["link_scorer_bias"],
            link_scaler=data["link_scaler"],
            metrics=data["metrics"],
            baseline_metrics=data["baseline_metrics"],
            quality=data["quality"], leakage=data["leakage"])

    @classmethod
    def from_json(cls, text: str) -> "ModelVersion":
        return cls.from_dict(json.loads(text))


def train_version(name: str, seed: int = 42, n_nodes: int = 240,
                  n_communities: int = 4, feature_dim: int = 8,
                  p_in: float = 0.12, p_out: float = 0.008,
                  logreg_epochs: int = 200, embedding_epochs: int = 2,
                  walks_per_node: int = 5, walk_length: int = 10,
                  embedding_dim: int = 16,
                  stability_seeds: int = 3) -> tuple[ModelVersion, dict]:
    """Train one version end to end; returns (version, report)."""
    graph = generate_graph(n_nodes=n_nodes, n_communities=n_communities,
                           feature_dim=feature_dim, p_in=p_in, p_out=p_out,
                           seed=seed)
    quality = check_quality(graph)
    if not quality["usable"]:
        raise ValueError(f"generated graph failed quality checks: {quality}")

    split = stratified_node_split(graph, seed=seed)
    leakage = leakage_screen(graph, split["train"], split["test"])
    if not leakage["clean"]:
        raise ValueError(f"leakage screen failed: {leakage}")

    # --- node classification -------------------------------------------------
    X_train = build_feature_matrix(graph, split["train"])
    X_test = build_feature_matrix(graph, split["test"])
    X_val = build_feature_matrix(graph, split["val"])
    y_train = [graph.labels[n] for n in split["train"]]
    y_test = [graph.labels[n] for n in split["test"]]
    y_val = [graph.labels[n] for n in split["val"]]

    scaler = Standardizer().fit(X_train)
    logreg = LogisticRegression(n_classes=n_communities, epochs=logreg_epochs,
                                seed=seed)
    logreg.fit(scaler.transform(X_train), y_train)
    test_proba = logreg.predict_proba(scaler.transform(X_test))
    test_pred = logreg.predict(scaler.transform(X_test))
    val_pred = logreg.predict(scaler.transform(X_val))

    lp = LabelPropagation(n_classes=n_communities)
    lp.fit(graph, split["train"], y_train)
    lp_test_pred = lp.predict(split["test"])
    lp_test_proba = lp.predict_proba(split["test"])

    baseline = MajorityBaseline().fit(y_train)
    base_pred = baseline.predict(len(y_test))

    metrics = {
        "logreg_test_accuracy": round(accuracy(y_test, test_pred), 4),
        "logreg_val_accuracy": round(accuracy(y_val, val_pred), 4),
        "logreg_macro_f1": round(macro_f1(y_test, test_pred, n_communities), 4),
        "logreg_roc_auc": round(
            roc_auc_multiclass(y_test, test_proba, n_communities), 4),
        "logreg_min_per_class_recall": round(
            min(per_class_recall(y_test, test_pred, n_communities)), 4),
        "label_prop_test_accuracy": round(
            accuracy(y_test, lp_test_pred), 4),
        "label_prop_roc_auc": round(
            roc_auc_multiclass(y_test, lp_test_proba, n_communities), 4),
    }
    baseline_metrics = {
        "majority_test_accuracy": round(accuracy(y_test, base_pred), 4),
        "majority_class": baseline.majority,
    }

    # Stability across seeds: same graph, re-split + re-train per seed.
    seed_accuracies = []
    for offset in range(stability_seeds):
        s = seed + 100 + offset
        sp = stratified_node_split(graph, seed=s)
        Xs_tr = build_feature_matrix(graph, sp["train"])
        Xs_te = build_feature_matrix(graph, sp["test"])
        sc = Standardizer().fit(Xs_tr)
        m = LogisticRegression(n_classes=n_communities,
                               epochs=logreg_epochs, seed=s)
        m.fit(sc.transform(Xs_tr), [graph.labels[n] for n in sp["train"]])
        seed_accuracies.append(accuracy(
            [graph.labels[n] for n in sp["test"]],
            m.predict(sc.transform(Xs_te))))
    mean_acc = sum(seed_accuracies) / len(seed_accuracies)
    var = sum((a - mean_acc) ** 2 for a in seed_accuracies) / len(seed_accuracies)
    metrics["seed_accuracies"] = [round(a, 4) for a in seed_accuracies]
    metrics["seed_accuracy_std"] = round(var ** 0.5, 4)

    # --- embeddings + link prediction ----------------------------------------
    # Nested edge split: the embeddings are trained on a graph that
    # excludes the link scorer's training positives, so embedding
    # similarity is scored under identical conditions for training
    # positives and held-out test edges. Training embeddings on a graph
    # that contains the scorer positives would inflate their similarity
    # and the scorer would overfit to it.
    train_graph, test_edges, _test_negs = edge_split(graph, seed=seed)
    emb_graph, scorer_pos_edges, _ = edge_split(
        train_graph, test_frac=0.25, seed=seed + 1)
    walks = random_walks(emb_graph, walk_length=walk_length,
                         walks_per_node=walks_per_node, seed=seed)
    embeddings = SkipGramEmbeddings(dim=embedding_dim,
                                    epochs=embedding_epochs, seed=seed)
    embeddings.fit(emb_graph, walks=walks)

    rng = random.Random(seed)
    edge_set = {tuple(sorted(e)) for e in graph.edge_list()}
    # Endpoint-matched negatives (share an endpoint with a positive),
    # mirroring the evaluation protocol.
    neg_sample: list[tuple[int, int]] = []
    attempts = 0
    while len(neg_sample) < len(scorer_pos_edges) and attempts < 20000:
        attempts += 1
        u, _v = rng.choice(scorer_pos_edges)
        w = rng.randrange(n_nodes)
        if w == u:
            continue
        pair = (min(u, w), max(u, w))
        if pair in edge_set or pair in neg_sample:
            continue
        neg_sample.append(pair)
    scorer = LearnedLinkScorer(epochs=200, seed=seed)
    scorer.fit(emb_graph, scorer_pos_edges, neg_sample, embeddings)

    def _scorer_fn(g: Graph, u: int, v: int,
                   emb: SkipGramEmbeddings | None) -> float:
        return scorer.score(g, u, v, emb)

    link_metrics = evaluate_link_prediction(
        train_graph, test_edges, _scorer_fn, embeddings, seed=seed)
    # Best single heuristic, for the "learned beats heuristic" comparison.
    heuristic_metrics = evaluate_link_prediction(
        train_graph, test_edges,
        lambda g, u, v, _e: adamic_adar(g, u, v), None, seed=seed)
    metrics["link_hits_at_10"] = link_metrics["hits_at_k"]
    metrics["link_mrr"] = link_metrics["mrr"]
    metrics["link_roc_auc"] = link_metrics["roc_auc"]
    metrics["link_random_hits_at_10"] = link_metrics["random_hits_at_k"]
    metrics["link_heuristic_hits_at_10"] = heuristic_metrics["hits_at_k"]

    version = ModelVersion(
        name=name, seed=seed, n_classes=n_communities,
        feature_dim=len(X_train[0]),
        logreg_weights=logreg.weights, logreg_bias=logreg.bias,
        scaler=scaler.to_dict(),
        embeddings={str(k): v for k, v in embeddings.all_embeddings().items()},
        link_scorer_weights=scorer.model.weights if scorer.model else [],
        link_scorer_bias=scorer.model.bias if scorer.model else [],
        link_scaler=scorer.scaler.to_dict(),
        metrics=metrics, baseline_metrics=baseline_metrics,
        quality=quality, leakage=leakage)
    report = {
        "version": name,
        "graph": {"n_nodes": n_nodes, "n_edges": graph.n_edges(),
                  "n_communities": n_communities},
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "quality": quality,
        "leakage": leakage,
    }
    return version, report
