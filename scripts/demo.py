"""End-to-end demo: train v1 + v2, gate the promotion, serve requests,
score link predictions, and show embedding drift on a shifted graph.

Every number printed is computed live from the seeded synthetic graph —
no hard-coded metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphml.drift import DriftMonitor  # noqa: E402
from graphml.embeddings import SkipGramEmbeddings, random_walks  # noqa: E402
from graphml.graph import generate_graph  # noqa: E402
from graphml.pipeline import train_version  # noqa: E402
from graphml.registry import ModelRegistry  # noqa: E402
from graphml.serving import ServingConfig, ServingLayer  # noqa: E402
from graphml.splits import edge_split  # noqa: E402


def run_demo() -> None:
    print("== Graph ML Platform — end-to-end demo ==\n")

    # 1. Train v1.
    v1, report1 = train_version("v1", seed=42)
    m1 = report1["metrics"]
    b1 = report1["baseline_metrics"]
    print(f"[1] v1 trained on a {report1['graph']['n_nodes']}-node, "
          f"{report1['graph']['n_edges']}-edge community graph "
          f"({report1['graph']['n_communities']} communities)")
    print(f"    quality: mean degree {report1['quality']['mean_degree']}, "
          f"isolated nodes {report1['quality']['n_isolated']}, "
          f"leakage screen clean: {report1['leakage']['clean']}")
    print(f"    node classification — logreg test accuracy: "
          f"{m1['logreg_test_accuracy']:.3f} "
          f"(macro F1 {m1['logreg_macro_f1']:.3f}, "
          f"ROC-AUC {m1['logreg_roc_auc']:.3f})")
    print(f"    label propagation test accuracy: "
          f"{m1['label_prop_test_accuracy']:.3f}")
    print(f"    majority-class baseline accuracy: "
          f"{b1['majority_test_accuracy']:.3f} "
          f"-> learned improvement: "
          f"{m1['logreg_test_accuracy'] - b1['majority_test_accuracy']:+.3f}")
    print(f"    link prediction — learned Hits@10: {m1['link_hits_at_10']:.3f} "
          f"(heuristic Adamic-Adar {m1['link_heuristic_hits_at_10']:.3f}, "
          f"random {m1['link_random_hits_at_10']:.3f}), "
          f"MRR {m1['link_mrr']:.3f}")
    print(f"    stability across seeds: {m1['seed_accuracies']} "
          f"(std {m1['seed_accuracy_std']:.3f})")

    # 2. Registry: register v1 as champion, train a stronger v2, promote.
    registry = ModelRegistry()
    registry.register(v1, role="champion")
    v2, report2 = train_version("v2", seed=42, logreg_epochs=300,
                                embedding_epochs=3, walks_per_node=8)
    m2 = report2["metrics"]
    print(f"\n[2] v2 trained (same graph, more epochs/walks): "
          f"test accuracy {m2['logreg_test_accuracy']:.3f}, "
          f"Hits@10 {m2['link_hits_at_10']:.3f}")
    registry.register(v2, role="challenger")
    result = registry.promote_challenger()
    print(f"    promotion gates: {result.summary()}")
    for gate, passed in result.checks.items():
        print(f"      {'PASS' if passed else 'FAIL'}  {gate}")
    print(f"    champion is now: {registry.champion_name}")

    # 3. Serve: champion predictions + link scores + embeddings.
    graph = generate_graph(n_nodes=240, seed=42)
    monitor = DriftMonitor(
        {int(k): list(v) for k, v in v1.embeddings.items()})
    layer = ServingLayer(registry, ServingConfig(canary_pct=0.0),
                         graph=graph, drift_monitor=monitor)
    node = 0
    pred = layer.predict_node(node, request_id="demo-1")
    print(f"\n[3] serving: node {node} (true label {graph.labels[node]}) -> "
          f"predicted {pred.value} via {pred.arm}, "
          f"confidence {pred.confidence:.2f}, {pred.latency_ms:.1f} ms")
    u, v = 0, 1
    link = layer.predict_link(u, v, request_id="demo-2")
    print(f"    link ({u}, {v}) — actually an edge: {graph.has_edge(u, v)} "
          f"-> score {link.value}, {link.latency_ms:.1f} ms")
    emb = layer.embed(0)
    print(f"    embedding for node 0: {len(emb.value)} dims, "
          f"first values {[round(x, 3) for x in emb.value[:4]]}")
    print(f"    serving stats: {layer.stats_dict()}")

    # 4. Drift: same graph re-trained (no drift) vs dissolved communities.
    # The re-train reproduces v1's nested embedding split exactly, so
    # its embeddings are identical and drift scores exactly zero.
    train_graph, _e, _n = edge_split(graph, seed=42)
    emb_graph, _sp, _sn = edge_split(train_graph, test_frac=0.25, seed=43)
    same = SkipGramEmbeddings(epochs=2, seed=42)
    same.fit(emb_graph, walks=random_walks(emb_graph, seed=42))
    same_report = monitor.score(same.all_embeddings())
    print(f"\n[4] drift — same graph re-trained: flagged="
          f"{same_report['flagged']}, mean cosine distance "
          f"{same_report['mean_cosine_distance']}")
    shifted = generate_graph(n_nodes=240, p_in=0.03, p_out=0.05, seed=42)
    shifted_train, _e2, _n2 = edge_split(shifted, seed=42)
    shifted_emb_graph, _sp2, _sn2 = edge_split(
        shifted_train, test_frac=0.25, seed=43)
    shifted_emb = SkipGramEmbeddings(epochs=2, seed=42)
    shifted_emb.fit(shifted_emb_graph,
                    walks=random_walks(shifted_emb_graph, seed=42))
    drift_report = monitor.score(shifted_emb.all_embeddings())
    print(f"    drift — communities dissolved (p_in 0.12->0.03, "
          f"p_out 0.008->0.05): flagged={drift_report['flagged']}, "
          f"centroid shift {drift_report['centroid_shift']}, "
          f"mean cosine distance {drift_report['mean_cosine_distance']}")
    print(f"    drift monitor: {monitor.report()['n_flagged']} flagged of "
          f"{monitor.report()['n_observations']} observations")
    print("\nDemo complete.")


if __name__ == "__main__":
    run_demo()
