"""Tests for the latency-budgeted serving layer."""
from graphml.serving import ServingConfig, ServingLayer


def test_predict_node_returns_champion_label(serving):
    pred = serving.predict_node(0, request_id="t1")
    assert pred.arm == "champion"
    assert pred.value in (0, 1, 2, 3)
    assert 0.0 < pred.confidence <= 1.0
    assert pred.latency_ms >= 0.0


def test_predict_node_matches_true_label_often(serving):
    from graphml.graph import generate_graph
    graph = generate_graph(n_nodes=120, p_in=0.2, seed=42)
    correct = sum(
        1 for node in range(120)
        if serving.predict_node(node, request_id=f"t{node}").value
        == graph.labels[node])
    assert correct / 120 > 0.6


def test_zero_latency_budget_forces_fallback(registry, drift_monitor):
    from graphml.graph import generate_graph
    graph = generate_graph(n_nodes=120, p_in=0.2, seed=42)
    layer = ServingLayer(registry, ServingConfig(latency_budget_ms=0.0),
                         graph=graph, drift_monitor=drift_monitor)
    pred = layer.predict_node(0, request_id="t-fb")
    assert pred.arm == "fallback-baseline"
    assert pred.fallback_reason == "latency-budget-exceeded"


def test_predict_node_bad_node_is_error(serving):
    pred = serving.predict_node(9999, request_id="t-bad")
    assert pred.arm == "error"
    assert pred.value is None


def test_predict_link_returns_probability(serving):
    pred = serving.predict_link(0, 1, request_id="t-link")
    assert pred.arm == "champion"
    assert 0.0 <= pred.value <= 1.0


def test_predict_link_rejects_self_pair(serving):
    pred = serving.predict_link(0, 0, request_id="t-self")
    assert pred.arm == "error"


def test_embed_returns_vector(serving):
    pred = serving.embed(0)
    assert pred.arm == "champion"
    assert len(pred.value) == 16


def test_embed_unknown_node_is_error(serving):
    pred = serving.embed(9999)
    assert pred.arm == "error"
    assert pred.value is None


def test_canary_bucket_routes_to_challenger(registry, drift_monitor,
                                            trained_version):
    import copy
    from graphml.graph import generate_graph
    v2 = copy.deepcopy(trained_version)
    v2.name = "v2-canary"
    registry.register(v2, role="challenger")
    graph = generate_graph(n_nodes=120, p_in=0.2, seed=42)
    layer = ServingLayer(registry, ServingConfig(canary_pct=100.0),
                         graph=graph, drift_monitor=drift_monitor)
    pred = layer.predict_node(0, request_id="canary-user")
    assert pred.arm == "canary"


def test_stats_track_requests(serving):
    before = serving.stats["total_requests"]
    serving.predict_node(1, request_id="t-stats")
    serving.predict_link(1, 2, request_id="t-stats2")
    assert serving.stats["total_requests"] == before + 2
    stats = serving.stats_dict()
    assert stats["mean_latency_ms"] >= 0.0
    assert stats["champion"] == "v1-test"
