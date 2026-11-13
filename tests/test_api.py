"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from graphml.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["champion"] == "v1-seed42"


def test_predict_node():
    resp = client.post("/predict-node", json={
        "node": 0, "request_id": "api-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["value"] in (0, 1, 2, 3)
    assert 0.0 < data["confidence"] <= 1.0
    assert data["arm"] in ("champion", "canary", "fallback-baseline")


def test_predict_node_bad_node_is_error():
    resp = client.post("/predict-node", json={"node": 99999})
    assert resp.status_code == 200
    assert resp.json()["arm"] == "error"


def test_predict_link():
    resp = client.post("/predict-link", json={"u": 0, "v": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["value"] <= 1.0
    assert data["arm"] == "champion"


def test_embed():
    resp = client.post("/embed", json={"node": 0})
    assert resp.status_code == 200
    assert len(resp.json()["value"]) == 16


def test_embed_unknown_node_404():
    resp = client.post("/embed", json={"node": 99999})
    assert resp.status_code == 404


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion"] == "v1-seed42"
    assert any(v["name"] == "v1-seed42" for v in data["versions"])


def test_stats_endpoint():
    client.post("/predict-node", json={"node": 1})
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] >= 1
    assert "drift" in data


def test_promote_without_challenger_rejected():
    resp = client.post("/promote")
    assert resp.status_code == 400
