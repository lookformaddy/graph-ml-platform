"""FastAPI service for the graph ML platform.

Endpoints:
    GET  /health        — liveness + champion version + drift summary
    POST /predict-node  — classify one node (champion/canary/fallback)
    POST /predict-link  — link-existence probability for a node pair
    POST /embed         — embedding vector for one node
    GET  /models        — registry versions, champion, challenger
    GET  /stats         — serving stats: arms, fallbacks, latency, drift
    POST /promote       — promote the challenger if the registry gates pass

The module-level ``app`` is built at import with a champion trained on
the seeded synthetic graph, so ``TestClient(app)`` works out of the box
in tests. For production wiring use :func:`build_app` with your own
ServingLayer.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphml.drift import DriftMonitor
from graphml.graph import generate_graph
from graphml.pipeline import train_version
from graphml.registry import ModelRegistry
from graphml.serving import ServingConfig, ServingLayer

__all__ = ["app", "build_app"]


class NodeRequest(BaseModel):
    node: int = Field(description="Node id to classify")
    request_id: str | None = Field(default=None)


class LinkRequest(BaseModel):
    u: int = Field(description="First node id")
    v: int = Field(description="Second node id")
    request_id: str | None = Field(default=None)


class EmbedRequest(BaseModel):
    node: int = Field(description="Node id to embed")


class PromoteResponse(BaseModel):
    promoted: bool
    summary: str
    checks: dict


def build_app(layer: ServingLayer) -> FastAPI:
    app = FastAPI(
        title="Graph ML Platform",
        description="From-scratch graph ML: node classification, link "
                    "prediction and node embeddings on a synthetic "
                    "community graph — versioned registry with gated "
                    "champion promotion, latency budgets with baseline "
                    "fallback, and embedding-drift monitoring.",
        version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        registry = layer.registry
        drift = (layer.drift_monitor.report()
                 if layer.drift_monitor is not None else None)
        return {"status": "ok",
                "champion": registry.champion_name,
                "challenger": registry.challenger_name,
                "drift": drift}

    @app.post("/predict-node")
    def predict_node(req: NodeRequest) -> dict:
        pred = layer.predict_node(req.node,
                                  request_id=req.request_id or "api")
        return pred.to_dict()

    @app.post("/predict-link")
    def predict_link(req: LinkRequest) -> dict:
        pred = layer.predict_link(req.u, req.v,
                                  request_id=req.request_id or "api")
        return pred.to_dict()

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict:
        pred = layer.embed(req.node)
        if pred.value is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown node {req.node}")
        return pred.to_dict()

    @app.get("/models")
    def models() -> dict:
        return {"versions": layer.registry.list_versions(),
                "champion": layer.registry.champion_name,
                "challenger": layer.registry.challenger_name,
                "history": layer.registry.history}

    @app.get("/stats")
    def stats() -> dict:
        data = layer.stats_dict()
        if layer.drift_monitor is not None:
            data["drift"] = layer.drift_monitor.report()
        return data

    @app.post("/promote", response_model=PromoteResponse)
    def promote() -> PromoteResponse:
        try:
            result = layer.registry.promote_challenger()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PromoteResponse(promoted=result.passed,
                               summary=result.summary(),
                               checks=result.checks)

    return app


def _default_layer() -> ServingLayer:
    """Seeded default: champion v1 trained on the deterministic graph."""
    version, _report = train_version("v1-seed42", seed=42, n_nodes=200,
                                     logreg_epochs=200, embedding_epochs=2)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    graph = generate_graph(n_nodes=200, seed=42)
    monitor = DriftMonitor(
        {int(k): list(v) for k, v in version.embeddings.items()})
    return ServingLayer(registry, ServingConfig(), graph=graph,
                        drift_monitor=monitor)


app = build_app(_default_layer())
