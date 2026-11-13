"""Latency-budgeted serving layer with baseline fallback.

The layer wraps a :class:`graphml.registry.ModelRegistry` and a
reference graph. Each request is timed; if the champion path exceeds
the latency budget (or raises), the request falls back to the
majority-class baseline so the service always answers. Canary traffic
(hash-bucketed by request id) is scored by the challenger when one is
registered. Serving stats (arms, fallbacks, latency) are tracked for
the ``/stats`` endpoint.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from graphml.drift import DriftMonitor
from graphml.features import aggregate_features
from graphml.graph import Graph
from graphml.models import MajorityBaseline
from graphml.registry import ModelRegistry


@dataclass
class ServingConfig:
    latency_budget_ms: float = 100.0
    canary_pct: float = 0.0


@dataclass
class Prediction:
    kind: str
    value: object
    confidence: float
    arm: str
    latency_ms: float
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value,
                "confidence": round(self.confidence, 4), "arm": self.arm,
                "latency_ms": round(self.latency_ms, 2),
                "fallback_reason": self.fallback_reason}


class ServingLayer:
    """Champion/canary serving with latency budgets and fallback."""

    def __init__(self, registry: ModelRegistry, config: ServingConfig,
                 graph: Graph | None = None,
                 drift_monitor: DriftMonitor | None = None) -> None:
        self.registry = registry
        self.config = config
        self.graph = graph
        self.drift_monitor = drift_monitor
        self.stats = {"total_requests": 0, "champion": 0, "canary": 0,
                      "fallback_baseline": 0, "errors": 0,
                      "latency_ms_total": 0.0, "latency_ms_max": 0.0}

    # -- helpers ------------------------------------------------------------
    def _bucket(self, request_id: str) -> float:
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        return (int(digest[:8], 16) % 10000) / 100.0

    def _record(self, arm: str, latency_ms: float) -> None:
        self.stats["total_requests"] += 1
        self.stats[arm] = self.stats.get(arm, 0) + 1
        self.stats["latency_ms_total"] += latency_ms
        self.stats["latency_ms_max"] = max(self.stats["latency_ms_max"],
                                           latency_ms)

    def _softmax(self, logits: list[float]) -> list[float]:
        import math
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _classify_with(self, version, node: int) -> tuple[int, float]:
        feats = aggregate_features(self.graph, node)
        scaled = [(feats[d] - version.scaler["mean"][d])
                  / version.scaler["std"][d]
                  for d in range(len(feats))]
        logits = [sum(version.logreg_weights[c][d] * scaled[d]
                      for d in range(len(scaled))) + version.logreg_bias[c]
                  for c in range(version.n_classes)]
        probs = self._softmax(logits)
        label = max(range(version.n_classes), key=lambda c: probs[c])
        return label, probs[label]

    def _fallback_node(self, version, node: int) -> Prediction:
        majority = version.baseline_metrics.get("majority_class", 0)
        return Prediction(kind="node", value=int(majority), confidence=0.0,
                          arm="fallback-baseline", latency_ms=0.0,
                          fallback_reason="latency-or-error")

    # -- endpoints ----------------------------------------------------------
    def predict_node(self, node: int,
                     request_id: str = "default") -> Prediction:
        start = time.perf_counter()
        champion = self.registry.champion()
        if champion is None or self.graph is None \
                or not (0 <= node < self.graph.n_nodes):
            self.stats["errors"] += 1
            return Prediction(kind="node", value=None, confidence=0.0,
                              arm="error", latency_ms=0.0,
                              fallback_reason="no-champion-or-bad-node")
        try:
            version = champion
            arm = "champion"
            if self.registry.challenger_name is not None \
                    and self._bucket(request_id) < self.config.canary_pct:
                version = self.registry.versions[self.registry.challenger_name]
                arm = "canary"
            label, conf = self._classify_with(version, node)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if latency_ms > self.config.latency_budget_ms:
                fallback = self._fallback_node(champion, node)
                fallback.latency_ms = latency_ms
                fallback.fallback_reason = "latency-budget-exceeded"
                self._record("fallback_baseline", latency_ms)
                return fallback
            self._record(arm, latency_ms)
            return Prediction(kind="node", value=label, confidence=conf,
                              arm=arm, latency_ms=latency_ms)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            fallback = self._fallback_node(champion, node)
            fallback.latency_ms = latency_ms
            self._record("fallback_baseline", latency_ms)
            return fallback

    def predict_link(self, u: int, v: int,
                     request_id: str = "default") -> Prediction:
        start = time.perf_counter()
        champion = self.registry.champion()
        if champion is None or self.graph is None \
                or not (0 <= u < self.graph.n_nodes) \
                or not (0 <= v < self.graph.n_nodes) or u == v:
            self.stats["errors"] += 1
            return Prediction(kind="link", value=None, confidence=0.0,
                              arm="error", latency_ms=0.0,
                              fallback_reason="no-champion-or-bad-pair")
        try:
            from graphml.embeddings import SkipGramEmbeddings  # local: speed
            from graphml.linkpred import heuristic_features
            emb = SkipGramEmbeddings(dim=len(next(iter(
                champion.embeddings.values()))))
            emb.vectors = {int(k): list(val)
                           for k, val in champion.embeddings.items()}
            feats = heuristic_features(self.graph, u, v, emb)
            scaled = [(feats[d] - champion.link_scaler["mean"][d])
                      / champion.link_scaler["std"][d]
                      for d in range(len(feats))]
            logits = [sum(champion.link_scorer_weights[c][d] * scaled[d]
                          for d in range(len(scaled)))
                      + champion.link_scorer_bias[c] for c in range(2)]
            probs = self._softmax(logits)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._record("champion", latency_ms)
            return Prediction(kind="link", value=round(probs[1], 4),
                              confidence=round(probs[1], 4),
                              arm="champion", latency_ms=latency_ms)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.stats["errors"] += 1
            return Prediction(kind="link", value=None, confidence=0.0,
                              arm="error", latency_ms=latency_ms,
                              fallback_reason="scoring-error")

    def embed(self, node: int) -> Prediction:
        start = time.perf_counter()
        champion = self.registry.champion()
        if champion is None or str(node) not in champion.embeddings:
            self.stats["errors"] += 1
            return Prediction(kind="embedding", value=None, confidence=0.0,
                              arm="error", latency_ms=0.0,
                              fallback_reason="unknown-node")
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record("champion", latency_ms)
        return Prediction(kind="embedding",
                          value=champion.embeddings[str(node)],
                          confidence=1.0, arm="champion",
                          latency_ms=latency_ms)

    def stats_dict(self) -> dict:
        total = self.stats["total_requests"]
        return {
            **self.stats,
            "mean_latency_ms": round(
                self.stats["latency_ms_total"] / total, 2) if total else 0.0,
            "latency_budget_ms": self.config.latency_budget_ms,
            "canary_pct": self.config.canary_pct,
            "champion": self.registry.champion_name,
            "challenger": self.registry.challenger_name,
        }
