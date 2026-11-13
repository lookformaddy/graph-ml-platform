"""Embedding-drift monitoring.

The monitor is fitted on a reference embedding set (the champion's).
For a new embedding set over the same node ids it reports:
- ``centroid_shift`` — Euclidean distance between the reference and new
  embedding centroids, normalised by the reference centroid norm;
- ``mean_norm_shift`` — relative change in mean embedding norm;
- ``mean_cosine_distance`` — average per-node 1 - cosine similarity
  between the reference and new vectors for the same node.

A graph is flagged as drifted when any of the three exceeds its
threshold. Because embeddings are deterministic given the seed, a
re-train on the same graph scores exactly zero drift, while a graph
whose community structure has dissolved scores far above the
thresholds. No labels are needed — this is unsupervised monitoring.
"""

from __future__ import annotations

import math


def _centroid(embeddings: dict[int, list[float]]) -> list[float]:
    vecs = list(embeddings.values())
    dim = len(vecs[0])
    return [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    # Clamp: float rounding can push the ratio a hair past 1.0.
    return max(-1.0, min(1.0, dot / (na * nb)))


class DriftMonitor:
    """Embedding-drift monitor fitted on reference embeddings."""

    def __init__(self, reference_embeddings: dict[int, list[float]],
                 centroid_threshold: float = 0.30,
                 norm_shift_threshold: float = 0.25,
                 cosine_distance_threshold: float = 0.15) -> None:
        self.reference = {int(k): list(v)
                          for k, v in reference_embeddings.items()}
        self.centroid_threshold = centroid_threshold
        self.norm_shift_threshold = norm_shift_threshold
        self.cosine_distance_threshold = cosine_distance_threshold
        self._ref_centroid = _centroid(self.reference)
        self._ref_norm = _norm(self._ref_centroid)
        self._ref_mean_norm = sum(_norm(v) for v in self.reference.values()) \
            / len(self.reference)
        self.n_observations = 0
        self.n_flagged = 0
        self.last_report: dict | None = None

    def score(self, embeddings: dict[int, list[float]]) -> dict:
        """Score one new embedding set; returns the drift report."""
        new = {int(k): list(v) for k, v in embeddings.items()}
        common = [k for k in new if k in self.reference]
        if not common:
            report = {"flagged": True, "reason": "no overlapping nodes",
                      "centroid_shift": None, "mean_norm_shift": None,
                      "mean_cosine_distance": None,
                      "n_common_nodes": 0}
            self._record(report)
            return report
        new_centroid = _centroid({k: new[k] for k in common})
        ref_centroid = _centroid({k: self.reference[k] for k in common})
        diff = [a - b for a, b in zip(new_centroid, ref_centroid)]
        centroid_shift = (_norm(diff) / _norm(ref_centroid)
                          if _norm(ref_centroid) else 0.0)
        new_mean_norm = sum(_norm(new[k]) for k in common) / len(common)
        ref_mean_norm = sum(_norm(self.reference[k]) for k in common) \
            / len(common)
        mean_norm_shift = (abs(new_mean_norm - ref_mean_norm) / ref_mean_norm
                           if ref_mean_norm else 0.0)
        mean_cos_dist = sum(
            1.0 - _cosine(self.reference[k], new[k]) for k in common) \
            / len(common)
        flagged = (centroid_shift > self.centroid_threshold
                   or mean_norm_shift > self.norm_shift_threshold
                   or mean_cos_dist > self.cosine_distance_threshold)
        report = {
            "flagged": flagged,
            "centroid_shift": round(centroid_shift, 4),
            "mean_norm_shift": round(mean_norm_shift, 4),
            "mean_cosine_distance": round(mean_cos_dist, 4),
            "n_common_nodes": len(common),
            "thresholds": {
                "centroid": self.centroid_threshold,
                "norm_shift": self.norm_shift_threshold,
                "cosine_distance": self.cosine_distance_threshold,
            },
        }
        self._record(report)
        return report

    def _record(self, report: dict) -> None:
        self.n_observations += 1
        if report["flagged"]:
            self.n_flagged += 1
        self.last_report = report

    def report(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_flagged": self.n_flagged,
            "last": self.last_report,
        }
