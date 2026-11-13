"""Versioned model registry with gated champion promotion.

Promotion gates (all must pass; every check is recorded in the audit
history):
- ``accuracy_floor`` — test accuracy at or above the floor;
- ``must_beat_baseline`` — test accuracy beats the majority-class
  baseline by at least ``min_improvement_over_baseline``;
- ``per_class_recall_floor`` — the worst class recall is at or above
  the floor (no class silently sacrificed);
- ``seed_stability`` — the standard deviation of test accuracy across
  re-training seeds is at or below ``max_seed_std``;
- ``link_quality_floor`` — link-prediction Hits@10 at or above the
  floor, and above the random baseline.

The registry also records an audited promotion history and supports
rollback to the previous champion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphml.pipeline import ModelVersion


@dataclass
class PromotionResult:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, object]

    def summary(self) -> str:
        verdict = "PROMOTED" if self.passed else "REJECTED"
        failed = [k for k, v in self.checks.items() if not v]
        if self.passed:
            return f"{verdict}: all {len(self.checks)} gates passed"
        return f"{verdict}: failed gates: {', '.join(failed)}"


class ModelRegistry:
    """In-memory versioned registry with promotion gates and rollback."""

    def __init__(self, accuracy_floor: float = 0.55,
                 min_improvement_over_baseline: float = 0.10,
                 min_per_class_recall: float = 0.30,
                 max_seed_std: float = 0.05,
                 min_link_hits_at_10: float = 0.30) -> None:
        self.accuracy_floor = accuracy_floor
        self.min_improvement_over_baseline = min_improvement_over_baseline
        self.min_per_class_recall = min_per_class_recall
        self.max_seed_std = max_seed_std
        self.min_link_hits_at_10 = min_link_hits_at_10
        self.versions: dict[str, ModelVersion] = {}
        self.champion_name: str | None = None
        self.challenger_name: str | None = None
        self.history: list[dict] = []

    def register(self, version: ModelVersion,
                 role: str = "challenger") -> None:
        self.versions[version.name] = version
        if role == "champion":
            self.champion_name = version.name
        elif role == "challenger":
            self.challenger_name = version.name
        else:
            raise ValueError(f"unknown role {role!r}")

    def list_versions(self) -> list[dict]:
        return [{"name": v.name, "seed": v.seed, "metrics": v.metrics,
                 "baseline_metrics": v.baseline_metrics,
                 "role": ("champion" if v.name == self.champion_name
                          else "challenger" if v.name == self.challenger_name
                          else "archived")}
                for v in self.versions.values()]

    def evaluate_gates(self, version: ModelVersion) -> PromotionResult:
        m = version.metrics
        b = version.baseline_metrics
        checks = {
            "accuracy_floor": m["logreg_test_accuracy"] >= self.accuracy_floor,
            "must_beat_baseline": (
                m["logreg_test_accuracy"] - b["majority_test_accuracy"]
                >= self.min_improvement_over_baseline),
            "per_class_recall_floor": (
                m["logreg_min_per_class_recall"] >= self.min_per_class_recall),
            "seed_stability": m["seed_accuracy_std"] <= self.max_seed_std,
            "link_quality_floor": (
                m["link_hits_at_10"] >= self.min_link_hits_at_10
                and m["link_hits_at_10"] > m["link_random_hits_at_10"]),
        }
        details = {
            "test_accuracy": m["logreg_test_accuracy"],
            "baseline_accuracy": b["majority_test_accuracy"],
            "improvement_over_baseline": round(
                m["logreg_test_accuracy"] - b["majority_test_accuracy"], 4),
            "min_per_class_recall": m["logreg_min_per_class_recall"],
            "seed_accuracy_std": m["seed_accuracy_std"],
            "link_hits_at_10": m["link_hits_at_10"],
            "link_random_hits_at_10": m["link_random_hits_at_10"],
            "champion": self.champion_name,
        }
        return PromotionResult(passed=all(checks.values()),
                               checks=checks, details=details)

    def promote_challenger(self) -> PromotionResult:
        if self.challenger_name is None:
            raise ValueError("no challenger registered")
        challenger = self.versions[self.challenger_name]
        result = self.evaluate_gates(challenger)
        previous = self.champion_name
        if result.passed:
            self.champion_name = self.challenger_name
            self.challenger_name = None
        self.history.append({
            "version": challenger.name, "passed": result.passed,
            "previous_champion": previous,
            "checks": result.checks, "details": result.details,
        })
        return result

    def rollback(self) -> str:
        """Restore the most recent previously-promoted champion."""
        for entry in reversed(self.history):
            if entry["passed"] and entry["previous_champion"]:
                self.champion_name = entry["previous_champion"]
                self.challenger_name = None
                return self.champion_name
        raise ValueError("no previous champion to roll back to")

    def champion(self) -> ModelVersion | None:
        if self.champion_name is None:
            return None
        return self.versions[self.champion_name]
