"""Tests for the registry's promotion gates, history and rollback."""
import copy

import pytest

from graphml.registry import ModelRegistry


def _weak_version(version):
    weak = copy.deepcopy(version)
    weak.name = "weak"
    weak.metrics = dict(version.metrics)
    weak.metrics["logreg_test_accuracy"] = 0.30
    weak.metrics["logreg_min_per_class_recall"] = 0.10
    weak.metrics["seed_accuracy_std"] = 0.20
    weak.metrics["link_hits_at_10"] = 0.05
    return weak


def test_gates_pass_for_trained_version(registry, trained_version):
    result = registry.evaluate_gates(trained_version)
    assert result.passed is True
    assert all(result.checks.values())


def test_gates_fail_for_weak_version(trained_version):
    reg = ModelRegistry()
    result = reg.evaluate_gates(_weak_version(trained_version))
    assert result.passed is False
    assert result.checks["accuracy_floor"] is False
    assert result.checks["must_beat_baseline"] is False
    assert result.checks["seed_stability"] is False


def test_promote_moves_challenger_to_champion(trained_version):
    reg = ModelRegistry()
    v2 = copy.deepcopy(trained_version)
    v2.name = "v2"
    reg.register(trained_version, role="champion")
    reg.register(v2, role="challenger")
    result = reg.promote_challenger()
    assert result.passed is True
    assert reg.champion_name == "v2"
    assert reg.challenger_name is None
    assert len(reg.history) == 1
    assert reg.history[0]["version"] == "v2"


def test_failed_promotion_keeps_champion(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    reg.register(_weak_version(trained_version), role="challenger")
    result = reg.promote_challenger()
    assert result.passed is False
    assert reg.champion_name == trained_version.name
    assert reg.challenger_name == "weak"


def test_promote_without_challenger_raises(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    with pytest.raises(ValueError, match="no challenger"):
        reg.promote_challenger()


def test_rollback_restores_previous_champion(trained_version):
    reg = ModelRegistry()
    v2 = copy.deepcopy(trained_version)
    v2.name = "v2"
    reg.register(trained_version, role="champion")
    reg.register(v2, role="challenger")
    reg.promote_challenger()
    assert reg.rollback() == trained_version.name
    assert reg.champion_name == trained_version.name


def test_rollback_without_history_raises(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    with pytest.raises(ValueError, match="no previous champion"):
        reg.rollback()


def test_list_versions_marks_roles(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    versions = reg.list_versions()
    assert versions[0]["role"] == "champion"
    assert versions[0]["name"] == trained_version.name
