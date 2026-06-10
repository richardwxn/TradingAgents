"""Tests for the gates-enforcement and leak-sanity helpers in eval_ml_models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_eval_module():
    """Load `scripts/eval_ml_models.py` as a module without running main()."""

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_eval_ml_models_for_tests",
        repo_root / "scripts" / "eval_ml_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_gates_check_passes_when_ml_beats_baseline():
    mod = _load_eval_module()
    leaderboard = [
        {"horizon": "ret_60d", "model": "factor_v1_8", "cohort": "all",
         "alpha_hit_rate": 0.55, "precision_at_top_k": 0.50},
        {"horizon": "ret_60d", "model": "elastic_logit", "cohort": "all",
         "alpha_hit_rate": 0.58, "precision_at_top_k": 0.54},
    ]
    out = mod._compute_gates_check(
        leaderboard,
        gates={"min_alpha_hit_lift": 0.01, "min_top_k_precision_lift": 0.01,
               "max_horizon_regression": 0.01},
        horizons=["ret_60d"],
        model_names=["elastic_logit"],
        cohorts=["all"],
    )
    assert len(out) == 1
    assert out[0]["gate_pass"] is True
    assert out[0]["alpha_hit_lift"] > 0


def test_compute_gates_check_fails_when_ml_loses_to_baseline():
    mod = _load_eval_module()
    leaderboard = [
        {"horizon": "ret_60d", "model": "factor_v1_8", "cohort": "all",
         "alpha_hit_rate": 0.60, "precision_at_top_k": 0.55},
        {"horizon": "ret_60d", "model": "elastic_logit", "cohort": "all",
         "alpha_hit_rate": 0.59, "precision_at_top_k": 0.54},
    ]
    out = mod._compute_gates_check(
        leaderboard,
        gates={"min_alpha_hit_lift": 0.01, "min_top_k_precision_lift": 0.01,
               "max_horizon_regression": 0.01},
        horizons=["ret_60d"],
        model_names=["elastic_logit"],
        cohorts=["all"],
    )
    assert out[0]["gate_pass"] is False


def test_compute_gates_check_fails_on_other_horizon_regression():
    """Even with a 60d lift, a >1pp drop at 20d means gates fail."""
    mod = _load_eval_module()
    leaderboard = [
        {"horizon": "ret_20d", "model": "factor_v1_8", "cohort": "all",
         "alpha_hit_rate": 0.55, "precision_at_top_k": 0.50},
        {"horizon": "ret_20d", "model": "ml", "cohort": "all",
         "alpha_hit_rate": 0.52, "precision_at_top_k": 0.51},  # -3pp regression
        {"horizon": "ret_60d", "model": "factor_v1_8", "cohort": "all",
         "alpha_hit_rate": 0.55, "precision_at_top_k": 0.50},
        {"horizon": "ret_60d", "model": "ml", "cohort": "all",
         "alpha_hit_rate": 0.60, "precision_at_top_k": 0.55},
    ]
    out = mod._compute_gates_check(
        leaderboard,
        gates={"min_alpha_hit_lift": 0.01, "min_top_k_precision_lift": 0.01,
               "max_horizon_regression": 0.01},
        horizons=["ret_20d", "ret_60d"],
        model_names=["ml"],
        cohorts=["all"],
    )
    sixty = next(r for r in out if r["horizon"] == "ret_60d")
    assert sixty["gate_pass"] is False
    assert sixty["worst_horizon_alpha_delta"] is not None
    assert sixty["worst_horizon_alpha_delta"] < 0


def test_compute_gates_check_per_cohort_independent():
    """Core can pass while canary fails (or vice versa)."""
    mod = _load_eval_module()
    leaderboard = [
        # Core: ML beats baseline.
        {"horizon": "ret_60d", "model": "factor_v1_8", "cohort": "core",
         "alpha_hit_rate": 0.55, "precision_at_top_k": 0.50},
        {"horizon": "ret_60d", "model": "ml", "cohort": "core",
         "alpha_hit_rate": 0.60, "precision_at_top_k": 0.55},
        # Canary: ML loses.
        {"horizon": "ret_60d", "model": "factor_v1_8", "cohort": "canary",
         "alpha_hit_rate": 0.60, "precision_at_top_k": 0.55},
        {"horizon": "ret_60d", "model": "ml", "cohort": "canary",
         "alpha_hit_rate": 0.58, "precision_at_top_k": 0.53},
    ]
    out = mod._compute_gates_check(
        leaderboard,
        gates={"min_alpha_hit_lift": 0.01, "min_top_k_precision_lift": 0.01,
               "max_horizon_regression": 0.01},
        horizons=["ret_60d"],
        model_names=["ml"],
        cohorts=["core", "canary"],
    )
    by_cohort = {r["cohort"]: r["gate_pass"] for r in out}
    assert by_cohort == {"core": True, "canary": False}


def test_load_universe_cohorts_reads_universe_yaml():
    """End-to-end: loader picks up core+canary from real universe.yaml."""
    mod = _load_eval_module()
    lookup = mod._load_universe_cohorts("configs/universe.yaml")
    assert lookup.get("NVDA") == "core"
    assert lookup.get("JPM") == "canary"
    # Symbols not in either list aren't in the lookup.
    assert "SPY" not in lookup


def test_load_universe_cohorts_missing_file_returns_empty():
    mod = _load_eval_module()
    assert mod._load_universe_cohorts("/nonexistent/path.yaml") == {}


def test_diff_handles_none():
    import pytest

    mod = _load_eval_module()
    assert mod._diff(0.6, 0.5) == pytest.approx(0.1)
    assert mod._diff(None, 0.5) is None
    assert mod._diff(0.6, None) is None
