"""Tests for walk_forward_calibration_reliability (frozen-vector OOS check)."""

from tradingagents.analysis_only.scoring import (
    walk_forward_calibration_reliability,
)


def _obs(n_per_day, days, hit_fn):
    """Build observations spanning `days` days, n_per_day each.

    hit_fn(composite) -> probability; we make it deterministic-ish by
    thresholding a cycling composite so the calibration has real signal.
    """
    from datetime import date, timedelta

    start = date(2022, 1, 1)
    out = []
    for d in range(days):
        day = (start + timedelta(days=d)).isoformat()
        for i in range(n_per_day):
            # composite cycles through a few values; high composite -> hit.
            comp = [-0.5, -0.1, 0.1, 0.5][i % 4]
            hit = 1 if comp > 0 else 0  # perfectly separable signal
            direction = "bullish" if comp > 0 else "bearish"
            out.append((day, comp, direction, hit, 1.0))
    return out


def test_oos_runs_and_reports_structure():
    obs = _obs(n_per_day=8, days=900, hit_fn=None)
    res = walk_forward_calibration_reliability(
        obs, train_window_days=360, test_window_days=90, step_days=90,
        min_train_obs=100, min_obs_per_direction=10,
    )
    assert res["status"] == "ok"
    assert res["n_windows"] >= 2
    assert res["n_oos_obs"] > 0
    assert "max_gap_pp" in res and "brier_oos" in res
    assert isinstance(res["gate_reliability_5pp"], bool)
    assert isinstance(res["gate_brier_beats_heuristic"], bool)


def test_oos_separable_signal_calibrates_well():
    # A perfectly separable signal should yield a strong OOS Brier and pass
    # the reliability gate (predictions ~0/1 match observed).
    obs = _obs(n_per_day=8, days=900, hit_fn=None)
    res = walk_forward_calibration_reliability(
        obs, train_window_days=360, test_window_days=90, step_days=90,
        min_train_obs=100, min_obs_per_direction=10,
    )
    assert res["status"] == "ok"
    # Isotonic should beat the base-rate Brier substantially on separable data.
    assert res["brier_oos"] < res["brier_base_rate"]
    assert res["gate_reliability_5pp"] is True


def test_oos_no_observations():
    assert walk_forward_calibration_reliability([])["status"] == "no_observations"


def test_oos_insufficient_history_reports_status():
    # Too few days to form a train+test window.
    obs = _obs(n_per_day=5, days=30, hit_fn=None)
    res = walk_forward_calibration_reliability(
        obs, train_window_days=360, test_window_days=90, step_days=30,
        min_train_obs=100,
    )
    assert res["status"] in ("insufficient_windows", "no_observations")


def test_oos_does_not_leak_test_labels():
    # Even with a constant composite (no signal), OOS predictions must come
    # from the train base rate, not the test labels — so Brier should be ~
    # base-rate, never near-zero.
    from datetime import date, timedelta

    start = date(2022, 1, 1)
    obs = []
    for d in range(900):
        day = (start + timedelta(days=d)).isoformat()
        for i in range(8):
            # Constant composite; hit is random-ish (alternating) → no signal.
            hit = (d + i) % 2
            obs.append((day, 0.0, "neutral", hit, 1.0))
    res = walk_forward_calibration_reliability(
        obs, train_window_days=360, test_window_days=90, step_days=90,
        min_train_obs=100, min_obs_per_direction=10,
    )
    assert res["status"] == "ok"
    # No-signal data: OOS Brier cannot be much better than base-rate (~0.25).
    assert res["brier_oos"] > 0.2
