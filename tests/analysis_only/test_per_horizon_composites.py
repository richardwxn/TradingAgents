"""Tests for per-horizon composite emission (compute_composite_signed +
compute_per_horizon_composites).

The signed-weight composite must match the walk-forward gate's
`backtest.rebuild_records_with_weights`, and fallback horizons must stay
identical to the primary composite.
"""

import pytest

from tradingagents.analysis_only.scoring import (
    PER_HORIZON_WEIGHTS,
    compute_composite,
    compute_composite_signed,
    compute_per_horizon_composites,
)
from tradingagents.analysis_only.backtest import (
    BacktestRecord,
    rebuild_records_with_weights,
)


def _factor_scores():
    # Mixed signs / availability; one factor missing data.
    return [
        {"factor": "market_vix_regime", "pillar": "market", "score": 0.5,
         "weight": 0.03, "weighted_score": 0.015, "data_available": True},
        {"factor": "options_iv_skew", "pillar": "options", "score": -0.4,
         "weight": 0.0, "weighted_score": 0.0, "data_available": True},
        {"factor": "market_fear_greed_regime", "pillar": "market", "score": 0.2,
         "weight": 0.05, "weighted_score": 0.01, "data_available": True},
        {"factor": "peer_relative_valuation", "pillar": "valuation", "score": 0.8,
         "weight": 0.06, "weighted_score": 0.048, "data_available": True},
        {"factor": "momentum_rsi", "pillar": "momentum", "score": -0.6,
         "weight": 0.06, "weighted_score": -0.036, "data_available": True},
        {"factor": "trend_price_vs_sma20", "pillar": "technical", "score": 1.0,
         "weight": 0.08, "weighted_score": 0.0, "data_available": False},
    ]


def test_signed_composite_matches_gate_rebuild():
    """compute_composite_signed must equal rebuild_records_with_weights."""
    fs = _factor_scores()
    weights = PER_HORIZON_WEIGHTS["ret_20d"]
    mine = compute_composite_signed(fs, weights)["composite_score"]

    rec = BacktestRecord(
        symbol="X", as_of_date="2026-01-01", direction="", confidence=0.0,
        composite_score=0.0, forward_returns={}, factor_scores=fs,
    )
    gate = rebuild_records_with_weights([rec], weights=weights)[0].composite_score
    assert mine == round(gate, 4)


def test_signed_composite_all_positive_equals_recompute():
    """With positive weights, the signed form equals score·weight aggregation."""
    fs = _factor_scores()
    weights = {f["factor"]: abs(f["weight"]) for f in fs if f["weight"]}
    signed = compute_composite_signed(fs, weights)["composite_score"]
    # Hand recompute: sum(score*w)/sum(w) over available, weight>0.
    num = sum(f["score"] * weights[f["factor"]] for f in fs
              if f["data_available"] and weights.get(f["factor"]))
    den = sum(weights[f["factor"]] for f in fs
              if f["data_available"] and weights.get(f["factor"]))
    assert signed == round(max(-1.0, min(1.0, num / den)), 4)


def test_negative_weight_inverts_contribution():
    fs = [
        {"factor": "a", "score": 0.8, "weight": 1.0, "data_available": True},
    ]
    pos = compute_composite_signed(fs, {"a": 0.1})["composite_score"]
    neg = compute_composite_signed(fs, {"a": -0.1})["composite_score"]
    assert pos == pytest.approx(0.8)
    assert neg == pytest.approx(-0.8)


def test_zero_and_missing_weights_excluded():
    fs = _factor_scores()
    res = compute_composite_signed(fs, {"market_vix_regime": 0.1, "options_iv_skew": 0.0})
    # options_iv_skew has weight 0 -> excluded; only 1 factor counts.
    assert res["n_factors"] == 1


def test_per_horizon_fallback_reuses_primary_verbatim():
    fs = _factor_scores()
    gw = {f["factor"]: abs(f["weight"]) for f in fs if f["weight"]}
    primary = 0.1234
    out = compute_per_horizon_composites(fs, gw, global_composite=primary)
    assert set(out) == {"ret_5d", "ret_20d", "ret_60d"}
    # 5d / 60d reuse the primary exactly; 20d is the override.
    assert out["ret_5d"]["composite_score"] == primary
    assert out["ret_60d"]["composite_score"] == primary
    assert out["ret_5d"]["weight_source"] == "global"
    assert out["ret_20d"]["weight_source"] == "per_horizon"


def test_per_horizon_20d_differs_from_primary():
    fs = _factor_scores()
    gw = {f["factor"]: abs(f["weight"]) for f in fs if f["weight"]}
    primary = compute_composite(fs, weights=gw)["composite_score"]
    out = compute_per_horizon_composites(fs, gw, global_composite=primary)
    # The 20d override should generally move the composite off the primary.
    assert out["ret_20d"]["composite_score"] != primary


def test_per_horizon_without_global_composite_computes_fallback():
    fs = _factor_scores()
    gw = {f["factor"]: abs(f["weight"]) for f in fs if f["weight"]}
    out = compute_per_horizon_composites(fs, gw)  # no global_composite
    # Falls back to computing via signed form; equals positive-weight aggregate.
    expected = compute_composite_signed(fs, gw)["composite_score"]
    assert out["ret_60d"]["composite_score"] == expected
