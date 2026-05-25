from __future__ import annotations

import math

import pytest

from tradingagents.analysis_only.options_iv import (
    compute_iv_history_features,
    compute_iv_surface,
)


def _contract(
    *,
    side: str,
    strike: float,
    dte: int,
    iv: float | None,
    spot: float = 100.0,
    delta: float | None = None,
    mid: float | None = None,
    expiry: str | None = None,
) -> dict:
    if expiry is None:
        expiry = f"2026-01-{1 + dte:02d}"[:10]
    return {
        "type": side,
        "strike": strike,
        "dte": dte,
        "expiry": expiry,
        "implied_volatility": iv,
        "delta": delta,
        "mid": mid,
        "bid": mid,
        "ask": mid,
        "last": mid,
        "spot_distance_pct": (strike - spot) / spot if spot else None,
    }


# ---------- status / edge cases ----------


def test_returns_unavailable_when_spot_missing():
    out = compute_iv_surface(
        contracts=[_contract(side="call", strike=100, dte=30, iv=0.30)],
        spot=None,
        realized_vol_daily_20d=0.02,
    )
    assert out["status"] == "unavailable"
    assert "Spot" in out["reason"]


def test_returns_unavailable_when_contracts_empty():
    out = compute_iv_surface(
        contracts=[],
        spot=100.0,
        realized_vol_daily_20d=0.02,
    )
    assert out["status"] == "unavailable"
    assert "empty" in out["reason"].lower()


def test_returns_unavailable_when_no_iv_in_any_tenor():
    contracts = [
        _contract(side="call", strike=100, dte=200, iv=None),
        _contract(side="put", strike=100, dte=200, iv=None),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=0.02,
    )
    assert out["status"] == "unavailable"
    # All fields still present in shape.
    assert "atm_iv_30d" in out
    assert out["atm_iv_30d"] is None


def test_always_returns_full_field_shape():
    out = compute_iv_surface(
        contracts=[],
        spot=None,
        realized_vol_daily_20d=None,
    )
    for k in (
        "status",
        "atm_iv_30d",
        "atm_iv_60d",
        "atm_iv_90d",
        "term_structure_slope_30_to_60",
        "term_structure_is_backwardation",
        "skew_25d_30d",
        "skew_25d_30d_put_iv",
        "skew_25d_30d_call_iv",
        "realized_vol_annual_20d",
        "implied_realized_ratio",
        "implied_realized_signal",
        "earnings_implied_move",
        "earnings_implied_move_expiry",
    ):
        assert k in out


# ---------- ATM IV per tenor ----------


def test_atm_iv_30d_averages_call_and_put_at_nearest_strike():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="2026-01-30"),
        _contract(side="put", strike=100, dte=30, iv=0.32, expiry="2026-01-30"),
        # Far OTM should be ignored.
        _contract(side="call", strike=130, dte=30, iv=0.55, expiry="2026-01-30"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["status"] == "ok"
    assert math.isclose(out["atm_iv_30d"], 0.31, abs_tol=1e-6)


def test_atm_iv_picks_strike_closest_to_spot_when_no_exact_match():
    contracts = [
        _contract(side="call", strike=98, dte=30, iv=0.28, expiry="2026-01-30"),
        _contract(side="put", strike=98, dte=30, iv=0.30, expiry="2026-01-30"),
        _contract(side="call", strike=103, dte=30, iv=0.50, expiry="2026-01-30"),
        _contract(side="put", strike=103, dte=30, iv=0.50, expiry="2026-01-30"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    # 98 is closer (|−2|) than 103 (|+3|); 5% band keeps both eligible.
    assert math.isclose(out["atm_iv_30d"], 0.29, abs_tol=1e-6)


def test_atm_iv_skips_strikes_outside_atm_band():
    contracts = [
        _contract(side="call", strike=120, dte=30, iv=0.80, expiry="2026-01-30"),
        _contract(side="put", strike=80, dte=30, iv=0.80, expiry="2026-01-30"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    # Both 20% from spot, outside the ±5% ATM band.
    assert out["atm_iv_30d"] is None


def test_atm_iv_works_with_only_one_side_present():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.35, expiry="2026-01-30"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["status"] == "ok"
    assert math.isclose(out["atm_iv_30d"], 0.35, abs_tol=1e-6)


def test_three_tenors_populate_independently():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="2026-01-30"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="2026-01-30"),
        _contract(side="call", strike=100, dte=60, iv=0.35, expiry="2026-02-28"),
        _contract(side="put", strike=100, dte=60, iv=0.35, expiry="2026-02-28"),
        _contract(side="call", strike=100, dte=90, iv=0.40, expiry="2026-03-30"),
        _contract(side="put", strike=100, dte=90, iv=0.40, expiry="2026-03-30"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["atm_iv_30d"] == 0.30
    assert out["atm_iv_60d"] == 0.35
    assert out["atm_iv_90d"] == 0.40


# ---------- term structure ----------


def test_term_structure_upward_slope():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="call", strike=100, dte=60, iv=0.36, expiry="B"),
        _contract(side="put", strike=100, dte=60, iv=0.36, expiry="B"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert math.isclose(
        out["term_structure_slope_30_to_60"], (0.36 - 0.30) / 0.30, abs_tol=1e-6
    )
    assert out["term_structure_is_backwardation"] is False


def test_term_structure_backwardation_detected():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.50, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.50, expiry="A"),
        _contract(side="call", strike=100, dte=60, iv=0.40, expiry="B"),
        _contract(side="put", strike=100, dte=60, iv=0.40, expiry="B"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["term_structure_is_backwardation"] is True
    assert out["term_structure_slope_30_to_60"] < 0


def test_term_structure_null_when_60d_missing():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["term_structure_slope_30_to_60"] is None
    assert out["term_structure_is_backwardation"] is None


# ---------- 25-delta skew ----------


def test_skew_with_greeks_picks_25_delta_contracts():
    contracts = [
        # 30d ATM pair for the atm_iv_30d computation.
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A", delta=0.50),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A", delta=-0.50),
        # 25-delta OTM call (delta=+0.25) and 25-delta put (delta=-0.25).
        _contract(side="call", strike=110, dte=30, iv=0.28, expiry="A", delta=0.25),
        _contract(side="put", strike=90, dte=30, iv=0.40, expiry="A", delta=-0.25),
        # Distractor far-OTM.
        _contract(side="call", strike=130, dte=30, iv=0.60, expiry="A", delta=0.05),
        _contract(side="put", strike=70, dte=30, iv=0.80, expiry="A", delta=-0.05),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert math.isclose(out["skew_25d_30d_call_iv"], 0.28, abs_tol=1e-6)
    assert math.isclose(out["skew_25d_30d_put_iv"], 0.40, abs_tol=1e-6)
    assert math.isclose(out["skew_25d_30d"], 0.12, abs_tol=1e-6)


def test_skew_falls_back_to_strike_when_greeks_missing():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="call", strike=105, dte=30, iv=0.25, expiry="A"),  # +5% strike
        _contract(side="put", strike=95, dte=30, iv=0.34, expiry="A"),   # −5% strike
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert math.isclose(out["skew_25d_30d_call_iv"], 0.25, abs_tol=1e-6)
    assert math.isclose(out["skew_25d_30d_put_iv"], 0.34, abs_tol=1e-6)
    assert math.isclose(out["skew_25d_30d"], 0.09, abs_tol=1e-6)


def test_skew_null_when_one_side_unavailable():
    contracts = [
        _contract(side="call", strike=105, dte=30, iv=0.30, expiry="A", delta=0.25),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["skew_25d_30d"] is None
    assert out["skew_25d_30d_call_iv"] == 0.30
    assert out["skew_25d_30d_put_iv"] is None


# ---------- implied vs realized ----------


def test_implied_realized_ratio_rich():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.60, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.60, expiry="A"),
    ]
    # Daily realized vol = 0.015 → annualized ≈ 0.238 → ratio ≈ 2.52 → rich.
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=0.015,
    )
    assert out["implied_realized_signal"] == "iv_rich"
    assert out["implied_realized_ratio"] > 1.3


def test_implied_realized_ratio_cheap():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.10, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.10, expiry="A"),
    ]
    # Daily realized = 0.03 → annualized ≈ 0.476 → ratio ≈ 0.21 → cheap.
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=0.03,
    )
    assert out["implied_realized_signal"] == "iv_cheap"
    assert out["implied_realized_ratio"] < 0.9


def test_implied_realized_ratio_neutral_band():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A"),
    ]
    # Daily realized = 0.0175 → annualized ≈ 0.278 → ratio ≈ 1.08 → neutral.
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=0.0175,
    )
    assert out["implied_realized_signal"] == "neutral"
    assert 0.9 < out["implied_realized_ratio"] < 1.3


def test_realized_block_null_when_realized_missing():
    contracts = [
        _contract(side="call", strike=100, dte=30, iv=0.30, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.30, expiry="A"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
    )
    assert out["realized_vol_annual_20d"] is None
    assert out["implied_realized_ratio"] is None
    assert out["implied_realized_signal"] is None


# ---------- earnings implied move ----------


def test_earnings_implied_move_from_atm_straddle():
    contracts = [
        # Front-month expiry just after earnings.
        _contract(
            side="call", strike=100, dte=12, iv=0.55,
            expiry="2026-02-06", mid=4.5,
        ),
        _contract(
            side="put", strike=100, dte=12, iv=0.55,
            expiry="2026-02-06", mid=4.0,
        ),
        # 30d ATM pair so the surface has ok status.
        _contract(side="call", strike=100, dte=30, iv=0.40, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.40, expiry="A"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
        earnings_in_30_days=True,
        next_earnings_dte=10,
    )
    # Straddle 4.5 + 4.0 = 8.5; 8.5 / 100 = 0.085.
    assert math.isclose(out["earnings_implied_move"], 0.085, abs_tol=1e-6)
    assert out["earnings_implied_move_expiry"] == "2026-02-06"


def test_earnings_implied_move_null_when_no_earnings_flag():
    contracts = [
        _contract(
            side="call", strike=100, dte=12, iv=0.55,
            expiry="2026-02-06", mid=4.5,
        ),
        _contract(
            side="put", strike=100, dte=12, iv=0.55,
            expiry="2026-02-06", mid=4.0,
        ),
        _contract(side="call", strike=100, dte=30, iv=0.40, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.40, expiry="A"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
        earnings_in_30_days=False,
    )
    assert out["earnings_implied_move"] is None


def test_earnings_implied_move_null_when_no_paired_atm_at_event():
    # Only a call near the event window — straddle requires both legs.
    contracts = [
        _contract(
            side="call", strike=100, dte=12, iv=0.55,
            expiry="2026-02-06", mid=4.5,
        ),
        _contract(side="call", strike=100, dte=30, iv=0.40, expiry="A"),
        _contract(side="put", strike=100, dte=30, iv=0.40, expiry="A"),
    ]
    out = compute_iv_surface(
        contracts=contracts,
        spot=100.0,
        realized_vol_daily_20d=None,
        earnings_in_30_days=True,
        next_earnings_dte=10,
    )
    assert out["earnings_implied_move"] is None


# ---------- IV history features ----------


def test_iv_history_unavailable_when_current_missing():
    out = compute_iv_history_features(
        current_atm_iv_30d=None,
        history=[{"atm_iv_30d": 0.30}] * 25,
    )
    assert out["iv_history_status"] == "unavailable"
    assert out["iv_rank_252d"] is None
    assert out["iv_percentile_252d"] is None


def test_iv_history_insufficient_when_below_min_observations():
    out = compute_iv_history_features(
        current_atm_iv_30d=0.30,
        history=[{"atm_iv_30d": 0.25}] * 5,
        min_observations=20,
    )
    assert out["iv_history_status"] == "insufficient_history"
    assert out["iv_history_observations"] == 5
    assert out["iv_rank_252d"] is None


def test_iv_rank_at_top_of_range_is_one():
    history = [{"atm_iv_30d": 0.20 + 0.01 * i} for i in range(25)]
    out = compute_iv_history_features(
        current_atm_iv_30d=0.50,  # above the historical max of 0.44
        history=history,
        min_observations=20,
    )
    assert out["iv_history_status"] == "ok"
    assert out["iv_rank_252d"] == 1.0
    assert out["iv_percentile_252d"] == 1.0


def test_iv_rank_at_bottom_of_range_is_zero():
    history = [{"atm_iv_30d": 0.20 + 0.01 * i} for i in range(25)]
    out = compute_iv_history_features(
        current_atm_iv_30d=0.10,  # below the historical min of 0.20
        history=history,
        min_observations=20,
    )
    assert out["iv_rank_252d"] == 0.0
    assert out["iv_percentile_252d"] == 0.0


def test_iv_rank_at_midpoint():
    # Use ints/100 to avoid float arithmetic in test data.
    history = [{"atm_iv_30d": v / 100} for v in range(20, 41)]
    # min=0.20, max=0.40, midpoint between equals 0.30.
    out = compute_iv_history_features(
        current_atm_iv_30d=0.30,
        history=history,
        min_observations=20,
    )
    assert math.isclose(out["iv_rank_252d"], 0.5, abs_tol=1e-4)
    # 11 of 21 values (0.20..0.30 inclusive) are ≤ 0.30.
    assert math.isclose(out["iv_percentile_252d"], 11 / 21, abs_tol=1e-4)


def test_iv_history_rank_is_half_when_all_equal():
    history = [{"atm_iv_30d": 0.30}] * 25
    out = compute_iv_history_features(
        current_atm_iv_30d=0.30,
        history=history,
        min_observations=20,
    )
    # No range to rank against — fall back to 0.5 (neutral).
    assert out["iv_rank_252d"] == 0.5
    # All values equal current → 100th percentile.
    assert out["iv_percentile_252d"] == 1.0


def test_iv_history_ignores_null_iv_rows():
    history = [{"atm_iv_30d": None}] * 30 + [{"atm_iv_30d": 0.30}] * 25
    out = compute_iv_history_features(
        current_atm_iv_30d=0.30,
        history=history,
        min_observations=20,
    )
    # Nulls drop out; only 25 usable observations remain.
    assert out["iv_history_observations"] == 25
    assert out["iv_history_status"] == "ok"
