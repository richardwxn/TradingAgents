from __future__ import annotations

import math

import pytest

from tradingagents.analysis_only.forecast import (
    build_decision_summary,
    build_option_strategies,
    build_trade_plan,
    estimate_price_ranges,
    estimate_price_target_scenarios,
)


# ---------- estimate_price_ranges ----------


def _baseline_ranges(**overrides):
    args = dict(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.0,
        composite_score=0.0,
        implied_vol_annual=0.3,
    )
    args.update(overrides)
    return estimate_price_ranges(**args)


def test_returns_three_horizons():
    out = _baseline_ranges()
    assert set(out.keys()) == {"1w", "1m", "3m"}
    assert out["1w"]["trading_days"] == 5
    assert out["1m"]["trading_days"] == 21
    assert out["3m"]["trading_days"] == 63


def test_no_spot_returns_empty():
    assert estimate_price_ranges(
        spot=None,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.0,
        composite_score=0.0,
    ) == {}


def test_bands_widen_with_horizon():
    out = _baseline_ranges()
    width_1w = out["1w"]["upper_80"] - out["1w"]["lower_80"]
    width_1m = out["1m"]["upper_80"] - out["1m"]["lower_80"]
    width_3m = out["3m"]["upper_80"] - out["3m"]["lower_80"]
    assert width_1w < width_1m < width_3m


def test_band_widths_roughly_sqrt_of_time():
    out = _baseline_ranges()
    w5 = out["1w"]["upper_80"] - out["1w"]["lower_80"]
    w21 = out["1m"]["upper_80"] - out["1m"]["lower_80"]
    # With blended (vol+atr+iv) the ratio of widths should be close to sqrt(21/5).
    ratio = w21 / w5
    expected = math.sqrt(21 / 5)
    assert math.isclose(ratio, expected, rel_tol=0.02)


def test_z_score_ordering():
    out = _baseline_ranges()
    for label in ("1w", "1m", "3m"):
        b = out[label]
        # 60% band should be narrower than 80%, which is narrower than 95%.
        w60 = b["upper_60"] - b["lower_60"]
        w80 = b["upper_80"] - b["lower_80"]
        w95 = b["upper_95"] - b["lower_95"]
        assert w60 < w80 < w95


def test_event_multiplier_widens_bands():
    base = _baseline_ranges()
    bumped = _baseline_ranges(event_risk_multiplier=1.5)
    bw = base["1w"]["upper_80"] - base["1w"]["lower_80"]
    iw = bumped["1w"]["upper_80"] - bumped["1w"]["lower_80"]
    assert iw > bw
    assert math.isclose(iw / bw, 1.5, rel_tol=0.01)


def test_positive_drift_pushes_center_above_spot():
    out = estimate_price_ranges(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.05,  # +5% over 20d
        composite_score=1.0,
        implied_vol_annual=0.3,
    )
    assert out["1m"]["center_price"] > 100.0


def test_negative_drift_pushes_center_below_spot():
    out = estimate_price_ranges(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=-0.05,
        composite_score=-1.0,
        implied_vol_annual=0.3,
    )
    assert out["1m"]["center_price"] < 100.0


def test_drift_clamped_at_extreme_input():
    # Even with an absurd 50% 20d return, daily drift is capped at +/-0.6%.
    out = estimate_price_ranges(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.50,
        composite_score=1.0,
        implied_vol_annual=0.3,
    )
    # ~0.006 * 21 = ~0.126 drift max
    assert out["1m"]["center_price"] <= 100.0 * 1.20


def test_no_implied_vol_still_returns_bands():
    out = estimate_price_ranges(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.0,
        composite_score=0.0,
        implied_vol_annual=None,
    )
    assert out["1w"]["assumptions"]["implied_vol_annual"] is None
    assert out["1w"]["upper_80"] > out["1w"]["lower_80"]


def test_lower_bands_never_negative():
    out = estimate_price_ranges(
        spot=5.0,
        vol_daily=0.20,
        atr_14=1.0,
        ret_20d=-0.50,
        composite_score=-1.0,
        implied_vol_annual=2.0,
    )
    for b in out.values():
        assert b["lower_95"] >= 0.01


# ---------- build_trade_plan ----------


@pytest.fixture
def baseline_forecast():
    return estimate_price_ranges(
        spot=100.0,
        vol_daily=0.02,
        atr_14=2.0,
        ret_20d=0.0,
        composite_score=0.0,
        implied_vol_annual=0.3,
    )


def test_insufficient_data_when_no_spot(baseline_forecast):
    plan = build_trade_plan(
        direction="bullish",
        confidence=0.8,
        composite_score=0.5,
        spot=None,
        price_range_forecast=baseline_forecast,
    )
    assert plan["strategy_profile"] == "insufficient_data"
    assert plan["entry_strategy"] == []
    assert plan["exit_strategy"] == []


def test_bullish_plan_has_long_accumulation(baseline_forecast):
    plan = build_trade_plan(
        direction="bullish",
        confidence=0.8,
        composite_score=0.6,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    assert plan["strategy_profile"] == "long_accumulation"
    # Has a stop_loss exit.
    assert any(
        x["label"] == "stop_loss" and "trigger_price_lte" in x
        for x in plan["exit_strategy"]
    )


def test_bearish_plan_is_defensive(baseline_forecast):
    plan = build_trade_plan(
        direction="bearish",
        confidence=0.7,
        composite_score=-0.5,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    assert plan["strategy_profile"] == "defensive_reduce"
    # No new longs.
    assert plan["entry_strategy"][0]["allocation_pct_of_target_position"] == 0.0


def test_neutral_plan_is_small_range(baseline_forecast):
    plan = build_trade_plan(
        direction="neutral",
        confidence=0.55,
        composite_score=0.0,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    assert plan["strategy_profile"] == "range_trade_small_size"


def test_high_confidence_bullish_uses_larger_starter(baseline_forecast):
    high = build_trade_plan(
        direction="bullish",
        confidence=0.85,
        composite_score=0.6,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    low = build_trade_plan(
        direction="bullish",
        confidence=0.55,
        composite_score=0.3,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    high_starter = high["entry_strategy"][0]["allocation_pct_of_target_position"]
    low_starter = low["entry_strategy"][0]["allocation_pct_of_target_position"]
    assert high_starter > low_starter


def test_target_position_pct_in_band(baseline_forecast):
    plan = build_trade_plan(
        direction="bullish",
        confidence=0.95,
        composite_score=0.99,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    target = plan["position_sizing"]["target_position_pct_of_portfolio"]
    assert 4.0 <= target <= 30.0


# ---------- estimate_price_target_scenarios ----------


def _baseline_price_target(**overrides):
    args = dict(
        spot=100.0,
        direction="bullish",
        confidence=0.7,
        composite_score=0.5,
        fundamentals={
            "forward_pe": 20.0,
            "trailing_pe": 25.0,
            "revenue_growth": 0.20,
            "earnings_growth": 0.25,
            "profit_margins": 0.30,
        },
        technicals={"return_20d": 0.04},
        price_range_forecast=estimate_price_ranges(
            spot=100.0,
            vol_daily=0.02,
            atr_14=2.0,
            ret_20d=0.04,
            composite_score=0.5,
            implied_vol_annual=0.30,
        ),
        analyst_consensus={
            "status": "ok",
            "price_targets": {
                "low": 90.0,
                "mean": 130.0,
                "median": 128.0,
                "high": 160.0,
            },
            "analyst_count": 35,
            "consensus_pit_status": "live",
        },
        competitor_analysis={
            "peer_fundamentals_pit_status": "live",
            "summary": {
                "peer_count": 4,
                "peer_forward_pe_median": 24.0,
                "peer_trailing_pe_median": 30.0,
            },
        },
        earnings_calendar={
            "status": "ok",
            "earnings_in_30_days": False,
            "earnings_in_90_days": True,
            "forward_eps_estimates_pit_status": "live",
            "upcoming_earnings": [{"date": "2026-06-01", "eps_estimate": 5.50}],
        },
        market_context={
            "spy_return_20d": 0.02,
            "vix_level": 18.0,
            "fear_greed_rating": "Neutral",
            "fear_greed_score": 50,
            "fear_greed_pit_status": "pit",
        },
        industry_news_context={
            "status": "ok",
            "pit_status": "pit",
            "ranked_themes": [
                {"theme": "ai_accelerator_demand", "count": 4},
                {"theme": "capex_infrastructure", "count": 2},
                {"theme": "supply_chain_geopolitics", "count": 1},
            ],
        },
        options_flow={"atm_iv_30d": 0.30, "unusual_count": 0},
    )
    args.update(overrides)
    return estimate_price_target_scenarios(**args)


def test_price_target_returns_scenarios():
    out = _baseline_price_target()
    assert out["status"] == "ok"
    assert out["bear"] < out["base"] < out["bull"]
    assert out["base_upside_pct"] > 0
    assert out["time_horizon"] == "3m"
    assert out["source_weights"]


def test_price_target_uses_analyst_anchor():
    with_analyst = _baseline_price_target(
        analyst_consensus={
            "status": "ok",
            "price_targets": {
                "low": 120.0,
                "mean": 180.0,
                "median": 175.0,
                "high": 220.0,
            },
            "analyst_count": 35,
            "consensus_pit_status": "live",
        }
    )
    no_analyst = _baseline_price_target(
        analyst_consensus={"status": "unavailable"}
    )
    assert with_analyst["base"] > no_analyst["base"]


def test_price_target_handles_missing_spot():
    out = _baseline_price_target(spot=None)
    assert out["status"] == "unavailable"


def test_price_target_confidence_caps_with_low_coverage():
    out = _baseline_price_target(
        analyst_consensus={"status": "unavailable"},
        competitor_analysis={"summary": {}},
        earnings_calendar={"upcoming_earnings": []},
        price_range_forecast={},
    )
    assert out["confidence"] <= 0.45
    assert out["missing_data"]


# ---------- build_decision_summary ----------


def test_decision_summary_buy_for_bullish_upside(baseline_forecast):
    price_target = {
        "status": "ok",
        "base": 120.0,
        "bull": 140.0,
        "bear": 90.0,
        "base_upside_pct": 0.20,
        "confidence": 0.70,
    }
    plan = build_trade_plan(
        direction="bullish",
        confidence=0.75,
        composite_score=0.5,
        spot=100.0,
        price_range_forecast=baseline_forecast,
    )
    out = build_decision_summary(
        direction="bullish",
        confidence=0.75,
        composite_score=0.5,
        coverage=0.9,
        spot=100.0,
        price_target=price_target,
        price_range_forecast=baseline_forecast,
        trade_plan=plan,
        risk_flags=[],
    )
    assert out["status"] == "ok"
    assert out["action"] == "buy"
    assert out["estimated_win_probability"] > 0.5
    assert out["entry"]["starter_buy_at_or_below"] == 100.0


def test_decision_summary_hold_when_upside_too_small(baseline_forecast):
    out = build_decision_summary(
        direction="bullish",
        confidence=0.75,
        composite_score=0.5,
        coverage=0.9,
        spot=100.0,
        price_target={"base_upside_pct": 0.02, "confidence": 0.6},
        price_range_forecast=baseline_forecast,
        trade_plan={},
        risk_flags=[],
    )
    assert out["action"] == "hold"


def test_decision_summary_handles_missing_spot(baseline_forecast):
    out = build_decision_summary(
        direction="bullish",
        confidence=0.75,
        composite_score=0.5,
        coverage=0.9,
        spot=None,
        price_target={},
        price_range_forecast=baseline_forecast,
        trade_plan={},
        risk_flags=[],
    )
    assert out["status"] == "unavailable"


# ---------- build_option_strategies ----------


def _option_contracts():
    return [
        {
            "type": "put",
            "expiry": "2026-06-19",
            "dte": 26,
            "strike": 92.0,
            "bid": 2.8,
            "ask": 3.2,
            "mid": 3.0,
            "delta": -0.28,
            "open_interest": 1000,
            "volume": 200,
        },
        {
            "type": "call",
            "expiry": "2026-06-19",
            "dte": 26,
            "strike": 105.0,
            "bid": 3.8,
            "ask": 4.2,
            "mid": 4.0,
            "delta": 0.45,
            "open_interest": 1500,
            "volume": 250,
        },
        {
            "type": "call",
            "expiry": "2026-06-19",
            "dte": 26,
            "strike": 115.0,
            "bid": 1.3,
            "ask": 1.7,
            "mid": 1.5,
            "delta": 0.25,
            "open_interest": 1200,
            "volume": 180,
        },
        {
            "type": "call",
            "expiry": "2027-01-15",
            "dte": 236,
            "strike": 95.0,
            "bid": 18.5,
            "ask": 19.5,
            "mid": 19.0,
            "delta": 0.62,
            "open_interest": 900,
            "volume": 40,
        },
    ]


def _option_strategy_args(**overrides):
    args = {
        "spot": 100.0,
        "direction": "bullish",
        "decision_summary": {
            "entry": {
                "preferred_buy_zone": {"low": 90.0, "high": 94.0},
            },
            "exit": {"take_profit_1": 112.0},
        },
        "price_target": {"base": 118.0},
        "contracts": _option_contracts(),
        "portfolio_context": {
            "account": {
                "cash": 50_000.0,
                "margin_remaining": 200_000.0,
                "short_put_margin_utilization": 0.20,
            },
            "holding": {
                "has_position": True,
                "shares": 200.0,
                "average_cost": 100.0,
            },
        },
        "earnings_calendar": {"earnings_in_30_days": False},
    }
    args.update(overrides)
    return args


def test_option_strategies_include_four_core_candidates():
    out = build_option_strategies(**_option_strategy_args())
    assert out["status"] == "ok"
    assert {s["type"] for s in out["strategies"]} == {
        "sell_put",
        "sell_call",
        "buy_call_spread",
        "leap_call",
    }
    assert out["recommended"] in {"sell_put", "sell_call", "buy_call_spread", "leap_call"}


def test_covered_call_uses_cost_basis_breakeven_when_available():
    out = build_option_strategies(**_option_strategy_args())
    sell_call = next(s for s in out["strategies"] if s["type"] == "sell_call")
    assert sell_call["cost_basis_breakeven"] == 98.5
    assert sell_call["premium_adjusted_reference_price"] == 98.5
    assert sell_call["max_profit_basis"] == "vs_cost_basis"


def test_option_strategies_make_short_put_conditional_when_margin_tight():
    out = build_option_strategies(
        **_option_strategy_args(
            portfolio_context={
                "account": {
                    "cash": 50_000.0,
                    "margin_remaining": 40_000.0,
                    "short_put_margin_utilization": 0.90,
                },
                "holding": {"has_position": True, "shares": 200.0},
            }
        )
    )
    sell_put = next(s for s in out["strategies"] if s["type"] == "sell_put")
    assert out["capital_warning"] is True
    assert sell_put["verdict"] == "conditional"
    assert sell_put["requires_capital_action"] is True


def test_option_strategies_do_not_sell_naked_call_without_shares():
    out = build_option_strategies(
        **_option_strategy_args(
            portfolio_context={
                "account": {
                    "cash": 50_000.0,
                    "margin_remaining": 200_000.0,
                    "short_put_margin_utilization": 0.20,
                },
                "holding": {"has_position": False, "shares": 0.0},
            }
        )
    )
    sell_call = next(s for s in out["strategies"] if s["type"] == "sell_call")
    assert sell_call["verdict"] == "unavailable"
    assert "naked calls" in sell_call["reason"]


def test_option_strategies_wait_for_bullish_spread_when_not_bullish():
    out = build_option_strategies(
        **_option_strategy_args(direction="neutral")
    )
    spread = next(s for s in out["strategies"] if s["type"] == "buy_call_spread")
    assert spread["verdict"] == "wait"
