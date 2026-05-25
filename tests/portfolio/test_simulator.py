"""Unit tests for portfolio/simulator.py (pure functions).

Strategy: build a tiny known-input scenario (1 ticker, 4 weeks, known
prices and signals) and verify the equity curve matches what we can
compute by hand. Then a 2-ticker scenario verifies the cost model
charges turnover correctly when the policy reshuffles between names.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portfolio.simulator import (
    SimulationConfig,
    WeeklyObservation,
    compute_metrics,
    excess_vs_benchmark,
    render_policy_comparison_markdown,
    run_simulation,
)
from portfolio.sizing import SizingConfig


# ---------- fixtures ----------


WEEKS_4 = [date(2026, 1, 2), date(2026, 1, 9), date(2026, 1, 16), date(2026, 1, 23)]


def _obs_bullish(ticker: str, week: date) -> WeeklyObservation:
    return WeeklyObservation(
        ticker=ticker, week=week, direction="bullish",
        composite=0.5, confidence=0.7, composite_age_weeks=0,
    )


def _obs_neutral(ticker: str, week: date) -> WeeklyObservation:
    return WeeklyObservation(
        ticker=ticker, week=week, direction="neutral",
        composite=0.0, confidence=0.5, composite_age_weeks=0,
    )


# ---------- single-ticker known-arithmetic case ----------


def test_single_ticker_bullish_returns_match_hand_calc():
    """One ticker, always bullish, max_per_name=max_long=10% so target
    weight is 10% every week. Equity should grow by 0.10 * weekly_return
    (minus a tiny first-week turnover cost on the new 10% position)."""
    sizing = SizingConfig(
        max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0,
    )
    obs = {w: {"AAA": _obs_bullish("AAA", w)} for w in WEEKS_4}
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0, 108.9]},
        index=[d for d in WEEKS_4],
    )
    sim_cfg = SimulationConfig(initial_capital=10_000.0, cost_per_side_bps=5.0)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=sim_cfg, policy_name="test_single",
    )
    # Week 0: target 10% → turnover 10% → cost = 10000 * 0.10 * 0.0005 = $0.50.
    # Post-trade equity = 9999.50.
    # Week 0 → 1: AAA +10% → portfolio return = 0.10 * 0.10 = 0.01 → 9999.50 * 1.01 ≈ 10099.4975.
    s0 = res.states[0]
    assert s0.one_way_turnover == pytest.approx(0.10)
    assert s0.cost_paid == pytest.approx(0.50)
    assert s0.equity_post_trade == pytest.approx(10_000.0 * 1.01 - 0.50 * 1.01, rel=1e-6)
    # Week 1: target unchanged → no further turnover cost.
    s1 = res.states[1]
    assert s1.one_way_turnover == pytest.approx(0.0)
    assert s1.cost_paid == pytest.approx(0.0)


def test_neutral_signals_keep_equity_flat():
    """All neutral → 0% target → no realized return, no cost."""
    sizing = SizingConfig()
    obs = {w: {"AAA": _obs_neutral("AAA", w)} for w in WEEKS_4}
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 99.0, 108.9]}, index=WEEKS_4)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=SimulationConfig(initial_capital=10_000.0),
    )
    for s in res.states:
        assert s.one_way_turnover == 0.0
        assert s.equity_post_trade == 10_000.0


# ---------- cost model: turnover ----------


def test_reshuffle_between_names_charges_turnover_each_week():
    """Week 0 → AAA only; week 1 → BBB only. Turnover at week 1 =
    |0 - 0.10| + |0.10 - 0| = 0.20 → cost = equity * 0.20 * 5bps."""
    sizing = SizingConfig(
        max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0,
    )
    obs = {
        WEEKS_4[0]: {"AAA": _obs_bullish("AAA", WEEKS_4[0]), "BBB": _obs_neutral("BBB", WEEKS_4[0])},
        WEEKS_4[1]: {"AAA": _obs_neutral("AAA", WEEKS_4[1]), "BBB": _obs_bullish("BBB", WEEKS_4[1])},
        WEEKS_4[2]: {"AAA": _obs_neutral("AAA", WEEKS_4[2]), "BBB": _obs_bullish("BBB", WEEKS_4[2])},
    }
    weeks = WEEKS_4[:3]
    prices = pd.DataFrame(
        {"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]},
        index=weeks,
    )
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=SimulationConfig(initial_capital=10_000.0),
    )
    # Week 0 turnover: 10% (new AAA). Cost: 10000 * 0.10 * 0.0005 = $0.50.
    # Week 1 turnover: |0 - 0.10| (AAA dropped) + |0.10 - 0| (BBB added) = 0.20.
    #   Cost: equity_at_w1 * 0.20 * 0.0005.
    assert res.states[0].one_way_turnover == pytest.approx(0.10)
    assert res.states[1].one_way_turnover == pytest.approx(0.20)
    # Week 2 turnover: 0 (BBB unchanged).
    assert res.states[2].one_way_turnover == pytest.approx(0.0)


# ---------- benchmark-only mode ----------


def test_benchmark_only_buys_and_holds_spy():
    sizing = SizingConfig()
    obs = {w: {} for w in WEEKS_4}
    prices = pd.DataFrame({"SPY": [400.0, 420.0, 410.0, 430.5]}, index=WEEKS_4)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=SimulationConfig(initial_capital=10_000.0),
        benchmark_only=True,
    )
    # Week 0: 100% SPY → turnover 100% → cost = 10000 * 1.0 * 0.0005 = $5.
    # SPY 400 → 420 (+5%) → equity after = (10000 - 5) * 1.05.
    expected_w0 = (10_000.0 - 5.0) * (420.0 / 400.0)
    assert res.states[0].equity_post_trade == pytest.approx(expected_w0)
    # Week 1+: target unchanged → 0 turnover.
    for s in res.states[1:]:
        assert s.one_way_turnover == 0.0


# ---------- stale composites ----------


def test_stale_composite_is_neutralized():
    """If composite_age_weeks > stale_composite_weeks, the simulator
    treats the signal as neutral → no target allocation."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    stale_obs = WeeklyObservation(
        ticker="AAA", week=WEEKS_4[0], direction="bullish",
        composite=0.5, confidence=0.7, composite_age_weeks=5,  # too old
    )
    obs = {WEEKS_4[0]: {"AAA": stale_obs}, WEEKS_4[1]: {"AAA": stale_obs}}
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=WEEKS_4[:2])
    res = run_simulation(
        weeks=WEEKS_4[:2], observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(initial_capital=10_000.0, stale_composite_weeks=2),
    )
    # AAA is neutralized → weight 0 (key may or may not be present;
    # the contract is "no allocation", not "absent key").
    assert res.states[0].target_weights.get("AAA", 0.0) == 0.0
    assert res.states[0].equity_post_trade == 10_000.0


# ---------- metrics ----------


def test_compute_metrics_smoke():
    """Sanity: positive returns give positive CAGR, non-zero Sharpe."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    obs = {w: {"AAA": _obs_bullish("AAA", w)} for w in WEEKS_4}
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 133.1]}, index=WEEKS_4)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=SimulationConfig(initial_capital=10_000.0),
    )
    m = res.metrics
    assert m["end_equity"] > m["start_equity"]
    assert m["cagr"] > 0
    assert m["max_drawdown"] <= 0  # by definition <= 0
    assert m["n_weeks"] == 4


def test_excess_vs_benchmark_zero_when_identical_runs():
    """If portfolio and benchmark are the same SimulationResult, excess = 0."""
    sizing = SizingConfig()
    obs = {w: {} for w in WEEKS_4}
    prices = pd.DataFrame({"SPY": [400.0, 410.0, 415.0, 420.0]}, index=WEEKS_4)
    sim_cfg = SimulationConfig(initial_capital=10_000.0, cost_per_side_bps=0.0)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing, sim_config=sim_cfg, benchmark_only=True,
    )
    ex = excess_vs_benchmark(res, res)
    assert ex["excess_cagr"] == pytest.approx(0.0, abs=1e-9)
    assert ex["annualized_excess_return"] == pytest.approx(0.0, abs=1e-9)


# ---------- rendering ----------


def test_render_policy_comparison_markdown_lists_policies():
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    obs = {w: {"AAA": _obs_bullish("AAA", w)} for w in WEEKS_4}
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 99.0, 108.9]}, index=WEEKS_4)
    res = run_simulation(
        weeks=WEEKS_4, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(initial_capital=10_000.0),
        policy_name="alpha",
    )
    md = render_policy_comparison_markdown([res], title="Test", slice_label="4 weeks")
    assert "alpha" in md
    assert "CAGR" in md
    assert "Sharpe" in md
    assert "4 weeks" in md
