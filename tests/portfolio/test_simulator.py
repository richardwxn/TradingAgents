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
    # Position-level stop-loss column should be present in the table.
    assert "Stops hit" in md


# ---------- position-level stop-loss (Section 16) ----------


def _obs_bullish_with_atr(ticker: str, week: date, atr: float) -> WeeklyObservation:
    return WeeklyObservation(
        ticker=ticker, week=week, direction="bullish",
        composite=0.5, confidence=0.7, composite_age_weeks=0, atr_14=atr,
    )


def test_stop_fires_when_intra_week_min_pierces_level():
    """Entry $100, ATR=$5, multiple=1.5 → stop=$92.50. Min close $90 is
    below the stop, so the position exits at $92.50 (NOT at the next
    Friday's close, even if it's higher). n_stops_hit should be 1 for
    the entry week."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:2]
    obs = {weeks[0]: {"AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0)}, weeks[1]: {}}
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=weeks)
    # Mid-week min close pierces the $92.50 stop.
    min_close = {weeks[0]: {"AAA": 90.0}}
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
    )
    s0 = res.states[0]
    # Realized return = 0.10 * (92.50/100 - 1) = 0.10 * -0.075 = -0.0075
    assert s0.n_stops_hit == 1
    assert s0.realized_return == pytest.approx(0.10 * (92.5 / 100.0 - 1.0))
    assert res.metrics["n_stops_hit"] == 1


def test_stop_does_not_fire_when_min_stays_above_level():
    """ATR=$5, mult=1.5 → stop=$92.50. Min close $95 stays above, so the
    position rides to next Friday's close. No stop hit."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:2]
    obs = {weeks[0]: {"AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0)}, weeks[1]: {}}
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=weeks)
    min_close = {weeks[0]: {"AAA": 95.0}}  # stays above $92.50
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
    )
    s0 = res.states[0]
    assert s0.n_stops_hit == 0
    # Realized return = 0.10 * (110/100 - 1) = 0.01
    assert s0.realized_return == pytest.approx(0.01)
    assert res.metrics["n_stops_hit"] == 0


def test_stop_disabled_when_multiplier_zero():
    """`stop_loss_atr_multiple=0` disables stops entirely — even a deep
    intra-week pierce can't trigger an exit."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:2]
    obs = {weeks[0]: {"AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0)}, weeks[1]: {}}
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=weeks)
    min_close = {weeks[0]: {"AAA": 50.0}}  # massive intra-week dip
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=0.0,
        ),
        intra_week_min_close=min_close,
    )
    s0 = res.states[0]
    assert s0.n_stops_hit == 0
    # Realized return rides to next close: 0.10 * (110/100 - 1) = 0.01
    assert s0.realized_return == pytest.approx(0.01)
    assert res.metrics["n_stops_hit"] == 0


def test_stop_skipped_when_atr_missing():
    """If the observation doesn't carry an ATR, the stop check is a
    no-op even when min_close data would otherwise trigger one."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:2]
    obs = {weeks[0]: {"AAA": _obs_bullish("AAA", weeks[0])}, weeks[1]: {}}  # no atr_14
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=weeks)
    min_close = {weeks[0]: {"AAA": 50.0}}
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
    )
    assert res.states[0].n_stops_hit == 0
    assert res.states[0].realized_return == pytest.approx(0.01)


def test_stop_skipped_when_intra_week_data_absent():
    """If `intra_week_min_close` is None, no stops fire even with ATR
    present — the simulator needs the min data to evaluate."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:2]
    obs = {weeks[0]: {"AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0)}, weeks[1]: {}}
    prices = pd.DataFrame({"AAA": [100.0, 80.0]}, index=weeks)  # next close $80 (under stop)
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=None,
    )
    # Realized = 0.10 * (80/100 - 1) = -0.02 (no stop applied, full close-to-close move)
    assert res.states[0].n_stops_hit == 0
    assert res.states[0].realized_return == pytest.approx(-0.02)


def test_mixed_positions_some_stop_some_dont():
    """Two positions: AAA hits its stop, BBB does not. n_stops_hit=1 for
    that week and the realized return mixes the stop exit on AAA with
    next-Friday close on BBB."""
    sizing = SizingConfig(
        max_per_name=0.10, max_long_exposure=0.20, min_position_weight=0.0,
    )
    weeks = WEEKS_4[:2]
    obs = {
        weeks[0]: {
            "AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0),  # stop at $92.50
            "BBB": _obs_bullish_with_atr("BBB", weeks[0], atr=2.0),  # stop at $47
        },
        weeks[1]: {},
    }
    prices = pd.DataFrame(
        {"AAA": [100.0, 105.0], "BBB": [50.0, 52.0]}, index=weeks,
    )
    min_close = {weeks[0]: {"AAA": 90.0, "BBB": 49.0}}  # only AAA pierces
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
    )
    s0 = res.states[0]
    assert s0.n_stops_hit == 1
    # AAA: weight 0.10 * (92.50/100 - 1) = -0.0075
    # BBB: weight 0.10 * (52/50 - 1)     = +0.004
    expected = 0.10 * (92.5 / 100.0 - 1.0) + 0.10 * (52.0 / 50.0 - 1.0)
    assert s0.realized_return == pytest.approx(expected)


def test_stops_persisted_in_metrics_and_markdown():
    """n_stops_hit appears as a metric and renders into the comparison
    table."""
    sizing = SizingConfig(max_per_name=0.10, max_long_exposure=0.10, min_position_weight=0.0)
    weeks = WEEKS_4[:3]
    obs = {
        weeks[0]: {"AAA": _obs_bullish_with_atr("AAA", weeks[0], atr=5.0)},
        weeks[1]: {"AAA": _obs_bullish_with_atr("AAA", weeks[1], atr=5.0)},
        weeks[2]: {},
    }
    prices = pd.DataFrame({"AAA": [100.0, 105.0, 110.0]}, index=weeks)
    # Both weeks pierce the stop.
    min_close = {weeks[0]: {"AAA": 90.0}, weeks[1]: {"AAA": 95.0}}  # 95 < 105-7.5=97.5
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
        policy_name="stoptest",
    )
    assert res.metrics["n_stops_hit"] == 2
    md = render_policy_comparison_markdown([res], title="T")
    # Both the header and the stop count cell should be present.
    assert "Stops hit" in md
    assert " 2 " in md or md.rstrip().endswith("| 2 |")


def test_benchmark_only_run_never_applies_stops():
    """benchmark_only=True is a buy-and-hold of SPY; stops should not
    fire even if ATR + min_close would otherwise trigger them."""
    sizing = SizingConfig()
    weeks = WEEKS_4[:2]
    obs = {w: {} for w in weeks}
    prices = pd.DataFrame({"SPY": [400.0, 420.0]}, index=weeks)
    min_close = {weeks[0]: {"SPY": 100.0}}  # absurd dip
    res = run_simulation(
        weeks=weeks, observations=obs, prices=prices,
        sizing_config=sizing,
        sim_config=SimulationConfig(
            initial_capital=10_000.0, cost_per_side_bps=0.0,
            stop_loss_atr_multiple=1.5,
        ),
        intra_week_min_close=min_close,
        benchmark_only=True,
    )
    assert res.states[0].n_stops_hit == 0
    assert res.metrics["n_stops_hit"] == 0
