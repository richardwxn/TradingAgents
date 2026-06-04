from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portfolio.execution_simulator import (
    ExecutionPolicyCandidate,
    ExecutionSignal,
    default_policy_grid,
    generate_walk_forward_windows,
    render_execution_comparison_markdown,
    summarize_walk_forward_acceptance,
    run_daily_execution_simulation,
)
from portfolio.sizing import SizingConfig
from portfolio_execution_simulate import write_execution_artifacts, write_walk_forward_artifacts


D0 = date(2026, 1, 1)
D1 = date(2026, 1, 2)
D2 = date(2026, 1, 5)
D3 = date(2026, 1, 6)


def _config() -> SizingConfig:
    return SizingConfig(
        policy="equal_weight_bullish",
        max_per_name=0.50,
        max_long_exposure=0.50,
        min_position_weight=0.0,
        universe=("AAA",),
    )


def _signal(day: date, direction: str = "bullish", symbol: str = "AAA") -> ExecutionSignal:
    return ExecutionSignal(
        symbol=symbol,
        as_of_date=day,
        direction=direction,
        composite=0.5 if direction == "bullish" else 0.0,
        confidence=0.7,
    )


def _bars(values: list[tuple[date, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [v[1] for v in values],
            "high": [v[2] for v in values],
            "low": [v[3] for v in values],
            "close": [v[4] for v in values],
            "volume": [1000 for _ in values],
        },
        index=pd.to_datetime([v[0] for v in values]),
    )


def test_buy_limit_fills_only_when_daily_low_reaches_limit():
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="heuristic",
        stop_atr_multiple=None,
    )
    signals = {D0: {"AAA": _signal(D0)}}
    fill_prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 102, 103, 99, 101),
        (D2, 101, 102, 100, 101),
    ])}
    filled = run_daily_execution_simulation(
        dates=[D0, D1, D2],
        signals_by_date=signals,
        prices=fill_prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert filled.fills[0].side == "buy"
    assert filled.fills[0].price == pytest.approx(100.0)

    miss_prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 102, 103, 101, 102),
        (D2, 102, 103, 101, 102),
    ])}
    missed = run_daily_execution_simulation(
        dates=[D0, D1],
        signals_by_date=signals,
        prices=miss_prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert missed.fills == []
    assert missed.missed_orders[0].reason == "entry_limit_not_reached"


def test_trim_limit_fills_only_when_daily_high_reaches_limit():
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        exit_mode="atr_limit",
        exit_patience_atrs=1.0,
        stop_atr_multiple=None,
    )
    signals = {
        D0: {"AAA": _signal(D0, "bullish")},
        D1: {"AAA": _signal(D1, "neutral")},
    }
    prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 100, 101, 99, 100),
        (D2, 100, 103, 99, 101),
        (D3, 101, 102, 99, 100),
    ])}
    result = run_daily_execution_simulation(
        dates=[D0, D1, D2, D3],
        signals_by_date=signals,
        prices=prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert [f.side for f in result.fills] == ["buy", "sell"]
    assert result.fills[1].reason == "rebalance_exit"
    assert result.fills[1].price == pytest.approx(102.0)

    no_exit_prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 100, 101, 99, 100),
        (D2, 100, 101, 99, 100),
    ])}
    missed = run_daily_execution_simulation(
        dates=[D0, D1, D2],
        signals_by_date=signals,
        prices=no_exit_prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert [f.side for f in missed.fills] == ["buy"]
    assert missed.missed_orders[-1].reason == "exit_limit_not_reached"


def test_stop_loss_triggers_before_normal_rebalance_exit():
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        exit_mode="close",
        stop_atr_multiple=1.5,
    )
    signals = {
        D0: {"AAA": _signal(D0, "bullish")},
        D1: {"AAA": _signal(D1, "neutral")},
    }
    prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 100, 101, 99, 100),
        (D2, 100, 105, 89, 104),
    ])}
    result = run_daily_execution_simulation(
        dates=[D0, D1, D2],
        signals_by_date=signals,
        prices=prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert [f.reason for f in result.fills] == ["entry", "stop_loss"]
    assert result.metrics["n_stop_outs"] == 1
    assert result.states[-1].positions == {}


def test_missing_ohlc_skips_fill_without_fabricating_price():
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        stop_atr_multiple=None,
    )
    prices = {"AAA": _bars([(D0, 100, 101, 99, 100), (D2, 101, 102, 100, 101)])}
    result = run_daily_execution_simulation(
        dates=[D0, D1],
        signals_by_date={D0: {"AAA": _signal(D0)}},
        prices=prices,
        base_config=_config(),
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    assert result.fills == []
    assert result.states[-1].equity == pytest.approx(100_000.0)


def test_turnover_and_cost_match_hand_calculation():
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        stop_atr_multiple=None,
    )
    prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 100, 101, 99, 100),
        (D2, 100, 101, 99, 100),
    ])}
    result = run_daily_execution_simulation(
        dates=[D0, D1, D2],
        signals_by_date={D0: {"AAA": _signal(D0)}},
        prices=prices,
        base_config=_config(),
        candidate=candidate,
        initial_capital=10_000.0,
        cost_per_side_bps=5.0,
    )
    buy = result.fills[0]
    assert buy.quantity == 50
    assert buy.notional == pytest.approx(5_000.0)
    assert buy.cost_paid == pytest.approx(2.5)
    assert result.states[1].one_way_turnover == pytest.approx(0.5)
    assert result.states[1].equity == pytest.approx(9_997.5)


def test_candidate_grid_can_rank_better_execution_policy_above_baseline():
    cfg = _config()
    prices = {"AAA": _bars([
        (D0, 100, 101, 99, 100),
        (D1, 100, 112, 101, 110),  # baseline limit 100 misses; open fills.
        (D2, 110, 121, 109, 120),
    ])}
    signals = {D0: {"AAA": _signal(D0)}}
    candidates = default_policy_grid(cfg, mode="small")
    results = [
        run_daily_execution_simulation(
            dates=[D0, D1, D2],
            signals_by_date=signals,
            prices=prices,
            base_config=cfg,
            candidate=c,
            cost_per_side_bps=0.0,
        )
        for c in candidates
    ]
    baseline = next(r for r in results if r.policy.name == "baseline_current_config")
    open_entry = next(r for r in results if r.policy.name == "entry_open")
    assert open_entry.metrics["end_equity"] > baseline.metrics["end_equity"]


def test_cli_artifact_writer_outputs_markdown_and_json(tmp_path):
    cfg = _config()
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        stop_atr_multiple=None,
    )
    prices = {"AAA": _bars([(D0, 100, 101, 99, 100), (D1, 100, 101, 99, 100)])}
    result = run_daily_execution_simulation(
        dates=[D0, D1],
        signals_by_date={D0: {"AAA": _signal(D0)}},
        prices=prices,
        base_config=cfg,
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    write_execution_artifacts(
        results=[result],
        benchmark=None,
        output_dir=tmp_path,
        max_markdown_rows=10,
    )
    assert (tmp_path / "policy_comparison.md").exists()
    assert (tmp_path / "sim_baseline_current_config.json").exists()
    md = (tmp_path / "policy_comparison.md").read_text()
    assert "Action/execution policy comparison" in md
    assert "baseline_current_config" in md


def test_render_execution_comparison_contains_required_metrics():
    cfg = _config()
    candidate = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        stop_atr_multiple=None,
    )
    prices = {"AAA": _bars([(D0, 100, 101, 99, 100), (D1, 100, 101, 99, 100)])}
    result = run_daily_execution_simulation(
        dates=[D0, D1],
        signals_by_date={D0: {"AAA": _signal(D0)}},
        prices=prices,
        base_config=cfg,
        candidate=candidate,
        cost_per_side_bps=0.0,
    )
    md = render_execution_comparison_markdown([result])
    assert "Missed" in md
    assert "Stops" in md
    assert "Turnover" in md


def test_walk_forward_acceptance_requires_window_win_rate(tmp_path):
    dates = [date(2026, 1, day) for day in range(1, 8)]
    windows = generate_walk_forward_windows(
        dates,
        train_days=2,
        test_days=2,
        step_days=2,
    )
    assert len(windows) == 2

    cfg = _config()
    baseline = ExecutionPolicyCandidate(
        name="baseline_current_config",
        sizing_policy="equal_weight_bullish",
        entry_mode="heuristic",
        stop_atr_multiple=None,
    )
    open_entry = ExecutionPolicyCandidate(
        name="entry_open",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        stop_atr_multiple=None,
    )
    prices = {"AAA": _bars([
        (date(2026, 1, 1), 100, 101, 99, 100),
        (date(2026, 1, 2), 100, 101, 99, 100),
        (date(2026, 1, 3), 100, 112, 101, 110),
        (date(2026, 1, 4), 110, 121, 109, 120),
        (date(2026, 1, 5), 120, 130, 119, 128),
        (date(2026, 1, 6), 128, 140, 127, 138),
        (date(2026, 1, 7), 138, 150, 137, 148),
    ])}
    signals = {date(2026, 1, 1): {"AAA": _signal(date(2026, 1, 1))}}
    window_results = []
    for window in windows:
        window_results.append({
            "window": window,
            "results": [
                run_daily_execution_simulation(
                    dates=window["test_dates"],
                    signals_by_date=signals,
                    prices=prices,
                    base_config=cfg,
                    candidate=c,
                    cost_per_side_bps=0.0,
                )
                for c in (baseline, open_entry)
            ],
        })
    summaries = summarize_walk_forward_acceptance(window_results)
    entry = next(s for s in summaries if s["policy_name"] == "entry_open")
    assert entry["n_windows"] == 2
    assert entry["win_rate"] >= 0.5

    write_walk_forward_artifacts(
        summaries=summaries,
        output_dir=tmp_path,
        max_markdown_rows=10,
    )
    assert (tmp_path / "walk_forward_acceptance.json").exists()
    assert (tmp_path / "walk_forward_acceptance.md").exists()
