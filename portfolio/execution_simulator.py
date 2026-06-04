"""Daily-OHLC execution simulator for the action layer.

This module validates the layer that turns calibrated signals into
practical actions: target weights, entry/exit prices, stop reminders,
fills, missed orders, turnover, costs, and realized equity curves.

It is deliberately pure and I/O-free. The root-level
``portfolio_execution_simulate.py`` CLI handles report loading and price
fetching, then delegates to these helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Mapping

import pandas as pd

from portfolio.sizing import (
    SizingConfig,
    buy_limit_price,
    compute_target_weights,
    stop_loss_price,
    trim_limit_price,
)


ENTRY_MODES = ("heuristic", "open", "close", "pullback_pct")
EXIT_MODES = ("heuristic", "open", "close", "atr_limit")


@dataclass(frozen=True)
class ExecutionSignal:
    symbol: str
    as_of_date: date
    direction: str
    composite: float | None
    confidence: float | None


@dataclass(frozen=True)
class ExecutionPolicyCandidate:
    """One concrete policy to validate against daily OHLC bars."""

    name: str
    sizing_policy: str
    entry_mode: str = "heuristic"
    max_entry_pullback_pct: float = 0.05
    exit_mode: str = "heuristic"
    exit_patience_atrs: float = 1.0
    stop_atr_multiple: float | None = 1.5
    stale_signal_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.entry_mode not in ENTRY_MODES:
            raise ValueError(f"unsupported entry_mode {self.entry_mode!r}")
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"unsupported exit_mode {self.exit_mode!r}")
        if not 0 <= self.stale_signal_decay <= 1:
            raise ValueError("stale_signal_decay must be in [0, 1]")
        if self.stop_atr_multiple is not None and self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be positive or None")


@dataclass(frozen=True)
class ExecutionFill:
    date: date
    symbol: str
    side: str
    reason: str
    quantity: int
    price: float
    notional: float
    cost_paid: float


@dataclass(frozen=True)
class MissedOrder:
    date: date
    symbol: str
    side: str
    reason: str
    quantity: int
    limit_price: float
    close_price: float | None


@dataclass(frozen=True)
class DailyExecutionState:
    date: date
    equity: float
    cash: float
    positions: dict[str, int]
    target_weights: dict[str, float]
    fills: list[ExecutionFill] = field(default_factory=list)
    missed_orders: list[MissedOrder] = field(default_factory=list)
    stop_outs: int = 0
    one_way_turnover: float = 0.0
    cost_paid: float = 0.0
    daily_return: float = 0.0


@dataclass(frozen=True)
class DailyExecutionResult:
    policy: ExecutionPolicyCandidate
    config: SizingConfig
    states: list[DailyExecutionState]
    fills: list[ExecutionFill]
    missed_orders: list[MissedOrder]
    metrics: dict[str, float]

    def equity_curve(self) -> pd.Series:
        return pd.Series(
            [s.equity for s in self.states],
            index=pd.to_datetime([s.date for s in self.states]),
            name=self.policy.name,
        )


def candidate_from_sizing_config(
    config: SizingConfig,
    *,
    name: str = "baseline_current_config",
) -> ExecutionPolicyCandidate:
    return ExecutionPolicyCandidate(
        name=name,
        sizing_policy=config.policy,
        entry_mode="heuristic",
        max_entry_pullback_pct=config.max_entry_pullback_pct,
        exit_mode="heuristic",
        exit_patience_atrs=config.exit_patience_atrs,
        stop_atr_multiple=config.stop_loss_atr_multiple,
        stale_signal_decay=config.stale_signal_decay,
    )


def sizing_for_candidate(
    base: SizingConfig,
    candidate: ExecutionPolicyCandidate,
) -> SizingConfig:
    return replace(
        base,
        policy=candidate.sizing_policy,
        max_entry_pullback_pct=candidate.max_entry_pullback_pct,
        exit_patience_atrs=candidate.exit_patience_atrs,
        stop_loss_atr_multiple=(
            candidate.stop_atr_multiple
            if candidate.stop_atr_multiple is not None
            else base.stop_loss_atr_multiple
        ),
        stale_signal_decay=candidate.stale_signal_decay,
    )


def default_policy_grid(
    base: SizingConfig,
    *,
    mode: str = "full",
) -> list[ExecutionPolicyCandidate]:
    """Return the fixed action/execution candidate grid.

    ``full`` is the Cartesian grid requested in the plan. ``small`` is a
    quicker one-axis smoke grid for local development and CI.
    """
    baseline = candidate_from_sizing_config(base)
    if mode == "baseline":
        return [baseline]

    entry_variants = [
        ("heuristic", base.max_entry_pullback_pct),
        ("open", 0.0),
        ("close", 0.0),
        ("pullback_pct", 0.02),
        ("pullback_pct", 0.05),
        ("pullback_pct", 0.08),
    ]
    exit_variants = [
        ("heuristic", base.exit_patience_atrs),
        ("open", 0.0),
        ("close", 0.0),
        ("atr_limit", 0.5),
        ("atr_limit", 1.5),
    ]
    stop_variants: list[float | None] = [None, 1.0, base.stop_loss_atr_multiple, 2.0]
    sizing_variants = [
        (base.policy, base.stale_signal_decay),
        ("equal_weight_bullish", base.stale_signal_decay),
        ("confidence_weighted", base.stale_signal_decay),
        (base.policy, 1.0),
        (base.policy, 0.5),
    ]

    candidates: list[ExecutionPolicyCandidate] = [baseline]
    if mode == "small":
        candidates.extend([
            replace(baseline, name="entry_open", entry_mode="open", max_entry_pullback_pct=0.0),
            replace(baseline, name="entry_close", entry_mode="close", max_entry_pullback_pct=0.0),
            replace(baseline, name="entry_pullback_2pct", entry_mode="pullback_pct", max_entry_pullback_pct=0.02),
            replace(baseline, name="exit_close", exit_mode="close", exit_patience_atrs=0.0),
            replace(baseline, name="stop_disabled", stop_atr_multiple=None),
            replace(baseline, name="stop_1atr", stop_atr_multiple=1.0),
            replace(baseline, name="equal_weight", sizing_policy="equal_weight_bullish"),
            replace(baseline, name="stale_decay_off", stale_signal_decay=1.0),
        ])
        return _dedupe_candidates(candidates)
    if mode != "full":
        raise ValueError("policy grid mode must be 'baseline', 'small', or 'full'")

    for entry_mode, pullback in entry_variants:
        for exit_mode, exit_atrs in exit_variants:
            for stop_mult in stop_variants:
                for sizing_policy, stale_decay in sizing_variants:
                    name = _candidate_name(
                        sizing_policy=sizing_policy,
                        stale_decay=stale_decay,
                        entry_mode=entry_mode,
                        pullback=pullback,
                        exit_mode=exit_mode,
                        exit_atrs=exit_atrs,
                        stop_mult=stop_mult,
                    )
                    candidates.append(ExecutionPolicyCandidate(
                        name=name,
                        sizing_policy=sizing_policy,
                        entry_mode=entry_mode,
                        max_entry_pullback_pct=pullback,
                        exit_mode=exit_mode,
                        exit_patience_atrs=exit_atrs,
                        stop_atr_multiple=stop_mult,
                        stale_signal_decay=stale_decay,
                    ))
    return _dedupe_candidates(candidates)


def run_daily_execution_simulation(
    *,
    dates: list[date],
    signals_by_date: Mapping[date, Mapping[str, ExecutionSignal]],
    prices: Mapping[str, pd.DataFrame],
    base_config: SizingConfig,
    candidate: ExecutionPolicyCandidate,
    initial_capital: float = 100_000.0,
    cost_per_side_bps: float = 5.0,
    slippage_bps: float = 0.0,
    rebalance_threshold_pp: float = 0.01,
) -> DailyExecutionResult:
    """Simulate action-layer decisions against daily OHLC bars.

    Report-date signals are made tradable only on the next bar by using
    ``signal.as_of_date < current_date``. This avoids same-day lookahead.
    """
    dates = sorted(dates)
    sim_config = sizing_for_candidate(base_config, candidate)
    cost_rate = float(cost_per_side_bps) / 10_000.0
    slip_rate = float(slippage_bps) / 10_000.0
    universe = list(base_config.universe) or sorted(prices)

    cash = float(initial_capital)
    positions: dict[str, int] = {sym: 0 for sym in universe}
    avg_cost: dict[str, float] = {sym: 0.0 for sym in universe}
    stop_levels: dict[str, float | None] = {sym: None for sym in universe}
    active_signals: dict[str, ExecutionSignal] = {}
    signal_dates = sorted(signals_by_date)
    signal_i = 0
    states: list[DailyExecutionState] = []
    all_fills: list[ExecutionFill] = []
    all_missed: list[MissedOrder] = []
    previous_equity = float(initial_capital)

    for d in dates:
        while signal_i < len(signal_dates) and signal_dates[signal_i] < d:
            for sym, sig in signals_by_date[signal_dates[signal_i]].items():
                if sym in universe:
                    active_signals[sym] = sig
            signal_i += 1

        fills: list[ExecutionFill] = []
        missed: list[MissedOrder] = []
        stop_outs = 0
        cost_paid = 0.0
        turnover_notional = 0.0

        # Stop-loss reminders become executable exits in this validator and
        # fire before normal rebalance actions on the same day.
        for sym in universe:
            qty = positions.get(sym, 0)
            level = stop_levels.get(sym)
            bar = _bar_for(prices, sym, d)
            if qty <= 0 or level is None or bar is None:
                continue
            if _num(bar.get("low")) is not None and float(bar["low"]) <= level:
                fill_price = _sell_price(level, slip_rate)
                fill, cash_delta, paid = _sell_fill(
                    d, sym, qty, fill_price, "stop_loss", cost_rate
                )
                cash += cash_delta
                cost_paid += paid
                turnover_notional += fill.notional
                positions[sym] = 0
                avg_cost[sym] = 0.0
                stop_levels[sym] = None
                fills.append(fill)
                stop_outs += 1

        target_weights = _target_weights_for_day(
            d,
            universe=universe,
            active_signals=active_signals,
            config=sim_config,
        )
        equity_pre_trade = _mark_equity(cash, positions, prices, d, use_previous=True)

        for sym in universe:
            bar = _bar_for(prices, sym, d)
            if bar is None:
                continue
            ref_close = _previous_close(prices, sym, d)
            if ref_close is None or ref_close <= 0 or equity_pre_trade <= 0:
                continue
            current_value = positions.get(sym, 0) * ref_close
            current_weight = current_value / equity_pre_trade
            target_weight = float(target_weights.get(sym, 0.0))
            if abs(target_weight - current_weight) <= rebalance_threshold_pp:
                _refresh_stop(sym, stop_levels, avg_cost, candidate, prices, d)
                continue
            target_shares = int(math.floor((target_weight * equity_pre_trade) / ref_close))
            delta = target_shares - positions.get(sym, 0)
            if delta > 0:
                fill_price, limit_price = _entry_order_price(
                    candidate, bar, ref_close, _sma(prices, sym, d, 20)
                )
                if fill_price is None:
                    if limit_price is not None:
                        missed.append(MissedOrder(
                            date=d,
                            symbol=sym,
                            side="buy",
                            reason="entry_limit_not_reached",
                            quantity=delta,
                            limit_price=limit_price,
                            close_price=_num(bar.get("close")),
                        ))
                    continue
                fill_price = _buy_price(fill_price, slip_rate)
                max_qty = int(cash // (fill_price * (1.0 + cost_rate)))
                qty = max(0, min(delta, max_qty))
                if qty <= 0:
                    continue
                fill, cash_delta, paid = _buy_fill(
                    d, sym, qty, fill_price, "entry", cost_rate
                )
                old_qty = positions.get(sym, 0)
                old_basis = avg_cost.get(sym, 0.0) * old_qty
                positions[sym] = old_qty + qty
                avg_cost[sym] = (old_basis + fill.price * qty) / positions[sym]
                cash += cash_delta
                cost_paid += paid
                turnover_notional += fill.notional
                fills.append(fill)
                stop_levels[sym] = _new_stop_level(
                    fill.price, _atr(prices, sym, d), candidate
                )
            elif delta < 0 and positions.get(sym, 0) > 0:
                qty = min(abs(delta), positions[sym])
                fill_price, limit_price = _exit_order_price(
                    candidate, bar, ref_close, _atr(prices, sym, d)
                )
                if fill_price is None:
                    if limit_price is not None:
                        missed.append(MissedOrder(
                            date=d,
                            symbol=sym,
                            side="sell",
                            reason="exit_limit_not_reached",
                            quantity=qty,
                            limit_price=limit_price,
                            close_price=_num(bar.get("close")),
                        ))
                    continue
                fill_price = _sell_price(fill_price, slip_rate)
                fill, cash_delta, paid = _sell_fill(
                    d, sym, qty, fill_price, "rebalance_exit", cost_rate
                )
                positions[sym] -= qty
                if positions[sym] <= 0:
                    positions[sym] = 0
                    avg_cost[sym] = 0.0
                    stop_levels[sym] = None
                cash += cash_delta
                cost_paid += paid
                turnover_notional += fill.notional
                fills.append(fill)
                if positions[sym] > 0:
                    _refresh_stop(sym, stop_levels, avg_cost, candidate, prices, d)
            else:
                _refresh_stop(sym, stop_levels, avg_cost, candidate, prices, d)

        equity = _mark_equity(cash, positions, prices, d, use_previous=True)
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        turnover = turnover_notional / equity_pre_trade if equity_pre_trade > 0 else 0.0
        state = DailyExecutionState(
            date=d,
            equity=equity,
            cash=cash,
            positions={k: int(v) for k, v in positions.items() if v},
            target_weights=dict(target_weights),
            fills=fills,
            missed_orders=missed,
            stop_outs=stop_outs,
            one_way_turnover=turnover,
            cost_paid=cost_paid,
            daily_return=daily_return,
        )
        states.append(state)
        all_fills.extend(fills)
        all_missed.extend(missed)
        previous_equity = equity

    metrics = compute_execution_metrics(
        states,
        initial_capital=initial_capital,
        trading_days_per_year=252,
    )
    return DailyExecutionResult(
        policy=candidate,
        config=sim_config,
        states=states,
        fills=all_fills,
        missed_orders=all_missed,
        metrics=metrics,
    )


def run_benchmark_simulation(
    *,
    dates: list[date],
    prices: Mapping[str, pd.DataFrame],
    benchmark: str,
    initial_capital: float = 100_000.0,
    cost_per_side_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> DailyExecutionResult:
    policy = ExecutionPolicyCandidate(
        name=f"{benchmark}_baseline",
        sizing_policy="equal_weight_bullish",
        entry_mode="open",
        exit_mode="close",
        stop_atr_multiple=None,
    )
    cost_rate = cost_per_side_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    cash = float(initial_capital)
    shares = 0
    states: list[DailyExecutionState] = []
    fills: list[ExecutionFill] = []
    previous_equity = float(initial_capital)
    for i, d in enumerate(sorted(dates)):
        bar = _bar_for(prices, benchmark, d)
        day_fills: list[ExecutionFill] = []
        cost_paid = 0.0
        turnover = 0.0
        if i == 0 and bar is not None and _num(bar.get("open")):
            price = _buy_price(float(bar["open"]), slip_rate)
            shares = int(cash // (price * (1.0 + cost_rate)))
            if shares > 0:
                fill, cash_delta, paid = _buy_fill(
                    d, benchmark, shares, price, "benchmark_entry", cost_rate
                )
                cash += cash_delta
                cost_paid += paid
                turnover = fill.notional / initial_capital
                fills.append(fill)
                day_fills.append(fill)
        close = _num(bar.get("close")) if bar is not None else None
        equity = cash + shares * (close if close is not None else 0.0)
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        states.append(DailyExecutionState(
            date=d,
            equity=equity,
            cash=cash,
            positions={benchmark: shares} if shares else {},
            target_weights={benchmark: 1.0},
            fills=day_fills,
            one_way_turnover=turnover,
            cost_paid=cost_paid,
            daily_return=daily_return,
        ))
        previous_equity = equity
    cfg = SizingConfig(policy="equal_weight_bullish", max_per_name=1.0, max_long_exposure=1.0)
    return DailyExecutionResult(
        policy=policy,
        config=cfg,
        states=states,
        fills=fills,
        missed_orders=[],
        metrics=compute_execution_metrics(states, initial_capital=initial_capital),
    )


def compute_execution_metrics(
    states: list[DailyExecutionState],
    *,
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    if not states:
        return {}
    equity = [s.equity for s in states]
    returns = [s.daily_return for s in states[1:]]
    n_days = len(states)
    years = max(1e-9, (n_days - 1) / trading_days_per_year)
    end = equity[-1]
    cagr = (end / initial_capital) ** (1.0 / years) - 1.0 if initial_capital > 0 else 0.0
    if len(returns) >= 2:
        mean_d = sum(returns) / len(returns)
        var = sum((r - mean_d) ** 2 for r in returns) / (len(returns) - 1)
        std_d = math.sqrt(var)
        sharpe = (mean_d * trading_days_per_year) / (std_d * math.sqrt(trading_days_per_year)) if std_d > 0 else 0.0
    else:
        sharpe = 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        dd = value / peak - 1.0 if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    fills = sum(len(s.fills) for s in states)
    missed = sum(len(s.missed_orders) for s in states)
    stop_outs = sum(s.stop_outs for s in states)
    costs = sum(s.cost_paid for s in states)
    turnover = sum(s.one_way_turnover for s in states)
    return {
        "start_equity": float(initial_capital),
        "end_equity": float(end),
        "total_return": float(end / initial_capital - 1.0) if initial_capital > 0 else 0.0,
        "cagr": float(cagr),
        "sharpe_annualized": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_days": int(n_days),
        "n_fills": int(fills),
        "n_missed_orders": int(missed),
        "n_stop_outs": int(stop_outs),
        "total_cost_paid": float(costs),
        "total_one_way_turnover": float(turnover),
        "start_date": states[0].date.isoformat(),
        "end_date": states[-1].date.isoformat(),
    }


def excess_vs_benchmark(
    result: DailyExecutionResult,
    benchmark: DailyExecutionResult,
) -> dict[str, float]:
    joined = pd.concat([result.equity_curve(), benchmark.equity_curve()], axis=1).dropna()
    if len(joined) < 3:
        return {}
    p_ret = joined.iloc[:, 0].pct_change().dropna()
    b_ret = joined.iloc[:, 1].pct_change().dropna()
    excess = p_ret - b_ret
    std = float(excess.std(ddof=1))
    mean = float(excess.mean())
    info = (mean * 252) / (std * math.sqrt(252)) if std > 0 else 0.0
    return {
        "excess_cagr": float(result.metrics.get("cagr", 0.0) - benchmark.metrics.get("cagr", 0.0)),
        "annualized_excess_return": float(mean * 252),
        "information_ratio": float(info),
    }


def evaluate_acceptance(
    results: list[DailyExecutionResult],
    *,
    baseline_name: str = "baseline_current_config",
) -> list[dict[str, Any]]:
    baseline = next((r for r in results if r.policy.name == baseline_name), None)
    if baseline is None:
        return []
    base = baseline.metrics
    out: list[dict[str, Any]] = []
    for result in results:
        if result.policy.name == baseline_name:
            continue
        m = result.metrics
        sharpe_lift = m.get("sharpe_annualized", 0.0) - base.get("sharpe_annualized", 0.0)
        dd_delta = m.get("max_drawdown", 0.0) - base.get("max_drawdown", 0.0)
        turnover_ratio = (
            m.get("total_one_way_turnover", 0.0) / base.get("total_one_way_turnover", 1.0)
            if base.get("total_one_way_turnover", 0.0) > 0
            else 1.0
        )
        return_lift = m.get("total_return", 0.0) - base.get("total_return", 0.0)
        turnover_ok = turnover_ratio <= 1.25 or return_lift > 0
        accepted = sharpe_lift > 0 and dd_delta >= -0.02 and turnover_ok and return_lift >= 0
        out.append({
            "policy_name": result.policy.name,
            "accepted": accepted,
            "sharpe_lift": sharpe_lift,
            "return_lift": return_lift,
            "drawdown_delta": dd_delta,
            "turnover_ratio": turnover_ratio,
        })
    return sorted(out, key=lambda x: (not x["accepted"], -x["sharpe_lift"], -x["return_lift"]))


def generate_walk_forward_windows(
    dates: list[date],
    *,
    train_days: int = 126,
    test_days: int = 21,
    step_days: int = 21,
) -> list[dict[str, Any]]:
    """Generate rolling train/test windows over trading dates.

    The execution policies are fixed, so the train slice is recorded for
    discipline and future tuning, while the simulator evaluates each
    candidate on the test slice.
    """
    dates = sorted(dates)
    out: list[dict[str, Any]] = []
    i = 0
    while i + train_days + test_days <= len(dates):
        train = dates[i:i + train_days]
        test = dates[i + train_days:i + train_days + test_days]
        out.append({
            "train_start": train[0],
            "train_end": train[-1],
            "test_start": test[0],
            "test_end": test[-1],
            "test_dates": test,
        })
        i += max(1, step_days)
    return out


def summarize_walk_forward_acceptance(
    window_results: list[dict[str, Any]],
    *,
    baseline_name: str = "baseline_current_config",
) -> list[dict[str, Any]]:
    """Aggregate candidate acceptance gates across walk-forward windows."""
    by_policy: dict[str, list[dict[str, float]]] = {}
    for block in window_results:
        results: list[DailyExecutionResult] = block.get("results", [])
        baseline = next((r for r in results if r.policy.name == baseline_name), None)
        if baseline is None:
            continue
        base = baseline.metrics
        for result in results:
            if result.policy.name == baseline_name:
                continue
            m = result.metrics
            turnover_ratio = (
                m.get("total_one_way_turnover", 0.0)
                / base.get("total_one_way_turnover", 1.0)
                if base.get("total_one_way_turnover", 0.0) > 0
                else 1.0
            )
            by_policy.setdefault(result.policy.name, []).append({
                "sharpe_lift": m.get("sharpe_annualized", 0.0) - base.get("sharpe_annualized", 0.0),
                "return_lift": m.get("total_return", 0.0) - base.get("total_return", 0.0),
                "drawdown_delta": m.get("max_drawdown", 0.0) - base.get("max_drawdown", 0.0),
                "turnover_ratio": turnover_ratio,
                "win": 1.0 if m.get("end_equity", 0.0) >= base.get("end_equity", 0.0) else 0.0,
            })
    summaries: list[dict[str, Any]] = []
    for policy_name, rows in by_policy.items():
        median_sharpe = _median([r["sharpe_lift"] for r in rows])
        median_return = _median([r["return_lift"] for r in rows])
        worst_dd_delta = min(r["drawdown_delta"] for r in rows)
        median_turnover = _median([r["turnover_ratio"] for r in rows])
        win_rate = sum(r["win"] for r in rows) / len(rows) if rows else 0.0
        turnover_ok = median_turnover <= 1.25 or median_return > 0
        accepted = (
            median_sharpe > 0
            and worst_dd_delta >= -0.02
            and turnover_ok
            and win_rate >= 0.60
        )
        summaries.append({
            "policy_name": policy_name,
            "accepted": accepted,
            "n_windows": len(rows),
            "win_rate": win_rate,
            "median_sharpe_lift": median_sharpe,
            "median_return_lift": median_return,
            "worst_drawdown_delta": worst_dd_delta,
            "median_turnover_ratio": median_turnover,
        })
    return sorted(
        summaries,
        key=lambda x: (
            not x["accepted"],
            -x["median_sharpe_lift"],
            -x["win_rate"],
            -x["median_return_lift"],
        ),
    )


def render_walk_forward_acceptance_markdown(
    summaries: list[dict[str, Any]],
    *,
    title: str = "Walk-forward execution acceptance",
    max_rows: int = 30,
) -> str:
    lines = [f"# {title}", ""]
    if not summaries:
        return "\n".join(lines + ["_No walk-forward windows were available._"]) + "\n"
    lines.append(
        "- Acceptance gates: positive median Sharpe lift, max drawdown no worse by more than 2pp, "
        "turnover controlled, and win-rate at least 60%."
    )
    lines.append("")
    lines.append("| Policy | Accepted | Windows | Win rate | Median Sharpe lift | Median return lift | Worst DD delta | Median turnover |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries[:max_rows]:
        lines.append(
            "| "
            + " | ".join([
                f"`{row['policy_name']}`",
                "yes" if row["accepted"] else "no",
                str(row["n_windows"]),
                f"{row['win_rate'] * 100:.1f}%",
                f"{row['median_sharpe_lift']:+.2f}",
                f"{row['median_return_lift'] * 100:+.2f}%",
                f"{row['worst_drawdown_delta'] * 100:+.2f}%",
                f"{row['median_turnover_ratio']:.2f}x",
            ])
            + " |"
        )
    if len(summaries) > max_rows:
        lines.append("")
        lines.append(f"_Showing top {max_rows} of {len(summaries)} candidates._")
    winners = [s for s in summaries if s["accepted"]]
    lines.append("")
    if winners:
        lines.append(f"Recommendation: `{winners[0]['policy_name']}` clears walk-forward gates.")
    else:
        lines.append("Recommendation: no candidate clears walk-forward gates; keep current production config.")
    return "\n".join(lines).rstrip() + "\n"


def render_execution_comparison_markdown(
    results: list[DailyExecutionResult],
    *,
    benchmark: DailyExecutionResult | None = None,
    title: str = "Action/execution policy comparison",
    max_rows: int = 30,
) -> str:
    if not results:
        return f"# {title}\n\n_No simulations to compare._\n"
    sorted_results = sorted(
        results,
        key=lambda r: (
            r.policy.name != "baseline_current_config",
            -r.metrics.get("sharpe_annualized", 0.0),
            -r.metrics.get("total_return", 0.0),
        ),
    )
    baseline = next((r for r in results if r.policy.name == "baseline_current_config"), None)
    if baseline is not None:
        sorted_results = [baseline] + [r for r in sorted_results if r is not baseline]
    lines = [f"# {title}", ""]
    first = results[0]
    lines.append(
        f"- Slice: {first.metrics.get('start_date')} -> {first.metrics.get('end_date')} "
        f"({int(first.metrics.get('n_days', 0))} trading days)"
    )
    lines.append("- Report-date signals are tradable starting on the next bar.")
    lines.append("")
    headers = [
        "Policy", "End equity", "Return", "Sharpe", "Max DD",
        "Turnover", "Costs", "Fills", "Missed", "Stops",
    ]
    if benchmark is not None:
        headers += ["Excess CAGR", "Info ratio"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    rows = sorted_results[:max_rows]
    for result in rows:
        m = result.metrics
        cells = [
            f"`{result.policy.name}`",
            f"${m.get('end_equity', 0.0):,.0f}",
            f"{m.get('total_return', 0.0) * 100:+.2f}%",
            f"{m.get('sharpe_annualized', 0.0):+.2f}",
            f"{m.get('max_drawdown', 0.0) * 100:+.2f}%",
            f"{m.get('total_one_way_turnover', 0.0):.2f}x",
            f"${m.get('total_cost_paid', 0.0):,.0f}",
            f"{int(m.get('n_fills', 0))}",
            f"{int(m.get('n_missed_orders', 0))}",
            f"{int(m.get('n_stop_outs', 0))}",
        ]
        if benchmark is not None:
            ex = excess_vs_benchmark(result, benchmark)
            cells += [
                f"{ex.get('excess_cagr', 0.0) * 100:+.2f}%",
                f"{ex.get('information_ratio', 0.0):+.2f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    if len(sorted_results) > len(rows):
        lines.append("")
        lines.append(f"_Showing top {len(rows)} of {len(sorted_results)} policies._")
    accepted = evaluate_acceptance(results)
    if accepted:
        winners = [x for x in accepted if x["accepted"]]
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        if winners:
            best = winners[0]
            lines.append(
                f"- Candidate `{best['policy_name']}` passes single-slice gates "
                f"(Sharpe lift {best['sharpe_lift']:+.2f}, return lift "
                f"{best['return_lift']*100:+.2f}%)."
            )
            lines.append("- Treat this as provisional until walk-forward slices confirm the same result.")
        else:
            lines.append("- No candidate clears the acceptance gates; keep current production config.")
    return "\n".join(lines).rstrip() + "\n"


def result_to_json(result: DailyExecutionResult) -> dict[str, Any]:
    return {
        "policy": result.policy.__dict__,
        "metrics": result.metrics,
        "states": [
            {
                "date": s.date.isoformat(),
                "equity": s.equity,
                "cash": s.cash,
                "positions": s.positions,
                "target_weights": s.target_weights,
                "one_way_turnover": s.one_way_turnover,
                "cost_paid": s.cost_paid,
                "daily_return": s.daily_return,
                "stop_outs": s.stop_outs,
                "fills": [f.__dict__ | {"date": f.date.isoformat()} for f in s.fills],
                "missed_orders": [m.__dict__ | {"date": m.date.isoformat()} for m in s.missed_orders],
            }
            for s in result.states
        ],
    }


def _target_weights_for_day(
    day: date,
    *,
    universe: list[str],
    active_signals: Mapping[str, ExecutionSignal],
    config: SizingConfig,
) -> dict[str, float]:
    inputs: dict[str, dict[str, Any]] = {}
    for sym in universe:
        sig = active_signals.get(sym)
        if sig is None:
            inputs[sym] = {"direction": "neutral", "composite": None, "confidence": None, "age_days": None}
            continue
        direction = sig.direction
        if direction == "bearish" and not config.enable_bearish:
            direction = "neutral"
        inputs[sym] = {
            "direction": direction,
            "composite": sig.composite,
            "confidence": sig.confidence,
            "age_days": (day - sig.as_of_date).days,
        }
    return compute_target_weights(inputs, config=config)


def _entry_order_price(
    candidate: ExecutionPolicyCandidate,
    bar: Mapping[str, Any],
    ref_close: float,
    sma20: float | None,
) -> tuple[float | None, float | None]:
    if candidate.entry_mode == "open":
        price = _num(bar.get("open"))
        return price, price
    if candidate.entry_mode == "close":
        price = _num(bar.get("close"))
        return price, price
    if candidate.entry_mode == "pullback_pct":
        limit = round(ref_close * (1.0 - candidate.max_entry_pullback_pct), 2)
    else:
        limit = buy_limit_price(
            ref_close,
            sma20,
            pullback_to="sma20",
            max_pullback_pct=candidate.max_entry_pullback_pct,
        )
    low = _num(bar.get("low"))
    if limit is None or low is None or low > limit:
        return None, limit
    return limit, limit


def _exit_order_price(
    candidate: ExecutionPolicyCandidate,
    bar: Mapping[str, Any],
    ref_close: float,
    atr14: float | None,
) -> tuple[float | None, float | None]:
    if candidate.exit_mode == "open":
        price = _num(bar.get("open"))
        return price, price
    if candidate.exit_mode == "close":
        price = _num(bar.get("close"))
        return price, price
    atrs = candidate.exit_patience_atrs
    limit = trim_limit_price(ref_close, atr14, atrs=atrs)
    high = _num(bar.get("high"))
    if limit is None or high is None or high < limit:
        return None, limit
    return limit, limit


def _new_stop_level(
    entry_price: float,
    atr14: float | None,
    candidate: ExecutionPolicyCandidate,
) -> float | None:
    if candidate.stop_atr_multiple is None:
        return None
    return stop_loss_price(
        entry_price,
        atr14,
        atr_multiple=candidate.stop_atr_multiple,
    )


def _refresh_stop(
    sym: str,
    stop_levels: dict[str, float | None],
    avg_cost: Mapping[str, float],
    candidate: ExecutionPolicyCandidate,
    prices: Mapping[str, pd.DataFrame],
    day: date,
) -> None:
    if candidate.stop_atr_multiple is None or avg_cost.get(sym, 0.0) <= 0:
        return
    stop_levels[sym] = stop_loss_price(
        avg_cost[sym],
        _atr(prices, sym, day),
        atr_multiple=candidate.stop_atr_multiple,
    )


def _buy_fill(
    day: date,
    sym: str,
    qty: int,
    price: float,
    reason: str,
    cost_rate: float,
) -> tuple[ExecutionFill, float, float]:
    notional = qty * price
    cost = notional * cost_rate
    fill = ExecutionFill(day, sym, "buy", reason, qty, price, notional, cost)
    return fill, -(notional + cost), cost


def _sell_fill(
    day: date,
    sym: str,
    qty: int,
    price: float,
    reason: str,
    cost_rate: float,
) -> tuple[ExecutionFill, float, float]:
    notional = qty * price
    cost = notional * cost_rate
    fill = ExecutionFill(day, sym, "sell", reason, qty, price, notional, cost)
    return fill, notional - cost, cost


def _buy_price(price: float, slippage_rate: float) -> float:
    return float(price) * (1.0 + slippage_rate)


def _sell_price(price: float, slippage_rate: float) -> float:
    return float(price) * (1.0 - slippage_rate)


def _mark_equity(
    cash: float,
    positions: Mapping[str, int],
    prices: Mapping[str, pd.DataFrame],
    day: date,
    *,
    use_previous: bool,
) -> float:
    total = float(cash)
    for sym, qty in positions.items():
        if qty == 0:
            continue
        bar = _bar_for(prices, sym, day)
        close = _num(bar.get("close")) if bar is not None else None
        if close is None and use_previous:
            close = _previous_close(prices, sym, day)
        total += qty * float(close or 0.0)
    return total


def _bar_for(
    prices: Mapping[str, pd.DataFrame],
    sym: str,
    day: date,
) -> dict[str, Any] | None:
    df = prices.get(sym)
    if df is None or df.empty:
        return None
    key = pd.Timestamp(day)
    if key not in df.index:
        key = day
    if key not in df.index:
        return None
    row = df.loc[key]
    return {str(k).lower(): v for k, v in row.items()}


def _previous_close(
    prices: Mapping[str, pd.DataFrame],
    sym: str,
    day: date,
) -> float | None:
    df = prices.get(sym)
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index)
    mask = idx < pd.Timestamp(day)
    if not mask.any():
        return None
    rows = df.loc[df.index[mask]]
    if rows.empty:
        return None
    value = rows.iloc[-1].get("close", rows.iloc[-1].get("Close"))
    return _num(value)


def _sma(
    prices: Mapping[str, pd.DataFrame],
    sym: str,
    day: date,
    window: int,
) -> float | None:
    closes = _prior_closes(prices, sym, day, window)
    if len(closes) < 1:
        return None
    return float(sum(closes) / len(closes))


def _atr(
    prices: Mapping[str, pd.DataFrame],
    sym: str,
    day: date,
    window: int = 14,
) -> float | None:
    df = prices.get(sym)
    if df is None or df.empty:
        return None
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame = frame.loc[pd.to_datetime(frame.index) < pd.Timestamp(day)]
    if len(frame) < 2:
        return None
    frame = frame.tail(window + 1)
    trs: list[float] = []
    prev_close: float | None = None
    for _idx, row in frame.iterrows():
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        close = _num(row.get("close"))
        if high is None or low is None:
            continue
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    if not trs:
        return None
    return float(sum(trs[-window:]) / len(trs[-window:]))


def _prior_closes(
    prices: Mapping[str, pd.DataFrame],
    sym: str,
    day: date,
    limit: int,
) -> list[float]:
    df = prices.get(sym)
    if df is None or df.empty:
        return []
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame = frame.loc[pd.to_datetime(frame.index) < pd.Timestamp(day)]
    closes = [_num(v) for v in frame.tail(limit).get("close", pd.Series(dtype=float)).tolist()]
    return [float(x) for x in closes if x is not None]


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _candidate_name(
    *,
    sizing_policy: str,
    stale_decay: float,
    entry_mode: str,
    pullback: float,
    exit_mode: str,
    exit_atrs: float,
    stop_mult: float | None,
) -> str:
    stop = "nostop" if stop_mult is None else f"stop{stop_mult:g}atr"
    entry = entry_mode if entry_mode in ("open", "close") else f"{entry_mode}{pullback:g}"
    exit_name = exit_mode if exit_mode in ("open", "close") else f"{exit_mode}{exit_atrs:g}"
    return f"{sizing_policy}_stale{stale_decay:g}_{entry}_{exit_name}_{stop}"


def _dedupe_candidates(candidates: list[ExecutionPolicyCandidate]) -> list[ExecutionPolicyCandidate]:
    seen: set[tuple[Any, ...]] = set()
    out: list[ExecutionPolicyCandidate] = []
    for c in candidates:
        key = (
            c.sizing_policy,
            c.entry_mode,
            c.max_entry_pullback_pct,
            c.exit_mode,
            c.exit_patience_atrs,
            c.stop_atr_multiple,
            c.stale_signal_decay,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
