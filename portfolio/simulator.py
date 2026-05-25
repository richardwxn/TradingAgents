"""Weekly-rebalance portfolio simulator (Section 17).

Walks forward over a set of weekly snapshots (every Friday in the
corpus), applies a sizing policy to derive target weights, charges
turnover-based transaction costs, and tracks the resulting equity
curve. Compares against a SPY buy-and-hold baseline.

Design contract:
- Pure functions. No yfinance, no file I/O. The CLI in
  `portfolio_simulate.py` loads the JSON reports + fetches prices and
  hands pre-built data structures into `run_simulation`.
- One simulation per (policy, slice) combination. Output is a
  `SimulationResult` with everything needed to render comparison
  markdown.
- Cost model: turnover (sum of absolute weight changes) × bps per
  side. Default 5 bps per side ≈ realistic for liquid large-caps.
- Stale-composite handling: any composite older than
  `stale_composite_weeks` weeks is treated as neutral (drops the
  ticker from the bullish set), mirroring the daily signals layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import pandas as pd

from portfolio.sizing import SizingConfig, compute_target_weights


# ---------- data shapes ----------


@dataclass(frozen=True)
class WeeklyObservation:
    """One (ticker, week) signal observation. Price comes from the
    PriceTable, not from this struct, so the simulator can compute
    returns across weeks where no fresh composite was emitted."""

    ticker: str
    week: date
    direction: str  # "bullish" | "bearish" | "neutral"
    composite: float | None
    confidence: float | None
    composite_age_weeks: int  # 0 = fresh this week; >0 = carried forward


@dataclass(frozen=True)
class SimulationConfig:
    initial_capital: float = 100_000.0
    cost_per_side_bps: float = 5.0
    benchmark: str = "SPY"
    stale_composite_weeks: int = 2
    risk_free_annual: float = 0.0
    trading_weeks_per_year: int = 52


@dataclass(frozen=True)
class WeeklyState:
    week: date
    equity_pre_trade: float
    equity_post_trade: float
    target_weights: dict[str, float]
    realized_return: float  # this week's return contribution
    one_way_turnover: float
    cost_paid: float


@dataclass(frozen=True)
class SimulationResult:
    policy_name: str
    config: SimulationConfig
    states: list[WeeklyState]
    metrics: dict[str, float]

    def equity_curve(self) -> pd.Series:
        idx = [s.week for s in self.states]
        vals = [s.equity_post_trade for s in self.states]
        return pd.Series(vals, index=pd.to_datetime(idx), name=self.policy_name)


# ---------- policy adapter ----------


def _signals_dict_from_observations(
    obs_by_ticker: dict[str, WeeklyObservation],
    *,
    config: SimulationConfig,
) -> dict[str, dict[str, Any]]:
    """Convert per-ticker observations into the dict shape
    `compute_target_weights` expects.

    Two-tier staleness handling:
    - Hard cliff at `stale_composite_weeks` → drop to neutral entirely
      (the data is too old to trust at all).
    - Below the cliff, `age_days` is passed through so sizing can apply
      its own soft decay (`stale_signal_decay`) when age >
      `stale_composite_days`.
    """
    out: dict[str, dict[str, Any]] = {}
    for t, o in obs_by_ticker.items():
        age_days = int(o.composite_age_weeks) * 7
        if o.composite_age_weeks > config.stale_composite_weeks:
            out[t] = {
                "direction": "neutral", "composite": None, "confidence": None,
                "age_days": age_days,
            }
        else:
            out[t] = {
                "direction": o.direction,
                "composite": o.composite,
                "confidence": o.confidence,
                "age_days": age_days,
            }
    return out


# ---------- core walk-forward simulation ----------


def run_simulation(
    *,
    weeks: list[date],
    observations: dict[date, dict[str, WeeklyObservation]],
    prices: pd.DataFrame,  # index = week dates (Fridays), columns = tickers
    sizing_config: SizingConfig,
    sim_config: SimulationConfig | None = None,
    policy_name: str | None = None,
    benchmark_only: bool = False,
) -> SimulationResult:
    """Walk forward week-by-week through `weeks`, applying the policy.

    `benchmark_only=True` ignores the sizing policy and runs a
    100%-`sim_config.benchmark` buy-and-hold instead. The benchmark
    must be a column in `prices`.

    Returns a fully-populated SimulationResult. Returns over the last
    week of `weeks` are computed if a next-week price is available;
    otherwise the final week records target weights but no realized
    return (the equity curve ends one week early to avoid fabricating
    a partial-week return).
    """
    sim_config = sim_config or SimulationConfig()
    policy_name = policy_name or (
        sim_config.benchmark + "_baseline" if benchmark_only else sizing_config.policy
    )
    equity = float(sim_config.initial_capital)
    prev_weights: dict[str, float] = {}
    states: list[WeeklyState] = []

    cost_rate = float(sim_config.cost_per_side_bps) / 10_000.0

    for i, w in enumerate(weeks):
        if benchmark_only:
            target = {sim_config.benchmark: 1.0}
        else:
            obs = observations.get(w, {})
            sizing_input = _signals_dict_from_observations(obs, config=sim_config)
            # Restrict sizing to tickers that have a valid price this week,
            # so we don't allocate to a ticker we can't actually buy.
            sizing_input = {
                t: v for t, v in sizing_input.items()
                if t in prices.columns and not pd.isna(prices.at[w, t])
            }
            target = compute_target_weights(sizing_input, config=sizing_config)

        # Turnover cost is paid against the *current* equity before this
        # week's market returns are applied (i.e. you trade at this Friday's
        # close, then sit until next Friday's close).
        keys = set(target) | set(prev_weights)
        turnover = sum(abs(target.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in keys)
        cost_paid = equity * turnover * cost_rate
        equity_pre_trade = equity
        equity -= cost_paid

        # Realized weekly return: weighted sum of per-ticker close-to-close
        # returns from this Friday to next Friday. The final week has no
        # next observation → realized return defaults to 0.
        realized = 0.0
        if i + 1 < len(weeks):
            w_next = weeks[i + 1]
            for t, weight in target.items():
                if weight == 0.0:
                    continue
                if t not in prices.columns:
                    continue
                p_now = prices.at[w, t] if w in prices.index else None
                p_next = prices.at[w_next, t] if w_next in prices.index else None
                if p_now is None or p_next is None or pd.isna(p_now) or pd.isna(p_next) or p_now <= 0:
                    continue
                realized += float(weight) * (float(p_next) / float(p_now) - 1.0)
            equity *= (1.0 + realized)

        states.append(WeeklyState(
            week=w,
            equity_pre_trade=equity_pre_trade,
            equity_post_trade=equity,
            target_weights=dict(target),
            realized_return=realized,
            one_way_turnover=turnover,
            cost_paid=cost_paid,
        ))
        prev_weights = target

    metrics = compute_metrics(states, sim_config=sim_config)
    return SimulationResult(
        policy_name=policy_name,
        config=sim_config,
        states=states,
        metrics=metrics,
    )


# ---------- metrics ----------


def compute_metrics(
    states: list[WeeklyState],
    *,
    sim_config: SimulationConfig,
) -> dict[str, float]:
    """CAGR (annualized), Sharpe (annualized, weekly returns), max
    drawdown, total turnover, average weekly cost, n_weeks."""
    if not states:
        return {}
    equity = [s.equity_post_trade for s in states]
    weeks = [s.week for s in states]
    returns = [s.realized_return for s in states[:-1]]  # last week has no realized
    n_weeks = len(states)
    weeks_per_year = sim_config.trading_weeks_per_year

    start = equity[0]
    end = equity[-1]
    years = max(1e-9, (n_weeks - 1) / weeks_per_year)
    cagr = (end / sim_config.initial_capital) ** (1.0 / years) - 1.0 if sim_config.initial_capital > 0 else 0.0

    # Sharpe from weekly returns (subtract weekly risk-free).
    rf_weekly = sim_config.risk_free_annual / weeks_per_year
    excess = [r - rf_weekly for r in returns]
    if len(excess) >= 2:
        mean_w = sum(excess) / len(excess)
        var = sum((r - mean_w) ** 2 for r in excess) / (len(excess) - 1)
        std_w = math.sqrt(var)
        sharpe = (mean_w * weeks_per_year) / (std_w * math.sqrt(weeks_per_year)) if std_w > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown on equity curve.
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    total_turnover = sum(s.one_way_turnover for s in states)
    total_cost = sum(s.cost_paid for s in states)
    avg_long_exposure = (
        sum(sum(max(0.0, w) for w in s.target_weights.values()) for s in states) / n_weeks
    )

    return {
        "start_equity": float(sim_config.initial_capital),
        "end_equity": float(end),
        "cagr": float(cagr),
        "sharpe_annualized": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_weeks": int(n_weeks),
        "total_one_way_turnover": float(total_turnover),
        "total_cost_paid": float(total_cost),
        "avg_long_exposure": float(avg_long_exposure),
        "start_week": weeks[0].isoformat() if isinstance(weeks[0], date) else str(weeks[0]),
        "end_week": weeks[-1].isoformat() if isinstance(weeks[-1], date) else str(weeks[-1]),
    }


def excess_vs_benchmark(
    portfolio: SimulationResult, benchmark: SimulationResult
) -> dict[str, float]:
    """Excess CAGR, information ratio, beta (rough)."""
    if not portfolio.states or not benchmark.states:
        return {}
    p_eq = portfolio.equity_curve()
    b_eq = benchmark.equity_curve()
    joined = pd.concat([p_eq, b_eq], axis=1).dropna()
    if len(joined) < 3:
        return {}
    p_ret = joined.iloc[:, 0].pct_change().dropna()
    b_ret = joined.iloc[:, 1].pct_change().dropna()
    if len(p_ret) < 2:
        return {}
    excess = p_ret - b_ret
    excess_mean_weekly = float(excess.mean())
    excess_std_weekly = float(excess.std(ddof=1))
    weeks_per_year = portfolio.config.trading_weeks_per_year
    excess_annual = excess_mean_weekly * weeks_per_year
    info_ratio = (
        excess_annual / (excess_std_weekly * math.sqrt(weeks_per_year))
        if excess_std_weekly > 0
        else 0.0
    )
    return {
        "excess_cagr": float(portfolio.metrics.get("cagr", 0.0) - benchmark.metrics.get("cagr", 0.0)),
        "annualized_excess_return": float(excess_annual),
        "information_ratio": float(info_ratio),
    }


# ---------- helpers for the CLI ----------


def fridays_from_observations(
    observations: dict[date, dict[str, WeeklyObservation]],
) -> list[date]:
    """Sorted unique week dates from the observations dict."""
    return sorted(observations.keys())


def filter_weeks_by_range(
    weeks: list[date], *, date_from: date | None, date_to: date | None
) -> list[date]:
    out = weeks
    if date_from is not None:
        out = [w for w in out if w >= date_from]
    if date_to is not None:
        out = [w for w in out if w <= date_to]
    return out


def render_policy_comparison_markdown(
    results: list[SimulationResult],
    *,
    benchmark: SimulationResult | None = None,
    title: str = "Portfolio policy comparison",
    slice_label: str | None = None,
) -> str:
    """One row per policy. Includes vs-benchmark excess columns when a
    benchmark is supplied (typically SPY)."""
    lines = [f"# {title}", ""]
    if slice_label:
        lines.append(f"_{slice_label}_")
        lines.append("")
    if not results:
        return "\n".join(lines + ["_No simulations to compare._"]) + "\n"
    first = results[0]
    lines.append(
        f"- Cost model: {first.config.cost_per_side_bps:.1f} bps per side "
        f"({first.config.cost_per_side_bps / 100:.2f}% round-trip per 100% turnover)"
    )
    lines.append(f"- Initial capital: ${first.config.initial_capital:,.0f}")
    if benchmark:
        lines.append(f"- Benchmark: {benchmark.policy_name}")
    lines.append("")
    headers = [
        "Policy", "End equity", "CAGR", "Sharpe", "Max DD",
        "Avg long", "Total turnover", "Costs paid",
    ]
    excess_cols = ["Excess CAGR", "Info ratio"]
    if benchmark:
        headers += excess_cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    rows = list(results)
    if benchmark and benchmark not in rows:
        rows = rows + [benchmark]
    for r in rows:
        m = r.metrics or {}
        end = m.get("end_equity", 0.0)
        cells = [
            f"`{r.policy_name}`",
            f"${end:,.0f}",
            f"{m.get('cagr', 0.0) * 100:+.2f}%",
            f"{m.get('sharpe_annualized', 0.0):+.2f}",
            f"{m.get('max_drawdown', 0.0) * 100:+.2f}%",
            f"{m.get('avg_long_exposure', 0.0) * 100:.1f}%",
            f"{m.get('total_one_way_turnover', 0.0):.2f}x",
            f"${m.get('total_cost_paid', 0.0):,.0f}",
        ]
        if benchmark:
            if r.policy_name == benchmark.policy_name:
                cells += ["—", "—"]
            else:
                ex = excess_vs_benchmark(r, benchmark)
                cells += [
                    f"{ex.get('excess_cagr', 0.0) * 100:+.2f}%",
                    f"{ex.get('information_ratio', 0.0):+.2f}",
                ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
