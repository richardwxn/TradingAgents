"""Portfolio simulator CLI (Section 17).

Loads the weekly composite reports + fetches weekly close prices for
each ticker + the benchmark, then runs the configured sizing policies
side-by-side. Outputs a markdown comparison table + per-policy JSONs
under `backtest/results/simulator_*`.

Example:

    source .venv/bin/activate
    python portfolio_simulate.py \\
        --reports-glob "reports/analysis_mvp/*.json" \\
        --policies equal_weight_bullish top_n_bullish confidence_weighted \\
        --include-benchmark \\
        --output-dir backtest/results/simulator_full

Train/test slicing reuses the same flags as backtest.py:

    --date-from 2025-11-21 --date-to 2026-02-27 → TRAIN
    --date-from 2026-02-28 --date-to 2026-05-22 → TEST
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from portfolio.signals import _load_signal_from_json
from portfolio.simulator import (
    SimulationConfig,
    WeeklyObservation,
    filter_weeks_by_range,
    fridays_from_observations,
    render_policy_comparison_markdown,
    run_simulation,
)
from portfolio.sizing import SizingConfig, sizing_config_from_dict


def _atr_from_report(path: Path) -> float | None:
    """Best-effort extraction of `key_features.technical.atr_14` from a
    composite report JSON. Returns None on any parse/missing failure."""
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    tech = ((payload.get("key_features") or {}).get("technical") or {})
    val = tech.get("atr_14")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


DEFAULT_POLICIES = ("equal_weight_bullish", "top_n_bullish", "confidence_weighted")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    p.add_argument("--sizing-config", default="configs/sizing.yaml")
    p.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES),
                   help="Sizing policies to compare (from portfolio.sizing.SUPPORTED_POLICIES).")
    p.add_argument("--include-benchmark", action="store_true", default=True,
                   help="Run a benchmark buy-and-hold simulation (default: True).")
    p.add_argument("--no-benchmark", dest="include_benchmark", action="store_false")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="Transaction cost per side in bps (default: 5).")
    p.add_argument("--top-n", type=int, default=5,
                   help="For top_n_bullish: pick top N composites each week.")
    p.add_argument("--date-from", default=None,
                   help="Inclusive lower bound on rebalance week (YYYY-MM-DD).")
    p.add_argument("--date-to", default=None,
                   help="Inclusive upper bound on rebalance week (YYYY-MM-DD).")
    p.add_argument("--stop-loss-atr-multiple", type=float, default=None,
                   help="Override stop_loss_atr_multiple from sizing.yaml. "
                        "Set to 0 to disable intra-week stops (legacy behavior).")
    p.add_argument("--output-dir", default="backtest/results/simulator_full")
    return p.parse_args()


def _parse_date_opt(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_observations(
    report_paths: list[Path],
    *,
    universe: list[str],
    cutoff: date | None = None,
) -> dict[date, dict[str, WeeklyObservation]]:
    """Group composites by (week_date, ticker). Carries forward the most
    recent composite for tickers that didn't refresh on a given Friday,
    annotated with `composite_age_weeks` so the simulator can age-out
    stale signals consistently with the daily layer.

    Also extracts `key_features.technical.atr_14` so the simulator can
    apply position-level stop-loss exits (Section 16). ATR is carried
    forward with the rest of the composite when a week has no fresh
    report for a given ticker."""
    by_ticker_week: dict[str, dict[date, dict]] = defaultdict(dict)
    for p in report_paths:
        sig = _load_signal_from_json(p)
        if sig is None or sig.symbol not in universe:
            continue
        try:
            d = datetime.strptime(sig.as_of_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        if cutoff and d > cutoff:
            continue
        by_ticker_week[sig.symbol][d] = {
            "direction": sig.direction,
            "composite": sig.composite,
            "confidence": sig.confidence,
            "atr_14": _atr_from_report(p),
        }

    all_weeks = sorted({w for entries in by_ticker_week.values() for w in entries})
    if not all_weeks:
        return {}
    out: dict[date, dict[str, WeeklyObservation]] = {w: {} for w in all_weeks}

    for ticker, entries in by_ticker_week.items():
        latest: dict | None = None
        latest_week: date | None = None
        for w in all_weeks:
            if w in entries:
                latest = entries[w]
                latest_week = w
            if latest is None or latest_week is None:
                continue
            age_weeks = max(0, (w - latest_week).days // 7)
            out[w][ticker] = WeeklyObservation(
                ticker=ticker,
                week=w,
                direction=latest["direction"],
                composite=latest["composite"],
                confidence=latest["confidence"],
                composite_age_weeks=age_weeks,
                atr_14=latest.get("atr_14"),
            )
    return out


def _build_intra_week_min_close(
    daily_closes: pd.DataFrame,
    weeks: list[date],
) -> dict[date, dict[str, float]]:
    """Build `{friday: {ticker: min_close_between_(friday, next_friday]}}`.

    Excludes the entry Friday itself (we entered at that close) and
    INCLUDES the next Friday (we'd exit at that close, so a pierce on
    the exit day still counts). Stops on the last week of the window
    are skipped because there's no next-Friday to define the interval.
    Returns an empty dict if `daily_closes` is empty.
    """
    out: dict[date, dict[str, float]] = {}
    if daily_closes is None or daily_closes.empty:
        return out
    idx = pd.to_datetime(daily_closes.index)
    df = daily_closes.copy()
    df.index = idx
    for i, w in enumerate(weeks):
        if i + 1 >= len(weeks):
            continue
        w_next = weeks[i + 1]
        # Inclusive of next Friday's close; exclusive of entry Friday.
        mask = (df.index > pd.Timestamp(w)) & (df.index <= pd.Timestamp(w_next))
        slab = df.loc[mask]
        if slab.empty:
            continue
        mins = slab.min(axis=0, skipna=True)
        out[w] = {
            sym: float(v) for sym, v in mins.items()
            if pd.notna(v)
        }
    return out


def _fetch_weekly_close(
    symbols: list[str],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch daily closes via yfinance, then forward-fill onto the
    Friday calendar so every Friday has a close per ticker (using the
    most recent trading day on or before that Friday). Defensive
    against per-ticker yfinance failures."""
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            raw = yf.download(
                sym,
                start=(start - timedelta(days=14)).isoformat(),
                end=(end + timedelta(days=7)).isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        if raw is None or getattr(raw, "empty", True):
            continue
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try:
            close = df["Close"].dropna()
        except Exception:
            continue
        if close.empty:
            continue
        close.index = pd.to_datetime(close.index)
        out[sym] = close
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def _project_to_weeks(
    daily_closes: pd.DataFrame,
    weeks: list[date],
) -> pd.DataFrame:
    """Reindex daily close columns onto the Friday calendar using
    forward-fill — picks the most recent trading day's close at or
    before each Friday so we don't drop weeks that fall on a holiday."""
    if daily_closes.empty:
        return pd.DataFrame(index=pd.to_datetime([pd.Timestamp(w) for w in weeks]))
    target_index = pd.to_datetime([pd.Timestamp(w) for w in weeks])
    out = daily_closes.reindex(daily_closes.index.union(target_index)).sort_index()
    out = out.ffill()
    out = out.loc[target_index]
    out.index = [d.date() for d in out.index]
    return out


def main() -> None:
    args = _parse_args()
    sizing_cfg_raw = yaml.safe_load(Path(args.sizing_config).read_text()) or {}
    base_sizing = sizing_config_from_dict(sizing_cfg_raw)
    universe = list(base_sizing.universe)
    if not universe:
        raise SystemExit("No universe configured in sizing.yaml.")

    stop_mult = (
        float(args.stop_loss_atr_multiple)
        if args.stop_loss_atr_multiple is not None
        else float(base_sizing.stop_loss_atr_multiple)
    )
    sim_config = SimulationConfig(
        initial_capital=args.initial_capital,
        cost_per_side_bps=args.cost_bps,
        benchmark=args.benchmark,
        stop_loss_atr_multiple=stop_mult,
    )

    date_from = _parse_date_opt(args.date_from)
    date_to = _parse_date_opt(args.date_to)

    print(f"Loading composites from {args.reports_glob}...")
    paths = [Path(p) for p in glob.glob(args.reports_glob)]
    observations = _load_observations(paths, universe=universe, cutoff=date_to)
    weeks_all = fridays_from_observations(observations)
    weeks = filter_weeks_by_range(weeks_all, date_from=date_from, date_to=date_to)
    if len(weeks) < 2:
        raise SystemExit(
            f"Need at least 2 rebalance weeks; got {len(weeks)} after filters."
        )
    print(f"Resolved {len(weeks)} rebalance weeks "
          f"({weeks[0].isoformat()} → {weeks[-1].isoformat()}) over {len(universe)} tickers.")

    fetch_symbols = sorted(set(universe) | {args.benchmark})
    print(f"Fetching daily closes for {len(fetch_symbols)} symbols via yfinance...")
    daily_closes = _fetch_weekly_close(fetch_symbols, start=weeks[0], end=weeks[-1])
    if daily_closes.empty:
        raise SystemExit("yfinance returned no usable price data; aborting.")
    weekly_prices = _project_to_weeks(daily_closes, weeks)
    n_priced = (~weekly_prices.isna()).any(axis=0).sum()
    print(f"Got weekly close coverage for {int(n_priced)}/{len(fetch_symbols)} symbols.")
    intra_week_min_close = _build_intra_week_min_close(daily_closes, weeks)
    if stop_mult > 0:
        n_with_mins = sum(1 for w in weeks if intra_week_min_close.get(w))
        print(
            f"Intra-week min closes built for {n_with_mins}/{len(weeks) - 1} weeks; "
            f"stop_loss_atr_multiple={stop_mult:.2f}."
        )
    else:
        print("Stops disabled (stop_loss_atr_multiple=0).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for policy in args.policies:
        sizing_for_policy = SizingConfig(
            policy=policy,
            max_per_name=base_sizing.max_per_name,
            max_long_exposure=base_sizing.max_long_exposure,
            min_position_weight=base_sizing.min_position_weight,
            enable_bearish=base_sizing.enable_bearish,
            stop_loss_atr_multiple=base_sizing.stop_loss_atr_multiple,
            entry_pullback_to=base_sizing.entry_pullback_to,
            max_entry_pullback_pct=base_sizing.max_entry_pullback_pct,
            exit_patience_atrs=base_sizing.exit_patience_atrs,
            stale_composite_days=base_sizing.stale_composite_days,
            stale_signal_decay=base_sizing.stale_signal_decay,
            composite_threshold=base_sizing.composite_threshold,
            top_n=args.top_n,
            universe=base_sizing.universe,
        )
        print(f"Running policy `{policy}`...")
        res = run_simulation(
            weeks=weeks,
            observations=observations,
            prices=weekly_prices,
            sizing_config=sizing_for_policy,
            sim_config=sim_config,
            policy_name=policy,
            intra_week_min_close=intra_week_min_close,
        )
        results.append(res)
        _write_per_policy_json(res, out_dir / f"sim_{policy}.json")

    benchmark_result = None
    if args.include_benchmark and args.benchmark in weekly_prices.columns:
        print(f"Running benchmark `{args.benchmark}` buy-and-hold...")
        bench_sizing = SizingConfig(
            policy="equal_weight_bullish",
            max_per_name=1.0,
            max_long_exposure=1.0,
            min_position_weight=0.0,
            universe=(args.benchmark,),
        )
        benchmark_result = run_simulation(
            weeks=weeks,
            observations=observations,
            prices=weekly_prices,
            sizing_config=bench_sizing,
            sim_config=sim_config,
            policy_name=f"{args.benchmark}_baseline",
            benchmark_only=True,
        )
        _write_per_policy_json(benchmark_result, out_dir / f"sim_{args.benchmark}_baseline.json")

    slice_label = (
        f"Slice: {weeks[0].isoformat()} → {weeks[-1].isoformat()} "
        f"({len(weeks)} weeks, {len(universe)} tickers)"
    )
    md = render_policy_comparison_markdown(
        results,
        benchmark=benchmark_result,
        title="Portfolio policy comparison",
        slice_label=slice_label,
    )
    (out_dir / "policy_comparison.md").write_text(md)
    print(f"Wrote: {out_dir / 'policy_comparison.md'}")

    print("\n=== summary ===")
    for r in results + ([benchmark_result] if benchmark_result else []):
        m = r.metrics
        print(
            f"  {r.policy_name:30s}  end=${m['end_equity']:>11,.0f}  "
            f"CAGR={m['cagr']*100:+6.2f}%  Sharpe={m['sharpe_annualized']:+5.2f}  "
            f"MaxDD={m['max_drawdown']*100:+6.2f}%  TO={m['total_one_way_turnover']:.1f}x  "
            f"Stops={int(m.get('n_stops_hit', 0))}"
        )


def _write_per_policy_json(res, path: Path) -> None:
    payload = {
        "policy_name": res.policy_name,
        "config": {
            "initial_capital": res.config.initial_capital,
            "cost_per_side_bps": res.config.cost_per_side_bps,
            "benchmark": res.config.benchmark,
            "stale_composite_weeks": res.config.stale_composite_weeks,
            "stop_loss_atr_multiple": res.config.stop_loss_atr_multiple,
        },
        "metrics": res.metrics,
        "states": [
            {
                "week": s.week.isoformat(),
                "equity_pre_trade": s.equity_pre_trade,
                "equity_post_trade": s.equity_post_trade,
                "realized_return": s.realized_return,
                "one_way_turnover": s.one_way_turnover,
                "cost_paid": s.cost_paid,
                "target_weights": s.target_weights,
                "n_stops_hit": s.n_stops_hit,
            }
            for s in res.states
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
