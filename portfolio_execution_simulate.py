"""Daily action/execution simulator CLI.

Validates the daily signals action layer against daily OHLC bars:
sizing, entry limits, exit limits, stop-loss reminders, turnover, costs,
missed orders, and realized equity curves.

Example:

    python portfolio_execution_simulate.py \\
        --reports-glob "reports/analysis_mvp/*.json" \\
        --sizing-config configs/sizing.yaml \\
        --output-dir backtest/results/execution_full \\
        --policy-grid full
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

from portfolio.execution_simulator import (
    ExecutionSignal,
    default_policy_grid,
    generate_walk_forward_windows,
    render_execution_comparison_markdown,
    render_walk_forward_acceptance_markdown,
    result_to_json,
    run_benchmark_simulation,
    run_daily_execution_simulation,
    summarize_walk_forward_acceptance,
)
from portfolio.signals import _load_signal_from_json
from portfolio.sizing import sizing_config_from_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reports-glob", required=True)
    parser.add_argument("--sizing-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date-from", default=None, help="Inclusive YYYY-MM-DD lower bound.")
    parser.add_argument("--date-to", default=None, help="Inclusive YYYY-MM-DD upper bound.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--policy-grid",
        choices=("baseline", "small", "full"),
        default="full",
        help="Candidate grid size. Use small for quick local smoke runs.",
    )
    parser.add_argument(
        "--max-markdown-rows",
        type=int,
        default=30,
        help="Rows to show in policy_comparison.md; all policy JSONs are still written.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run rolling test windows and write walk_forward_acceptance.{json,md}.",
    )
    parser.add_argument("--train-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    return parser.parse_args()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_execution_signals(
    paths: list[Path],
    *,
    universe: set[str],
    date_to: date | None = None,
) -> dict[date, dict[str, ExecutionSignal]]:
    out: dict[date, dict[str, ExecutionSignal]] = defaultdict(dict)
    for path in paths:
        sig = _load_signal_from_json(path)
        if sig is None or sig.symbol not in universe:
            continue
        try:
            as_of = datetime.strptime(sig.as_of_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        if date_to is not None and as_of > date_to:
            continue
        out[as_of][sig.symbol] = ExecutionSignal(
            symbol=sig.symbol,
            as_of_date=as_of,
            direction=sig.direction,
            composite=sig.composite,
            confidence=sig.confidence,
        )
    return dict(out)


def fetch_daily_ohlc(
    symbols: list[str],
    *,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            raw = yf.download(
                sym,
                start=(start - timedelta(days=30)).isoformat(),
                end=(end + timedelta(days=2)).isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        frame = normalize_ohlc(raw)
        if frame.empty:
            continue
        prices[sym] = frame
    return prices


def normalize_ohlc(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    renamed = {str(c).lower(): c for c in frame.columns}
    required = ["open", "high", "low", "close"]
    if not all(k in renamed for k in required):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.DataFrame({
        "open": frame[renamed["open"]],
        "high": frame[renamed["high"]],
        "low": frame[renamed["low"]],
        "close": frame[renamed["close"]],
        "volume": frame[renamed["volume"]] if "volume" in renamed else 0,
    })
    out.index = pd.to_datetime(out.index)
    return out.dropna(subset=["open", "high", "low", "close"])


def write_execution_artifacts(
    *,
    results,
    benchmark,
    output_dir: Path,
    max_markdown_rows: int = 30,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (output_dir / f"sim_{_safe_name(result.policy.name)}.json").write_text(
            json.dumps(result_to_json(result), indent=2, default=str)
        )
    if benchmark is not None:
        (output_dir / f"sim_{_safe_name(benchmark.policy.name)}.json").write_text(
            json.dumps(result_to_json(benchmark), indent=2, default=str)
        )
    markdown = render_execution_comparison_markdown(
        results,
        benchmark=benchmark,
        max_rows=max_markdown_rows,
    )
    (output_dir / "policy_comparison.md").write_text(markdown)


def write_walk_forward_artifacts(
    *,
    summaries: list[dict],
    output_dir: Path,
    max_markdown_rows: int = 30,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "walk_forward_acceptance.json").write_text(
        json.dumps(summaries, indent=2, default=str)
    )
    (output_dir / "walk_forward_acceptance.md").write_text(
        render_walk_forward_acceptance_markdown(
            summaries,
            max_rows=max_markdown_rows,
        )
    )


def _safe_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep)


def _date_range_from_prices(
    prices: dict[str, pd.DataFrame],
    *,
    date_from: date | None,
    date_to: date | None,
) -> list[date]:
    all_dates = sorted({
        ts.date()
        for frame in prices.values()
        for ts in pd.to_datetime(frame.index)
    })
    if date_from is not None:
        all_dates = [d for d in all_dates if d >= date_from]
    if date_to is not None:
        all_dates = [d for d in all_dates if d <= date_to]
    return all_dates


def main() -> None:
    args = _parse_args()
    date_from = _parse_date(args.date_from)
    date_to = _parse_date(args.date_to)

    sizing_raw = yaml.safe_load(Path(args.sizing_config).read_text()) or {}
    base_config = sizing_config_from_dict(sizing_raw)
    universe = set(base_config.universe)
    if not universe:
        raise SystemExit("No universe configured in sizing config.")

    report_paths = [Path(p) for p in glob.glob(args.reports_glob)]
    signals = load_execution_signals(report_paths, universe=universe, date_to=date_to)
    if not signals:
        raise SystemExit("No usable analysis reports found for the configured universe.")

    signal_min = min(signals)
    signal_max = max(signals)
    fetch_start = date_from or signal_min
    fetch_end = date_to or (signal_max + timedelta(days=90))
    symbols = sorted(universe | {args.benchmark})

    print(f"Fetching daily OHLC for {len(symbols)} symbols...")
    prices = fetch_daily_ohlc(symbols, start=fetch_start, end=fetch_end)
    dates = _date_range_from_prices(prices, date_from=date_from, date_to=date_to)
    if len(dates) < 3:
        raise SystemExit(f"Need at least 3 trading days; got {len(dates)}.")
    print(f"Simulating {len(dates)} trading days ({dates[0]} -> {dates[-1]}).")

    candidates = default_policy_grid(base_config, mode=args.policy_grid)
    print(f"Running {len(candidates)} policy candidate(s)...")
    results = [
        run_daily_execution_simulation(
            dates=dates,
            signals_by_date=signals,
            prices=prices,
            base_config=base_config,
            candidate=candidate,
            initial_capital=args.initial_capital,
            cost_per_side_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
        )
        for candidate in candidates
    ]

    benchmark = None
    if args.benchmark in prices:
        benchmark = run_benchmark_simulation(
            dates=dates,
            prices=prices,
            benchmark=args.benchmark,
            initial_capital=args.initial_capital,
            cost_per_side_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
        )

    out_dir = Path(args.output_dir)
    write_execution_artifacts(
        results=results,
        benchmark=benchmark,
        output_dir=out_dir,
        max_markdown_rows=args.max_markdown_rows,
    )
    if args.walk_forward:
        windows = generate_walk_forward_windows(
            dates,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
        )
        print(f"Running walk-forward acceptance over {len(windows)} window(s)...")
        window_results = []
        for window in windows:
            window_results.append({
                "window": {
                    "train_start": window["train_start"].isoformat(),
                    "train_end": window["train_end"].isoformat(),
                    "test_start": window["test_start"].isoformat(),
                    "test_end": window["test_end"].isoformat(),
                },
                "results": [
                    run_daily_execution_simulation(
                        dates=window["test_dates"],
                        signals_by_date=signals,
                        prices=prices,
                        base_config=base_config,
                        candidate=candidate,
                        initial_capital=args.initial_capital,
                        cost_per_side_bps=args.cost_bps,
                        slippage_bps=args.slippage_bps,
                    )
                    for candidate in candidates
                ],
            })
        summaries = summarize_walk_forward_acceptance(window_results)
        write_walk_forward_artifacts(
            summaries=summaries,
            output_dir=out_dir,
            max_markdown_rows=args.max_markdown_rows,
        )
        print(f"Wrote: {out_dir / 'walk_forward_acceptance.md'}")
    print(f"Wrote: {out_dir / 'policy_comparison.md'}")


if __name__ == "__main__":
    main()
