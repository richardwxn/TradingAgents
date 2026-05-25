"""Daily signals CLI.

Reads the latest weekly analysis report per ticker plus the
user-maintained position ledger and emits a single markdown file per
day with explicit per-name actions, limit prices, and stop-loss
reminders. Composites refresh weekly (via `analysis_mvp.py`); this CLI
is the daily layer that diffs the user's actual positions against the
target portfolio derived from the latest composites and current prices.

Example:

    source .venv/bin/activate
    python daily_signals.py \\
        --reports-glob "reports/analysis_mvp/*.json" \\
        --positions portfolio/positions.json \\
        --sizing-config configs/sizing.yaml \\
        --output-dir reports/daily_signals

By default `--as-of` is today. Pass `--as-of YYYY-MM-DD` to back-date a
report (useful for replay; only composites with `as_of_date <= --as-of`
are considered, so no future data leaks in).
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from portfolio.signals import (
    PriceContext,
    Position,
    compute_actions,
    format_daily_report,
    load_latest_signals,
)
from portfolio.sizing import SizingConfig, sizing_config_from_dict


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports-glob", default="reports/analysis_mvp/*.json",
                   help="Glob for weekly analysis report JSONs.")
    p.add_argument("--positions", default="portfolio/positions.json",
                   help="Path to the user-maintained position ledger.")
    p.add_argument("--sizing-config", default="configs/sizing.yaml",
                   help="YAML config consumed by portfolio/sizing.py.")
    p.add_argument("--output-dir", default="reports/daily_signals",
                   help="Directory to write the markdown report into.")
    p.add_argument("--as-of", default=None,
                   help="Override the run date (YYYY-MM-DD); defaults to today.")
    p.add_argument("--no-prices", action="store_true",
                   help="Skip yfinance price fetching (offline / smoke-test mode).")
    return p.parse_args()


def _load_positions(path: Path) -> tuple[float, dict[str, Position]]:
    payload = json.loads(path.read_text())
    cash = float(payload.get("cash") or 0.0)
    positions: dict[str, Position] = {}
    for sym, raw in (payload.get("positions") or {}).items():
        positions[sym.upper()] = Position(
            shares=float(raw.get("shares") or 0),
            avg_cost=float(raw.get("avg_cost") or 0.0),
        )
    return cash, positions


def _load_sizing_config(path: Path) -> SizingConfig:
    data = yaml.safe_load(path.read_text()) or {}
    return sizing_config_from_dict(data)


def _atr14(df: pd.DataFrame) -> float | None:
    """Wilder-style ATR(14) is fine, but a simple rolling mean of True
    Range is close enough at swing horizons and avoids the recursive
    update. Returns None when there's not enough history."""
    try:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        tr_clean = tr.dropna()
        if len(tr_clean) < 14:
            return None
        return float(tr_clean.rolling(14).mean().iloc[-1])
    except Exception:
        return None


def fetch_price_context(symbol: str) -> PriceContext:
    """Fetch last close + SMA20 + ATR(14) for one ticker.

    Defensive: yfinance is unreliable (401s, threading bugs, malformed
    payloads). Any failure returns an all-None PriceContext so the
    report still renders.
    """
    try:
        raw = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return PriceContext(None, None, None)
    if raw is None or getattr(raw, "empty", True):
        return PriceContext(None, None, None)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        close = df["Close"].dropna()
        last_close = float(close.iloc[-1])
    except Exception:
        return PriceContext(None, None, None)
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    if sma20 is not None and (pd.isna(sma20) or sma20 <= 0):
        sma20 = None
    atr14 = _atr14(df)
    return PriceContext(last_close=last_close, sma20=sma20, atr14=atr14)


def fetch_prices_for_universe(symbols: list[str]) -> dict[str, PriceContext]:
    out: dict[str, PriceContext] = {}
    for sym in symbols:
        out[sym.upper()] = fetch_price_context(sym)
    return out


def main() -> None:
    args = _parse_args()

    as_of = date.today()
    if args.as_of:
        try:
            as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"--as-of must be YYYY-MM-DD; got {args.as_of!r}")

    sizing_config = _load_sizing_config(Path(args.sizing_config))
    cash, positions = _load_positions(Path(args.positions))

    universe = list(sizing_config.universe) or sorted(positions.keys())
    if not universe:
        raise SystemExit("No universe configured and no positions held; nothing to do.")

    print(f"Loading reports from {args.reports_glob}...")
    report_paths = [Path(p) for p in glob.glob(args.reports_glob)]
    signals = load_latest_signals(report_paths, universe=universe, as_of=as_of)
    found = sum(1 for s in signals.values() if s is not None)
    print(f"Resolved {found}/{len(universe)} latest signals (as of {as_of.isoformat()}).")

    # Fetch prices for the universe + any extra held names.
    fetch_symbols = sorted(set(universe) | set(positions.keys()))
    if args.no_prices:
        print("Skipping price fetch (--no-prices); price context will be empty.")
        prices = {sym: PriceContext(None, None, None) for sym in fetch_symbols}
    else:
        print(f"Fetching prices for {len(fetch_symbols)} symbols via yfinance...")
        prices = fetch_prices_for_universe(fetch_symbols)
        n_priced = sum(1 for p in prices.values() if p.last_close is not None)
        print(f"Got prices for {n_priced}/{len(fetch_symbols)} symbols.")

    actions, summary = compute_actions(
        signals=signals,
        positions=positions,
        prices=prices,
        config=sizing_config,
        cash=cash,
        as_of=as_of,
    )

    report_md = format_daily_report(actions, summary, config=sizing_config, as_of=as_of)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{as_of.isoformat()}.md"
    json_path = output_dir / f"{as_of.isoformat()}.json"

    md_path.write_text(report_md)
    json_path.write_text(
        json.dumps(
            {
                "as_of": as_of.isoformat(),
                "config": {
                    "policy": sizing_config.policy,
                    "max_per_name": sizing_config.max_per_name,
                    "max_long_exposure": sizing_config.max_long_exposure,
                    "enable_bearish": sizing_config.enable_bearish,
                    "composite_threshold": sizing_config.composite_threshold,
                },
                "summary": summary,
                "actions": [a.__dict__ for a in actions],
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")

    # Brief one-line console summary for the impatient.
    counts: dict[str, int] = {}
    for a in actions:
        counts[a.action] = counts.get(a.action, 0) + 1
    counts_str = " · ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"Summary: {counts_str}")


if __name__ == "__main__":
    main()
