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
    format_option_positions_section,
    load_latest_signals,
)
from portfolio.snapshot import load_positions_payload
from portfolio.options import (
    book_greeks,
    enrich_with_chain,
    fetch_current_chain,
    load_option_positions,
)
from portfolio.sizing import SizingConfig, sizing_config_from_dict
from portfolio.risk import SectorShock


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
    p.add_argument(
        "--confidence-calibration-path",
        default="configs/confidence_calibration.json",
        help=(
            "Path to the Phase-5 isotonic calibration JSON. When the file "
            "exists, heuristic confidence values baked into older reports "
            "are REPLACED with calibrated values at read time. Pass an "
            "empty string to disable (use heuristic as-emitted)."
        ),
    )
    p.add_argument(
        "--no-shock-refresh", action="store_true",
        help=(
            "Disable the mid-week shock-triggered composite refresh. When "
            "enabled (default), a VIX spike or sector ETF crash on the "
            "current trading session triggers a fresh `analysis_mvp.py` "
            "run for every held position before signals are loaded — so the "
            "daily layer sees post-shock composites instead of last Friday's "
            "stale ones. Back-dated --as-of runs always skip the refresh."
        ),
    )
    p.add_argument(
        "--shock-vix-pct-threshold", type=float, default=0.15,
        help=(
            "VIX percent rise vs prior close that triggers a mid-week "
            "refresh. Default 0.15 (15%%). Combined OR with the absolute "
            "threshold."
        ),
    )
    p.add_argument(
        "--shock-vix-absolute-threshold", type=float, default=25.0,
        help=(
            "Absolute VIX level that triggers a mid-week refresh. Default "
            "25 — the upper edge of the calm-trend regime classifier in "
            "scoring.py. Combined OR with the pct threshold."
        ),
    )
    p.add_argument(
        "--paper-log-dir", default="reports/paper_trading",
        help=(
            "Where to append per-day recommendation JSONL files. Used by "
            "scripts/paper_trading_report.py to compute follow-rate + "
            "attribution. Default reports/paper_trading."
        ),
    )
    p.add_argument(
        "--no-paper-log", action="store_true",
        help=(
            "Disable the paper-trading recommendation logger. Mostly for "
            "smoke tests; for production use keep the log on."
        ),
    )
    p.add_argument(
        "--ml-shadow-config",
        default="configs/ml_models.yaml",
        help=(
            "Optional shadow-ML config. When present, daily paper-trading "
            "recommendations include non-actionable ML scores under "
            "`ml_shadow`; production actions are unchanged."
        ),
    )
    p.add_argument(
        "--no-ml-shadow", action="store_true",
        help="Disable shadow-ML scoring in the paper-trading log.",
    )
    return p.parse_args()


def _load_positions(path: Path) -> tuple[float, dict[str, Position]]:
    payload = json.loads(path.read_text())
    loaded = load_positions_payload(payload)
    return loaded.cash, loaded.positions


def _load_sizing_config(path: Path) -> SizingConfig:
    data = yaml.safe_load(path.read_text()) or {}
    return sizing_config_from_dict(data)


def _load_sector_map(universe_path: Path = Path("configs/universe.yaml")) -> dict[str, str]:
    """Load the `sectors:` block from universe.yaml. Returns empty dict
    when the file is missing — caller treats that as 'no sector tags.'"""
    try:
        data = yaml.safe_load(universe_path.read_text()) or {}
    except Exception:
        return {}
    raw = data.get("sectors") or {}
    return {str(k).upper(): str(v) for k, v in raw.items() if v}


def _load_risk_limits(path: Path):
    """Read the optional `risk_limits:` block from sizing.yaml. Returns
    `None` when missing — the caller treats that as "no risk caps."""
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    block = data.get("risk_limits")
    if not isinstance(block, dict):
        return None
    from portfolio.risk import RiskLimits
    allowed = {"max_sector_exposure", "max_portfolio_beta", "max_pair_correlation"}
    kwargs = {k: float(v) for k, v in block.items() if k in allowed}
    try:
        return RiskLimits(**kwargs)
    except (TypeError, ValueError):
        return None


def fetch_betas_and_correlations(
    symbols: list[str],
    *,
    benchmark: str = "SPY",
    lookback_days: int = 90,
) -> tuple[dict[str, float | None], dict[str, dict[str, float]]]:
    """Fetch ~60-trading-day return series for `symbols` + benchmark, then
    compute per-symbol beta vs benchmark and the pairwise correlation
    matrix. yfinance failures degrade gracefully — missing symbols just
    show up with `None` beta / absent from the correlation matrix."""
    from portfolio.risk import (
        compute_beta_vs_benchmark,
        compute_pairwise_correlations,
    )
    syms = sorted(set([s.upper() for s in symbols] + [benchmark]))
    returns: dict[str, list[float]] = {}
    for sym in syms:
        try:
            raw = yf.download(
                sym, period=f"{lookback_days}d", interval="1d",
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception:
            continue
        if raw is None or getattr(raw, "empty", True):
            continue
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try:
            closes = df["Close"].dropna()
            rets = closes.pct_change().dropna().tolist()
            if len(rets) >= 20:
                returns[sym] = [float(r) for r in rets]
        except Exception:
            continue
    bench_returns = returns.get(benchmark, [])
    betas: dict[str, float | None] = {}
    for sym in symbols:
        sym_u = sym.upper()
        rs = returns.get(sym_u)
        if rs is None or not bench_returns:
            betas[sym_u] = None
            continue
        betas[sym_u] = compute_beta_vs_benchmark(rs, bench_returns)
    corr = compute_pairwise_correlations(
        {s: returns[s] for s in symbols if s.upper() in returns}
    )
    return betas, corr


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
        return PriceContext(None, None, None, source="unavailable")
    if raw is None or getattr(raw, "empty", True):
        return PriceContext(None, None, None, source="unavailable")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        close = df["Close"].dropna()
        last_close = float(close.iloc[-1])
    except Exception:
        return PriceContext(None, None, None, source="unavailable")
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    if sma20 is not None and (pd.isna(sma20) or sma20 <= 0):
        sma20 = None
    atr14 = _atr14(df)
    return PriceContext(last_close=last_close, sma20=sma20, atr14=atr14, source="yfinance")


def fetch_prices_for_universe(symbols: list[str]) -> dict[str, PriceContext]:
    out: dict[str, PriceContext] = {}
    for sym in symbols:
        out[sym.upper()] = fetch_price_context(sym)
    return out


def _fetch_day_pct_change(symbol: str) -> float | None:
    """Return latest daily percent change from yfinance, or None.

    During the live session Yahoo's latest daily bar may be an in-progress
    bar. That is exactly what the same-day circuit breaker wants. Historical
    backtests avoid this helper entirely unless `as_of == today`.
    """
    try:
        raw = yf.download(
            symbol,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if raw is None or getattr(raw, "empty", True):
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        prev = float(closes.iloc[-2])
        latest = float(closes.iloc[-1])
    except Exception:
        return None
    if prev <= 0:
        return None
    return (latest / prev) - 1.0


def detect_vix_spike(
    *,
    pct_change_threshold: float = 0.15,
    absolute_threshold: float = 25.0,
) -> tuple[bool, float | None, float | None, float | None]:
    """Detect a VIX spike vs prior close.

    Returns ``(spiked, current_vix, prior_close, pct_change)``. ``spiked``
    is True when EITHER:
      - VIX rose by `pct_change_threshold` (default 15%) vs prior close,
      - OR absolute VIX > `absolute_threshold` (default 25 — the upper
        edge of the calm-trend regime classifier in `scoring.py`).
    Returns ``(False, None, None, None)`` on yfinance failure.
    """
    try:
        raw = yf.download(
            "^VIX",
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return (False, None, None, None)
    if raw is None or getattr(raw, "empty", True):
        return (False, None, None, None)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return (False, None, None, None)
        prior = float(closes.iloc[-2])
        current = float(closes.iloc[-1])
    except Exception:
        return (False, None, None, None)
    if prior <= 0:
        return (False, current, prior, None)
    pct = (current / prior) - 1.0
    spiked = (pct >= pct_change_threshold) or (current >= absolute_threshold)
    return (bool(spiked), current, prior, pct)


def maybe_refresh_held_composites(
    *,
    positions: dict[str, Position],
    as_of: date,
    reports_dir: Path = Path("reports/analysis_mvp"),
    config: SizingConfig | None = None,
    vix_pct_threshold: float = 0.15,
    vix_absolute_threshold: float = 25.0,
    disable: bool = False,
) -> dict[str, str]:
    """If a market shock condition is met, refresh weekly composites for
    held positions via `analysis_mvp.py` BEFORE today's daily signals fire.

    The default daily-signals flow reads composites that were generated last
    Friday. On a sudden shock (VIX spike, sector ETF crash) those composites
    are stale and won't reflect the new vol regime. This helper detects the
    shock conditions and runs `analysis_mvp.py` for each held ticker so the
    daily layer sees fresh composites.

    Skip conditions (no refresh):
    - `disable=True`
    - `as_of != today` (back-dated replay — no shock possible)
    - No positions (nothing to refresh)
    - No VIX spike AND no sector ETF shock above threshold

    Returns a dict mapping symbol → result string ("refreshed" | "failed:
    <reason>"). Empty dict means no refresh was triggered.
    """
    if disable or as_of != date.today() or not positions:
        return {}
    spiked, vix_now, vix_prior, vix_pct = detect_vix_spike(
        pct_change_threshold=vix_pct_threshold,
        absolute_threshold=vix_absolute_threshold,
    )
    sector_threshold = (
        float(getattr(config, "sector_shock_drop_pct", 0.03))
        if config is not None else 0.03
    )
    sector_etfs = (
        dict(getattr(config, "sector_shock_etfs", {}) or {})
        if config is not None else {}
    )
    sector_shocked_etfs: list[tuple[str, float]] = []
    for sector, etf in sector_etfs.items():
        etf_u = str(etf).upper()
        if not etf_u:
            continue
        pct = _fetch_day_pct_change(etf_u)
        if pct is not None and pct <= -sector_threshold:
            sector_shocked_etfs.append((etf_u, pct))
    if not spiked and not sector_shocked_etfs:
        return {}

    # Shock detected. Surface the trigger inline so operators see WHY a
    # mid-week regen kicked off (rather than wondering at a slow daily run).
    trigger_parts: list[str] = []
    if spiked:
        trigger_parts.append(
            f"VIX={vix_now:.1f} (prior {vix_prior:.1f}, "
            f"Δ {vix_pct*100:+.1f}%)"
        )
    if sector_shocked_etfs:
        s = ", ".join(f"{etf} {pct*100:+.1f}%" for etf, pct in sector_shocked_etfs[:3])
        trigger_parts.append(f"sector ETFs: {s}")
    print(f"  ⚠ Shock detected ({'; '.join(trigger_parts)}) — refreshing held composites")

    # Refresh composites for HELD positions only (typically <20 tickers, fast).
    # Calls analysis_mvp.py via subprocess so this script doesn't import the
    # heavy pipeline. Uses the same defaults (data-provider polygon,
    # calibration + regime weights paths auto-loaded).
    import subprocess
    import sys
    today_iso = as_of.isoformat()
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for sym in sorted(positions.keys()):
        try:
            r = subprocess.run(
                [
                    sys.executable, "analysis_mvp.py",
                    "--ticker", sym,
                    "--date", today_iso,
                    "--no-markdown",
                    "--no-json-stdout",
                    "--output-dir", str(reports_dir),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            results[sym] = "refreshed" if r.returncode == 0 else f"failed: rc={r.returncode}"
        except subprocess.TimeoutExpired:
            results[sym] = "failed: timeout"
        except Exception as exc:
            results[sym] = f"failed: {type(exc).__name__}"
        print(f"    {sym}: {results[sym]}")
    return results


def fetch_sector_shocks(
    signals: dict[str, object],
    *,
    config: SizingConfig,
    as_of: date,
) -> dict[str, SectorShock]:
    """Detect configured same-day sector ETF shocks for active sectors."""
    if not getattr(config, "sector_shock_guard_enabled", True):
        return {}
    if as_of != date.today():
        return {}
    sector_etfs = getattr(config, "sector_shock_etfs", {}) or {}
    threshold = float(getattr(config, "sector_shock_drop_pct", 0.03))
    sectors = sorted({
        getattr(sig, "sector", None)
        for sig in signals.values()
        if sig is not None and getattr(sig, "sector", None)
    })
    shocks: dict[str, SectorShock] = {}
    pct_by_etf: dict[str, float | None] = {}
    for sector in sectors:
        etf = str(sector_etfs.get(str(sector), "") or "").upper()
        if not etf:
            continue
        if etf not in pct_by_etf:
            pct_by_etf[etf] = _fetch_day_pct_change(etf)
        pct_change = pct_by_etf.get(etf)
        if pct_change is None or pct_change > -threshold:
            continue
        shocks[str(sector)] = SectorShock(
            sector=str(sector),
            trigger_symbol=etf,
            pct_change=float(pct_change),
            threshold=threshold,
        )
    return shocks


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

    # Mid-week shock refresh: if VIX spikes or a sector ETF crashes today,
    # refresh held-position composites BEFORE loading signals so the daily
    # layer sees the post-shock fear_greed_regime / IV factors instead of
    # last Friday's stale composites. Skipped on back-dated runs.
    maybe_refresh_held_composites(
        positions=positions,
        as_of=as_of,
        reports_dir=Path("reports/analysis_mvp"),
        config=sizing_config,
        vix_pct_threshold=float(args.shock_vix_pct_threshold),
        vix_absolute_threshold=float(args.shock_vix_absolute_threshold),
        disable=bool(args.no_shock_refresh),
    )

    print(f"Loading reports from {args.reports_glob}...")
    report_paths = [Path(p) for p in glob.glob(args.reports_glob)]
    cal_path = args.confidence_calibration_path or None
    if cal_path and not Path(cal_path).exists():
        cal_path = None
    signals = load_latest_signals(
        report_paths,
        universe=universe,
        as_of=as_of,
        calibration_path=cal_path,
    )
    if cal_path:
        print(f"Calibrated confidence at read time via {cal_path}")
    found = sum(1 for s in signals.values() if s is not None)
    print(f"Resolved {found}/{len(universe)} latest signals (as of {as_of.isoformat()}).")

    # Fall back to authoritative sector tags from configs/universe.yaml when
    # the report's industry_context.sector is None (yfinance often fails to
    # populate it under threading). Section 26.
    sector_map = _load_sector_map()
    if sector_map:
        from dataclasses import replace
        for sym, sig in list(signals.items()):
            if sig is None or sig.sector is not None:
                continue
            fallback = sector_map.get(sym.upper())
            if fallback:
                signals[sym] = replace(sig, sector=fallback)

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

    # Section 26 portfolio risk caps. Optional — only fires when the
    # sizing.yaml carries a `risk_limits:` block. Betas + correlations
    # come from 60-day yfinance returns.
    risk_limits = _load_risk_limits(Path(args.sizing_config))
    beta_map: dict[str, float | None] = {}
    corr_matrix: dict[str, dict[str, float]] = {}
    if risk_limits is not None and not args.no_prices:
        print(f"Fetching betas + correlations for {len(universe)} symbols vs SPY...")
        beta_map, corr_matrix = fetch_betas_and_correlations(
            universe, benchmark="SPY", lookback_days=90,
        )
        n_betas = sum(1 for b in beta_map.values() if b is not None)
        print(f"Computed {n_betas}/{len(universe)} betas + correlation matrix.")

    sector_shocks: dict[str, SectorShock] = {}
    if not args.no_prices and sizing_config.sector_shock_guard_enabled:
        print("Checking same-day sector shock guards...")
        sector_shocks = fetch_sector_shocks(
            signals,
            config=sizing_config,
            as_of=as_of,
        )
        if sector_shocks:
            triggered = ", ".join(
                f"{s} ({sh.trigger_symbol} {sh.pct_change*100:.1f}%)"
                for s, sh in sorted(sector_shocks.items())
            )
            print(f"Sector shock guard triggered: {triggered}")

    actions, summary = compute_actions(
        signals=signals,
        positions=positions,
        prices=prices,
        config=sizing_config,
        cash=cash,
        as_of=as_of,
        risk_limits=risk_limits,
        beta_map=beta_map,
        correlation_matrix=corr_matrix,
        sector_shocks=sector_shocks,
    )

    ml_shadow_by_symbol: dict[str, dict[str, object]] = {}
    ml_shadow_config = getattr(args, "ml_shadow_config", None)
    if (
        not getattr(args, "no_ml_shadow", False)
        and ml_shadow_config
        and Path(ml_shadow_config).exists()
        and not args.no_prices
    ):
        try:
            from tradingagents.analysis_only.ml_shadow import compute_shadow_predictions

            print("Computing non-actionable ML shadow predictions for paper log...")
            ml_shadow_by_symbol = compute_shadow_predictions(
                report_paths=report_paths,
                signals=signals,
                as_of_date=as_of.isoformat(),
                config_path=ml_shadow_config,
            )
            print(f"ML shadow predictions attached for {len(ml_shadow_by_symbol)} symbol(s).")
        except Exception as exc:
            print(f"  ⚠ ML shadow scoring failed: {type(exc).__name__}: {exc}")
            ml_shadow_by_symbol = {}
    elif getattr(args, "no_ml_shadow", False):
        print("Skipping ML shadow scoring (--no-ml-shadow).")
    elif args.no_prices:
        print("Skipping ML shadow scoring (--no-prices).")

    # Load option positions from the same positions ledger (Section 27).
    # Backward-compatible: positions without an `options` field yield an
    # empty dict and the section renders to "" (no-op).
    positions_path = Path(args.positions)
    try:
        positions_payload = json.loads(positions_path.read_text())
    except Exception:
        positions_payload = {}
    options_by_symbol = load_option_positions(positions_payload)

    enriched_by_symbol: dict[str, list] = {}
    book_greeks_by_symbol: dict[str, object] = {}
    if options_by_symbol:
        print(
            f"Loading current option chains for {len(options_by_symbol)} "
            f"symbol(s) with option positions..."
        )
        for sym, opts in options_by_symbol.items():
            chain = fetch_current_chain(sym)
            if not chain:
                print(f"  {sym}: chain unavailable; positions render without Greeks.")
            enriched = enrich_with_chain(opts, chain)
            enriched_by_symbol[sym] = enriched
            shares_held = int(positions.get(sym, Position(0, 0.0)).shares or 0)
            bg = book_greeks(enriched, shares=shares_held)
            if bg is not None:
                book_greeks_by_symbol[sym] = bg

    report_md = format_daily_report(actions, summary, config=sizing_config, as_of=as_of)
    options_section = format_option_positions_section(
        options_by_symbol=options_by_symbol,
        enriched_by_symbol=enriched_by_symbol,
        book_greeks_by_symbol=book_greeks_by_symbol,
        as_of=as_of,
    )
    if options_section:
        # Insert the options section before the trailing newline.
        report_md = report_md.rstrip() + "\n\n" + options_section
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

    # Paper-trading dry-run log: persist recommendations to JSONL so the
    # weekly reporter (scripts/paper_trading_report.py) can join them to
    # executions (logged via scripts/log_execution.py) and produce a
    # follow-rate + attribution report. Skipped on --no-paper-log.
    if not getattr(args, "no_paper_log", False):
        try:
            from portfolio.paper_trading import (
                RecommendationRecord,
                now_utc_iso,
                write_recommendations_for_day,
            )
            paper_log_dir = Path(getattr(args, "paper_log_dir", "reports/paper_trading"))
            now_iso = now_utc_iso()
            recs = [
                RecommendationRecord(
                    as_of_date=as_of.isoformat(),
                    symbol=a.symbol,
                    action=a.action,
                    direction=a.direction,
                    composite=a.composite,
                    confidence=a.confidence,
                    target_weight=float(a.target_weight),
                    current_weight=float(a.current_weight),
                    delta_pp=float(a.delta_pp),
                    target_shares=int(a.target_shares),
                    current_shares=int(a.current_shares),
                    delta_shares=int(a.delta_shares),
                    limit_price=a.limit_price,
                    stop_loss=a.stop_loss,
                    last_close=a.last_close,
                    sma20=a.sma20,
                    atr14=a.atr14,
                    signal_age_days=a.signal_age_days,
                    price_source=a.price_source,
                    notes=list(a.notes or []),
                    review_gate_status=getattr(a, "review_gate_status", None),
                    review_gate_reason=getattr(a, "review_gate_reason", None),
                    ml_shadow=dict(ml_shadow_by_symbol.get(a.symbol.upper(), {})),
                    generated_at_utc=now_iso,
                )
                for a in actions
            ]
            pt_path = write_recommendations_for_day(
                recs, base_dir=paper_log_dir, as_of_date=as_of.isoformat(),
            )
            print(f"Paper-trading log: wrote {len(recs)} recommendations → {pt_path}")
        except Exception as exc:
            # Logging must NEVER break the daily-signals run. Surface the
            # error and continue — the user still gets their report.
            print(f"  ⚠ paper-trading log failed: {type(exc).__name__}: {exc}")

    # Brief one-line console summary for the impatient.
    counts: dict[str, int] = {}
    for a in actions:
        counts[a.action] = counts.get(a.action, 0) + 1
    counts_str = " · ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"Summary: {counts_str}")


if __name__ == "__main__":
    main()
