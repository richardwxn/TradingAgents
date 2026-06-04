"""Score past analysis-only reports against forward returns.

Usage:
    python backtest.py                              # default reports/analysis_mvp/*.json
    python backtest.py --reports-glob 'out/**/*.json'
    python backtest.py --horizons 5 20 60 --output-dir backtest/results

What this does:
- Load every `AnalysisReport` JSON matched by `--reports-glob`.
- For each (symbol, as_of_date), pull forward returns at the requested
  trading-day horizons via yfinance (price data is PIT, so this is
  safe even for old reports).
- Aggregate hit-rate, mean / median / P25 / P75 by direction, confidence
  bucket, and composite-score bucket.
- Write `summary.json` + `summary.md` (plus a per-record CSV so you can
  pivot it yourself).

Hit definition:
- bullish  → forward_return > 0
- bearish  → forward_return < 0
- neutral  → |forward_return| < `--neutral-band` (default 2%)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from tradingagents.analysis_only.backtest import (
    BacktestRecord,
    explode_records_to_factors,
    ic_signed_weights,
    rebuild_records_with_weights,
    recommend_asymmetric_thresholds,
    recommend_direction_threshold,
    regime_walk_forward_backtest,
    regime_walk_forward_timeline_to_dict,
    render_asymmetric_sweep_markdown,
    render_factor_by_ticker_markdown,
    render_factor_summary_markdown,
    render_summary_markdown,
    render_threshold_sweep_markdown,
    summarize_all,
    summarize_factors,
    summarize_factors_by_ticker,
    sweep_direction_threshold,
    sweep_direction_threshold_asymmetric,
    walk_forward_backtest,
    walk_forward_timeline_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest past analysis-only reports vs forward returns."
    )
    parser.add_argument(
        "--reports-glob",
        default="reports/analysis_mvp/*.json",
        help="Glob pattern for report JSONs (default: reports/analysis_mvp/*.json)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 20, 60],
        help="Forward horizons in trading days (default: 5 20 60)",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.02,
        help="Absolute return threshold counted as a 'hit' for neutral calls",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest/results",
        help="Output directory for summary.json / summary.md / records.csv",
    )
    parser.add_argument(
        "--by-factor",
        action="store_true",
        help=(
            "Also compute per-factor IC + long-short + hit-rate against "
            "forward returns. Writes factor_summary.{json,md}."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help=(
            "Benchmark symbol for benchmark-adjusted returns "
            "(default: SPY). Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--by-ticker",
        action="store_true",
        help=(
            "Also stratify per-factor IC by ticker (single-ticker "
            "robustness check). Implies --by-factor."
        ),
    )
    parser.add_argument(
        "--rebuild-with-ic-weights",
        action="store_true",
        help=(
            "Run a counterfactual: rebuild each record's composite + "
            "direction using IC-derived signed weights (from the 20d "
            "factor IC), then write a side-by-side before/after summary."
        ),
    )
    parser.add_argument(
        "--rebuild-horizon",
        default="ret_20d",
        help="Horizon to source IC weights from when --rebuild-with-ic-weights is set.",
    )
    parser.add_argument(
        "--rebuild-min-ic",
        type=float,
        default=0.05,
        help="Minimum |IC| for a factor to be kept in the IC-weighted model.",
    )
    parser.add_argument(
        "--rebuild-min-n",
        type=int,
        default=50,
        help="Minimum n_paired observations for a factor to be kept in the IC-weighted model.",
    )
    parser.add_argument(
        "--weights-override",
        default=None,
        help=(
            "Path to a JSON file mapping factor -> weight. If set, also "
            "rebuilds composites using these weights and writes "
            "summary_override.{json,md} alongside the standard outputs. "
            "Useful for testing a proposed `DEFAULT_FACTOR_WEIGHTS` change "
            "without regenerating reports."
        ),
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help=(
            "Optional inclusive lower bound on as_of_date (YYYY-MM-DD). "
            "Reports with as_of_date < this value are dropped before "
            "summarization. Use with --date-to for train/test splits."
        ),
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help=(
            "Optional inclusive upper bound on as_of_date (YYYY-MM-DD). "
            "Reports with as_of_date > this value are dropped before "
            "summarization."
        ),
    )
    parser.add_argument(
        "--sweep-direction-threshold",
        nargs="+",
        type=float,
        default=None,
        metavar="THR",
        help=(
            "Sweep direction_for_composite threshold across these "
            "candidate values. Writes threshold_sweep.{json,md} with "
            "direction-conditional hit-rate + mean return per "
            "threshold × horizon. If --weights-override is set, "
            "composites are first rebuilt under those weights; "
            "otherwise composites are taken as-emitted from the JSONs. "
            "Example: --sweep-direction-threshold 0.05 0.10 0.15 0.20 0.25"
        ),
    )
    parser.add_argument(
        "--sweep-recommend-horizon",
        default="ret_20d",
        help="Horizon used by the threshold recommender (default: ret_20d).",
    )
    parser.add_argument(
        "--sweep-min-n-bullish",
        type=int,
        default=30,
        help="Minimum bullish-call count for a threshold to be recommendable.",
    )
    parser.add_argument(
        "--sweep-min-n-bearish",
        type=int,
        default=5,
        help="Minimum bearish-call count for a threshold to be recommendable.",
    )
    parser.add_argument(
        "--sweep-bullish-thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="THR",
        help=(
            "Bullish-side thresholds for the asymmetric 2-D sweep. "
            "Must be paired with --sweep-bearish-thresholds. Writes "
            "threshold_sweep_asymmetric.{json,md}. Example: "
            "--sweep-bullish-thresholds 0.10 0.125 0.15 0.175 0.20"
        ),
    )
    parser.add_argument(
        "--sweep-bearish-thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="THR",
        help=(
            "Bearish-side thresholds (absolute magnitudes; negated "
            "internally) for the asymmetric 2-D sweep. Example: "
            "--sweep-bearish-thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.40"
        ),
    )
    parser.add_argument(
        "--sweep-bearish-precision-floor",
        type=float,
        default=0.50,
        help=(
            "Minimum bearish hit-rate at the recommender's horizon for "
            "a bearish threshold to be recommendable (default: 0.50). "
            "If no cell clears this floor the recommender returns "
            "bearish_pick=None — explicit signal that the bearish side "
            "cannot be fixed by threshold alone."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Run the Phase 3 walk-forward backtest. Rolling refit of IC "
            "weights with a gap, scoring records out-of-sample only. "
            "Writes summary_walk_forward.{json,md} and "
            "walk_forward_weights_timeline.json. Implies --by-factor "
            "(needs factor_scores)."
        ),
    )
    parser.add_argument(
        "--wf-refit-freq-weeks",
        type=int, default=4,
        help="Refit cadence in weeks (default: 4).",
    )
    parser.add_argument(
        "--wf-train-window-weeks",
        type=int, default=52,
        help="Trailing training window in weeks (default: 52).",
    )
    parser.add_argument(
        "--wf-gap-weeks",
        type=int, default=4,
        help=(
            "Gap (weeks) between train-window end and the refit anchor. "
            "Must be >= longest forward horizon in weeks to prevent label "
            "overlap. Default 4 covers ret_20d (20 trading days ~ 4 weeks)."
        ),
    )
    parser.add_argument(
        "--wf-first-refit-after-weeks",
        type=int, default=26,
        help=(
            "Skip the first N weeks of the corpus so the very first "
            "training window has enough samples (default: 26)."
        ),
    )
    parser.add_argument(
        "--wf-horizon",
        default="ret_20d",
        help="Forward-return column used to fit IC weights (default: ret_20d).",
    )
    parser.add_argument(
        "--wf-min-abs-ic",
        type=float, default=0.05,
        help="Minimum |IC| for a factor to enter the walk-forward weights.",
    )
    parser.add_argument(
        "--wf-min-n",
        type=int, default=50,
        help="Minimum n_paired observations per training window.",
    )
    parser.add_argument(
        "--regime-walk-forward",
        action="store_true",
        help=(
            "Run the Phase 4 regime-conditional walk-forward backtest. "
            "Fits per-regime weights when a regime has enough samples, "
            "falls back to global weights otherwise. Writes "
            "summary_walk_forward_by_regime.{json,md} and "
            "walk_forward_regime_timeline.json. Implies --walk-forward."
        ),
    )
    parser.add_argument(
        "--regime-min-samples", type=int, default=250,
        help=(
            "Minimum training-window samples per regime to ship a "
            "regime-specific weight vector (Phase 4 gate: 250)."
        ),
    )
    parser.add_argument(
        "--regime-min-ic-lift", type=float, default=0.02,
        help=(
            "Minimum train-window IC lift vs global IC required to ship "
            "a regime-specific weight vector (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--eligible-regimes",
        nargs="+",
        default=["chop", "trend_on"],
        help=(
            "Regimes allowed to receive regime-specific weights. Others "
            "fall back to global weights (default: chop trend_on)."
        ),
    )
    parser.add_argument(
        "--no-require-regime-ic-ge-global",
        action="store_true",
        help=(
            "Disable the regime IC >= global IC + lift gate. Intended for "
            "diagnostics only."
        ),
    )
    parser.add_argument(
        "--include-llm-critic", action="store_true",
        help=(
            "Phase 6: load the `llm_critic` block from each report (if "
            "present and status=ok) and emit critic_* columns in "
            "records.csv. Has no effect on the summary metrics."
        ),
    )
    return parser.parse_args()


def load_records(
    report_paths: list[str],
    horizons: list[int],
    *,
    capture_factor_scores: bool = False,
    capture_market_context: bool = False,
    capture_llm_critic: bool = False,
    benchmark_symbol: str | None = None,
) -> list[BacktestRecord]:
    by_symbol: dict[str, list[BacktestRecord]] = {}
    records: list[BacktestRecord] = []
    for path in sorted(report_paths):
        try:
            payload = json.loads(Path(path).read_text())
        except Exception:
            print(f"[skip] cannot read {path}")
            continue
        symbol = (payload.get("symbol") or "").upper()
        as_of = payload.get("as_of_date")
        if not symbol or not as_of:
            continue
        direction = (payload.get("direction") or "neutral").lower()
        model_scoring = (
            (payload.get("key_features") or {}).get("model_scoring") or {}
        )
        composite = model_scoring.get("composite_score")
        factor_scores = None
        if capture_factor_scores:
            factor_scores = model_scoring.get("factor_scores") or []
        market_context = None
        if capture_market_context:
            market_context = (
                (payload.get("key_features") or {}).get("market_context") or {}
            )
        critic_out = None
        disagreement = None
        if capture_llm_critic:
            block = payload.get("llm_critic")
            if isinstance(block, dict) and block.get("status") == "ok":
                critic_out = block.get("output") or None
            multi = payload.get("llm_critic_multi") or {}
            if isinstance(multi, dict):
                d = multi.get("disagreement")
                if isinstance(d, dict):
                    disagreement = d
        rec = BacktestRecord(
            symbol=symbol,
            as_of_date=as_of,
            direction=direction,
            confidence=payload.get("confidence"),
            composite_score=composite,
            factor_scores=factor_scores,
            market_context=market_context,
            llm_critic=critic_out,
            llm_disagreement=disagreement,
            source_path=path,
        )
        records.append(rec)
        by_symbol.setdefault(symbol, []).append(rec)

    # Fetch prices once per symbol over the union span needed.
    for symbol, syms in by_symbol.items():
        attach_forward_returns(symbol, syms, horizons=horizons)

    if benchmark_symbol:
        attach_benchmark_adjusted_returns(
            records, benchmark_symbol=benchmark_symbol, horizons=horizons
        )
    return records


def attach_benchmark_adjusted_returns(
    records: list[BacktestRecord],
    *,
    benchmark_symbol: str,
    horizons: list[int],
) -> None:
    """Compute `stock_ret - benchmark_ret` per record per horizon.

    The benchmark price series is anchored on each record's `as_of_date`
    so the horizon matches the stock side trade-day-for-trade-day.
    Records where the benchmark anchor or future close is unavailable
    get `None` for that horizon's adjusted return.
    """
    if not records:
        return
    as_of_dates = sorted({r.as_of_date for r in records})
    try:
        earliest = datetime.strptime(as_of_dates[0], "%Y-%m-%d").date()
        latest = datetime.strptime(as_of_dates[-1], "%Y-%m-%d").date()
    except ValueError:
        return
    end_pad = (max(horizons) if horizons else 0) * 2 + 7
    start = earliest - timedelta(days=5)
    end = latest + timedelta(days=end_pad)

    try:
        df = yf.download(
            benchmark_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    except Exception:
        df = pd.DataFrame()
    if df.empty or "Close" not in df.columns:
        print(f"[warn] no benchmark price data for {benchmark_symbol}")
        return
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    bench_closes: pd.Series = pd.to_numeric(df["Close"], errors="coerce")

    for record in records:
        try:
            cutoff = pd.Timestamp(record.as_of_date).normalize()
        except Exception:
            continue
        on_or_before = df[df["Date"] <= cutoff]
        if on_or_before.empty:
            continue
        anchor_idx = int(on_or_before.index[-1])
        anchor_close = float(bench_closes.iloc[anchor_idx])
        if anchor_close <= 0:
            continue
        for h in horizons:
            field = f"ret_{h}d"
            stock_ret = record.forward_returns.get(field)
            target_idx = anchor_idx + h
            if stock_ret is None or target_idx >= len(bench_closes):
                record.benchmark_adjusted_returns[field] = None
                continue
            future_close = float(bench_closes.iloc[target_idx])
            if future_close <= 0:
                record.benchmark_adjusted_returns[field] = None
                continue
            bench_ret = (future_close / anchor_close) - 1.0
            record.benchmark_adjusted_returns[field] = round(
                float(stock_ret) - bench_ret, 6
            )


def attach_forward_returns(
    symbol: str,
    records: list[BacktestRecord],
    horizons: list[int],
) -> None:
    if not records:
        return
    as_of_dates = sorted({r.as_of_date for r in records})
    try:
        earliest = datetime.strptime(as_of_dates[0], "%Y-%m-%d").date()
        latest = datetime.strptime(as_of_dates[-1], "%Y-%m-%d").date()
    except ValueError:
        return
    # Pad the right side enough to cover the longest horizon plus weekend slack.
    max_horizon_days = max(horizons) if horizons else 0
    end_pad = max_horizon_days * 2 + 7
    start = earliest - timedelta(days=5)
    end = latest + timedelta(days=end_pad)

    try:
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    except Exception:
        df = pd.DataFrame()
    if df.empty or "Close" not in df.columns:
        print(f"[warn] no price data for {symbol}")
        return
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    closes: pd.Series = pd.to_numeric(df["Close"], errors="coerce")

    for record in records:
        try:
            cutoff = pd.Timestamp(record.as_of_date).normalize()
        except Exception:
            continue
        # Closest trading day on or before cutoff = anchor close.
        on_or_before = df[df["Date"] <= cutoff]
        if on_or_before.empty:
            continue
        anchor_idx = int(on_or_before.index[-1])
        anchor_close = float(closes.iloc[anchor_idx])
        if anchor_close <= 0:
            continue
        for h in horizons:
            target_idx = anchor_idx + h
            if target_idx >= len(closes):
                record.forward_returns[f"ret_{h}d"] = None
                continue
            future_close = float(closes.iloc[target_idx])
            if future_close <= 0:
                record.forward_returns[f"ret_{h}d"] = None
                continue
            record.forward_returns[f"ret_{h}d"] = round(
                (future_close / anchor_close) - 1.0, 6
            )


def filter_by_date_range(
    records: list[BacktestRecord],
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[BacktestRecord]:
    """Inclusive temporal filter on `as_of_date` (YYYY-MM-DD strings)."""
    if not date_from and not date_to:
        return records
    out: list[BacktestRecord] = []
    for r in records:
        if date_from and r.as_of_date < date_from:
            continue
        if date_to and r.as_of_date > date_to:
            continue
        out.append(r)
    return out


def write_records_csv(records: list[BacktestRecord], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        horizon_keys = sorted(
            {k for r in records for k in r.forward_returns.keys()}
        )
        adj_keys = sorted(
            {k for r in records for k in r.benchmark_adjusted_returns.keys()}
        )
        adj_cols = [f"{k}_vs_bench" for k in adj_keys]
        # Phase 6 critic columns. Always emitted (None when no critic block)
        # so the CSV schema is stable across runs with and without critic.
        critic_cols = [
            "critic_invalidation_prob_30d",
            "critic_confidence_adjustment",
            "critic_veto",
            "critic_blindspot_count",
        ]
        # Phase 7 disagreement columns.
        disagreement_cols = [
            "disagreement_confidence_adjustment_stdev",
            "disagreement_invalidation_prob_30d_stdev",
            "disagreement_blindspots_jaccard_mean",
            "disagreement_veto_agreement_rate",
            "disagreement_n_models",
        ]
        writer.writerow(
            ["symbol", "as_of_date", "direction", "confidence",
             "composite_score", "source_path"]
            + horizon_keys + adj_cols + critic_cols + disagreement_cols
        )
        for r in records:
            critic = r.llm_critic or {}
            dis = r.llm_disagreement or {}
            writer.writerow(
                [
                    r.symbol,
                    r.as_of_date,
                    r.direction,
                    r.confidence,
                    r.composite_score,
                    r.source_path,
                ]
                + [r.forward_returns.get(k) for k in horizon_keys]
                + [r.benchmark_adjusted_returns.get(k) for k in adj_keys]
                + [
                    critic.get("invalidation_prob_30d"),
                    critic.get("confidence_adjustment"),
                    critic.get("veto"),
                    len(critic.get("factor_blindspots") or [])
                    if critic else None,
                ]
                + [
                    dis.get("confidence_adjustment_stdev"),
                    dis.get("invalidation_prob_30d_stdev"),
                    dis.get("blindspots_jaccard_mean"),
                    dis.get("veto_agreement_rate"),
                    dis.get("n_models"),
                ]
            )


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        raise SystemExit(f"No reports matched: {args.reports_glob}")
    print(f"Loading {len(paths)} reports...")
    benchmark = (args.benchmark or "").strip().upper() or None
    asym_sweep_requested = (
        args.sweep_bullish_thresholds is not None
        and args.sweep_bearish_thresholds is not None
    )
    need_factor_scores = (
        args.by_factor
        or args.by_ticker
        or args.rebuild_with_ic_weights
        or args.weights_override is not None
        or args.walk_forward
        or args.regime_walk_forward
        or (args.sweep_direction_threshold is not None and args.weights_override is not None)
        or (asym_sweep_requested and args.weights_override is not None)
    )
    records = load_records(
        paths,
        horizons=args.horizons,
        capture_factor_scores=need_factor_scores,
        capture_market_context=args.regime_walk_forward,
        capture_llm_critic=args.include_llm_critic,
        benchmark_symbol=benchmark,
    )
    print(f"Loaded {len(records)} records.")
    if args.date_from or args.date_to:
        before = len(records)
        records = filter_by_date_range(
            records, date_from=args.date_from, date_to=args.date_to
        )
        print(
            f"Date filter ({args.date_from or '—'} .. {args.date_to or '—'}): "
            f"{before} → {len(records)} records."
        )
        if not records:
            raise SystemExit("No records remain after date filter.")
    if benchmark:
        n_adj = sum(
            1
            for r in records
            if any(v is not None for v in r.benchmark_adjusted_returns.values())
        )
        print(f"Benchmark-adjusted ({benchmark}) returns attached to {n_adj}/{len(records)} records.")

    return_fields = [f"ret_{h}d" for h in args.horizons]
    summary = summarize_all(
        records,
        return_fields=return_fields,
        neutral_band=args.neutral_band,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(render_summary_markdown(summary))
    write_records_csv(records, out_dir / "records.csv")
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'summary.md'}")
    print(f"Wrote: {out_dir / 'records.csv'}")

    factor_summary_raw: dict | None = None
    if args.by_factor or args.by_ticker or args.rebuild_with_ic_weights:
        by_factor = explode_records_to_factors(records)
        if not by_factor:
            print(
                "[warn] no factor_scores found in any report; "
                "skipping factor analysis."
            )
            return
        n_per_factor = {k: len(v) for k, v in by_factor.items()}
        print(f"Found {len(by_factor)} factors; observations: {n_per_factor}")

        factor_summary_raw = summarize_factors(
            by_factor, return_fields=return_fields, use_benchmark_adjusted=False
        )
        (out_dir / "factor_summary.json").write_text(
            json.dumps(factor_summary_raw, indent=2)
        )
        (out_dir / "factor_summary.md").write_text(
            render_factor_summary_markdown(
                factor_summary_raw,
                return_fields=return_fields,
                title="Per-factor IC summary (raw forward returns)",
                return_label="raw",
            )
        )
        print(f"Wrote: {out_dir / 'factor_summary.json'}")
        print(f"Wrote: {out_dir / 'factor_summary.md'}")

        if benchmark:
            factor_summary_adj = summarize_factors(
                by_factor,
                return_fields=return_fields,
                use_benchmark_adjusted=True,
            )
            (out_dir / "factor_summary_vs_benchmark.json").write_text(
                json.dumps(factor_summary_adj, indent=2)
            )
            (out_dir / "factor_summary_vs_benchmark.md").write_text(
                render_factor_summary_markdown(
                    factor_summary_adj,
                    return_fields=return_fields,
                    title=f"Per-factor IC summary (vs {benchmark})",
                    return_label=f"benchmark-adjusted vs {benchmark}",
                )
            )
            print(f"Wrote: {out_dir / 'factor_summary_vs_benchmark.json'}")
            print(f"Wrote: {out_dir / 'factor_summary_vs_benchmark.md'}")

        if args.by_ticker:
            by_ticker_summary = summarize_factors_by_ticker(
                by_factor, return_fields=return_fields, use_benchmark_adjusted=False
            )
            (out_dir / "factor_summary_by_ticker.json").write_text(
                json.dumps(by_ticker_summary, indent=2)
            )
            (out_dir / "factor_summary_by_ticker.md").write_text(
                render_factor_by_ticker_markdown(
                    by_ticker_summary,
                    return_fields=return_fields,
                    title="Per-factor IC stratified by ticker (raw returns)",
                    return_label="raw",
                )
            )
            print(f"Wrote: {out_dir / 'factor_summary_by_ticker.json'}")
            print(f"Wrote: {out_dir / 'factor_summary_by_ticker.md'}")

    if args.rebuild_with_ic_weights and factor_summary_raw is not None:
        ic_weights = ic_signed_weights(
            factor_summary_raw,
            horizon=args.rebuild_horizon,
            min_abs_ic=args.rebuild_min_ic,
            min_n=args.rebuild_min_n,
        )
        if not ic_weights:
            print(
                "[warn] no factors passed the IC threshold; "
                "skipping counterfactual rebuild."
            )
            return
        print(
            f"Counterfactual: using {len(ic_weights)} factors with "
            f"|IC|≥{args.rebuild_min_ic} at {args.rebuild_horizon}."
        )
        (out_dir / "ic_weights.json").write_text(
            json.dumps(ic_weights, indent=2, sort_keys=True)
        )

        rebuilt = rebuild_records_with_weights(records, weights=ic_weights)
        rebuilt_summary = summarize_all(
            rebuilt,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        (out_dir / "summary_rebuilt.json").write_text(
            json.dumps(rebuilt_summary, indent=2)
        )
        (out_dir / "summary_rebuilt.md").write_text(
            render_summary_markdown(rebuilt_summary)
        )
        print(f"Wrote: {out_dir / 'ic_weights.json'}")
        print(f"Wrote: {out_dir / 'summary_rebuilt.json'}")
        print(f"Wrote: {out_dir / 'summary_rebuilt.md'}")

    if args.weights_override is not None:
        override_path = Path(args.weights_override)
        try:
            override_weights = json.loads(override_path.read_text())
        except Exception as exc:
            print(f"[error] failed to read --weights-override {override_path}: {exc}")
            return
        if not isinstance(override_weights, dict):
            print(f"[error] --weights-override must be a JSON object; got {type(override_weights)}")
            return
        # Coerce to floats; drop anything non-numeric for resilience.
        clean_weights: dict[str, float] = {}
        for key, value in override_weights.items():
            try:
                clean_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        print(
            f"Override: rebuilding composites from {override_path} with "
            f"{len(clean_weights)} factor weights (sum={sum(clean_weights.values()):.4f})."
        )
        overridden = rebuild_records_with_weights(records, weights=clean_weights)
        overridden_summary = summarize_all(
            overridden,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        (out_dir / "summary_override.json").write_text(
            json.dumps(overridden_summary, indent=2)
        )
        (out_dir / "summary_override.md").write_text(
            render_summary_markdown(overridden_summary)
        )
        print(f"Wrote: {out_dir / 'summary_override.json'}")
        print(f"Wrote: {out_dir / 'summary_override.md'}")

    if args.sweep_direction_threshold:
        # If a weights override was provided, sweep under those weights;
        # otherwise sweep against the as-emitted composites from the JSONs.
        sweep_weights: dict[str, float] | None = None
        if args.weights_override is not None:
            try:
                raw = json.loads(Path(args.weights_override).read_text())
                sweep_weights = {
                    str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))
                }
            except Exception as exc:
                print(f"[error] re-reading --weights-override for sweep: {exc}")
                sweep_weights = None
        thresholds = sorted(set(args.sweep_direction_threshold))
        sweep = sweep_direction_threshold(
            records,
            weights=sweep_weights,
            thresholds=thresholds,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        recommendation = recommend_direction_threshold(
            sweep,
            horizon=args.sweep_recommend_horizon,
            min_n_bullish=args.sweep_min_n_bullish,
            min_n_bearish=args.sweep_min_n_bearish,
        )
        sweep["recommendation"] = recommendation
        (out_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
        (out_dir / "threshold_sweep.md").write_text(
            render_threshold_sweep_markdown(
                sweep,
                title=(
                    f"Direction-threshold sweep "
                    f"({'rebuilt' if sweep_weights else 'as-emitted'} composites)"
                ),
            )
        )
        print(f"Wrote: {out_dir / 'threshold_sweep.json'}")
        print(f"Wrote: {out_dir / 'threshold_sweep.md'}")
        bull = recommendation.get("bullish_pick")
        bear = recommendation.get("bearish_pick")
        print(
            f"Recommendation @ {recommendation.get('horizon')}: "
            f"bullish-best threshold={bull.get('threshold') if bull else 'n/a'} "
            f"(n={bull.get('n') if bull else 0}, hit={bull.get('hit_rate') if bull else 'n/a'}); "
            f"bearish-best threshold={bear.get('threshold') if bear else 'n/a'} "
            f"(n={bear.get('n') if bear else 0}, hit={bear.get('hit_rate') if bear else 'n/a'})"
        )

    if args.regime_walk_forward:
        rwf_rebuilt, rwf_timeline = regime_walk_forward_backtest(
            records,
            refit_freq_weeks=args.wf_refit_freq_weeks,
            train_window_weeks=args.wf_train_window_weeks,
            gap_weeks=args.wf_gap_weeks,
            first_refit_after_weeks=args.wf_first_refit_after_weeks,
            horizon=args.wf_horizon,
            min_abs_ic=args.wf_min_abs_ic,
            min_n=args.wf_min_n,
            min_samples_per_regime=args.regime_min_samples,
            min_regime_ic_lift=args.regime_min_ic_lift,
            eligible_regimes=args.eligible_regimes,
            require_regime_ic_ge_global=(
                not args.no_require_regime_ic_ge_global
            ),
        )
        rwf_summary = summarize_all(
            rwf_rebuilt,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        rwf_summary["regime_walk_forward_config"] = {
            "refit_freq_weeks": args.wf_refit_freq_weeks,
            "train_window_weeks": args.wf_train_window_weeks,
            "gap_weeks": args.wf_gap_weeks,
            "first_refit_after_weeks": args.wf_first_refit_after_weeks,
            "horizon": args.wf_horizon,
            "min_abs_ic": args.wf_min_abs_ic,
            "min_n": args.wf_min_n,
            "min_samples_per_regime": args.regime_min_samples,
            "min_regime_ic_lift": args.regime_min_ic_lift,
            "eligible_regimes": args.eligible_regimes,
            "require_regime_ic_ge_global": (
                not args.no_require_regime_ic_ge_global
            ),
            "n_refit_steps": len(rwf_timeline),
            "n_records_scored": len(rwf_rebuilt),
        }
        timeline_payload = regime_walk_forward_timeline_to_dict(rwf_timeline)
        (out_dir / "summary_walk_forward_by_regime.json").write_text(
            json.dumps(rwf_summary, indent=2)
        )
        (out_dir / "summary_walk_forward_by_regime.md").write_text(
            render_summary_markdown(rwf_summary)
        )
        (out_dir / "walk_forward_regime_timeline.json").write_text(
            json.dumps(timeline_payload, indent=2)
        )
        print(f"Wrote: {out_dir / 'summary_walk_forward_by_regime.json'}")
        print(f"Wrote: {out_dir / 'summary_walk_forward_by_regime.md'}")
        print(f"Wrote: {out_dir / 'walk_forward_regime_timeline.json'}")
        regimes_shipped = {
            r for s in rwf_timeline for r in s.regimes_used
        }
        regimes_fellback = {
            r for s in rwf_timeline for r in s.regimes_fellback_to_global
        }
        print(
            f"Regime walk-forward: {len(rwf_timeline)} steps, "
            f"{len(rwf_rebuilt)} OOS records. "
            f"regimes shipped at least once: {sorted(regimes_shipped) or '<none>'}; "
            f"fellback at least once: {sorted(regimes_fellback) or '<none>'}."
        )

    if args.walk_forward:
        wf_rebuilt, wf_timeline = walk_forward_backtest(
            records,
            refit_freq_weeks=args.wf_refit_freq_weeks,
            train_window_weeks=args.wf_train_window_weeks,
            gap_weeks=args.wf_gap_weeks,
            first_refit_after_weeks=args.wf_first_refit_after_weeks,
            horizon=args.wf_horizon,
            min_abs_ic=args.wf_min_abs_ic,
            min_n=args.wf_min_n,
        )
        wf_summary = summarize_all(
            wf_rebuilt,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        timeline_payload = walk_forward_timeline_to_dict(wf_timeline)
        wf_summary["walk_forward_config"] = {
            "refit_freq_weeks": args.wf_refit_freq_weeks,
            "train_window_weeks": args.wf_train_window_weeks,
            "gap_weeks": args.wf_gap_weeks,
            "first_refit_after_weeks": args.wf_first_refit_after_weeks,
            "horizon": args.wf_horizon,
            "min_abs_ic": args.wf_min_abs_ic,
            "min_n": args.wf_min_n,
            "n_refit_steps": len(wf_timeline),
            "n_records_scored": len(wf_rebuilt),
        }
        (out_dir / "summary_walk_forward.json").write_text(
            json.dumps(wf_summary, indent=2)
        )
        (out_dir / "summary_walk_forward.md").write_text(
            render_summary_markdown(wf_summary)
        )
        (out_dir / "walk_forward_weights_timeline.json").write_text(
            json.dumps(timeline_payload, indent=2)
        )
        print(f"Wrote: {out_dir / 'summary_walk_forward.json'}")
        print(f"Wrote: {out_dir / 'summary_walk_forward.md'}")
        print(f"Wrote: {out_dir / 'walk_forward_weights_timeline.json'}")
        print(
            f"Walk-forward: {len(wf_timeline)} refit steps, "
            f"{len(wf_rebuilt)} out-of-sample records scored."
        )

    if asym_sweep_requested:
        asym_weights: dict[str, float] | None = None
        if args.weights_override is not None:
            try:
                raw = json.loads(Path(args.weights_override).read_text())
                asym_weights = {
                    str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))
                }
            except Exception as exc:
                print(f"[error] re-reading --weights-override for asym sweep: {exc}")
                asym_weights = None
        bull_thrs = sorted(set(args.sweep_bullish_thresholds))
        bear_thrs = sorted(set(args.sweep_bearish_thresholds))
        asym_sweep = sweep_direction_threshold_asymmetric(
            records,
            weights=asym_weights,
            bullish_thresholds=bull_thrs,
            bearish_thresholds=bear_thrs,
            return_fields=return_fields,
            neutral_band=args.neutral_band,
        )
        asym_rec = recommend_asymmetric_thresholds(
            asym_sweep,
            horizon=args.sweep_recommend_horizon,
            min_n_bullish=args.sweep_min_n_bullish,
            min_n_bearish=args.sweep_min_n_bearish,
            bearish_precision_floor=args.sweep_bearish_precision_floor,
        )
        asym_sweep["recommendation"] = asym_rec
        (out_dir / "threshold_sweep_asymmetric.json").write_text(
            json.dumps(asym_sweep, indent=2)
        )
        (out_dir / "threshold_sweep_asymmetric.md").write_text(
            render_asymmetric_sweep_markdown(
                asym_sweep,
                title=(
                    f"Asymmetric direction-threshold sweep "
                    f"({'rebuilt' if asym_weights else 'as-emitted'} composites)"
                ),
            )
        )
        print(f"Wrote: {out_dir / 'threshold_sweep_asymmetric.json'}")
        print(f"Wrote: {out_dir / 'threshold_sweep_asymmetric.md'}")
        bp = asym_rec.get("bullish_pick")
        ep = asym_rec.get("bearish_pick")
        print(
            f"Asym recommendation @ {asym_rec.get('horizon')}: "
            f"bullish_threshold={bp.get('threshold') if bp else 'n/a'} "
            f"(n={bp.get('n') if bp else 0}, hit={bp.get('hit_rate') if bp else 'n/a'}); "
            f"bearish_threshold={ep.get('threshold') if ep else 'n/a'} "
            f"(precision floor {asym_rec.get('bearish_precision_floor')}; "
            f"n={ep.get('n') if ep else 0}, hit={ep.get('hit_rate') if ep else 'n/a'})"
        )


if __name__ == "__main__":
    main()
