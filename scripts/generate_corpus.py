"""Generate a historical corpus of analysis reports for backtesting.

Phase 1 corpus target: 12 tech tickers × 150 weekly Fridays = 1,800 reports.
Resumable: skips any (ticker, date) pair that already has a JSON report on
disk, so iterating from the legacy 11×26 corpus is incremental.

Usage:
    python scripts/generate_corpus.py                   # defaults (12 × 150)
    python scripts/generate_corpus.py --workers 4       # more parallelism
    python scripts/generate_corpus.py --tickers NVDA AMD  # subset
    python scripts/generate_corpus.py --pace-seconds 1.5 # extra throttle
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_UNIVERSE_PATH = _REPO_ROOT / "configs" / "universe.yaml"


def load_universe_tickers(path: Path = _UNIVERSE_PATH) -> list[str]:
    """Load tickers from configs/universe.yaml (core + canary, in order).

    Falls back to the original Phase 1 hardcoded list if the yaml is missing
    or unparseable so the CLI still works in a stripped environment.
    """
    fallback = [
        "NVDA", "AMD", "AVGO", "MU", "TSM",
        "ALAB", "COHR", "FIG", "GLW", "LEU",
        "NET", "RKLB",
    ]
    try:
        import yaml  # local import — keeps yaml optional for the fallback path
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        out: list[str] = []
        for cohort in ("core", "canary"):
            for t in data.get(cohort, []) or []:
                t_upper = str(t).strip().upper()
                if t_upper and t_upper not in out:
                    out.append(t_upper)
        return out or fallback
    except Exception:
        return fallback


# Phase 2 universe (configs/universe.yaml): 24 core tech + 6 cross-sector
# canaries. Canary tickers appear in the corpus and per-ticker IC table for
# diagnostic purposes — they do not drive weight tuning. See Section 22 in
# handoff.md for the protocol.
DEFAULT_TICKERS = load_universe_tickers()

# 2023-07-14 → 2026-05-22 inclusive = exactly 150 Fridays.
DEFAULT_START = "2023-07-14"
DEFAULT_END = "2026-05-22"


def weekly_fridays(start_iso: str, end_iso: str) -> list[str]:
    """All Fridays between start and end inclusive (ISO date strings)."""
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    # Snap start to the first Friday on or after start.
    days_to_friday = (4 - start.weekday()) % 7
    d = start + timedelta(days=days_to_friday)
    out: list[str] = []
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=7)
    return out


def report_path(output_dir: Path, ticker: str, date_str: str) -> Path:
    return output_dir / f"{ticker.upper()}_{date_str}.json"


def _generate_one(
    args: tuple[str, str, str, str, int, bool, str | None, float, bool],
) -> dict:
    """Worker: generate a single report. Returns a result dict for the parent
    process to aggregate. Defined at module level so it's picklable.

    yfinance is not thread-safe under load — Yahoo's anti-scraping returns
    401s and corrupted partial responses, which surface as `TypeError`s
    deep inside the pipeline. We retry the whole job with backoff so a
    transient yfinance failure doesn't drop the report.
    """
    (
        ticker,
        date_str,
        output_dir_str,
        data_provider,
        max_retries,
        force,
        state_store_path,
        pace_seconds,
        minimal_context,
    ) = args
    if pace_seconds > 0:
        time.sleep(pace_seconds)
    output_dir = Path(output_dir_str)
    result: dict = {
        "ticker": ticker,
        "date": date_str,
        "status": "unknown",
        "elapsed_s": None,
        "error": None,
        "attempts": 0,
    }

    target = report_path(output_dir, ticker, date_str)
    if target.exists() and not force:
        result["status"] = "skipped_existing"
        return result

    try:
        from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP
    except Exception as exc:
        result["status"] = "import_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    started = time.time()
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            mvp = AnalysisOnlyMVP(
                data_provider=data_provider,
                enable_llm_insights=False,
                enable_narrative=False,
                verbose=False,
                logger=logging.getLogger("worker"),
                state_store_path=state_store_path,
                enable_news_fetching=not minimal_context,
                enable_filings_fetching=not minimal_context,
            )
            report = mvp.run(symbol=ticker, as_of_date=date_str)
            mvp.save_report(report, output_dir=output_dir)
            result["status"] = "ok" if attempt == 1 else f"ok_retry_{attempt}"
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                # Exponential backoff to give yfinance / Yahoo time to recover
                # and to stagger threads. 5s, 15s, 45s by default.
                time.sleep(5 * (3 ** (attempt - 1)))

    if last_error is not None:
        result["status"] = "error"
        result["error"] = (
            f"{type(last_error).__name__}: {last_error}\n"
            + traceback.format_exc(limit=4)
        )
    result["elapsed_s"] = round(time.time() - started, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a historical corpus of analysis reports.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to analyze (default: 11-name semi/AI basket).",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="Start date (inclusive). Default: 2025-11-21",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help="End date (inclusive). Default: 2026-05-22",
    )
    parser.add_argument(
        "--output-dir", default="reports/analysis_mvp",
        help="Where report JSONs are written.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help=(
            "Thread pool size. Default: 2 (yfinance is unreliable above 2-3 "
            "concurrent threads — Yahoo throttles + returns Invalid Crumb "
            "401s that corrupt downstream parsing)."
        ),
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Retry attempts per job on failure (5s/15s/45s backoff).",
    )
    parser.add_argument(
        "--data-provider", default="polygon",
        choices=["auto", "polygon", "yfinance"],
        help="Price/intraday data provider. Default: polygon.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the job list and exit without running.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Regenerate reports even when a JSON already exists. Use this "
            "after a schema change (e.g. new factors / IV surface) so the "
            "corpus reflects the new fields."
        ),
    )
    parser.add_argument(
        "--state-store-path",
        default="state/analysis_state.sqlite",
        help=(
            "SQLite path for state_store (symbol_state, iv_history, news_seen, "
            "filing_seen). Pass an empty string to disable. Default matches "
            "analysis_mvp.py so each report contributes to the iv_history "
            "table needed for IV rank/percentile derivation."
        ),
    )
    parser.add_argument(
        "--errors-log",
        default="reports/corpus_errors.jsonl",
        help="JSONL file to append per-job errors to.",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.0,
        help=(
            "Per-job sleep (seconds) before starting each worker. Adds a "
            "soft rate limit so we don't burst-trip Polygon/yfinance "
            "throttling on the long Phase 1 backfill. Try 1.0-2.5 for the "
            "free Polygon tier (5 calls/min limit applies per minute, not "
            "per second)."
        ),
    )
    parser.add_argument(
        "--minimal-context", action="store_true",
        help=(
            "Skip news + SEC-filings fetches in each report. Useful for "
            "backfill runs since `filings_recency_signal` is weight=0 and "
            "news PIT-filters to near-empty on older dates anyway. Saves "
            "~0.5-1s per report (~20-40 min on a Phase-2-scale regen)."
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("POLYGON_API_KEY"):
        print(
            "WARNING: POLYGON_API_KEY not set. The Polygon data path will "
            "fall back to yfinance for everything, and PIT fundamentals "
            "will be unavailable.",
            file=sys.stderr,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_log = Path(args.errors_log)
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    dates = weekly_fridays(args.start, args.end)
    tickers = [t.upper() for t in args.tickers]
    # Date-major ordering so that all jobs for date D-1 are submitted (and
    # in a small pool, mostly completed) before any job for date D. This
    # matters for IV-history accumulation in state_store: each report's
    # IV-rank/percentile derivation queries history strictly < as_of_date,
    # so chronological execution is needed for the trailing window to
    # populate correctly during a first-time backfill.
    state_store_path = args.state_store_path or None
    JobTuple = tuple[str, str, str, str, int, bool, str | None, float, bool]
    jobs: list[JobTuple] = [
        (
            t,
            d,
            str(output_dir),
            args.data_provider,
            args.max_retries,
            args.force,
            state_store_path,
            args.pace_seconds,
            args.minimal_context,
        )
        for d in dates
        for t in tickers
    ]

    pending: list[JobTuple] = []
    already: list[tuple[str, str]] = []
    for job in jobs:
        if report_path(output_dir, job[0], job[1]).exists() and not args.force:
            already.append((job[0], job[1]))
        else:
            pending.append(job)

    print(f"Tickers ({len(tickers)}): {' '.join(tickers)}")
    print(f"Date range: {args.start} → {args.end}  ({len(dates)} Fridays)")
    print(f"Total jobs: {len(jobs)}  |  already done: {len(already)}  |  pending: {len(pending)}")
    print(f"Workers: {args.workers}  |  output_dir: {output_dir}")
    print(f"Errors log: {errors_log}")

    if args.dry_run:
        print("\n[dry-run] First 5 pending jobs:")
        for j in pending[:5]:
            print(f"  {j[0]} @ {j[1]}")
        return 0

    if not pending:
        print("\nNothing to do.")
        return 0

    started_all = time.time()
    completed = 0
    errors = 0
    skipped = 0
    print(f"\nStarting corpus generation at {datetime.now().isoformat(timespec='seconds')}\n")
    print(f"{'#':>4} {'ticker':<6} {'date':<10} {'status':<18} {'elapsed':>8}  {'eta':>10}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(_generate_one, job): job for job in pending}
        for fut in as_completed(future_map):
            result = fut.result()
            completed += 1
            if result["status"] == "ok":
                pass
            elif result["status"] == "skipped_existing":
                skipped += 1
            else:
                errors += 1
                with errors_log.open("a") as fh:
                    fh.write(json.dumps(result) + "\n")

            elapsed_all = time.time() - started_all
            done_jobs = completed
            remaining = len(pending) - done_jobs
            rate = done_jobs / max(elapsed_all, 0.001)
            eta_s = remaining / rate if rate > 0 else 0
            eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
            elapsed_s = result["elapsed_s"] or 0.0
            print(
                f"{done_jobs:>4} {result['ticker']:<6} {result['date']:<10} "
                f"{result['status']:<18} {elapsed_s:>7.1f}s  {eta_str:>10}"
            )

    total_s = time.time() - started_all
    ok = completed - skipped - errors
    print(
        f"\nDone. ok={ok}  errors={errors}  skipped={skipped}  "
        f"total_elapsed={int(total_s // 60)}m{int(total_s % 60):02d}s"
    )
    if errors:
        print(f"See {errors_log} for error details.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
