"""Record an actual trade for the paper-trading dry-run log.

Usage:

    .venv/bin/python scripts/log_execution.py \\
        --symbol NVDA --side BUY --shares 50 --fill-price 143.10 \\
        --ref-date 2026-06-07 --reason "model said buy, I bought"

Appends one line to `reports/paper_trading/executions/<trade-date>.jsonl`.
Multiple trades on the same date append to the same file.

Use `--list-pending` to show today's outstanding (unfilled-side)
recommendations that don't yet have a matching execution — quick check
on what the model wanted you to do.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.paper_trading import (  # noqa: E402
    ExecutionRecord,
    append_execution,
    load_executions,
    load_recommendations,
    now_utc_iso,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=False)

    # Default = log a trade. Flat args for ergonomics ("just type the trade").
    p.add_argument("--symbol")
    p.add_argument("--side", choices=["BUY", "SELL", "buy", "sell"])
    p.add_argument("--shares", type=float)
    p.add_argument("--fill-price", type=float)
    p.add_argument("--trade-date", default=None,
                   help="ISO date the trade executed. Defaults to today.")
    p.add_argument("--ref-date", default=None,
                   help=(
                       "ISO date of the recommendation this trade was "
                       "responding to. Use today's date when you follow "
                       "today's daily-signals output; omit for pure "
                       "discretionary trades."
                   ))
    p.add_argument("--reason", default=None,
                   help="Optional free-text override reason.")
    p.add_argument("--base-dir", default="reports/paper_trading",
                   help="Paper-trading log directory.")
    p.add_argument("--list-pending", action="store_true",
                   help="Show today's recommended actions that have no "
                        "matching execution yet. No-op for the log itself.")
    return p.parse_args()


def _today_iso() -> str:
    return date.today().isoformat()


def _list_pending(base_dir: Path, today: str) -> int:
    recs = load_recommendations(base_dir, today)
    if not recs:
        print(f"No recommendations on file for {today}.")
        return 0
    execs = load_executions(base_dir, today)
    # Build (symbol → side filled) from today's executions for a coarse
    # "did we trade this name yet?" check.
    filled: set[tuple[str, str]] = set()
    for e in execs:
        filled.add((e.symbol.upper(), e.side.upper()))
        if e.ref_recommendation_date == today:
            filled.add((e.symbol.upper(), "REF"))

    pending = []
    for r in recs:
        if r.action in ("HOLD", "SKIP", "REVIEW"):
            continue
        sym = r.symbol.upper()
        side = "BUY" if r.action in ("BUY", "ADD") else "SELL"
        if (sym, "REF") in filled or (sym, side) in filled:
            continue
        pending.append(r)
    if not pending:
        print(f"All actionable recommendations for {today} appear covered by executions.")
        return 0
    print(f"Pending recommendations for {today}:")
    print(
        f"  {'symbol':<6} {'action':<6} {'shares':>8} {'limit':>10} {'stop':>10} {'composite':>10}"
    )
    for r in pending:
        print(
            f"  {r.symbol:<6} {r.action:<6} "
            f"{abs(r.delta_shares):>8d} "
            f"{(r.limit_price or 0):>10.2f} "
            f"{(r.stop_loss or 0):>10.2f} "
            f"{(r.composite or 0):>+10.3f}"
        )
    return 0


def main() -> int:
    args = _parse_args()
    base_dir = Path(args.base_dir)

    if args.list_pending:
        return _list_pending(base_dir, _today_iso())

    # Validate required args for the log path.
    missing = [
        name for name, val in (
            ("--symbol", args.symbol),
            ("--side", args.side),
            ("--shares", args.shares),
            ("--fill-price", args.fill_price),
        ) if val in (None, "")
    ]
    if missing:
        print(
            f"Missing required args: {', '.join(missing)}. "
            f"Run with --help for usage or --list-pending to see "
            f"outstanding recommendations.",
            file=sys.stderr,
        )
        return 2

    trade_date = args.trade_date or _today_iso()
    try:
        datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        print(f"--trade-date must be YYYY-MM-DD; got {trade_date!r}", file=sys.stderr)
        return 2
    if args.ref_date:
        try:
            datetime.strptime(args.ref_date, "%Y-%m-%d")
        except ValueError:
            print(f"--ref-date must be YYYY-MM-DD; got {args.ref_date!r}", file=sys.stderr)
            return 2

    record = ExecutionRecord(
        trade_date=trade_date,
        symbol=str(args.symbol).upper(),
        side=str(args.side).upper(),
        shares=float(args.shares),
        fill_price=float(args.fill_price),
        ref_recommendation_date=args.ref_date,
        override_reason=args.reason,
        logged_at_utc=now_utc_iso(),
    )
    path = append_execution(record, base_dir=base_dir)
    print(
        f"Logged: {record.side} {record.shares} {record.symbol} @ "
        f"{record.fill_price} (trade_date={record.trade_date}, "
        f"ref={record.ref_recommendation_date or 'discretionary'}) → {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
