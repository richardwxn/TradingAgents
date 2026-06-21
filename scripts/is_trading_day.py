#!/usr/bin/env python
"""Trading-day gate for the morning automation.

Exit code 0 when the given date (default: today, US/Eastern) is a regular NYSE
trading session; non-zero otherwise (weekend or holiday). The morning report
runs pre-open, so a "is the market open right now?" check is unsuitable; this
uses a self-contained NYSE holiday calendar instead (no network, no extra deps).

Optionally, ``--live`` consults Polygon's market-status endpoint via the
existing PolygonMarketStatusProvider to confirm the venue is actually open
(useful for intraday callers, not the pre-open morning run).

Usage:
    python scripts/is_trading_day.py                 # today (US/Eastern)
    python scripts/is_trading_day.py --date 2026-07-03
    python scripts/is_trading_day.py --quiet && run_pipeline
    python scripts/is_trading_day.py --live          # also require market open now
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


def _easter(year: int) -> dt.date:
    """Anonymous Gregorian (Meeus/Jones/Butcher) Easter Sunday algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The nth (1-based) ``weekday`` (Mon=0) of ``month`` in ``year``."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The last ``weekday`` (Mon=0) of ``month`` in ``year``."""
    if month == 12:
        nxt = dt.date(year + 1, 1, 1)
    else:
        nxt = dt.date(year, month + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(date: dt.date) -> dt.date:
    """Apply NYSE weekend-observance: Sat holiday -> Fri, Sun holiday -> Mon."""
    if date.weekday() == 5:  # Saturday
        return date - dt.timedelta(days=1)
    if date.weekday() == 6:  # Sunday
        return date + dt.timedelta(days=1)
    return date


def nyse_holidays(year: int) -> dict[dt.date, str]:
    """NYSE full-day market holidays for ``year`` (observed dates)."""
    holidays: dict[dt.date, str] = {}

    def add(date: dt.date, name: str) -> None:
        holidays[_observed(date)] = name

    add(dt.date(year, 1, 1), "New Year's Day")
    add(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    add(_nth_weekday(year, 2, 0, 3), "Washington's Birthday")
    add(_easter(year) - dt.timedelta(days=2), "Good Friday")  # always a weekday
    add(_last_weekday(year, 5, 0), "Memorial Day")
    if year >= 2022:
        add(dt.date(year, 6, 19), "Juneteenth National Independence Day")
    add(dt.date(year, 7, 4), "Independence Day")
    add(_nth_weekday(year, 9, 0, 1), "Labor Day")
    add(_nth_weekday(year, 11, 3, 4), "Thanksgiving Day")
    add(dt.date(year, 12, 25), "Christmas Day")
    return holidays


def trading_day_status(date: dt.date) -> tuple[bool, str]:
    """Return (is_trading_day, reason)."""
    if date.weekday() >= 5:
        return False, f"{date.isoformat()} is a weekend ({date.strftime('%A')})"
    holiday = nyse_holidays(date.year).get(date)
    if holiday:
        return False, f"{date.isoformat()} is a market holiday ({holiday})"
    return True, f"{date.isoformat()} is a regular NYSE trading session"


def _market_open_now() -> tuple[bool, str]:
    try:
        from tradingagents.analysis_only.providers import PolygonMarketStatusProvider
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"could not load market-status provider: {exc}"
    is_open = PolygonMarketStatusProvider().is_market_open()
    return is_open, "market is open now" if is_open else "market is not open now"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYSE trading-day gate.")
    parser.add_argument(
        "--date",
        default=None,
        help="Date to check (YYYY-MM-DD). Defaults to today in US/Eastern.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also require that the market is open right now (Polygon status).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable status line; rely on exit code.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.date:
        try:
            date = dt.date.fromisoformat(args.date)
        except ValueError:
            print(f"invalid --date {args.date!r}; expected YYYY-MM-DD", file=sys.stderr)
            return 2
    else:
        date = dt.datetime.now(_EASTERN).date()

    is_trading, reason = trading_day_status(date)
    if is_trading and args.live:
        live_open, live_reason = _market_open_now()
        is_trading = is_trading and live_open
        reason = f"{reason}; {live_reason}"

    if not args.quiet:
        prefix = "TRADING DAY" if is_trading else "NO TRADE"
        print(f"[{prefix}] {reason}")
    return 0 if is_trading else 1


if __name__ == "__main__":
    raise SystemExit(main())
