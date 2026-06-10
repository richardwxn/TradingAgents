"""Fundamental data from Financial Modeling Prep (FMP).

Returns JSON-serialised payloads (like ``alpha_vantage_fundamentals``) so the
calling agent/LLM gets structured, parseable data. All statement endpoints are
point-in-time filtered against ``curr_date`` via ``filter_statements_by_date``.
"""

from __future__ import annotations

import json
from typing import Any

from .fmp_common import filter_statements_by_date, fmp_request

# Number of historical statement periods to request. FMP returns most-recent
# first; PIT filtering then drops anything published after ``curr_date``.
_STATEMENT_LIMIT = 12


def _period_for_freq(freq: str | None) -> str:
    """Map the codebase's freq convention to FMP's ``period`` query value."""
    return "quarter" if str(freq or "quarterly").lower().startswith("q") else "annual"


def _first(rows: Any) -> Any:
    """FMP profile/TTM endpoints return a single-element list; unwrap it."""
    if isinstance(rows, list):
        return rows[0] if rows else {}
    return rows


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Company overview: profile + TTM key metrics + TTM ratios.

    Args:
        ticker: Ticker symbol of the company.
        curr_date: Current trading date, yyyy-mm-dd. TTM snapshots are not
            period-stamped by FMP, so this is accepted for signature parity
            with the other vendors and applied only where a date exists.
    """
    profile = fmp_request(f"profile/{ticker}")
    key_metrics = fmp_request(f"key-metrics-ttm/{ticker}")
    ratios = fmp_request(f"ratios-ttm/{ticker}")

    payload = {
        "symbol": ticker.upper(),
        "as_of_date": curr_date,
        "profile": _first(profile),
        "key_metrics_ttm": _first(key_metrics),
        "ratios_ttm": _first(ratios),
    }
    return json.dumps(payload, indent=2, default=str)


def get_balance_sheet(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    """Balance sheet statements (most-recent first, PIT filtered)."""
    rows = fmp_request(
        f"balance-sheet-statement/{ticker}",
        {"period": _period_for_freq(freq), "limit": _STATEMENT_LIMIT},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)


def get_cashflow(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    """Cash flow statements (most-recent first, PIT filtered)."""
    rows = fmp_request(
        f"cash-flow-statement/{ticker}",
        {"period": _period_for_freq(freq), "limit": _STATEMENT_LIMIT},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)


def get_income_statement(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    """Income statements (most-recent first, PIT filtered)."""
    rows = fmp_request(
        f"income-statement/{ticker}",
        {"period": _period_for_freq(freq), "limit": _STATEMENT_LIMIT},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)
