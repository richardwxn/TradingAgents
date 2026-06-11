"""Fundamental data from Financial Modeling Prep (FMP).

Returns JSON-serialised payloads (like ``alpha_vantage_fundamentals``) so the
calling agent/LLM gets structured, parseable data. All statement endpoints are
point-in-time filtered against ``curr_date`` via ``filter_statements_by_date``.
"""

from __future__ import annotations

import json
from typing import Any

from .fmp_common import filter_statements_by_date, fmp_request, statement_limit


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
    profile = fmp_request("profile", {"symbol": ticker})
    key_metrics = fmp_request("key-metrics-ttm", {"symbol": ticker})
    ratios = fmp_request("ratios-ttm", {"symbol": ticker})

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
        "balance-sheet-statement",
        {"symbol": ticker, "period": _period_for_freq(freq), "limit": statement_limit()},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)


def get_cashflow(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    """Cash flow statements (most-recent first, PIT filtered)."""
    rows = fmp_request(
        "cash-flow-statement",
        {"symbol": ticker, "period": _period_for_freq(freq), "limit": statement_limit()},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)


def get_income_statement(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    """Income statements (most-recent first, PIT filtered)."""
    rows = fmp_request(
        "income-statement",
        {"symbol": ticker, "period": _period_for_freq(freq), "limit": statement_limit()},
    )
    rows = filter_statements_by_date(rows, curr_date)
    return json.dumps(rows, indent=2, default=str)
