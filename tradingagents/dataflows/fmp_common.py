"""Shared helpers for the Financial Modeling Prep (FMP) data vendor.

FMP (https://financialmodelingprep.com) offers richer, cleaner fundamentals
than the other vendors — company profiles, TTM key metrics/ratios, and
as-reported statements with explicit filing dates. This module centralises
auth, request handling, rate-limit detection, and point-in-time filtering so
the per-endpoint modules stay thin (mirrors ``alpha_vantage_common``).
"""

from __future__ import annotations

import os
from typing import Any

import requests

API_BASE_URL = "https://financialmodelingprep.com/api"


class FMPRateLimitError(Exception):
    """Raised when the FMP API usage/rate limit is exceeded.

    Shares intent with ``AlphaVantageRateLimitError``: ``route_to_vendor``
    catches it to fall through to the next vendor in the chain rather than
    failing the whole request.
    """


class FMPError(Exception):
    """Raised for non-rate-limit FMP API failures (bad symbol, plan gating)."""


def get_api_key() -> str:
    """Retrieve the FMP API key from the environment."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is not set.")
    return api_key


def fmp_request(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    version: str = "v3",
    session: Any | None = None,
    timeout: int = 30,
) -> Any:
    """Make an FMP API request and return the parsed JSON body.

    Args:
        path: Endpoint path after the version segment, e.g. ``"profile/AAPL"``.
        params: Query parameters (the API key is injected automatically).
        version: API version segment (``v3`` for most endpoints, ``v4`` for some).
        session: Optional object with a ``get`` method (used to inject a stub
            in tests); falls back to the ``requests`` module.
        timeout: Per-request timeout in seconds.

    Raises:
        FMPRateLimitError: On HTTP 429 or a body indicating the usage limit.
        FMPError: On other API-reported errors (e.g. invalid symbol/plan).
    """
    query = dict(params or {})
    query["apikey"] = get_api_key()
    url = f"{API_BASE_URL}/{version}/{path}"

    sess = session if session is not None else requests
    response = sess.get(url, params=query, timeout=timeout)

    status = getattr(response, "status_code", 200)
    if status == 429:
        raise FMPRateLimitError("FMP rate limit exceeded (HTTP 429)")
    response.raise_for_status()

    data = response.json()

    # FMP signals errors as a JSON object with an "Error Message" key, while
    # successful list endpoints return a JSON array. Usage-limit messages are
    # surfaced as rate-limit errors so the vendor fallback chain can continue.
    if isinstance(data, dict):
        message = data.get("Error Message") or data.get("Information") or ""
        if message:
            lowered = message.lower()
            if "limit" in lowered or "api key" in lowered or "rate" in lowered:
                raise FMPRateLimitError(f"FMP usage limit: {message}")
            raise FMPError(message)
    return data


def filter_statements_by_date(rows: Any, curr_date: str | None) -> Any:
    """Drop statement rows that became public after ``curr_date``.

    Prevents look-ahead bias. FMP statements expose ``fillingDate`` (the date
    the filing was submitted to the SEC and thus public); we prefer it over the
    fiscal ``date`` (period end), falling back to ``date`` when absent. A row
    with neither field is kept (cannot prove it is future-dated).
    """
    if not curr_date or not isinstance(rows, list):
        return rows
    kept = []
    for row in rows:
        if not isinstance(row, dict):
            kept.append(row)
            continue
        ref = row.get("fillingDate") or row.get("acceptedDate") or row.get("date")
        ref = str(ref)[:10] if ref else ""
        if ref and ref > curr_date:
            continue
        kept.append(row)
    return kept
