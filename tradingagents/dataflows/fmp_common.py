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

# FMP's current ("stable") API. Endpoints take the symbol as a query parameter
# (e.g. ``/stable/profile?symbol=AAPL``); the legacy ``/api/v3/<path>/<symbol>``
# surface is deprecated and returns HTTP 403 for newer keys.
API_BASE_URL = "https://financialmodelingprep.com/stable"


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


# The free plan caps statement endpoints to the 5 most recent periods (a larger
# ``limit`` returns HTTP 402). 5 quarters is enough for the pipeline's TTM
# synthesis; paid-plan users can raise this via ``FMP_STATEMENT_LIMIT``.
_DEFAULT_STATEMENT_LIMIT = 5


def statement_limit() -> int:
    """Periods to request from statement endpoints (env-overridable)."""
    try:
        return max(1, int(os.getenv("FMP_STATEMENT_LIMIT", _DEFAULT_STATEMENT_LIMIT)))
    except (TypeError, ValueError):
        return _DEFAULT_STATEMENT_LIMIT


def fmp_request(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    session: Any | None = None,
    timeout: int = 30,
) -> Any:
    """Make an FMP (stable) API request and return the parsed JSON body.

    Args:
        endpoint: Stable endpoint name, e.g. ``"profile"`` or
            ``"income-statement"``. The symbol is passed via ``params``
            (``{"symbol": "AAPL"}``), not in the path.
        params: Query parameters (the API key is injected automatically).
        session: Optional object with a ``get`` method (used to inject a stub
            in tests); falls back to the ``requests`` module.
        timeout: Per-request timeout in seconds.

    Raises:
        FMPRateLimitError: On HTTP 429 or a body indicating the usage limit.
        FMPError: On other API-reported errors (e.g. invalid symbol/plan).
    """
    query = dict(params or {})
    query["apikey"] = get_api_key()
    url = f"{API_BASE_URL}/{endpoint}"

    sess = session if session is not None else requests
    response = sess.get(url, params=query, timeout=timeout)

    status = getattr(response, "status_code", 200)
    if status == 429:
        raise FMPRateLimitError("FMP rate limit exceeded (HTTP 429)")
    if status == 402:
        raise FMPError(
            f"FMP endpoint '{endpoint}' requires a higher plan (HTTP 402). "
            "On the free tier, statement requests are limited to 5 periods "
            "(lower FMP_STATEMENT_LIMIT) and some endpoints (e.g. news) are "
            "unavailable."
        )
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
        # Stable API uses ``filingDate``; legacy v3 used ``fillingDate`` (sic).
        ref = (
            row.get("filingDate")
            or row.get("fillingDate")
            or row.get("acceptedDate")
            or row.get("date")
        )
        ref = str(ref)[:10] if ref else ""
        if ref and ref > curr_date:
            continue
        kept.append(row)
    return kept
