"""News from Financial Modeling Prep (FMP)."""

from __future__ import annotations

import json

from .fmp_common import fmp_request

# Cap matches the other news vendors' typical request size.
_NEWS_LIMIT = 50


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Ticker-scoped news articles between ``start_date`` and ``end_date``.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Inclusive start date, yyyy-mm-dd.
        end_date: Inclusive end date, yyyy-mm-dd.
    """
    rows = fmp_request(
        "news/stock",
        {
            "symbols": ticker,
            "from": start_date,
            "to": end_date,
            "limit": _NEWS_LIMIT,
        },
    )
    return json.dumps(rows, indent=2, default=str)
