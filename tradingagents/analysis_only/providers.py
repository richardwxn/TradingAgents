from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
import logging
import math
import os
import re
import threading
import time

import pandas as pd
import requests
import yfinance as yf


class SECFetchError(Exception):
    """Raised when a SEC EDGAR fetch fails transiently (HTTP error,
    rate-limit, parse error). Distinct from a successful response that
    simply contains no matching filings on/before the as_of_date — the
    latter is represented by `get_latest_filing` returning ``None``.

    Callers (e.g. pipeline cache layer) should treat this as a non-
    cacheable failure so a future run can retry.
    """


class PriceProvider(Protocol):
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        ...

    def get_intraday_bars(
        self,
        symbol: str,
        timespan: str = "hour",
        multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        ...


class NewsProvider(Protocol):
    def get_news(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        ...


class MarketStatusProvider(Protocol):
    def is_market_open(self) -> bool:
        ...


class FearGreedProvider:
    API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    SOURCE = "cnn_fear_and_greed"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Referer": "https://edition.cnn.com/",
        "Accept": "application/json",
    }
    INDICATOR_KEYS = [
        "market_momentum_sp500",
        "stock_price_strength",
        "stock_price_breadth",
        "put_call_options",
        "market_volatility_vix",
        "junk_bond_demand",
        "safe_haven_demand",
    ]

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = 10,
    ):
        self.session = session or requests.Session()
        self.session.headers.update(self.HEADERS)
        self.timeout = timeout

    def get_index(self, as_of_date: str | None = None) -> dict[str, Any]:
        try:
            response = self.session.get(self.API_URL, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {
                "status": "error",
                "pit_status": "error",
                "source": self.SOURCE,
                "error": str(exc),
            }
        return self.normalize(payload, as_of_date=as_of_date)

    def normalize(
        self,
        payload: dict[str, Any],
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        if not payload:
            return self._unavailable("empty_payload")
        if self._is_historical(as_of_date):
            return self._normalize_historical(payload, as_of_date or "")
        return self._normalize_current(payload)

    def _normalize_current(self, payload: dict[str, Any]) -> dict[str, Any]:
        fg = payload.get("fear_and_greed") or {}
        score = self._safe_float(fg.get("score"))
        rating = fg.get("rating")
        if score is None or rating is None:
            return self._unavailable("missing_current_index")
        timestamp = self._parse_timestamp(fg.get("timestamp"))
        return {
            "status": "ok",
            "pit_status": "live",
            "source": self.SOURCE,
            "score": score,
            "rating": str(rating),
            "timestamp": timestamp,
            "previous_1_week": self._safe_float(fg.get("previous_1_week")),
            "previous_1_month": self._safe_float(fg.get("previous_1_month")),
            "previous_1_year": self._safe_float(fg.get("previous_1_year")),
            "indicators": self._extract_indicators(payload),
        }

    def _normalize_historical(
        self,
        payload: dict[str, Any],
        as_of_date: str,
    ) -> dict[str, Any]:
        point = self._historical_point_on_or_before(payload, as_of_date)
        if not point:
            return self._unavailable("no_historical_point_on_or_before_date")
        return {
            "status": "ok",
            "pit_status": "pit",
            "source": self.SOURCE,
            "score": self._safe_float(point.get("y")),
            "rating": point.get("rating"),
            "timestamp": self._parse_timestamp(point.get("x")),
            "previous_1_week": None,
            "previous_1_month": None,
            "previous_1_year": None,
            "indicators": {},
        }

    def _historical_point_on_or_before(
        self,
        payload: dict[str, Any],
        as_of_date: str,
    ) -> dict[str, Any] | None:
        try:
            end_dt = datetime.strptime(as_of_date, "%Y-%m-%d").replace(
                hour=23,
                minute=59,
                second=59,
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
        rows = (
            (payload.get("fear_and_greed_historical") or {})
            .get("data", [])
        )
        candidates: list[tuple[datetime, dict[str, Any]]] = []
        for row in rows:
            ts = self._datetime_from_any(row.get("x"))
            if ts and ts <= end_dt:
                candidates.append((ts, row))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def _extract_indicators(
        self,
        payload: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        indicators: dict[str, dict[str, Any]] = {}
        for key in self.INDICATOR_KEYS:
            row = payload.get(key) or {}
            score = self._safe_float(row.get("score"))
            rating = row.get("rating")
            if score is None and rating is None:
                continue
            indicators[key] = {
                "score": score,
                "rating": str(rating) if rating is not None else None,
            }
        return indicators

    def _is_historical(self, as_of_date: str | None) -> bool:
        if not as_of_date:
            return False
        try:
            target = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        except ValueError:
            return False
        return target < datetime.now(timezone.utc).date()

    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "pit_status": "unavailable",
            "source": self.SOURCE,
            "reason": reason,
        }

    def _parse_timestamp(self, value: Any) -> str | None:
        dt = self._datetime_from_any(value)
        if not dt:
            return None
        return dt.isoformat()

    def _datetime_from_any(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None


# --- VIX-based fear/greed proxy ---
#
# Process-shared cache of the ^VIX daily-close series. Loaded once per
# process; threads share it. Same pattern as RiskFreeRateProvider's
# `_RATE_SERIES_CACHE`.
_VIX_SERIES_CACHE: dict[str, dict[str, float]] = {}
_VIX_SERIES_LOCK = threading.Lock()


def _load_vix_series(start: str, logger: logging.Logger) -> dict[str, float]:
    """Load ^VIX into a date→close dict. Returns {} on failure."""
    series: dict[str, float] = {}
    try:
        hist = yf.Ticker("^VIX").history(start=start)
        if hist is not None and not hist.empty:
            closes = hist["Close"].dropna()
            for ts, value in closes.items():
                d_str = ts.date().isoformat()
                try:
                    series[d_str] = float(value)
                except (TypeError, ValueError):
                    continue
        dates = sorted(series.keys())
        logger.info(
            "VIXFearGreedProvider loaded %d ^VIX observations from %s to %s",
            len(series), dates[0] if dates else "?", dates[-1] if dates else "?",
        )
    except Exception as exc:
        logger.warning(
            "VIXFearGreedProvider failed to load ^VIX (%s); proxy will "
            "return unavailable.", exc,
        )
    return series


def reset_vix_cache() -> None:
    """Clear the module-level ^VIX cache. For tests."""
    with _VIX_SERIES_LOCK:
        _VIX_SERIES_CACHE.clear()


class VIXFearGreedProvider:
    """Fear/greed proxy derived from VIX percentile rank.

    Maps the trailing-window percentile rank of VIX to a CNN-equivalent
    score in [0, 100], where higher = more greed, lower = more fear:

        cnn_score ≈ (1 - vix_percentile_252d) * 100

    Bucket cuts match `score_fear_greed_regime`:
      ≤25  → "extreme fear"     (VIX ≥ 95th percentile of trailing year)
      <45  → "fear"              (75-95th percentile)
      ~50  → "neutral"           (25-75th)
      >55  → "greed"             (5-25th)
      ≥75  → "extreme greed"     (≤ 5th percentile)

    Returns the same dict shape as `FearGreedProvider.get_index` so it
    can be used as a drop-in fallback. Source field stamps
    `vix_fear_greed_proxy` so downstream can tell which it got.

    PIT-correct: the percentile is computed using only ^VIX data with
    `date < as_of_date` (strictly before). A historical run never sees
    its own day's VIX in the trailing window.
    """

    SOURCE = "vix_fear_greed_proxy"

    def __init__(
        self,
        start: str = "2018-01-01",
        window_days: int = 252,
        logger: logging.Logger | None = None,
    ) -> None:
        self._start = start
        self._window_days = window_days
        self._logger = logger or logging.getLogger(__name__)
        self._series: dict[str, float] | None = None
        self._dates_sorted: list[str] = []

    def _ensure_loaded(self) -> None:
        if self._series is not None:
            return
        with _VIX_SERIES_LOCK:
            cached = _VIX_SERIES_CACHE.get(self._start)
            if cached is None:
                cached = _load_vix_series(self._start, self._logger)
                _VIX_SERIES_CACHE[self._start] = cached
        self._series = cached
        self._dates_sorted = sorted(self._series.keys())

    def get_index(self, as_of_date: str | None = None) -> dict[str, Any]:
        """Return the same dict shape as `FearGreedProvider.get_index`."""
        self._ensure_loaded()
        if not self._series:
            return self._unavailable("vix_series_unavailable")
        target = as_of_date or datetime.now(timezone.utc).date().isoformat()
        # Use the most recent close ≤ target (walks back for weekends/holidays).
        vix_today = self._latest_close_on_or_before(target)
        if vix_today is None:
            return self._unavailable("no_vix_close_on_or_before_target")
        # Trailing window: VIX closes strictly before target, capped at window_days
        window = self._trailing_window(target)
        if len(window) < 20:  # need a meaningful base of comparison
            return self._unavailable("insufficient_vix_history")
        pct_rank = self._percentile_of(vix_today, window)
        # Map percentile → CNN-equivalent score. High VIX percentile = fear → low score.
        cnn_score = max(0.0, min(100.0, (1.0 - pct_rank) * 100.0))
        rating = self._rating_for(cnn_score)
        return {
            "status": "ok",
            "pit_status": "pit",
            "source": self.SOURCE,
            "score": round(cnn_score, 2),
            "rating": rating,
            "timestamp": target,
            "previous_1_week": None,
            "previous_1_month": None,
            "previous_1_year": None,
            "indicators": {
                "market_volatility_vix": {
                    "score": round(cnn_score, 2),
                    "rating": rating,
                    "vix_close": round(vix_today, 4),
                    "vix_window_size": len(window),
                    "vix_percentile_252d": round(pct_rank, 4),
                },
            },
        }

    # ---------- internals ----------

    def _latest_close_on_or_before(self, target: str) -> float | None:
        if target in self._series:
            return self._series[target]
        # Walk back up to 10 calendar days for weekends/holidays.
        try:
            d = datetime.fromisoformat(target).date()
        except ValueError:
            return None
        for _ in range(10):
            d -= timedelta(days=1)
            key = d.isoformat()
            if key in self._series:
                return self._series[key]
        return None

    def _trailing_window(self, target: str) -> list[float]:
        """Closes with date strictly < target, in the trailing window_days."""
        try:
            target_d = datetime.fromisoformat(target).date()
        except ValueError:
            return []
        window_start = (
            target_d - timedelta(days=self._window_days)
        ).isoformat()
        return [
            v for d, v in self._series.items()
            if window_start <= d < target
        ]

    @staticmethod
    def _percentile_of(value: float, sample: list[float]) -> float:
        """Fraction of sample ≤ value, in [0, 1]."""
        if not sample:
            return 0.5
        count_le = sum(1 for x in sample if x <= value)
        return count_le / len(sample)

    @staticmethod
    def _rating_for(score: float) -> str:
        if score <= 25:
            return "extreme fear"
        if score < 45:
            return "fear"
        if score >= 75:
            return "extreme greed"
        if score > 55:
            return "greed"
        return "neutral"

    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "pit_status": "unavailable",
            "source": self.SOURCE,
            "score": None,
            "rating": None,
            "reason": reason,
        }


class FilingsProvider(Protocol):
    def get_latest_filing(
        self,
        symbol: str,
        as_of_date: str | None = None,
    ) -> dict[str, Any] | None:
        ...


class PolygonPriceProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._get_aggs(
            symbol=symbol,
            multiplier=1,
            timespan="day",
            start_date=start_date,
            end_date=end_date,
        )

    def get_intraday_bars(
        self,
        symbol: str,
        timespan: str = "hour",
        multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        now = datetime.now(timezone.utc).date()
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        return self._get_aggs(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            start_date=start_date,
            end_date=end_date,
        )

    def _get_aggs(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()
        url = (
            "https://api.polygon.io/v2/aggs/ticker/"
            f"{symbol.upper()}/range/{multiplier}/{timespan}/"
            f"{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return pd.DataFrame()
        rows = payload.get("results") or []
        if not rows:
            return pd.DataFrame()
        data = pd.DataFrame(rows).rename(
            columns={
                "t": "Date",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
            }
        )
        data["Date"] = pd.to_datetime(data["Date"], unit="ms", utc=True)
        data["Date"] = data["Date"].dt.tz_localize(None)
        return data


# --- Polygon daily-aggs cache for market-context symbols ---
#
# Process-shared cache of Polygon daily bars keyed by (symbol, start, end).
# Same pattern as `_VIX_SERIES_CACHE` / `_RATE_SERIES_CACHE`: the first call
# inside a process pays the network cost; later threads serve the same
# DataFrame from memory. Used by `pipeline._download_yfinance_daily_cached`
# to replace the noisy `yf.download("SPY"/"XLK"/...)` path that surfaces
# Yahoo's "HTTP 401 Invalid Crumb" errors under concurrent regen.
#
# Indices like `^VIX` / `^IRX` / `^TNX` are NOT on the Polygon Stocks plan
# (`I:VIX` returns 403 NOT_AUTHORIZED). Those keep using yfinance — see
# `is_polygon_supported_symbol`.
_POLYGON_DAILY_AGGS_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_POLYGON_DAILY_AGGS_LOCK = threading.RLock()


def is_polygon_supported_symbol(symbol: str) -> bool:
    """True if `symbol` can be fetched from Polygon's /v2/aggs endpoint.

    The Polygon Stocks plan covers stocks + ETFs but NOT index codes
    (`^VIX`, `^IRX`, `^TNX`, etc. — these require an Indices plan and
    return 403 NOT_AUTHORIZED on /v2/aggs). Yahoo-style `^`-prefixed
    tickers therefore stay on yfinance.
    """
    if not symbol:
        return False
    return not symbol.startswith("^")


def _load_polygon_daily_aggs(
    symbol: str,
    start: str,
    end: str,
    api_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Fetch adjusted daily aggregates for `symbol` from Polygon.

    Returns a DataFrame with `Date` as the index (matching yfinance's
    `auto_adjust=True` shape) and `Open/High/Low/Close/Volume` columns,
    or an empty DataFrame on any failure. `start`/`end` are ISO dates,
    both inclusive (matching Polygon's path semantics).
    """
    if not api_key:
        return pd.DataFrame()
    url = (
        "https://api.polygon.io/v2/aggs/ticker/"
        f"{symbol.upper()}/range/1/day/{start}/{end}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Polygon daily aggs fetch failed for %s [%s..%s]: %s",
            symbol, start, end, exc,
        )
        return pd.DataFrame()
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).rename(
        columns={
            "t": "Date",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        }
    )
    # Polygon timestamps are ms epoch UTC; convert to naive UTC date index
    # so downstream `data["Close"]` / `.iloc` access matches yfinance.
    frame["Date"] = pd.to_datetime(frame["Date"], unit="ms", utc=True)
    frame["Date"] = frame["Date"].dt.tz_localize(None).dt.normalize()
    frame = frame.set_index("Date").sort_index()
    # Keep only the OHLCV columns yfinance exposes; drop Polygon-specific
    # fields (vw, n) to avoid surprising callers.
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    return frame[keep]


def fetch_polygon_daily_aggs_cached(
    symbol: str,
    start: str,
    end: str,
    api_key: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Process-shared cached wrapper around `_load_polygon_daily_aggs`.

    Threads spawned from the same process (e.g. corpus-regen worker pool)
    share a single fetch per (symbol, start, end) tuple. Returns an empty
    DataFrame on failure — callers should treat as "data unavailable".
    """
    log = logger or logging.getLogger(__name__)
    key = (symbol.upper(), start, end)
    with _POLYGON_DAILY_AGGS_LOCK:
        cached = _POLYGON_DAILY_AGGS_CACHE.get(key)
        if cached is not None:
            return cached
    # Fetch outside the lock so concurrent fetches for different keys
    # don't serialize. We accept the rare double-fetch on the same key.
    frame = _load_polygon_daily_aggs(symbol, start, end, api_key, log)
    with _POLYGON_DAILY_AGGS_LOCK:
        # Only populate cache on success — an empty result might just be
        # a transient network blip; let the next caller retry.
        if not frame.empty:
            _POLYGON_DAILY_AGGS_CACHE[key] = frame
    return frame


def reset_polygon_daily_aggs_cache() -> None:
    """Clear the module-level Polygon daily aggs cache. For tests."""
    with _POLYGON_DAILY_AGGS_LOCK:
        _POLYGON_DAILY_AGGS_CACHE.clear()


# --- Polygon grouped-aggs (Future-work #6) ---
# Different access pattern from `fetch_polygon_daily_aggs_cached`: that one
# fetches a date RANGE for ONE symbol; this one fetches ONE date for ALL
# symbols on the US stocks market via `/v2/aggs/grouped/locale/us/market/
# stocks/{date}`. Right tool when the workload iterates many symbols at the
# same date — e.g. `scripts/build_screener_universe.py` filtering ~3000
# Nasdaq tickers by trailing-20d ADV. Replaces ~3000 per-symbol HTTP calls
# with ~20 per-date calls returning ALL symbols (5-10x speedup on that
# workload; modest impact on the regen which is already symbol-major).
_POLYGON_GROUPED_AGGS_CACHE: dict[str, dict[str, dict]] = {}
_POLYGON_GROUPED_AGGS_LOCK = threading.RLock()


def _load_polygon_grouped_aggs(
    date_iso: str,
    api_key: str,
    logger: logging.Logger,
) -> dict[str, dict]:
    """Fetch one day's daily aggs for ALL US stocks/ETFs from Polygon.

    Returns a dict keyed by ticker (uppercase) mapping to the raw Polygon
    row `{o, h, l, c, v, vw, n, t}` — caller normalizes as needed. Empty
    dict on failure or no results (e.g. weekend/holiday).
    """
    if not api_key:
        return {}
    url = (
        "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
        f"{date_iso}"
    )
    params = {"adjusted": "true", "apiKey": api_key}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Polygon grouped aggs fetch failed for %s: %s", date_iso, exc
        )
        return {}
    rows = payload.get("results") or []
    out: dict[str, dict] = {}
    for row in rows:
        ticker = (row.get("T") or "").upper()
        if ticker:
            out[ticker] = row
    return out


def fetch_polygon_grouped_aggs_cached(
    date_iso: str,
    api_key: str,
    logger: logging.Logger | None = None,
) -> dict[str, dict]:
    """Process-shared cached wrapper around `_load_polygon_grouped_aggs`.

    Threads spawned from the same process share a single fetch per date.
    Returns an empty dict on failure (callers should treat as data
    unavailable for that date — weekend/holiday/network blip).
    """
    log = logger or logging.getLogger(__name__)
    with _POLYGON_GROUPED_AGGS_LOCK:
        cached = _POLYGON_GROUPED_AGGS_CACHE.get(date_iso)
        if cached is not None:
            return cached
    # Fetch outside the lock so concurrent fetches for different dates
    # don't serialize. We accept the rare double-fetch on the same date.
    rows = _load_polygon_grouped_aggs(date_iso, api_key, log)
    with _POLYGON_GROUPED_AGGS_LOCK:
        # Cache even empty results when the lookup completed cleanly —
        # weekends/holidays legitimately return zero rows, no point in
        # retrying. Only skip cache when the fetch itself errored (caller
        # can't tell the difference today; future work could promote this
        # to an explicit fetch-status field).
        _POLYGON_GROUPED_AGGS_CACHE[date_iso] = rows
    return rows


def reset_polygon_grouped_aggs_cache() -> None:
    """Clear the module-level Polygon grouped-aggs cache. For tests."""
    with _POLYGON_GROUPED_AGGS_LOCK:
        _POLYGON_GROUPED_AGGS_CACHE.clear()


class YFinancePriceProvider:
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        try:
            data = yf.download(
                symbol.upper(),
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
        except Exception:
            return pd.DataFrame()
        if data.empty:
            return pd.DataFrame()
        return data.reset_index()

    def get_intraday_bars(
        self,
        symbol: str,
        timespan: str = "hour",
        multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        interval = "60m" if timespan == "hour" else "1d"
        period = "7d" if interval == "60m" else "1mo"
        try:
            data = yf.download(
                symbol.upper(),
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
        except Exception:
            return pd.DataFrame()
        if data.empty:
            return pd.DataFrame()
        return data.reset_index()


# --- Polygon news cache ---
#
# Process-shared cache keyed by (symbol, published_before, limit). Mirrors
# the `_VIX_SERIES_CACHE` pattern: load once, share across threads. Each
# call to `PolygonNewsProvider.get_news` with the same (symbol, as_of)
# tuple is a stable PIT-correct query, so caching is safe.
_POLYGON_NEWS_CACHE: dict[tuple[str, str | None, int], list[dict[str, Any]]] = {}
_POLYGON_NEWS_LOCK = threading.Lock()


def reset_polygon_news_cache() -> None:
    """Clear the module-level Polygon news cache. For tests."""
    with _POLYGON_NEWS_LOCK:
        _POLYGON_NEWS_CACHE.clear()


class PolygonNewsProvider:
    """Per-ticker historical news via Polygon /v2/reference/news.

    PIT-correct: callers pass `published_before` (an ISO date or RFC3339
    timestamp) and the server filters articles to those published on or
    before that instant. Returns the raw Polygon `results` list — each
    item is a dict with `title`, `description`, `published_utc`,
    `tickers`, and (when available) `insights` whose entries carry a
    `sentiment` classification in {"positive", "neutral", "negative"}.

    Module-level cache (`_POLYGON_NEWS_CACHE`) dedups repeated calls with
    the same (symbol, published_before, limit) tuple within a process —
    mirrors the `_VIX_SERIES_CACHE` pattern. Returns `[]` when the API key
    is missing or the HTTP call fails (no exception leaks).
    """

    BASE_URL = "https://api.polygon.io/v2/reference/news"

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.session = session or requests.Session()
        self.timeout = timeout

    def get_news(
        self,
        symbol: str,
        limit: int = 50,
        published_before: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        key = (symbol.upper(), published_before, int(limit))
        with _POLYGON_NEWS_LOCK:
            cached = _POLYGON_NEWS_CACHE.get(key)
            if cached is not None:
                # Return a shallow copy so callers can mutate the list
                # without poisoning the cache for the next request.
                return list(cached)
        params: dict[str, Any] = {
            "ticker": symbol.upper(),
            "limit": int(limit),
            "sort": "published_utc",
            "order": "desc",
            "apiKey": self.api_key,
        }
        if published_before:
            params["published_utc.lte"] = published_before
        try:
            response = self.session.get(
                self.BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            results: list[dict[str, Any]] = []
        else:
            results = payload.get("results") or []
        with _POLYGON_NEWS_LOCK:
            _POLYGON_NEWS_CACHE[key] = list(results)
        return list(results)


class YFinanceNewsProvider:
    def get_news(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        try:
            items = yf.Ticker(symbol.upper()).news or []
        except Exception:
            return []
        return items[:limit]


# Module-level cache for Polygon Financials: each (symbol, timeframe) pair
# gets fetched once per process. Threadsafe — first caller per key blocks
# others until the fetch completes. Pattern matches `_RATE_SERIES_CACHE` in
# this file (Section 23). See plans/screener_and_regen.md Future-work #3.
#
# Cache stores the FULL paginated result (up to 100 records — the Polygon
# per-page max). `_fetch` then client-side filters by `filing_date.lte
# as_of_date` and slices to the caller's `limit`. For the regen workload
# (30 tickers × 150 Fridays × 2 calls = 9k HTTP requests) this collapses
# to ~60 requests (30 tickers × 2 timeframes), saving ~15 minutes of HTTP
# latency. Live `analysis_mvp.py` runs see no change since they hit each
# (ticker, timeframe) once anyway.
_FINANCIALS_ALL_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_FINANCIALS_ALL_LOCK = threading.RLock()

_FMP_FINANCIALS_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_FMP_FINANCIALS_LOCK = threading.RLock()


def reset_financials_cache() -> None:
    """Clear the module-level financials cache. Test-only helper."""
    with _FINANCIALS_ALL_LOCK:
        _FINANCIALS_ALL_CACHE.clear()


def reset_fmp_financials_cache() -> None:
    """Clear the module-level FMP financials cache. Test-only helper."""
    with _FMP_FINANCIALS_LOCK:
        _FMP_FINANCIALS_CACHE.clear()


class PolygonFinancialsProvider:
    """Point-in-time fundamentals via Polygon /vX/reference/financials.

    Returns the most-recent statement(s) whose `filing_date` is on or before
    `as_of_date`, so that a historical run only sees data that was public
    at the time. This is the PIT-correct alternative to `yfinance` whose
    fundamentals always reflect "now".

    Docs: https://polygon.io/docs/rest/stocks/fundamentals/financials
    """

    BASE_URL = "https://api.polygon.io/vX/reference/financials"
    # Page size for the all-history fetch. Polygon caps at 100. We rely on
    # the latest 100 filings covering the relevant history for our backtest
    # window (2023-07 → present); for tickers public > ~25 years this may
    # truncate the oldest filings, but those are never queried by the
    # current corpus dates.
    _ALL_FETCH_LIMIT = 100

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_quarterly(
        self,
        symbol: str,
        as_of_date: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` quarterly statements filed on/before `as_of_date`,
        sorted newest first by `period_of_report_date`."""
        return self._fetch(symbol, as_of_date, timeframe="quarterly", limit=limit)

    def fetch_ttm(
        self,
        symbol: str,
        as_of_date: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return self._fetch(symbol, as_of_date, timeframe="ttm", limit=limit)

    def fetch_annual(
        self,
        symbol: str,
        as_of_date: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self._fetch(symbol, as_of_date, timeframe="annual", limit=limit)

    def _fetch_all(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        """Fetch (and cache) the full timeframe history for `symbol`.

        Cached per (symbol, timeframe) at module level so all threads sharing
        this Python process — including the regen worker pool — share results.
        """
        cache_key = (symbol.upper(), timeframe)
        with _FINANCIALS_ALL_LOCK:
            cached = _FINANCIALS_ALL_CACHE.get(cache_key)
            if cached is not None:
                return cached
            if not self.api_key:
                _FINANCIALS_ALL_CACHE[cache_key] = []
                return []
            params: dict[str, Any] = {
                "ticker": symbol.upper(),
                "timeframe": timeframe,
                "limit": self._ALL_FETCH_LIMIT,
                "sort": "period_of_report_date",
                "order": "desc",
                "apiKey": self.api_key,
            }
            try:
                response = self.session.get(
                    self.BASE_URL, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                # Negative cache: store the empty list so we don't retry
                # this (symbol, timeframe) within the same process on
                # transient failures. Restart-the-process if needed.
                _FINANCIALS_ALL_CACHE[cache_key] = []
                return []
            results = payload.get("results") or []
            _FINANCIALS_ALL_CACHE[cache_key] = results
            return results

    def _fetch(
        self,
        symbol: str,
        as_of_date: str,
        timeframe: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` filings on/before `as_of_date`, PIT-correct.

        Polygon may return forward-looking placeholder records with
        `filing_date=None` (e.g. a future quarter pre-populated for
        a fiscal year). We drop those client-side or risk catastrophic
        look-ahead bias.

        Reads from the module-level all-history cache (`_fetch_all`),
        filtering and slicing client-side.
        """
        if not self.api_key:
            return []
        cap = max(1, min(int(limit), self._ALL_FETCH_LIMIT))
        all_results = self._fetch_all(symbol, timeframe)
        filtered: list[dict[str, Any]] = []
        for record in all_results:
            filing_date = record.get("filing_date")
            if not filing_date:
                continue
            if str(filing_date) > as_of_date:
                continue
            end_date = record.get("end_date")
            if end_date and str(end_date) > as_of_date:
                # Period extends past as_of_date — even if the placeholder
                # was somehow filed earlier, the data inside reflects a
                # period whose final values weren't known.
                continue
            filtered.append(record)
            if len(filtered) >= cap:
                break
        return filtered

    @staticmethod
    def value_of(line: Any) -> float | None:
        """Polygon line items look like {'value': 1.23e10, 'unit': 'USD', ...}.
        Returns the value as float or None if missing/malformed."""
        if not isinstance(line, dict):
            return None
        v = line.get("value")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None


# FMP statement field -> Polygon-style GAAP line key. Adapting FMP records
# into Polygon's nested shape lets the pipeline's `_pf_pick` / TTM-synth logic
# consume either provider unchanged.
_FMP_INCOME_MAP = {
    "revenues": "revenue",
    "net_income_loss": "netIncome",
    "operating_income_loss": "operatingIncome",
    "gross_profit": "grossProfit",
}
_FMP_BALANCE_MAP = {
    "equity": "totalStockholdersEquity",
    "assets": "totalAssets",
    "liabilities": "totalLiabilities",
    "current_assets": "totalCurrentAssets",
    "current_liabilities": "totalCurrentLiabilities",
    "inventory": "inventory",
}
_FMP_CASHFLOW_MAP = {
    "net_cash_flow_from_operating_activities": "netCashProvidedByOperatingActivities",
    "net_cash_flow_from_investing_activities": "netCashUsedForInvestingActivites",
}


class FMPFinancialsProvider:
    """Point-in-time fundamentals via Financial Modeling Prep.

    A PIT-correct alternative to ``PolygonFinancialsProvider`` for the analysis
    pipeline. Fetches FMP income / balance-sheet / cash-flow statements, merges
    them by fiscal period, and adapts each period into Polygon's nested record
    shape (``record.financials.<statement>.<gaap_key> = {"value", "unit"}`` plus
    ``filing_date`` / ``end_date`` / ``period_of_report_date``) so the rest of
    the pipeline consumes it identically. Returns only statements whose
    ``fillingDate`` (when the filing became public) is on/before ``as_of_date``.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    _ALL_FETCH_LIMIT = 100

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_quarterly(
        self, symbol: str, as_of_date: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        return self._fetch(symbol, as_of_date, period="quarter", limit=limit)

    def fetch_annual(
        self, symbol: str, as_of_date: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        return self._fetch(symbol, as_of_date, period="annual", limit=limit)

    def fetch_ttm(
        self, symbol: str, as_of_date: str, limit: int = 4
    ) -> list[dict[str, Any]]:
        # FMP exposes no native trailing-twelve-month statement endpoint; the
        # pipeline synthesizes TTM from four quarters itself. Kept for
        # interface parity with PolygonFinancialsProvider.
        return []

    def _statement_url(self, statement: str, symbol: str) -> str:
        return f"{self.BASE_URL}/{statement}/{symbol.upper()}"

    def _fetch_statement(
        self, symbol: str, statement: str, period: str
    ) -> list[dict[str, Any]]:
        params = {
            "period": period,
            "limit": self._ALL_FETCH_LIMIT,
            "apikey": self.api_key,
        }
        response = self.session.get(
            self._statement_url(statement, symbol), params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    def _fetch_all(self, symbol: str, period: str) -> list[dict[str, Any]]:
        """Fetch + merge the three statements into Polygon-shaped records.

        Cached per (symbol, period) at module level so threads share results.
        On any error the empty list is negative-cached for the process.
        """
        cache_key = (symbol.upper(), period)
        with _FMP_FINANCIALS_LOCK:
            cached = _FMP_FINANCIALS_CACHE.get(cache_key)
            if cached is not None:
                return cached
            if not self.api_key:
                _FMP_FINANCIALS_CACHE[cache_key] = []
                return []
            try:
                income = self._fetch_statement(symbol, "income-statement", period)
                balance = self._fetch_statement(
                    symbol, "balance-sheet-statement", period
                )
                cashflow = self._fetch_statement(
                    symbol, "cash-flow-statement", period
                )
            except Exception:
                _FMP_FINANCIALS_CACHE[cache_key] = []
                return []
            records = self._merge_statements(income, balance, cashflow)
            _FMP_FINANCIALS_CACHE[cache_key] = records
            return records

    @staticmethod
    def _section(row: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
        section: dict[str, Any] = {}
        for gaap_key, fmp_key in field_map.items():
            value = row.get(fmp_key)
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None:
                section[gaap_key] = {"value": value, "unit": "USD"}
        return section

    def _merge_statements(
        self,
        income: list[dict[str, Any]],
        balance: list[dict[str, Any]],
        cashflow: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge per-statement rows keyed by fiscal period-end date.

        Income statements drive the record set (period date, filing date);
        balance-sheet and cash-flow lines are joined on the matching date.
        Records are returned newest-first (FMP's native order).
        """
        balance_by_date = {str(r.get("date")): r for r in balance if r.get("date")}
        cashflow_by_date = {str(r.get("date")): r for r in cashflow if r.get("date")}
        records: list[dict[str, Any]] = []
        for inc in income:
            date = str(inc.get("date") or "")
            if not date:
                continue
            bs = balance_by_date.get(date, {})
            cf = cashflow_by_date.get(date, {})
            records.append({
                "filing_date": inc.get("fillingDate")
                or inc.get("acceptedDate"),
                "end_date": date,
                "start_date": None,
                "period_of_report_date": date,
                "financials": {
                    "income_statement": self._section(inc, _FMP_INCOME_MAP),
                    "balance_sheet": self._section(bs, _FMP_BALANCE_MAP),
                    "cash_flow_statement": self._section(cf, _FMP_CASHFLOW_MAP),
                },
            })
        return records

    def _fetch(
        self, symbol: str, as_of_date: str, period: str, limit: int
    ) -> list[dict[str, Any]]:
        """Up to ``limit`` records filed on/before ``as_of_date``, PIT-correct."""
        if not self.api_key:
            return []
        cap = max(1, min(int(limit), self._ALL_FETCH_LIMIT))
        filtered: list[dict[str, Any]] = []
        for record in self._fetch_all(symbol, period):
            filing_date = record.get("filing_date")
            if not filing_date or str(filing_date) > as_of_date:
                continue
            end_date = record.get("end_date")
            if end_date and str(end_date) > as_of_date:
                continue
            filtered.append(record)
            if len(filtered) >= cap:
                break
        return filtered


class PolygonMarketStatusProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    def is_market_open(self) -> bool:
        if not self.api_key:
            return _fallback_market_open()
        url = "https://api.polygon.io/v1/marketstatus/now"
        params = {"apiKey": self.api_key}
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return _fallback_market_open()
        status = str(payload.get("market", "")).lower()
        return status == "open"


def strip_html_to_text(document: str) -> str:
    """Reduce a filing's HTML to readable plain text.

    Uses ``parsel`` (already a project dependency) to drop script/style nodes
    and collect text. Whitespace is collapsed and non-breaking spaces are
    normalised so downstream section regexes see clean ``Item N`` headers.
    Plain-text filings (no tags) pass through largely unchanged.
    """
    if not document:
        return ""
    from parsel import Selector

    try:
        selector = Selector(text=document)
        # Remove non-content nodes before extracting text.
        for bad in selector.xpath("//script | //style"):
            bad.root.getparent().remove(bad.root)
        parts = selector.xpath("//body//text()").getall()
        if not parts:
            parts = selector.xpath("//text()").getall()
        text = " ".join(parts)
    except Exception:
        # Fall back to a naive tag strip if parsing fails on malformed HTML.
        text = re.sub(r"<[^>]+>", " ", document)

    text = text.replace("\xa0", " ").replace("​", "")
    # Normalise curly quotes so title anchors like "management's discussion"
    # match regardless of the apostrophe glyph used in the filing.
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


# Per-form section maps. Sections are anchored on their TITLE text (more
# reliable than bare "Item N" numbers, which repeat across Part I/Part II of a
# 10-Q) and bounded by the next section's title or item marker. Matching is
# case-insensitive on a whitespace-normalised copy of the text.
_FILING_SECTION_MAP: dict[str, dict[str, dict[str, list[str]]]] = {
    "10-K": {
        "risk_factors": {
            "starts": ["risk factors"],
            "prefixes": ["item 1a"],
            "ends": [
                "item 1b", "unresolved staff comments", "item 2.",
                "management's discussion", "properties",
            ],
        },
        "mdna": {
            "starts": [
                "management's discussion and analysis", "management's discussion",
            ],
            "prefixes": ["item 7"],
            "ends": [
                "item 7a", "item 8", "quantitative and qualitative",
                "financial statements and supplementary",
            ],
        },
    },
    "10-Q": {
        "mdna": {
            "starts": [
                "management's discussion and analysis", "management's discussion",
            ],
            "prefixes": ["item 2"],
            "ends": [
                "item 3.", "item 4.", "quantitative and qualitative",
                "controls and procedures",
            ],
        },
        "risk_factors": {
            "starts": ["risk factors"],
            "prefixes": ["item 1a"],
            "ends": [
                "item 2.", "unregistered sales", "item 6", "exhibits",
                "defaults upon senior", "legal proceedings",
            ],
        },
    },
}

# Window (chars) after a title anchor inspected to decide whether the match is
# a table-of-contents entry rather than the real section heading.
_TOC_WINDOW = 60
_TOC_PAGE_NUM_RE = re.compile(r"^[\s.…]*\d{1,4}\b")
# Window (chars) before a title anchor searched for its "Item N" marker. The
# real section heading reads "Item 1A. Risk Factors"; an in-prose mention
# (e.g. a forward-looking-statements disclaimer) has no item marker just
# before it, so requiring one filters those out.
_PREFIX_WINDOW = 40


def _looks_like_toc(text_after_anchor: str) -> bool:
    """Heuristic: does the text right after a title anchor read like a TOC row?

    Table-of-contents rows are followed by dot leaders, a page number, and/or
    the next item heading; real sections are followed by prose.
    """
    head = text_after_anchor[:_TOC_WINDOW]
    lowered = head.lower()
    if "item " in lowered:  # another item heading immediately follows
        return True
    if "...." in head or "…" in head:  # dot leaders
        return True
    if _TOC_PAGE_NUM_RE.match(head):  # leading page number
        return True
    return False


def _has_item_prefix(lowered: str, anchor_start: int, prefixes: list[str]) -> bool:
    """True if an ``Item N`` marker sits just before the title anchor."""
    if not prefixes:
        return True  # no prefix requirement
    window = lowered[max(0, anchor_start - _PREFIX_WINDOW):anchor_start]
    return any(p in window for p in prefixes)


def _extract_section(
    text: str,
    lowered: str,
    starts: list[str],
    ends: list[str],
    *,
    prefixes: list[str] | None = None,
    min_len: int = 40,
) -> str | None:
    """Slice ``text`` for a section, skipping table-of-contents matches.

    Collects every start-anchor occurrence and drops the ones that look like
    TOC rows. Candidates whose title is preceded by the section's ``Item N``
    marker are preferred (the real heading); only if none qualify do we fall
    back to unprefixed candidates (some filings omit the marker). Among the
    chosen set, the longest slice to the nearest end anchor wins.
    """
    prefixes = prefixes or []
    prefixed: list[int] = []
    other: list[int] = []
    for anchor in starts:
        pos = 0
        while True:
            idx = lowered.find(anchor, pos)
            if idx == -1:
                break
            pos = idx + len(anchor)
            if _looks_like_toc(text[idx + len(anchor):]):
                continue
            if _has_item_prefix(lowered, idx, prefixes):
                prefixed.append(idx)
            else:
                other.append(idx)

    candidates = prefixed or other
    best: str | None = None
    for start in candidates:
        end_idx = len(text)
        for phrase in ends:
            phrase_idx = lowered.find(phrase, start + 1)
            if phrase_idx != -1:
                end_idx = min(end_idx, phrase_idx)
        section = text[start:end_idx].strip()
        if len(section) >= min_len and (best is None or len(section) > len(best)):
            best = section
    return best


def extract_filing_sections(
    document_text: str,
    form: str,
    *,
    max_section_chars: int = 20_000,
) -> dict[str, str]:
    """Extract the analytically-useful sections from a filing.

    * 10-K: Risk Factors (Item 1A) and MD&A (Item 7).
    * 10-Q: MD&A (Item 2) and Risk Factors (Part II, Item 1A).
    * 8-K (and anything unmapped): the full body, since 8-Ks are short and
      have no stable item structure worth slicing.

    Each section is truncated to ``max_section_chars``. Sections that cannot
    be located are simply omitted; an empty dict means nothing usable was
    found (the caller can fall back to the raw text).
    """
    text = strip_html_to_text(document_text)
    if not text:
        return {}
    form = str(form or "").upper()
    spec = _FILING_SECTION_MAP.get(form)
    if not spec:
        return {"body": text[:max_section_chars]}

    lowered = text.lower()
    out: dict[str, str] = {}
    for name, bounds in spec.items():
        section = _extract_section(
            text,
            lowered,
            bounds["starts"],
            bounds["ends"],
            prefixes=bounds.get("prefixes"),
        )
        if section:
            out[name] = section[:max_section_chars]
    if not out:
        # Item headers not found (unusual formatting) — give the caller the
        # head of the document rather than nothing.
        out["body"] = text[:max_section_chars]
    return out


class SECFilingsProvider:
    """Fetch SEC EDGAR submissions and pick the latest 10-Q/10-K/8-K
    filing on/before ``as_of_date``.

    SEC EDGAR requires a User-Agent that identifies the requester with a
    contact email — anonymous defaults like ``Python-requests/2.x`` and
    placeholder strings get blocked with HTTP 403 (``Request Rate
    Threshold Exceeded``). EDGAR also rate-limits to ~10 requests per
    second per IP. This provider:

    * Builds a compliant User-Agent from ``SEC_USER_AGENT`` (preferred),
      otherwise from ``SEC_CONTACT_EMAIL`` and ``SEC_CONTACT_NAME``.
    * Throttles requests to ``MIN_REQUEST_INTERVAL`` between calls.
    * Retries HTTP 429/5xx with exponential backoff.
    * Raises ``SECFetchError`` on transient failures so callers can
      avoid caching error responses.
    """

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    # Primary filing documents live under the EDGAR Archives tree. The CIK is
    # un-padded here and the accession number has its dashes stripped.
    ARCHIVES_URL = (
        "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
    )
    RELEVANT_FORMS = {"10-Q", "10-K", "8-K"}
    MIN_REQUEST_INTERVAL = 0.11  # ~9 req/s, under EDGAR's 10/s ceiling.
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each retry.

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or self._default_user_agent()
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        # `Host` is overridden per-request because the ticker map lives
        # on `www.sec.gov` while submissions live on `data.sec.gov`.
        self._ticker_to_cik: dict[str, str] | None = None
        self._last_request_ts: float = 0.0
        self._throttle_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _default_user_agent() -> str:
        explicit = os.getenv("SEC_USER_AGENT", "").strip()
        if explicit:
            return explicit
        email = os.getenv("SEC_CONTACT_EMAIL", "").strip()
        name = os.getenv("SEC_CONTACT_NAME", "TradingAgents Research").strip()
        if email:
            return f"{name} {email}"
        # SEC EDGAR requires a contact email; without one the request
        # gets blocked. Fall back to a generic-but-valid contact so the
        # request shape is at least compliant. Users should set
        # `SEC_USER_AGENT` (or `SEC_CONTACT_EMAIL`) to get reliable
        # service.
        return "TradingAgents Research contact@tradingagents.local"

    def get_latest_filing(
        self,
        symbol: str,
        as_of_date: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the most recent 10-Q/10-K/8-K filing on/before
        ``as_of_date``, or ``None`` if the issuer has no such filing in
        the recent submissions feed.

        Raises ``SECFetchError`` for transient HTTP/parse failures so
        the caller can avoid caching error responses.
        """

        cik = self._get_cik(symbol)
        if not cik:
            return None
        submissions_url = self.SUBMISSIONS_URL.format(cik=int(cik))
        payload = self._fetch_json(submissions_url, host="data.sec.gov")
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", []) or []
        accession_numbers = recent.get("accessionNumber", []) or []
        filing_dates = recent.get("filingDate", []) or []
        primary_docs = recent.get("primaryDocument", []) or []

        for idx, form in enumerate(forms):
            f = str(form).upper()
            if f not in self.RELEVANT_FORMS:
                continue
            filed = filing_dates[idx] if idx < len(filing_dates) else None
            if as_of_date and filed and filed > as_of_date:
                # Filing is after the analysis date; skip to keep PIT.
                continue
            accession = (
                accession_numbers[idx] if idx < len(accession_numbers) else None
            )
            doc = primary_docs[idx] if idx < len(primary_docs) else None
            return {
                "symbol": symbol.upper(),
                "cik": cik,
                "form": f,
                "accession": accession,
                "filing_date": filed,
                "primary_document": doc,
            }
        return None

    def filing_document_url(self, filing: dict[str, Any]) -> str | None:
        """Build the EDGAR Archives URL for a filing's primary document.

        Returns ``None`` if the filing dict is missing the cik, accession,
        or primary_document fields needed to locate the file.
        """
        cik = filing.get("cik")
        accession = filing.get("accession")
        document = filing.get("primary_document")
        if not (cik and accession and document):
            return None
        return self.ARCHIVES_URL.format(
            cik=int(cik),
            accession=str(accession).replace("-", ""),
            document=document,
        )

    def fetch_filing_document(
        self,
        filing: dict[str, Any],
        *,
        max_chars: int = 8_000_000,
    ) -> str | None:
        """Fetch the raw primary document (HTML/text) for a filing.

        Returns the document body (truncated to ``max_chars`` to bound memory
        for pathologically large filings) or ``None`` when the URL cannot be
        built. The cap is generous because a filing's HTML is ~10x its text,
        and in a 10-Q the MD&A sits *after* the bulky financial statements —
        too small a cap would silently drop the section we most want.

        Raises ``SECFetchError`` on transient HTTP/network failures so callers
        can avoid caching an error.
        """
        url = self.filing_document_url(filing)
        if not url:
            return None
        text = self._fetch_text(url, host="www.sec.gov")
        if max_chars and len(text) > max_chars:
            return text[:max_chars]
        return text

    def _fetch_text(self, url: str, *, host: str) -> str:
        """Fetch a URL returning its raw text body (mirrors ``_fetch_json``).

        Uses the same throttle/backoff/retry policy as the JSON fetcher so a
        document download stays within EDGAR's rate ceiling.
        """
        headers = dict(self.headers)
        headers["Host"] = host
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            self._throttle()
            try:
                response = requests.get(url, headers=headers, timeout=30)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            status = response.status_code
            if status == 200:
                return response.text
            if status == 429 or 500 <= status < 600:
                last_exc = SECFetchError(
                    f"SEC {url} returned HTTP {status} (attempt "
                    f"{attempt + 1}/{self.MAX_RETRIES})"
                )
                self._sleep_backoff(attempt)
                continue
            raise SECFetchError(
                f"SEC {url} returned HTTP {status} (non-retryable)"
            )
        raise SECFetchError(
            f"SEC fetch exhausted retries for {url}: {last_exc}"
        ) from last_exc

    def _get_cik(self, symbol: str) -> str | None:
        mapping = self._load_ticker_mapping()
        if not mapping:
            return None
        return mapping.get(symbol.upper())

    def _load_ticker_mapping(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        try:
            payload = self._fetch_json(self.TICKER_MAP_URL, host="www.sec.gov")
        except SECFetchError as exc:
            # Ticker map can't be cached as empty on transient failure —
            # that would poison every subsequent call. Re-raise so the
            # caller decides whether to retry later.
            self._logger.warning("SEC ticker map fetch failed: %s", exc)
            raise
        mapping: dict[str, str] = {}
        for _, row in payload.items():
            ticker = str(row.get("ticker", "")).upper()
            cik = str(row.get("cik_str", ""))
            if ticker and cik:
                mapping[ticker] = cik
        self._ticker_to_cik = mapping
        return mapping

    def _throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            wait = self.MIN_REQUEST_INTERVAL - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _fetch_json(self, url: str, *, host: str) -> dict[str, Any]:
        headers = dict(self.headers)
        headers["Host"] = host
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            self._throttle()
            try:
                response = requests.get(url, headers=headers, timeout=20)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SECFetchError(
                        f"SEC {url} returned non-JSON body (status 200)"
                    ) from exc
            if status == 429 or 500 <= status < 600:
                last_exc = SECFetchError(
                    f"SEC {url} returned HTTP {status} (attempt "
                    f"{attempt + 1}/{self.MAX_RETRIES})"
                )
                self._sleep_backoff(attempt)
                continue
            # 4xx other than 429 (403 forbidden, 404 not found) — no
            # point retrying with the same UA.
            raise SECFetchError(
                f"SEC {url} returned HTTP {status} (non-retryable)"
            )
        raise SECFetchError(
            f"SEC fetch exhausted retries for {url}: {last_exc}"
        ) from last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.RETRY_BACKOFF_BASE * (2**attempt)
        time.sleep(delay)


def _fallback_market_open() -> bool:
    now = datetime.now(timezone.utc)
    return now.weekday() < 5
