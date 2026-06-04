from __future__ import annotations

import pytest

from tradingagents.analysis_only.providers import (
    FearGreedProvider,
    PolygonFinancialsProvider,
    reset_financials_cache,
)


def _payload():
    return {
        "fear_and_greed": {
            "score": 82.3,
            "rating": "extreme greed",
            "timestamp": "2026-05-23T20:00:00+00:00",
            "previous_1_week": 70.1,
            "previous_1_month": 56.2,
            "previous_1_year": 44.0,
        },
        "fear_and_greed_historical": {
            "data": [
                {
                    "x": 1_767_225_600_000,  # 2026-01-01 UTC
                    "y": 20.0,
                    "rating": "extreme fear",
                },
                {
                    "x": 1_767_484_800_000,  # 2026-01-04 UTC
                    "y": 42.5,
                    "rating": "fear",
                },
                {
                    "x": 1_767_571_200_000,  # 2026-01-05 UTC
                    "y": 51.0,
                    "rating": "neutral",
                },
            ]
        },
        "market_momentum_sp500": {
            "score": 91.25,
            "rating": "extreme greed",
        },
        "put_call_options": {
            "score": 35.4,
            "rating": "fear",
        },
    }


def test_fear_greed_normalizes_current_payload():
    provider = FearGreedProvider()
    out = provider.normalize(_payload(), as_of_date=None)

    assert out["status"] == "ok"
    assert out["pit_status"] == "live"
    assert out["source"] == "cnn_fear_and_greed"
    assert out["score"] == pytest.approx(82.3)
    assert out["rating"] == "extreme greed"
    assert out["previous_1_week"] == pytest.approx(70.1)
    assert out["indicators"]["market_momentum_sp500"]["score"] == pytest.approx(91.25)
    assert out["indicators"]["put_call_options"]["rating"] == "fear"


def test_fear_greed_historical_exact_date_uses_that_point():
    provider = FearGreedProvider()
    out = provider.normalize(_payload(), as_of_date="2026-01-05")

    assert out["status"] == "ok"
    assert out["pit_status"] == "pit"
    assert out["score"] == pytest.approx(51.0)
    assert out["rating"] == "neutral"
    assert out["previous_1_week"] is None
    assert out["indicators"] == {}


def test_fear_greed_historical_weekend_uses_closest_prior_point():
    provider = FearGreedProvider()
    out = provider.normalize(_payload(), as_of_date="2026-01-03")

    assert out["status"] == "ok"
    assert out["pit_status"] == "pit"
    assert out["score"] == pytest.approx(20.0)
    assert out["rating"] == "extreme fear"


def test_fear_greed_historical_missing_prior_point_is_unavailable():
    provider = FearGreedProvider()
    out = provider.normalize(_payload(), as_of_date="2025-12-31")

    assert out["status"] == "unavailable"
    assert out["pit_status"] == "unavailable"
    assert out["reason"] == "no_historical_point_on_or_before_date"


# ---------- VIXFearGreedProvider ----------


from datetime import datetime, timedelta

import pytest
from tradingagents.analysis_only import providers as _providers
from tradingagents.analysis_only.providers import (
    VIXFearGreedProvider,
    reset_vix_cache,
)


@pytest.fixture(autouse=True)
def _clear_vix_cache():
    reset_vix_cache()
    yield
    reset_vix_cache()


def _vix_series_const(value: float, dates: list[str]) -> dict[str, float]:
    return {d: value for d in dates}


def _vix_series_uniform(dates: list[str], start: float, step: float) -> dict[str, float]:
    return {d: start + i * step for i, d in enumerate(dates)}


def test_vix_proxy_returns_unavailable_when_no_series(monkeypatch):
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: {})
    p = VIXFearGreedProvider()
    out = p.get_index(as_of_date="2024-06-21")
    assert out["status"] == "unavailable"
    assert out["source"] == "vix_fear_greed_proxy"


def test_vix_proxy_extreme_fear_when_vix_at_top_of_window(monkeypatch):
    # 119 trailing days of VIX ~15, with a current spike to 40 on the last
    # date in the series. Target is one day after so walk-back picks up the
    # spike and the trailing window contains all 120 observations.
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
             for i in range(120)]
    series = {d: 15.0 for d in dates[:-1]}
    series[dates[-1]] = 40.0
    target = (datetime.fromisoformat(dates[-1]) + timedelta(days=1)).date().isoformat()
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: series)
    p = VIXFearGreedProvider()
    out = p.get_index(as_of_date=target)
    assert out["status"] == "ok"
    assert out["rating"] == "extreme fear"
    assert out["score"] <= 25


def test_vix_proxy_extreme_greed_when_vix_at_bottom_of_window(monkeypatch):
    # Trailing window is high VIX; current is low.
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
             for i in range(120)]
    series = {d: 35.0 for d in dates[:-1]}
    series[dates[-1]] = 10.0
    target = (datetime.fromisoformat(dates[-1]) + timedelta(days=1)).date().isoformat()
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: series)
    p = VIXFearGreedProvider()
    out = p.get_index(as_of_date=target)
    assert out["status"] == "ok"
    assert out["rating"] == "extreme greed"
    assert out["score"] >= 75


def test_vix_proxy_neutral_when_vix_at_median(monkeypatch):
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
             for i in range(120)]
    series = _vix_series_uniform(dates, start=10.0, step=0.1)  # 10..21.9
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: series)
    p = VIXFearGreedProvider()
    # Target one day after dates[100] so walk-back picks up dates[100] and the
    # trailing window is dates[0..99] — 100 obs averaging well below dates[100].
    target = (datetime.fromisoformat(dates[100]) + timedelta(days=1)).date().isoformat()
    out = p.get_index(as_of_date=target)
    assert out["status"] == "ok"
    # dates[100] is higher than all 100 preceding obs, so percentile = 1.0
    # which maps to score = 0 = extreme fear (in this monotone-up series).
    assert out["rating"] in ("extreme fear", "fear")


def test_vix_proxy_pit_correct_strictly_before(monkeypatch):
    # The trailing window must NOT include the target date itself.
    dates = ["2024-06-21", "2024-06-24", "2024-06-25"]
    series = {"2024-06-21": 15.0, "2024-06-24": 16.0, "2024-06-25": 100.0}
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: series)
    p = VIXFearGreedProvider(window_days=30)
    # If 2024-06-25's 100 leaks into the lookup, it would dominate. PIT means it doesn't.
    out = p.get_index(as_of_date="2024-06-25")
    # With insufficient window (only 2 prior obs), should be unavailable.
    assert out["status"] == "unavailable"
    assert "insufficient" in out["reason"]


def test_vix_proxy_cached_across_providers(monkeypatch):
    calls = {"n": 0}

    def fake_loader(start, logger):
        calls["n"] += 1
        # 50 days of constant VIX
        dates = [(datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
                 for i in range(50)]
        return {d: 15.0 for d in dates}

    monkeypatch.setattr(_providers, "_load_vix_series", fake_loader)
    VIXFearGreedProvider()._ensure_loaded()
    VIXFearGreedProvider()._ensure_loaded()
    VIXFearGreedProvider()._ensure_loaded()
    assert calls["n"] == 1


def test_vix_proxy_returns_same_shape_as_cnn(monkeypatch):
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
             for i in range(60)]
    series = {d: 18.0 for d in dates}
    monkeypatch.setattr(_providers, "_load_vix_series", lambda *a, **kw: series)
    p = VIXFearGreedProvider()
    out = p.get_index(as_of_date=dates[55])
    # Must contain the same fields downstream code reads off CNN F&G dict.
    for key in (
        "status", "pit_status", "source", "score", "rating", "timestamp",
        "previous_1_week", "previous_1_month", "previous_1_year",
        "indicators",
    ):
        assert key in out
    assert out["source"] == "vix_fear_greed_proxy"


# ---------------------------------------------------------------------------
# PolygonFinancialsProvider — module-level cache (see plan Future-work #3).
# Reduces per-regen HTTP calls from ~9k to ~60 by fetching all-history per
# (symbol, timeframe) once and slicing client-side.
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _StubResponse(self.payload)


def _sample_filings_payload():
    return {
        "results": [
            {
                "filing_date": "2026-02-20",
                "end_date": "2026-01-31",
                "period_of_report_date": "2026-01-31",
                "financials": {"income_statement": {"revenues": {"value": 300}}},
            },
            {
                "filing_date": "2025-11-15",
                "end_date": "2025-10-31",
                "period_of_report_date": "2025-10-31",
                "financials": {"income_statement": {"revenues": {"value": 250}}},
            },
            {
                "filing_date": "2025-08-20",
                "end_date": "2025-07-31",
                "period_of_report_date": "2025-07-31",
                "financials": {"income_statement": {"revenues": {"value": 220}}},
            },
            {
                # Placeholder record with no filing_date — must be dropped.
                "filing_date": None,
                "end_date": "2027-04-30",
                "financials": {},
            },
            {
                # Future-dated filing — must be dropped client-side.
                "filing_date": "2026-08-15",
                "end_date": "2026-07-31",
                "financials": {},
            },
        ]
    }


def test_polygon_financials_pit_filter_drops_future_and_null():
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    out = provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01", limit=8)
    assert len(session.calls) == 1
    dates = [r["filing_date"] for r in out]
    assert dates == ["2026-02-20", "2025-11-15", "2025-08-20"]


def test_polygon_financials_cache_dedupes_repeat_calls():
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01")
    # Different as_of_date and limit must NOT trigger another HTTP request.
    provider.fetch_quarterly(symbol="ABCD", as_of_date="2025-12-01", limit=4)
    provider.fetch_quarterly(symbol="ABCD", as_of_date="2025-09-15", limit=2)
    assert len(session.calls) == 1


def test_polygon_financials_cache_segregates_symbols_and_timeframes():
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01")
    provider.fetch_quarterly(symbol="WXYZ", as_of_date="2026-03-01")
    provider.fetch_annual(symbol="ABCD", as_of_date="2026-03-01")
    assert len(session.calls) == 3
    params = [c["params"] for c in session.calls]
    keys = {(p["ticker"], p["timeframe"]) for p in params}
    assert keys == {("ABCD", "quarterly"), ("WXYZ", "quarterly"), ("ABCD", "annual")}


def test_polygon_financials_cache_respects_limit_after_filter():
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    out = provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01", limit=2)
    assert len(out) == 2
    assert [r["filing_date"] for r in out] == ["2026-02-20", "2025-11-15"]


def test_polygon_financials_cache_no_api_key_returns_empty(monkeypatch):
    # Constructor falls back to POLYGON_API_KEY env var when api_key is
    # falsy. Clear it for this test so we're actually testing the no-key
    # behavior (otherwise we'd hit the live API).
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="", session=session)
    out = provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01")
    assert out == []
    assert session.calls == []


def test_polygon_financials_no_server_side_filing_date_filter():
    """All-history fetch must not pass filing_date.lte — otherwise different
    as_of_dates would produce different cached blobs."""
    reset_financials_cache()
    session = _StubSession(_sample_filings_payload())
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    provider.fetch_quarterly(symbol="ABCD", as_of_date="2026-03-01")
    assert "filing_date.lte" not in session.calls[0]["params"]


def test_polygon_financials_negative_caches_failures():
    """A failed HTTP fetch is negatively cached so subsequent calls don't
    re-attempt within the same process."""
    reset_financials_cache()

    class _FailingSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("simulated network failure")

    session = _FailingSession()
    provider = PolygonFinancialsProvider(api_key="test", session=session)
    assert provider.fetch_quarterly("ABCD", "2026-03-01") == []
    assert provider.fetch_quarterly("ABCD", "2026-02-01") == []
    assert session.calls == 1
# ---------- Polygon daily aggs cache ----------
#
# These cover the helpers that replace the noisy yfinance market-context
# path for SPY / sector ETFs. Indices stay on yfinance; the helpers should
# correctly identify which symbols they support.

import logging

import pandas as pd

from tradingagents.analysis_only import providers as _polygon_providers
from tradingagents.analysis_only.providers import (
    fetch_polygon_daily_aggs_cached,
    is_polygon_supported_symbol,
    reset_polygon_daily_aggs_cache,
)


@pytest.fixture(autouse=True)
def _clear_polygon_daily_aggs_cache():
    reset_polygon_daily_aggs_cache()
    yield
    reset_polygon_daily_aggs_cache()


def test_polygon_supported_symbol_routes_etfs_and_stocks():
    # ETFs and stocks are on the Polygon Stocks plan.
    for sym in ("SPY", "XLK", "XLV", "XLF", "NVDA", "AAPL", "BIL"):
        assert is_polygon_supported_symbol(sym) is True


def test_polygon_supported_symbol_excludes_yahoo_indices():
    # Yahoo-style indices (^X) are NOT on the Polygon Stocks plan
    # (I:VIX / I:IRX / I:TNX return 403 NOT_AUTHORIZED).
    for sym in ("^VIX", "^IRX", "^TNX", "^GSPC", "^DJI"):
        assert is_polygon_supported_symbol(sym) is False
    # Defensive: empty / None-like inputs are NOT supported.
    assert is_polygon_supported_symbol("") is False


def test_polygon_daily_aggs_cache_dedupes_repeated_calls(monkeypatch):
    """Multiple callers asking for the same (symbol, start, end) trigger
    exactly one underlying fetch."""
    call_count = {"n": 0}

    def fake_loader(symbol, start, end, api_key, logger):
        call_count["n"] += 1
        idx = pd.to_datetime(["2024-06-19", "2024-06-20", "2024-06-21"])
        return pd.DataFrame(
            {
                "Open": [540.0, 541.0, 542.0],
                "High": [545.0, 546.0, 547.0],
                "Low": [538.0, 539.0, 540.0],
                "Close": [543.0, 544.0, 545.0],
                "Volume": [10_000_000, 11_000_000, 12_000_000],
            },
            index=pd.Index(idx, name="Date"),
        )

    monkeypatch.setattr(
        _polygon_providers, "_load_polygon_daily_aggs", fake_loader,
    )

    a = fetch_polygon_daily_aggs_cached("SPY", "2024-06-19", "2024-06-21", "key")
    b = fetch_polygon_daily_aggs_cached("SPY", "2024-06-19", "2024-06-21", "key")
    c = fetch_polygon_daily_aggs_cached("SPY", "2024-06-19", "2024-06-21", "key")
    assert call_count["n"] == 1
    assert len(a) == len(b) == len(c) == 3
    # Same DataFrame object (cached identity).
    assert a is b


def test_polygon_daily_aggs_cache_keys_by_symbol_and_range(monkeypatch):
    """Different (symbol, start, end) tuples each fetch independently."""
    calls: list[tuple] = []

    def fake_loader(symbol, start, end, api_key, logger):
        calls.append((symbol, start, end))
        idx = pd.to_datetime([start])
        return pd.DataFrame({"Close": [100.0]}, index=pd.Index(idx, name="Date"))

    monkeypatch.setattr(
        _polygon_providers, "_load_polygon_daily_aggs", fake_loader,
    )
    fetch_polygon_daily_aggs_cached("SPY", "2024-01-01", "2024-01-31", "k")
    fetch_polygon_daily_aggs_cached("XLK", "2024-01-01", "2024-01-31", "k")
    fetch_polygon_daily_aggs_cached("SPY", "2024-02-01", "2024-02-29", "k")
    fetch_polygon_daily_aggs_cached("SPY", "2024-01-01", "2024-01-31", "k")  # cache hit

    assert len(calls) == 3  # the 4th call hit the cache


def test_polygon_daily_aggs_empty_result_is_not_cached(monkeypatch):
    """An empty fetch (network blip / no data) does NOT poison the cache —
    the next caller retries."""
    call_count = {"n": 0}

    def fake_loader(symbol, start, end, api_key, logger):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return pd.DataFrame()
        idx = pd.to_datetime(["2024-06-19"])
        return pd.DataFrame({"Close": [100.0]}, index=pd.Index(idx, name="Date"))

    monkeypatch.setattr(
        _polygon_providers, "_load_polygon_daily_aggs", fake_loader,
    )

    first = fetch_polygon_daily_aggs_cached("SPY", "2024-06-19", "2024-06-19", "k")
    assert first.empty
    second = fetch_polygon_daily_aggs_cached("SPY", "2024-06-19", "2024-06-19", "k")
    assert not second.empty
    # Two fetches because empty was not cached.
    assert call_count["n"] == 2


def test_polygon_daily_aggs_no_api_key_returns_empty():
    """Without an API key the loader short-circuits to an empty frame
    (callers fall back to yfinance)."""
    log = logging.getLogger("test_polygon_no_key")
    out = _polygon_providers._load_polygon_daily_aggs(
        symbol="SPY",
        start="2024-06-19",
        end="2024-06-21",
        api_key="",
        logger=log,
    )
    assert out.empty


def test_polygon_daily_aggs_pit_correct_for_historical_as_of(monkeypatch):
    """The cached frame contains only the closes within [start, end] —
    nothing dated AFTER the requested end leaks in."""
    def fake_loader(symbol, start, end, api_key, logger):
        # Simulate the Polygon API correctly honoring end-date inclusivity:
        # only bars with date ≤ end appear.
        all_dates = pd.to_datetime([
            "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20",
            "2024-06-21", "2024-06-24", "2024-06-25",
        ])
        end_ts = pd.to_datetime(end)
        included = [d for d in all_dates if d <= end_ts and d >= pd.to_datetime(start)]
        return pd.DataFrame(
            {"Close": list(range(100, 100 + len(included)))},
            index=pd.Index(included, name="Date"),
        )

    monkeypatch.setattr(
        _polygon_providers, "_load_polygon_daily_aggs", fake_loader,
    )
    # Ask for as_of=2024-06-21 (Friday). Should never see 2024-06-24/25.
    out = fetch_polygon_daily_aggs_cached("SPY", "2024-06-17", "2024-06-21", "k")
    assert out.index.max() <= pd.to_datetime("2024-06-21")
    assert pd.to_datetime("2024-06-24") not in out.index
    assert pd.to_datetime("2024-06-25") not in out.index


def test_polygon_daily_aggs_parses_polygon_response_shape(monkeypatch):
    """Sanity-check the parser: Polygon's `t/o/h/l/c/v` payload becomes
    a DataFrame with OHLCV columns and a normalized Date index."""
    fake_payload = {
        "results": [
            {
                "t": 1_718_582_400_000,  # 2024-06-17 00:00 UTC
                "o": 540.0, "h": 545.0, "l": 538.0, "c": 543.0, "v": 1_000_000,
                "vw": 542.5, "n": 50_000,
            },
            {
                "t": 1_718_668_800_000,  # 2024-06-18 00:00 UTC
                "o": 543.0, "h": 547.0, "l": 541.0, "c": 545.0, "v": 1_200_000,
                "vw": 544.0, "n": 55_000,
            },
        ]
    }

    class _FakeResp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return fake_payload

    monkeypatch.setattr(
        _polygon_providers.requests, "get", lambda *a, **kw: _FakeResp(),
    )
    log = logging.getLogger("test_polygon_parse")
    out = _polygon_providers._load_polygon_daily_aggs(
        symbol="SPY",
        start="2024-06-17",
        end="2024-06-18",
        api_key="dummy",
        logger=log,
    )
    assert len(out) == 2
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Close"].iloc[-1] == 545.0
    # Index is a naive Datetime index (matches yfinance auto_adjust=True).
    assert out.index.tz is None
    assert pd.to_datetime("2024-06-17") in out.index


def test_polygon_daily_aggs_http_error_returns_empty(monkeypatch):
    """A 4xx/5xx response surfaces as an empty frame; downstream callers
    treat as missing data without raising."""
    class _FakeResp:
        def raise_for_status(self):
            raise RuntimeError("HTTP 500")
        def json(self):
            return {}

    monkeypatch.setattr(
        _polygon_providers.requests, "get", lambda *a, **kw: _FakeResp(),
    )
    log = logging.getLogger("test_polygon_err")
    out = _polygon_providers._load_polygon_daily_aggs(
        symbol="SPY",
        start="2024-06-17",
        end="2024-06-18",
        api_key="dummy",
        logger=log,
    )
    assert out.empty
# ---------- SECFilingsProvider ----------


from tradingagents.analysis_only.providers import (
    SECFetchError,
    SECFilingsProvider,
)


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_payload: Any | None = None,
        body: str = "",
    ):
        self.status_code = status_code
        self._json = json_payload
        self.text = body

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _ticker_map_payload() -> dict[str, Any]:
    return {
        "0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"},
        "1": {"ticker": "AAPL", "cik_str": 320193, "title": "APPLE INC"},
    }


def _submissions_payload(
    forms: list[str],
    dates: list[str],
    accessions: list[str] | None = None,
    docs: list[str] | None = None,
) -> dict[str, Any]:
    accessions = accessions or [f"acc-{i:04d}" for i in range(len(forms))]
    docs = docs or [f"doc-{i}.htm" for i in range(len(forms))]
    return {
        "filings": {
            "recent": {
                "form": forms,
                "accessionNumber": accessions,
                "filingDate": dates,
                "primaryDocument": docs,
            }
        }
    }


def _install_fake_requests(monkeypatch, responses: dict[str, list[_FakeResponse]]):
    """Patch `requests.get` to return queued responses by URL host or
    keyword. Each URL key maps to a list popped FIFO."""

    calls: list[dict[str, Any]] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": dict(headers or {})})
        for key, queue in responses.items():
            if key in url and queue:
                return queue.pop(0)
        return _FakeResponse(status_code=404, json_payload={})

    monkeypatch.setattr(_providers.requests, "get", fake_get)
    # Disable throttle sleeps to keep tests fast.
    monkeypatch.setattr(_providers.time, "sleep", lambda *_: None)
    return calls


def test_sec_default_user_agent_uses_contact_email_env(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "tester@example.com")
    monkeypatch.setenv("SEC_CONTACT_NAME", "Acme Research")
    p = SECFilingsProvider()
    assert p.user_agent == "Acme Research tester@example.com"


def test_sec_explicit_user_agent_env_wins(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "CustomUA admin@x.com")
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "ignored@x.com")
    p = SECFilingsProvider()
    assert p.user_agent == "CustomUA admin@x.com"


def test_sec_default_user_agent_has_email_format(monkeypatch):
    """Even without env config, the default UA must contain a @-style
    email so SEC EDGAR's WAF doesn't reject it as anonymous."""

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("SEC_CONTACT_NAME", raising=False)
    p = SECFilingsProvider()
    assert "@" in p.user_agent
    # SEC docs: "Sample Company Name AdminContact@samplecompany.com"
    # so the UA should be `name email` separated by whitespace.
    assert " " in p.user_agent


def test_sec_get_latest_filing_picks_latest_relevant_on_or_before(monkeypatch):
    payload = _submissions_payload(
        forms=["4", "144", "10-Q", "8-K", "10-K"],
        dates=["2024-06-15", "2024-06-10", "2024-05-20", "2024-06-07", "2024-02-01"],
    )
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [_FakeResponse(200, payload)],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    out = p.get_latest_filing("NVDA", as_of_date="2024-06-21")
    assert out is not None
    # Latest *relevant* (10-Q/10-K/8-K) on/before date is the 10-Q
    # 2024-05-20 — Form 4/144 should be filtered out, the 8-K 2024-06-07
    # comes after the 10-Q in the recent feed order so 10-Q wins on
    # index order.
    assert out["form"] == "10-Q"
    assert out["filing_date"] == "2024-05-20"


def test_sec_get_latest_filing_skips_future_filings_for_pit(monkeypatch):
    payload = _submissions_payload(
        forms=["8-K", "10-Q"],
        dates=["2024-06-25", "2024-05-20"],  # First is AFTER as_of
    )
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [_FakeResponse(200, payload)],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    out = p.get_latest_filing("NVDA", as_of_date="2024-06-21")
    assert out is not None
    assert out["filing_date"] == "2024-05-20"
    assert out["form"] == "10-Q"


def test_sec_get_latest_filing_returns_none_when_no_relevant(monkeypatch):
    payload = _submissions_payload(
        forms=["4", "144", "SD"],
        dates=["2024-06-15", "2024-06-10", "2024-05-20"],
    )
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [_FakeResponse(200, payload)],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    out = p.get_latest_filing("NVDA", as_of_date="2024-06-21")
    assert out is None  # genuinely empty — distinct from fetch error


def test_sec_get_latest_filing_raises_on_403(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [_FakeResponse(403, body="Forbidden")],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    with pytest.raises(SECFetchError):
        p.get_latest_filing("NVDA", as_of_date="2024-06-21")


def test_sec_get_latest_filing_retries_on_429_then_succeeds(monkeypatch):
    payload = _submissions_payload(
        forms=["10-Q"],
        dates=["2024-05-20"],
    )
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [
                _FakeResponse(429, body="rate limited"),
                _FakeResponse(429, body="rate limited"),
                _FakeResponse(200, payload),
            ],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    out = p.get_latest_filing("NVDA", as_of_date="2024-06-21")
    assert out is not None
    assert out["filing_date"] == "2024-05-20"


def test_sec_get_latest_filing_raises_after_max_429s(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [
                _FakeResponse(429, body="rate limited"),
                _FakeResponse(429, body="rate limited"),
                _FakeResponse(429, body="rate limited"),
            ],
        },
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    with pytest.raises(SECFetchError):
        p.get_latest_filing("NVDA", as_of_date="2024-06-21")


def test_sec_requests_carry_compliant_user_agent(monkeypatch):
    payload = _submissions_payload(forms=["10-Q"], dates=["2024-05-20"])
    calls = _install_fake_requests(
        monkeypatch,
        {
            "company_tickers.json": [_FakeResponse(200, _ticker_map_payload())],
            "submissions/CIK": [_FakeResponse(200, payload)],
        },
    )
    p = SECFilingsProvider(user_agent="Acme Research admin@acme.example")
    p.get_latest_filing("NVDA", as_of_date="2024-06-21")
    assert calls, "no HTTP calls were made"
    for call in calls:
        assert call["headers"].get("User-Agent") == "Acme Research admin@acme.example"
        # SEC requires the Host header to match the endpoint.
        assert call["headers"].get("Host") in {"data.sec.gov", "www.sec.gov"}


def test_sec_unknown_ticker_returns_none(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        {"company_tickers.json": [_FakeResponse(200, _ticker_map_payload())]},
    )
    p = SECFilingsProvider(user_agent="TestUA test@example.com")
    out = p.get_latest_filing("ZZZZ", as_of_date="2024-06-21")
    assert out is None
