"""Tests for `AnalysisOnlyMVP._download_yfinance_daily_cached` routing.

The method is the single chokepoint where SPY / sector-ETF lookups used to
hit `yf.download`, generating "HTTP 401 Invalid Crumb" log noise under
concurrent regen. It now routes to Polygon's process-shared cache for
supported symbols and falls back to yfinance only for indices (`^VIX`,
`^TNX`, `^IRX`) and when no `POLYGON_API_KEY` is set.

These tests cover routing only — the cache mechanics are exercised in
`test_providers.py::test_polygon_daily_aggs_cache_*`.
"""

from __future__ import annotations

from datetime import datetime
from unittest import mock

import pandas as pd
import pytest

from tradingagents.analysis_only import pipeline as _pipeline
from tradingagents.analysis_only import providers as _providers
from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP
from tradingagents.analysis_only.providers import reset_polygon_daily_aggs_cache


@pytest.fixture(autouse=True)
def _clear_polygon_cache():
    reset_polygon_daily_aggs_cache()
    yield
    reset_polygon_daily_aggs_cache()


def _spy_frame() -> pd.DataFrame:
    idx = pd.to_datetime([
        "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21",
    ])
    return pd.DataFrame(
        {
            "Open": [540, 541, 542, 543, 544],
            "High": [545, 546, 547, 548, 549],
            "Low": [538, 539, 540, 541, 542],
            "Close": [543.0, 544.0, 545.0, 546.0, 547.0],
            "Volume": [10, 11, 12, 13, 14],
        },
        index=pd.Index(idx, name="Date"),
    )


def _make_mvp(tmp_path) -> AnalysisOnlyMVP:
    # enable_data_cache=False keeps the per-instance disk cache out of the
    # picture so the routing decision is the only thing under test.
    return AnalysisOnlyMVP(enable_data_cache=False)


def test_spy_routes_to_polygon_when_api_key_set(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    yf_calls: list[tuple] = []
    polygon_calls: list[tuple] = []

    def _yf_fake(*args, **kwargs):
        yf_calls.append((args, kwargs))
        return pd.DataFrame()

    def _polygon_fake(symbol, start, end, api_key, logger=None):
        polygon_calls.append((symbol, start, end, api_key))
        return _spy_frame()

    monkeypatch.setattr(_pipeline.yf, "download", _yf_fake)
    monkeypatch.setattr(
        _pipeline, "fetch_polygon_daily_aggs_cached", _polygon_fake,
    )

    mvp = _make_mvp(tmp_path)
    out = mvp._download_yfinance_daily_cached(
        symbol="SPY",
        start_dt=datetime(2024, 6, 17),
        end_dt=datetime(2024, 6, 22),  # callers pass as_of + 1 day
        as_of_date="2024-06-21",
        namespace="market_context",
    )

    assert out is not None and not out.empty
    assert polygon_calls, "Polygon should be hit for SPY when key is set"
    # yfinance must NOT be touched on the SPY path.
    assert yf_calls == []
    # PIT: the Polygon `end` is the day BEFORE the yfinance-style `end_dt`,
    # so we never request a bar dated after `as_of_date`.
    assert polygon_calls[0][2] == "2024-06-21"


def test_vix_falls_back_to_yfinance_even_with_polygon_key(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    yf_calls: list[tuple] = []
    polygon_calls: list[tuple] = []

    def _yf_fake(*args, **kwargs):
        yf_calls.append((args, kwargs))
        return _spy_frame()

    def _polygon_fake(*args, **kwargs):
        polygon_calls.append((args, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(_pipeline.yf, "download", _yf_fake)
    monkeypatch.setattr(
        _pipeline, "fetch_polygon_daily_aggs_cached", _polygon_fake,
    )

    mvp = _make_mvp(tmp_path)
    out = mvp._download_yfinance_daily_cached(
        symbol="^VIX",
        start_dt=datetime(2024, 6, 17),
        end_dt=datetime(2024, 6, 22),
        as_of_date="2024-06-21",
        namespace="market_context",
    )

    assert out is not None and not out.empty
    # Polygon path NOT taken (would fail with 403 in real life).
    assert polygon_calls == []
    # yfinance IS used as the fallback.
    assert yf_calls, "^VIX must still use yfinance"


def test_no_api_key_falls_back_to_yfinance(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    yf_calls: list[tuple] = []
    polygon_calls: list[tuple] = []

    def _yf_fake(*args, **kwargs):
        yf_calls.append((args, kwargs))
        return _spy_frame()

    def _polygon_fake(*args, **kwargs):
        polygon_calls.append((args, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(_pipeline.yf, "download", _yf_fake)
    monkeypatch.setattr(
        _pipeline, "fetch_polygon_daily_aggs_cached", _polygon_fake,
    )

    mvp = _make_mvp(tmp_path)
    out = mvp._download_yfinance_daily_cached(
        symbol="SPY",
        start_dt=datetime(2024, 6, 17),
        end_dt=datetime(2024, 6, 22),
        as_of_date="2024-06-21",
        namespace="market_context",
    )

    assert out is not None and not out.empty
    # No POLYGON_API_KEY → graceful degradation to yfinance.
    assert polygon_calls == []
    assert yf_calls, "SPY without POLYGON_API_KEY must fall back to yfinance"


def test_sector_etfs_all_route_to_polygon(monkeypatch, tmp_path):
    """All eleven SPDR sector ETFs are Polygon-supported (verified manually
    against /v2/aggs)."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    polygon_symbols: list[str] = []

    def _polygon_fake(symbol, start, end, api_key, logger=None):
        polygon_symbols.append(symbol)
        return _spy_frame()

    def _yf_fake(*args, **kwargs):
        pytest.fail(
            "yfinance should not be touched for sector ETFs when POLYGON key set"
        )

    monkeypatch.setattr(_pipeline.yf, "download", _yf_fake)
    monkeypatch.setattr(
        _pipeline, "fetch_polygon_daily_aggs_cached", _polygon_fake,
    )

    mvp = _make_mvp(tmp_path)
    for etf in ("SPY", "XLK", "XLV", "XLF", "XLE", "XLY",
                "XLP", "XLI", "XLC", "XLU", "XLRE", "XLB"):
        mvp._download_yfinance_daily_cached(
            symbol=etf,
            start_dt=datetime(2024, 6, 17),
            end_dt=datetime(2024, 6, 22),
            as_of_date="2024-06-21",
            namespace="market_context",
        )

    assert polygon_symbols == [
        "SPY", "XLK", "XLV", "XLF", "XLE", "XLY",
        "XLP", "XLI", "XLC", "XLU", "XLRE", "XLB",
    ]


def test_empty_polygon_response_returns_empty_does_not_fall_back(monkeypatch, tmp_path):
    """If Polygon legitimately has no data (e.g. weekend / pre-IPO),
    return the empty result rather than triggering yfinance noise."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    yf_calls: list[tuple] = []

    def _yf_fake(*args, **kwargs):
        yf_calls.append((args, kwargs))
        return _spy_frame()

    def _polygon_empty(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(_pipeline.yf, "download", _yf_fake)
    monkeypatch.setattr(
        _pipeline, "fetch_polygon_daily_aggs_cached", _polygon_empty,
    )

    mvp = _make_mvp(tmp_path)
    out = mvp._download_yfinance_daily_cached(
        symbol="SPY",
        start_dt=datetime(2024, 6, 17),
        end_dt=datetime(2024, 6, 22),
        as_of_date="2024-06-21",
        namespace="market_context",
    )

    assert out is not None and out.empty
    assert yf_calls == []  # no yfinance fallback for transient empties
