"""Tests for the historical options reconstruction module.

Network-touching paths (`_load_irx_series`, the Polygon HTTP calls) are
not exercised directly here — they're tested by the higher-level pipeline
smoke runs. These tests focus on the pure logic: process-level rate cache
behavior, rate lookup walk-back, and date parsing.
"""

from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.analysis_only import options_historical as ohist
from tradingagents.analysis_only.options_historical import (
    RiskFreeRateProvider,
    reset_rate_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_rate_cache()
    yield
    reset_rate_cache()


def test_rate_series_cached_across_providers(monkeypatch):
    """Two providers with the same (start, end) share a single fetch."""
    call_count = {"n": 0}

    def fake_loader(start, end, logger):
        call_count["n"] += 1
        return {"2024-01-02": 0.054, "2024-01-03": 0.0541}

    monkeypatch.setattr(ohist, "_load_irx_series", fake_loader)

    p1 = RiskFreeRateProvider(start="2024-01-01")
    p1._ensure_loaded()
    p2 = RiskFreeRateProvider(start="2024-01-01")
    p2._ensure_loaded()
    p3 = RiskFreeRateProvider(start="2024-01-01")
    p3._ensure_loaded()

    assert call_count["n"] == 1
    # All three see the same data.
    assert p1.rate_for("2024-01-02") == 0.054
    assert p2.rate_for("2024-01-02") == 0.054
    assert p3.rate_for("2024-01-02") == 0.054


def test_different_date_ranges_get_separate_cache_entries(monkeypatch):
    """Different (start, end) tuples each trigger their own fetch."""
    calls: list[tuple] = []

    def fake_loader(start, end, logger):
        calls.append((start, end))
        return {"2024-01-02": 0.054}

    monkeypatch.setattr(ohist, "_load_irx_series", fake_loader)

    RiskFreeRateProvider(start="2020-01-01")._ensure_loaded()
    RiskFreeRateProvider(start="2024-01-01")._ensure_loaded()
    RiskFreeRateProvider(start="2024-01-01", end="2024-12-31")._ensure_loaded()
    RiskFreeRateProvider(start="2020-01-01")._ensure_loaded()  # cache hit

    assert len(calls) == 3  # not 4 — last call hit the cache


def test_rate_for_exact_date(monkeypatch):
    monkeypatch.setattr(
        ohist, "_load_irx_series",
        lambda *a, **kw: {"2024-06-21": 0.0522},
    )
    p = RiskFreeRateProvider()
    assert p.rate_for("2024-06-21") == 0.0522


def test_rate_for_walks_back_for_weekend(monkeypatch):
    # Friday → Saturday: Saturday isn't in the series, walk back one day.
    monkeypatch.setattr(
        ohist, "_load_irx_series",
        lambda *a, **kw: {"2024-06-21": 0.0522},  # Friday only
    )
    p = RiskFreeRateProvider()
    assert p.rate_for("2024-06-22") == 0.0522  # Saturday walks back to Friday
    assert p.rate_for("2024-06-23") == 0.0522  # Sunday walks back to Friday


def test_rate_for_falls_back_to_closest_when_outside_window(monkeypatch):
    # The full series has just one date; any target outside 10-trading-days
    # walk-back should fall through to "closest available."
    monkeypatch.setattr(
        ohist, "_load_irx_series",
        lambda *a, **kw: {"2024-06-21": 0.0522},
    )
    p = RiskFreeRateProvider()
    assert p.rate_for("2025-12-31") == 0.0522  # far future → closest


def test_rate_for_empty_series_returns_fallback(monkeypatch):
    monkeypatch.setattr(ohist, "_load_irx_series", lambda *a, **kw: {})
    p = RiskFreeRateProvider(fallback_rate=0.041)
    assert p.rate_for("2024-06-21") == 0.041


def test_rate_for_invalid_date_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        ohist, "_load_irx_series",
        lambda *a, **kw: {"2024-06-21": 0.0522},
    )
    p = RiskFreeRateProvider(fallback_rate=0.041)
    assert p.rate_for("not-a-date") == 0.041
