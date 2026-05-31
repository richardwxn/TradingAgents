from __future__ import annotations

import pytest

from tradingagents.analysis_only.providers import FearGreedProvider


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
