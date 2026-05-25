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
