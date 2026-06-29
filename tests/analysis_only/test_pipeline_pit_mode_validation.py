from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP


NY = ZoneInfo("America/New_York")


def _make_mvp_at(local_dt: datetime) -> AnalysisOnlyMVP:
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    mvp._market_now = lambda: local_dt.replace(tzinfo=NY)  # type: ignore[method-assign]
    return mvp


# Reference: Monday 2026-06-08 10:00 ET -> live reference date is 2026-06-08.
def _mvp() -> AnalysisOnlyMVP:
    return _make_mvp_at(datetime(2026, 6, 8, 10, 0))


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-date",
        "2026/06/08",
        "06-08-2026",
        "2026-13-01",
        "2026-06-32",
        "20260608",
        "today",
    ],
)
def test_non_empty_malformed_as_of_date_raises(bad: str):
    # (a) A malformed, non-empty as_of_date must fail loud rather than
    # silently selecting live mode (which would leak realtime data into a
    # supposedly point-in-time historical run).
    mvp = _mvp()
    with pytest.raises(ValueError, match="as_of_date must be YYYY-MM-DD"):
        mvp._resolve_pit_mode(bad)


def test_malformed_value_is_included_in_error_message():
    mvp = _mvp()
    with pytest.raises(ValueError, match=r"got 'banana'"):
        mvp._resolve_pit_mode("banana")


def test_valid_historical_date_returns_historical():
    # (b) A well-formed date strictly before the live reference is historical.
    mvp = _mvp()
    assert mvp._current_market_reference_date().isoformat() == "2026-06-08"
    assert mvp._resolve_pit_mode("2026-06-05") == "historical"
    assert mvp._resolve_pit_mode("2026-01-02") == "historical"


def test_current_and_future_date_returns_live():
    # (c) The reference date itself and any future date are live.
    mvp = _mvp()
    assert mvp._resolve_pit_mode("2026-06-08") == "live"
    assert mvp._resolve_pit_mode("2026-06-09") == "live"
    assert mvp._resolve_pit_mode("2030-01-01") == "live"


def test_falsy_as_of_date_still_resolves_live_sentinel():
    # (d) The pre-existing live sentinel -- a falsy as_of_date (empty string
    # or None) -- must keep resolving to live by design, not raise.
    mvp = _mvp()
    assert mvp._resolve_pit_mode("") == "live"
    assert mvp._resolve_pit_mode(None) == "live"  # type: ignore[arg-type]
    # And the dependent helper that builds on it stays live too.
    assert mvp._live_or_leak("") == "live"
