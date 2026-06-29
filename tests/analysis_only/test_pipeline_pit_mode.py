from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP


NY = ZoneInfo("America/New_York")


def _make_mvp_at(local_dt: datetime) -> AnalysisOnlyMVP:
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    mvp._market_now = lambda: local_dt.replace(tzinfo=NY)  # type: ignore[method-assign]
    return mvp


def test_sunday_night_after_utc_rollover_still_uses_friday_live_mode():
    # 2026-06-08 00:32 ET is Sunday night Pacific but Monday in UTC.
    # The latest regular US equity session is still Friday 2026-06-05.
    mvp = _make_mvp_at(datetime(2026, 6, 8, 0, 32))

    assert mvp._current_market_reference_date().isoformat() == "2026-06-05"
    assert mvp._resolve_pit_mode("2026-06-05") == "live"
    assert mvp._resolve_pit_mode("2026-06-06") == "live"
    assert mvp._resolve_pit_mode("2026-06-07") == "live"
    assert mvp._resolve_pit_mode("2026-06-04") == "historical"


def test_monday_after_market_open_uses_monday_as_live_reference():
    mvp = _make_mvp_at(datetime(2026, 6, 8, 10, 0))

    assert mvp._current_market_reference_date().isoformat() == "2026-06-08"
    assert mvp._resolve_pit_mode("2026-06-08") == "live"
    assert mvp._resolve_pit_mode("2026-06-07") == "historical"


def test_weekday_before_open_uses_previous_trading_session_reference():
    mvp = _make_mvp_at(datetime(2026, 6, 9, 8, 0))

    assert mvp._current_market_reference_date().isoformat() == "2026-06-08"
    assert mvp._resolve_pit_mode("2026-06-08") == "live"
    assert mvp._resolve_pit_mode("2026-06-07") == "historical"


# Sections whose loaders pull realtime third-party data and therefore stamp
# their pit_status via `_live_or_leak`. These are the exact keys the production
# loaders pass to `_set_pit(section, self._live_or_leak(as_of_date))` in
# pipeline.py (`_load_fundamentals`, `_load_industry_context`,
# `_load_industry_news`, `_load_earnings_calendar`, `_load_analyst_consensus`,
# `_load_competitor_analysis`). Exercising the labeling helpers directly keeps
# the test deterministic and offline (no data fetches needed).
_LIVE_OR_LEAK_SECTIONS = (
    "fundamentals",
    "industry_context.sector_labels",
    "industry_news_context",
    "earnings_calendar.forward_eps_estimates",
    "analyst_consensus",
    "competitor_analysis.peer_fundamentals",
)


def test_live_or_leak_sections_flag_non_pit_for_historical_as_of():
    # Monday 2026-06-08 10:00 ET -> live reference date is 2026-06-08, so an
    # earlier session (Friday 2026-06-05) is a historical backfill where a
    # realtime snapshot would leak post-as_of data.
    mvp = _make_mvp_at(datetime(2026, 6, 8, 10, 0))
    historical_as_of = "2026-06-05"

    assert mvp._resolve_pit_mode(historical_as_of) == "historical"
    assert mvp._live_or_leak(historical_as_of) == "non_pit_live_snapshot"

    # Mirror exactly what the production loaders do: stamp each section with the
    # `_live_or_leak` verdict for the requested as_of date.
    for section in _LIVE_OR_LEAK_SECTIONS:
        mvp._set_pit(section, mvp._live_or_leak(historical_as_of))

    for section in _LIVE_OR_LEAK_SECTIONS:
        assert mvp._pit_status[section] == "non_pit_live_snapshot"

    # Every leaked section must surface in data_quality.pit_warnings.
    warnings = mvp._pit_warnings()
    for section in _LIVE_OR_LEAK_SECTIONS:
        assert section in warnings


def test_live_or_leak_sections_are_live_for_current_as_of():
    mvp = _make_mvp_at(datetime(2026, 6, 8, 10, 0))
    live_as_of = "2026-06-08"

    assert mvp._resolve_pit_mode(live_as_of) == "live"
    assert mvp._live_or_leak(live_as_of) == "live"

    for section in _LIVE_OR_LEAK_SECTIONS:
        mvp._set_pit(section, mvp._live_or_leak(live_as_of))

    for section in _LIVE_OR_LEAK_SECTIONS:
        assert mvp._pit_status[section] == "live"

    # Live runs produce no PIT-leak warnings for these sections.
    assert mvp._pit_warnings() == []
