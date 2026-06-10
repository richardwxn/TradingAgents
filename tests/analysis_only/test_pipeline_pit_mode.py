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
