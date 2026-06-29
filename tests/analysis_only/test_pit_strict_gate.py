from __future__ import annotations

import pytest

from tradingagents.analysis_only.pipeline import (
    AnalysisReport,
    PitLeakError,
    _pit_strict_from_env,
    collect_pit_warnings,
    enforce_pit_strict,
    pit_as_of_mode,
)


def _report_dict(*, as_of_mode: str, pit_status: dict[str, str]) -> dict:
    """Minimal persisted-report shape exercised by the PIT helpers."""
    warnings = sorted(
        section
        for section, status in pit_status.items()
        if status == "non_pit_live_snapshot"
    )
    return {
        "symbol": "AAPL",
        "as_of_date": "2025-01-15",
        "key_features": {"pit_status": pit_status},
        "data_quality": {"as_of_mode": as_of_mode, "pit_warnings": warnings},
    }


def _analysis_report(*, as_of_mode: str, pit_status: dict[str, str]) -> AnalysisReport:
    d = _report_dict(as_of_mode=as_of_mode, pit_status=pit_status)
    return AnalysisReport(
        symbol="AAPL",
        horizon="swing_1_4_weeks",
        as_of_date=d["as_of_date"],
        thesis="",
        direction="neutral",
        confidence=0.5,
        bull_case=[],
        bear_case=[],
        key_features=d["key_features"],
        risk_flags=[],
        invalidation_conditions=[],
        data_quality=d["data_quality"],
        generated_at_utc="2025-01-15T00:00:00+00:00",
    )


LEAKY = {
    "price_data": "pit",
    "options_flow": "non_pit_live_snapshot",
    "market_context": "pit",
    "intraday_context": "non_pit_live_snapshot",
}
CLEAN = {"price_data": "pit", "market_context": "pit"}


def test_collect_pit_warnings_from_dict():
    report = _report_dict(as_of_mode="historical", pit_status=LEAKY)
    assert collect_pit_warnings(report) == ["intraday_context", "options_flow"]


def test_collect_pit_warnings_from_analysis_report():
    report = _analysis_report(as_of_mode="historical", pit_status=LEAKY)
    assert collect_pit_warnings(report) == ["intraday_context", "options_flow"]


def test_collect_pit_warnings_clean_is_empty():
    report = _report_dict(as_of_mode="live", pit_status=CLEAN)
    assert collect_pit_warnings(report) == []


def test_collect_pit_warnings_falls_back_to_data_quality_list():
    # No pit_status map; only the precomputed warnings list survives.
    report = {
        "as_of_date": "2025-01-15",
        "key_features": {},
        "data_quality": {
            "as_of_mode": "historical",
            "pit_warnings": ["options_flow"],
        },
    }
    assert collect_pit_warnings(report) == ["options_flow"]


def test_pit_as_of_mode():
    assert pit_as_of_mode(_report_dict(as_of_mode="historical", pit_status=CLEAN)) == "historical"
    assert pit_as_of_mode(_report_dict(as_of_mode="live", pit_status=CLEAN)) == "live"


def test_enforce_strict_off_is_noop_even_with_historical_leak():
    # Default OFF: current behavior preserved, no raise, warnings returned.
    report = _report_dict(as_of_mode="historical", pit_status=LEAKY)
    assert enforce_pit_strict(report) == ["intraday_context", "options_flow"]
    assert enforce_pit_strict(report, strict=False) == [
        "intraday_context",
        "options_flow",
    ]


def test_enforce_strict_on_raises_for_historical_leak():
    report = _report_dict(as_of_mode="historical", pit_status=LEAKY)
    with pytest.raises(PitLeakError) as exc:
        enforce_pit_strict(report, strict=True)
    assert exc.value.as_of_date == "2025-01-15"
    assert exc.value.sections == ["intraday_context", "options_flow"]


def test_enforce_strict_on_raises_for_analysis_report():
    report = _analysis_report(as_of_mode="historical", pit_status=LEAKY)
    with pytest.raises(PitLeakError):
        enforce_pit_strict(report, strict=True)


def test_enforce_strict_on_does_not_raise_for_live_run():
    # Live runs are point-in-time by definition; even a stray snapshot stamp
    # must not block a live report.
    report = _report_dict(as_of_mode="live", pit_status=LEAKY)
    assert enforce_pit_strict(report, strict=True) == [
        "intraday_context",
        "options_flow",
    ]


def test_enforce_strict_on_does_not_raise_when_clean():
    report = _report_dict(as_of_mode="historical", pit_status=CLEAN)
    assert enforce_pit_strict(report, strict=True) == []


def test_pit_strict_from_env(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_PIT_STRICT", raising=False)
    assert _pit_strict_from_env() is False
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TRADINGAGENTS_PIT_STRICT", truthy)
        assert _pit_strict_from_env() is True
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRADINGAGENTS_PIT_STRICT", falsy)
        assert _pit_strict_from_env() is False
