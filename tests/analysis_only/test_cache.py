from __future__ import annotations

from tradingagents.analysis_only.cache import (
    DiskCache,
    load_report_if_cache_hit,
)
from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP, AnalysisReport


def test_disk_cache_json_round_trip(tmp_path):
    cache = DiskCache(tmp_path)

    cache.set_json("example", "key", {"value": 3})

    assert cache.get_json("example", "key") == {"value": 3}


def test_load_report_if_cache_hit_marks_source(tmp_path):
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    key = mvp.report_cache_key("NVDA", "2026-05-26")
    report = AnalysisReport(
        symbol="NVDA",
        horizon="swing_1_4_weeks",
        as_of_date="2026-05-26",
        thesis="test",
        direction="neutral",
        confidence=0.5,
        bull_case=[],
        bear_case=[],
        key_features={},
        risk_flags=[],
        invalidation_conditions=[],
        data_quality={"analysis_cache": {"key": key, "source": "fresh"}},
        generated_at_utc="2026-05-26T00:00:00Z",
    )
    mvp.save_report(report, tmp_path)

    loaded = load_report_if_cache_hit(
        AnalysisReport,
        tmp_path,
        "NVDA",
        "2026-05-26",
        key,
    )

    assert loaded is not None
    assert loaded.data_quality["analysis_cache"]["source"] == "cache"


def test_load_report_if_cache_hit_rejects_mismatched_key(tmp_path):
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    key = mvp.report_cache_key("NVDA", "2026-05-26")
    report = AnalysisReport(
        symbol="NVDA",
        horizon="swing_1_4_weeks",
        as_of_date="2026-05-26",
        thesis="test",
        direction="neutral",
        confidence=0.5,
        bull_case=[],
        bear_case=[],
        key_features={},
        risk_flags=[],
        invalidation_conditions=[],
        data_quality={"analysis_cache": {"key": key, "source": "fresh"}},
        generated_at_utc="2026-05-26T00:00:00Z",
    )
    mvp.save_report(report, tmp_path)

    assert load_report_if_cache_hit(
        AnalysisReport,
        tmp_path,
        "NVDA",
        "2026-05-26",
        "different",
    ) is None
