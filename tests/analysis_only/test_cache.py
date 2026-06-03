from __future__ import annotations

from tradingagents.analysis_only.cache import (
    DiskCache,
    load_report_if_cache_hit,
)
from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP, AnalysisReport
from tradingagents.analysis_only.providers import SECFetchError


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


def test_load_filings_context_does_not_cache_fetch_errors(tmp_path, monkeypatch):
    """`_load_filings_context` must NOT cache a `SECFetchError` outcome.

    Historical-date cache entries have ttl=None (never expire), so a
    cached error would block future retries forever. Regression guard
    for the bug that left every report's `filings_context.status` as
    `unavailable` because a poisoned 403 response from SEC EDGAR was
    cached on first run.
    """

    monkeypatch.setenv("TRADINGAGENTS_ANALYSIS_CACHE_DIR", str(tmp_path))
    mvp = AnalysisOnlyMVP(enable_data_cache=True, cache_dir=str(tmp_path))

    call_count = {"n": 0}

    def fake_get_latest_filing(symbol, as_of_date=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SECFetchError("HTTP 403 simulated")
        return {
            "symbol": symbol.upper(),
            "cik": "1045810",
            "form": "10-Q",
            "accession": "acc-1234",
            "filing_date": "2024-05-20",
            "primary_document": "doc.htm",
        }

    monkeypatch.setattr(
        mvp.sec_filings_provider,
        "get_latest_filing",
        fake_get_latest_filing,
    )

    out1 = mvp._load_filings_context("NVDA", as_of_date="2024-06-21")
    assert out1["status"] == "error"

    # Second call must retry (i.e. not hit cache), then succeed.
    out2 = mvp._load_filings_context("NVDA", as_of_date="2024-06-21")
    assert out2["status"] == "ok"
    assert out2["latest_form"] == "10-Q"
    assert call_count["n"] == 2


def test_load_filings_context_caches_genuine_empty_result(tmp_path, monkeypatch):
    """A successful fetch that contains no relevant filings on/before
    the as_of_date IS a real PIT result and should be cached as
    ``unavailable`` so we don't keep re-hitting SEC for the same query.
    """

    monkeypatch.setenv("TRADINGAGENTS_ANALYSIS_CACHE_DIR", str(tmp_path))
    mvp = AnalysisOnlyMVP(enable_data_cache=True, cache_dir=str(tmp_path))

    call_count = {"n": 0}

    def fake_get_latest_filing(symbol, as_of_date=None):
        call_count["n"] += 1
        return None  # no matching filing in submissions feed

    monkeypatch.setattr(
        mvp.sec_filings_provider,
        "get_latest_filing",
        fake_get_latest_filing,
    )

    out1 = mvp._load_filings_context("NVDA", as_of_date="2024-06-21")
    out2 = mvp._load_filings_context("NVDA", as_of_date="2024-06-21")
    assert out1 == {"status": "unavailable"}
    assert out2 == {"status": "unavailable"}
    # Second call should be served from cache.
    assert call_count["n"] == 1
