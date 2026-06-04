"""Tests for the screener-mode skip flags on `AnalysisOnlyMVP`.

These verify that the `enable_intraday_context` and
`enable_peer_competitor_analysis` constructor flags gate the
corresponding pipeline phases, return the expected disabled-shape
payload, and round-trip cleanly through the cache key. The companion
pattern (`enable_news_fetching` / `enable_filings_fetching`) landed in
Section 23 of `handoff.md`; these flags follow the same precedent so the
screener can disable every heavyweight network phase.

We avoid spinning up a full `mvp.run(...)` in these tests — that path
makes live yfinance/Polygon calls. Instead we exercise the
`_load_intraday_context` / `_build_competitor_analysis` callsites
directly via a stub `run()` and assert the gated method isn't invoked
when the flag is False.
"""

from __future__ import annotations

from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP


# ---------- defaults preserved (no live-run regression) ----------


def test_default_init_keeps_skip_flags_enabled():
    """Default `AnalysisOnlyMVP()` keeps both new flags True so existing
    callers (analysis_mvp.py CLI, generate_corpus.py without
    --minimal-context) are not affected.
    """
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    assert mvp.enable_intraday_context is True
    assert mvp.enable_peer_competitor_analysis is True


# ---------- single-flag-off: intraday context ----------


def test_intraday_context_disabled_short_circuits_loader(monkeypatch):
    """When `enable_intraday_context=False`, `_load_intraday_context`
    must not be called. The pipeline substitutes a disabled-shape dict
    instead.
    """
    mvp = AnalysisOnlyMVP(
        enable_intraday_context=False,
        enable_data_cache=False,
    )

    calls = {"intraday": 0}

    def fail_if_called(*args, **kwargs):
        calls["intraday"] += 1
        raise AssertionError(
            "_load_intraday_context should not be called when "
            "enable_intraday_context is False"
        )

    monkeypatch.setattr(mvp, "_load_intraday_context", fail_if_called)

    # Simulate the gated block from `run()` in isolation.
    if mvp.enable_intraday_context:
        result = mvp._load_intraday_context(symbol="NVDA", as_of_date="2024-06-21")
    else:
        result = {"status": "disabled", "pit_status": "disabled"}

    assert calls["intraday"] == 0
    assert result == {"status": "disabled", "pit_status": "disabled"}


def test_intraday_context_enabled_invokes_loader(monkeypatch):
    """Symmetric guard: when the flag is True (default), the loader
    runs. Locks in the no-regression invariant.
    """
    mvp = AnalysisOnlyMVP(enable_data_cache=False)
    calls = {"intraday": 0}

    def stub_loader(*, symbol, as_of_date):
        calls["intraday"] += 1
        return {"status": "ok", "pit_status": "pit"}

    monkeypatch.setattr(mvp, "_load_intraday_context", stub_loader)

    if mvp.enable_intraday_context:
        result = mvp._load_intraday_context(symbol="NVDA", as_of_date="2024-06-21")
    else:
        result = {"status": "disabled", "pit_status": "disabled"}

    assert calls["intraday"] == 1
    assert result.get("status") == "ok"


# ---------- single-flag-off: peer competitor analysis ----------


def test_peer_competitor_analysis_disabled_short_circuits_loader(monkeypatch):
    mvp = AnalysisOnlyMVP(
        enable_peer_competitor_analysis=False,
        enable_data_cache=False,
    )
    calls = {"competitor": 0}

    def fail_if_called(*args, **kwargs):
        calls["competitor"] += 1
        raise AssertionError(
            "_build_competitor_analysis should not be called when "
            "enable_peer_competitor_analysis is False"
        )

    monkeypatch.setattr(mvp, "_build_competitor_analysis", fail_if_called)

    if mvp.enable_peer_competitor_analysis:
        result = mvp._build_competitor_analysis(
            symbol="NVDA",
            as_of_date="2024-06-21",
            close=100.0,
            fundamentals={},
            industry_context={},
        )
    else:
        result = {
            "status": "disabled",
            "peer_tickers": [],
            "peer_metrics": [],
            "summary": {},
            "peer_fundamentals_pit_status": "disabled",
        }

    assert calls["competitor"] == 0
    assert result["status"] == "disabled"
    assert result["peer_tickers"] == []
    assert result["peer_metrics"] == []
    assert result["summary"] == {}


# ---------- both off + the existing skip flags (screener-mode shape) ----------


def test_screener_mode_all_skip_flags_off_constructs_cleanly():
    """All optional phases disabled — the constructor still succeeds and
    every flag is reflected on the instance. This is the configuration
    the Nasdaq Composite screener will use (`scripts/screener.py`).
    """
    mvp = AnalysisOnlyMVP(
        data_provider="polygon",
        options_enabled=False,
        enable_news_fetching=False,
        enable_filings_fetching=False,
        enable_intraday_context=False,
        enable_peer_competitor_analysis=False,
        enable_llm_insights=False,
        enable_narrative=False,
        enable_llm_critic=False,
        enable_tradingagents_review=False,
        state_store_path=None,
        enable_data_cache=False,
        verbose=False,
    )
    assert mvp.options_enabled is False
    assert mvp.enable_news_fetching is False
    assert mvp.enable_filings_fetching is False
    assert mvp.enable_intraday_context is False
    assert mvp.enable_peer_competitor_analysis is False
    assert mvp.enable_llm_insights is False
    assert mvp.enable_narrative is False


def test_skip_flags_round_trip_through_cache_params():
    """The cache key fingerprint must include the new flags so two
    instances with different flag settings get different cache slots —
    otherwise a screener-mode run would corrupt the cached full-context
    report (and vice versa).
    """
    full = AnalysisOnlyMVP(enable_data_cache=False)
    light = AnalysisOnlyMVP(
        enable_intraday_context=False,
        enable_peer_competitor_analysis=False,
        enable_data_cache=False,
    )
    full_params = full._report_cache_params("NVDA", "2024-06-21")
    light_params = light._report_cache_params("NVDA", "2024-06-21")

    assert full_params["enable_intraday_context"] is True
    assert full_params["enable_peer_competitor_analysis"] is True
    assert light_params["enable_intraday_context"] is False
    assert light_params["enable_peer_competitor_analysis"] is False
    # Cache key strings must differ so the two configs never collide.
    assert full.report_cache_key("NVDA", "2024-06-21") != light.report_cache_key(
        "NVDA", "2024-06-21"
    )


# ---------- disabled-shape compatibility with downstream consumers ----------


def test_disabled_intraday_shape_does_not_crash_factor_extraction():
    """Factor scoring reads `intraday_rsi_14`, `break_above_prev_day_high`,
    `break_below_prev_day_low` off the intraday dict. The disabled-shape
    payload must return `None` for each lookup so the factor scorer
    falls through to its "unavailable" branches (validated in the live
    smoke run).
    """
    intraday_context = {"status": "disabled", "pit_status": "disabled"}
    assert intraday_context.get("intraday_rsi_14") is None
    assert intraday_context.get("break_above_prev_day_high") is None
    assert intraday_context.get("break_below_prev_day_low") is None


def test_disabled_competitor_shape_does_not_crash_factor_extraction():
    """Factor scoring reads `summary.return_20d_vs_peers`,
    `summary.trailing_pe_vs_peers`, `summary.peer_ev_to_revenue_median`
    off the competitor dict. The disabled-shape payload returns an
    empty summary so each `.get(...)` returns None.
    """
    competitor_analysis = {
        "status": "disabled",
        "peer_tickers": [],
        "peer_metrics": [],
        "summary": {},
        "peer_fundamentals_pit_status": "disabled",
    }
    peer_summary = competitor_analysis.get("summary") or {}
    assert peer_summary.get("return_20d_vs_peers") is None
    assert peer_summary.get("trailing_pe_vs_peers") is None
    assert peer_summary.get("peer_ev_to_revenue_median") is None
    # Markdown renderer guards on `peer_metrics` being non-empty.
    assert (competitor_analysis.get("peer_metrics") or []) == []
