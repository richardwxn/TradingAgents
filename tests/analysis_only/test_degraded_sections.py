from __future__ import annotations

from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP


def _mvp(**kwargs) -> AnalysisOnlyMVP:
    # enable_data_cache=False keeps construction fully offline/deterministic.
    return AnalysisOnlyMVP(enable_data_cache=False, **kwargs)


def test_fresh_instance_has_no_degraded_sections():
    mvp = _mvp()
    assert mvp.degraded_sections() == {}


def test_set_pit_unavailable_auto_marks_degraded():
    mvp = _mvp()
    mvp._set_pit("intraday_context", "unavailable")
    assert mvp.degraded_sections() == {"intraday_context": "unavailable"}


def test_set_pit_healthy_or_disabled_status_is_not_degraded():
    mvp = _mvp()
    mvp._set_pit("price_data", "pit")
    mvp._set_pit("options_flow", "live")
    mvp._set_pit("news", "disabled")
    mvp._set_pit("fundamentals", "disabled_historical")
    # Intentional config / healthy statuses must never count as degraded.
    assert mvp.degraded_sections() == {}


def test_mark_degraded_records_reason_and_keeps_first_nonempty():
    mvp = _mvp()
    mvp._mark_degraded("news_sentiment", "timeout")
    mvp._mark_degraded("news_sentiment")  # empty reason must not clobber.
    assert mvp.degraded_sections() == {"news_sentiment": "timeout"}


def test_degraded_sections_returns_a_copy():
    mvp = _mvp()
    mvp._mark_degraded("filings_context", "boom")
    snapshot = mvp.degraded_sections()
    snapshot["filings_context"] = "mutated"
    snapshot["extra"] = "x"
    # Mutating the returned dict must not affect internal registry state.
    assert mvp.degraded_sections() == {"filings_context": "boom"}


def test_guard_disabled_by_default_never_breaches():
    mvp = _mvp()
    for i in range(5):
        mvp._mark_degraded(f"section_{i}", "err")
    status = mvp.degraded_guard_status()
    assert status["enabled"] is False
    assert status["threshold"] is None
    assert status["count"] == 5
    assert status["breached"] is False
    assert status["sections"] == [
        "section_0",
        "section_1",
        "section_2",
        "section_3",
        "section_4",
    ]


def test_guard_via_method_argument_breaches_when_count_exceeds_threshold():
    mvp = _mvp()
    mvp._mark_degraded("a", "e")
    mvp._mark_degraded("b", "e")
    mvp._mark_degraded("c", "e")
    # threshold=2, count=3 -> breached.
    breached = mvp.degraded_guard_status(max_degraded=2)
    assert breached["enabled"] is True
    assert breached["threshold"] == 2
    assert breached["count"] == 3
    assert breached["breached"] is True
    # threshold=3, count=3 -> at limit, not breached.
    at_limit = mvp.degraded_guard_status(max_degraded=3)
    assert at_limit["breached"] is False


def test_guard_via_constructor_threshold():
    mvp = _mvp(max_degraded_sections=1)
    mvp._set_pit("x", "unavailable")
    assert mvp.degraded_guard_status()["breached"] is False
    mvp._set_pit("y", "unavailable")
    status = mvp.degraded_guard_status()
    assert status["enabled"] is True
    assert status["threshold"] == 1
    assert status["count"] == 2
    assert status["breached"] is True


def test_method_argument_overrides_constructor_threshold():
    mvp = _mvp(max_degraded_sections=10)
    mvp._mark_degraded("a", "e")
    mvp._mark_degraded("b", "e")
    # Constructor threshold 10 would not breach, but explicit arg 1 does.
    assert mvp.degraded_guard_status(max_degraded=1)["breached"] is True
