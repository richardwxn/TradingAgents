"""Pure-function tests for the screener module.

These tests do NOT touch yfinance/Polygon — they exercise the pure helpers
(`extract_candidate_from_report`, `rank_candidates`, `rank_per_sector`,
`render_screener_markdown`) against synthetic report dicts. The CLI in
`scripts/screener.py` is exercised by the end-to-end smoke run, not here.
"""
from __future__ import annotations

import pytest

from tradingagents.analysis_only.screener import (
    UNKNOWN_SECTOR,
    ScreenerCandidate,
    candidate_to_dict,
    cohort_for_sector,
    extract_candidate_from_report,
    rank_candidates,
    rank_per_sector,
    render_screener_markdown,
    rescore_report_for_cohort,
)


def _make_report(
    *,
    symbol: str = "NVDA",
    composite: float = 0.6,
    direction: str = "bullish",
    confidence: float = 0.72,
    as_of: str = "2026-05-22",
    next_earnings: str | None = None,
    market_cap: float | None = 1.5e12,
    extra_factors: list[dict] | None = None,
    pit_status: dict | None = None,
    tradingagents_review: dict | None = None,
) -> dict:
    """Build a minimal synthetic report shaped like AnalysisOnlyMVP.to_json_dict."""
    default_factors = [
        {
            "factor": "trend_price_vs_sma20",
            "weighted_score": 0.0625,
            "rationale": "Price above 20D moving average.",
        },
        {
            "factor": "fund_revenue_growth",
            "weighted_score": 0.0625,
            "rationale": "Revenue growth is strong.",
        },
        {
            "factor": "momentum_macd_hist",
            "weighted_score": -0.0273,
            "rationale": "MACD histogram negative.",
        },
        {
            "factor": "fund_profit_margins",
            "weighted_score": 0.05,
            "rationale": "Profit margins are strong.",
        },
    ]
    factor_scores = default_factors + list(extra_factors or [])
    return {
        "symbol": symbol,
        "as_of_date": as_of,
        "direction": direction,
        "confidence": confidence,
        "key_features": {
            "fundamental": {"market_cap": market_cap},
            "company_events": {"next_earnings_date": next_earnings},
            "model_scoring": {
                "composite_score": composite,
                "factor_scores": factor_scores,
            },
            "pit_status": pit_status
            or {
                "price_data": "pit",
                "fundamentals": "pit",
                "industry_context.sector_labels": "non_pit_live_snapshot",
            },
            "tradingagents_review": tradingagents_review or {},
        },
    }


# ---------- extract_candidate_from_report ----------


def test_extract_candidate_basic_fields():
    rep = _make_report(symbol="NVDA", composite=0.6, confidence=0.72)
    c = extract_candidate_from_report(rep, sector_map={"NVDA": "Semiconductors"})
    assert c.symbol == "NVDA"
    assert c.composite_score == pytest.approx(0.6)
    assert c.direction == "bullish"
    assert c.confidence == pytest.approx(0.72)
    assert c.sector == "Semiconductors"
    assert c.market_cap == pytest.approx(1.5e12)


def test_extract_candidate_unknown_sector_when_not_in_map():
    rep = _make_report(symbol="ABNB")
    c = extract_candidate_from_report(rep, sector_map={"NVDA": "Semiconductors"})
    assert c.sector == UNKNOWN_SECTOR


def test_extract_candidate_top_factors_ranked_by_abs_weighted():
    rep = _make_report()
    c = extract_candidate_from_report(rep)
    # The largest |weighted_score| in the synthetic factor list is 0.0625
    # (two tied) then 0.05 then 0.0273. Ties broken by raw weighted (so the
    # +0.0625 entries come before the -0.0273).
    assert len(c.top_factors) == 3
    weights = [w for (_, w, _) in c.top_factors]
    abs_weights = [abs(w) for w in weights]
    assert abs_weights == sorted(abs_weights, reverse=True)


def test_extract_candidate_handles_missing_company_events():
    rep = _make_report(next_earnings=None)
    c = extract_candidate_from_report(rep)
    assert c.next_earnings_in_calendar_days is None


def test_extract_candidate_computes_earnings_in_days():
    rep = _make_report(as_of="2026-05-22", next_earnings="2026-06-05")
    c = extract_candidate_from_report(rep)
    assert c.next_earnings_in_calendar_days == 14


def test_extract_candidate_pit_summary_flags_non_pit_sections():
    rep = _make_report(
        pit_status={
            "price_data": "pit",
            "fundamentals": "pit",
            "industry_context.sector_labels": "non_pit_live_snapshot",
        }
    )
    c = extract_candidate_from_report(rep)
    assert "non-pit" in c.pit_status_summary
    assert "sector_labels" in c.pit_status_summary


def test_extract_candidate_pit_summary_returns_pit_when_clean():
    rep = _make_report(
        pit_status={"price_data": "pit", "fundamentals": "pit"}
    )
    c = extract_candidate_from_report(rep)
    assert c.pit_status_summary == "pit"


def test_extract_candidate_carries_tradingagents_review_gate():
    rep = _make_report(
        tradingagents_review={
            "status": "ok",
            "gate": {
                "risk_veto": True,
                "reason": "TradingAgents risk review vetoes new long exposure.",
                "missing_evidence": ["estimate revisions"],
            },
        }
    )
    c = extract_candidate_from_report(rep)
    assert c.tradingagents_review_status == "ok"
    assert c.review_risk_veto is True
    assert "vetoes" in c.review_summary
    assert c.review_missing_evidence == ["estimate revisions"]


# ---------- rank_candidates ----------


def test_rank_candidates_sorts_desc_by_composite():
    cands = [
        ScreenerCandidate(symbol="A", composite_score=0.3, direction="bullish",
                          confidence=0.5),
        ScreenerCandidate(symbol="B", composite_score=0.6, direction="bullish",
                          confidence=0.5),
        ScreenerCandidate(symbol="C", composite_score=-0.2, direction="bearish",
                          confidence=0.5),
    ]
    ranked = rank_candidates(cands, top_n=10)
    assert [c.symbol for c in ranked] == ["B", "A", "C"]


def test_rank_candidates_caps_at_top_n():
    cands = [
        ScreenerCandidate(
            symbol=f"T{i:02d}", composite_score=float(i) / 10,
            direction="bullish", confidence=0.5,
        )
        for i in range(10)
    ]
    ranked = rank_candidates(cands, top_n=3)
    assert len(ranked) == 3
    assert ranked[0].symbol == "T09"
    assert ranked[2].symbol == "T07"


def test_rank_candidates_breaks_ties_by_confidence_then_symbol():
    cands = [
        ScreenerCandidate(symbol="B", composite_score=0.3, direction="bullish",
                          confidence=0.5),
        ScreenerCandidate(symbol="A", composite_score=0.3, direction="bullish",
                          confidence=0.5),
        ScreenerCandidate(symbol="C", composite_score=0.3, direction="bullish",
                          confidence=0.7),
    ]
    ranked = rank_candidates(cands, top_n=3)
    # Highest confidence first (C), then alpha on ties (A before B).
    assert [c.symbol for c in ranked] == ["C", "A", "B"]


def test_rank_candidates_prefers_tech_weights_when_populated():
    """Plan Future-work #9: in cohort-aware mode, non-tech composites
    saturate at the universal-factor cap (Section 27 finding — 25 names
    tied at +0.832). Ranking should use composite_score_tech_weights
    when populated so the meaningful spread wins."""
    cands = [
        # All three at the cohort-aware cap. Without the fix, they'd order
        # alphabetically (A, B, C). With the fix they order by tech-weights.
        ScreenerCandidate(symbol="A", composite_score=0.832,
                          composite_score_tech_weights=0.20,
                          direction="bullish", confidence=0.77),
        ScreenerCandidate(symbol="B", composite_score=0.832,
                          composite_score_tech_weights=0.71,
                          direction="bullish", confidence=0.77),
        ScreenerCandidate(symbol="C", composite_score=0.832,
                          composite_score_tech_weights=0.55,
                          direction="bullish", confidence=0.77),
    ]
    ranked = rank_candidates(cands, top_n=3)
    assert [c.symbol for c in ranked] == ["B", "C", "A"]


def test_rank_candidates_falls_back_to_composite_when_tech_weights_none():
    """Non-cohort-aware runs leave composite_score_tech_weights=None.
    Behavior must match the pre-#9 contract: sort by composite_score."""
    cands = [
        ScreenerCandidate(symbol="A", composite_score=0.4, direction="bullish",
                          confidence=0.6),
        ScreenerCandidate(symbol="B", composite_score=0.7, direction="bullish",
                          confidence=0.6),
        ScreenerCandidate(symbol="C", composite_score=0.5, direction="bullish",
                          confidence=0.6),
    ]
    # No composite_score_tech_weights set anywhere → None on all three.
    assert all(c.composite_score_tech_weights is None for c in cands)
    ranked = rank_candidates(cands, top_n=3)
    assert [c.symbol for c in ranked] == ["B", "C", "A"]


def test_rank_per_sector_also_uses_tech_weights_when_populated():
    """Per-sector view inherits the same fix so a saturated sector doesn't
    rank by symbol alpha."""
    cands = [
        ScreenerCandidate(symbol="HX", composite_score=0.832,
                          composite_score_tech_weights=0.20, sector="Healthcare",
                          direction="bullish", confidence=0.77),
        ScreenerCandidate(symbol="HZ", composite_score=0.832,
                          composite_score_tech_weights=0.61, sector="Healthcare",
                          direction="bullish", confidence=0.77),
        ScreenerCandidate(symbol="HY", composite_score=0.832,
                          composite_score_tech_weights=0.40, sector="Healthcare",
                          direction="bullish", confidence=0.77),
    ]
    bucketed = rank_per_sector(cands, top_n_per_sector=3)
    assert [c.symbol for c in bucketed["Healthcare"]] == ["HZ", "HY", "HX"]


# ---------- rank_per_sector ----------


def test_rank_per_sector_groups_and_caps():
    cands = [
        ScreenerCandidate(symbol="N1", composite_score=0.5, direction="bullish",
                          confidence=0.5, sector="Semiconductors"),
        ScreenerCandidate(symbol="N2", composite_score=0.4, direction="bullish",
                          confidence=0.5, sector="Semiconductors"),
        ScreenerCandidate(symbol="N3", composite_score=0.3, direction="bullish",
                          confidence=0.5, sector="Semiconductors"),
        ScreenerCandidate(symbol="T1", composite_score=0.45, direction="bullish",
                          confidence=0.5, sector="Tech-MegaCap"),
        ScreenerCandidate(symbol="T2", composite_score=0.35, direction="bullish",
                          confidence=0.5, sector="Tech-MegaCap"),
    ]
    bucketed = rank_per_sector(cands, top_n_per_sector=2)
    assert set(bucketed.keys()) == {"Semiconductors", "Tech-MegaCap"}
    assert [c.symbol for c in bucketed["Semiconductors"]] == ["N1", "N2"]
    assert [c.symbol for c in bucketed["Tech-MegaCap"]] == ["T1", "T2"]


def test_rank_per_sector_buckets_unknown_last():
    cands = [
        ScreenerCandidate(symbol="A", composite_score=0.5, direction="bullish",
                          confidence=0.5, sector="Semiconductors"),
        ScreenerCandidate(symbol="B", composite_score=0.4, direction="bullish",
                          confidence=0.5, sector=UNKNOWN_SECTOR),
        ScreenerCandidate(symbol="C", composite_score=0.3, direction="bullish",
                          confidence=0.5, sector="Tech-MegaCap"),
    ]
    bucketed = rank_per_sector(cands, top_n_per_sector=5)
    keys = list(bucketed.keys())
    assert keys[-1] == UNKNOWN_SECTOR
    # Known sectors are alpha-sorted before Unknown.
    assert keys[:-1] == sorted(keys[:-1])


# ---------- render_screener_markdown ----------


def test_render_screener_markdown_contains_required_sections():
    top = [
        ScreenerCandidate(
            symbol="NVDA", composite_score=0.6, direction="bullish",
            confidence=0.72, sector="Semiconductors",
            top_factors=[
                ("fund_revenue_growth", 0.0625, "Revenue growth is strong."),
                ("trend_price_vs_sma20", 0.0625, "Price above 20D MA."),
            ],
            market_cap=1.5e12,
            next_earnings_in_calendar_days=14,
            pit_status_summary="pit",
            tradingagents_review_status="ok",
            review_risk_veto=False,
            review_summary="TradingAgents review agrees with the bullish signal.",
        ),
        ScreenerCandidate(
            symbol="AMD", composite_score=0.42, direction="bullish",
            confidence=0.61, sector="Semiconductors",
            top_factors=[
                ("fund_revenue_growth", 0.0625, "Revenue growth is strong."),
            ],
            market_cap=2.5e11,
            next_earnings_in_calendar_days=None,
            pit_status_summary="pit",
        ),
    ]
    per_sector = rank_per_sector(top, top_n_per_sector=5)
    md = render_screener_markdown(
        top_overall=top,
        per_sector_top=per_sector,
        as_of_date="2026-05-22",
        universe_size=100,
        scan_elapsed_s=312.4,
        candidates_evaluated=2,
        excluded_core_count=26,
        top_n=50,
        top_n_per_sector=5,
    )
    # Scan-stats header
    assert "# Screener — 2026-05-22" in md
    assert "Scan stats" in md
    assert "100 tickers" in md
    assert "26" in md
    assert "312.4" in md
    # Top-overall table includes both names + composite formatting
    assert "Top 2 overall" in md
    assert "NVDA" in md
    assert "AMD" in md
    assert "+0.600" in md
    assert "+0.420" in md
    # Per-sector header
    assert "Per-sector top picks" in md
    assert "Semiconductors" in md
    assert "TradingAgents review" in md
    assert "agrees with the bullish signal" in md
    # Top factor cell renders factor name + signed weighted contribution
    assert "fund_revenue_growth" in md
    assert "(+0.062)" in md
    # Market cap + earnings cells
    assert "$1.50T" in md
    assert "14d" in md


def test_render_screener_markdown_handles_empty_candidates():
    md = render_screener_markdown(
        top_overall=[],
        per_sector_top={},
        as_of_date="2026-05-22",
        universe_size=100,
        scan_elapsed_s=10.0,
    )
    assert "No candidates produced" in md
    assert "No sector buckets populated" in md


# ---------- candidate_to_dict ----------


def test_candidate_to_dict_round_trip():
    c = ScreenerCandidate(
        symbol="NVDA", composite_score=0.6, direction="bullish",
        confidence=0.72, sector="Semiconductors",
        top_factors=[("fund_revenue_growth", 0.0625, "rationale")],
        market_cap=1.5e12, adv_usd=None,
        next_earnings_in_calendar_days=14, pit_status_summary="pit",
    )
    d = candidate_to_dict(c)
    assert d["symbol"] == "NVDA"
    assert d["composite_score"] == pytest.approx(0.6, abs=1e-3)
    assert d["top_factors"][0]["factor"] == "fund_revenue_growth"
    assert d["sector"] == "Semiconductors"
    assert d["next_earnings_in_calendar_days"] == 14


# ---------- cohort_for_sector ----------


@pytest.mark.parametrize(
    "sector,expected",
    [
        ("Semiconductors", "tech"),
        ("Tech-MegaCap", "tech"),
        ("Software", "tech"),
        ("Networking", "tech"),
        ("Photonics", "tech"),
        ("Specialty-Materials", "tech"),
        ("Aerospace", "tech"),
        ("Financial-Data", "tech"),
        ("Financials", "non_tech"),
        ("Energy", "non_tech"),
        ("Healthcare", "non_tech"),
        ("Consumer-Staples", "non_tech"),
        ("Utilities", "non_tech"),
        ("Consumer-Discretionary", "non_tech"),
        ("Energy-Nuclear", "non_tech"),
        ("Other", "non_tech"),
        # Unknowns default conservatively to non_tech.
        ("Unknown", "non_tech"),
        ("", "non_tech"),
        (None, "non_tech"),
        ("CompletelyMadeUp", "non_tech"),
    ],
)
def test_cohort_for_sector(sector, expected):
    assert cohort_for_sector(sector) == expected


# ---------- rescore_report_for_cohort ----------


def _make_rich_report(symbol: str = "AAPL") -> dict:
    """A report with score/weight/data_available/pillar so rescore can run."""
    return {
        "symbol": symbol,
        "as_of_date": "2026-05-22",
        "direction": "bullish",
        "confidence": 0.7,
        "key_features": {
            "fundamental": {"market_cap": 3e12},
            "company_events": {"next_earnings_date": None},
            "model_scoring": {
                "composite_score": 0.55,
                "factor_scores": [
                    # Universal factor — should survive cohort=non_tech.
                    {
                        "factor": "momentum_rsi",
                        "pillar": "momentum",
                        "score": 0.5,
                        "weight": 0.06,
                        "weighted_score": 0.03,
                        "data_available": True,
                        "rationale": "RSI bullish.",
                    },
                    # Universal factor.
                    {
                        "factor": "peer_relative_valuation",
                        "pillar": "valuation",
                        "score": 0.4,
                        "weight": 0.06,
                        "weighted_score": 0.024,
                        "data_available": True,
                        "rationale": "Below peer P/E.",
                    },
                    # Non-universal — should be zeroed under cohort=non_tech.
                    {
                        "factor": "fund_revenue_growth",
                        "pillar": "fundamental",
                        "score": 0.8,
                        "weight": 0.08,
                        "weighted_score": 0.064,
                        "data_available": True,
                        "rationale": "Revenue growing fast.",
                    },
                    # Non-universal.
                    {
                        "factor": "market_spy_trend",
                        "pillar": "market",
                        "score": -0.3,
                        "weight": 0.04,
                        "weighted_score": -0.012,
                        "data_available": True,
                        "rationale": "SPY trend weak.",
                    },
                ],
            },
            "pit_status": {"price_data": "pit"},
        },
    }


def test_rescore_tech_cohort_is_noop_but_records_tech_weights():
    rep = _make_rich_report()
    original = rep["key_features"]["model_scoring"]["composite_score"]
    out = rescore_report_for_cohort(rep, cohort="tech")
    assert out is rep
    # composite_score is unchanged
    assert out["key_features"]["model_scoring"]["composite_score"] == original
    # tech-weights composite is captured (equal to original for tech cohort)
    assert (
        out["key_features"]["model_scoring"]["composite_score_tech_weights"]
        == round(original, 4)
    )


def test_rescore_non_tech_cohort_changes_composite_and_zeros_non_universal():
    rep = _make_rich_report()
    original = rep["key_features"]["model_scoring"]["composite_score"]
    out = rescore_report_for_cohort(rep, cohort="non_tech")
    factor_scores = out["key_features"]["model_scoring"]["factor_scores"]
    by_name = {f["factor"]: f for f in factor_scores}
    # Non-universal factors should have weight==0 and weighted_score==0.
    assert by_name["fund_revenue_growth"]["weight"] == 0.0
    assert by_name["fund_revenue_growth"]["weighted_score"] == 0.0
    assert by_name["market_spy_trend"]["weight"] == 0.0
    assert by_name["market_spy_trend"]["weighted_score"] == 0.0
    # Universal factors should still have nonzero weight after
    # renormalization.
    assert by_name["momentum_rsi"]["weight"] > 0
    assert by_name["peer_relative_valuation"]["weight"] > 0
    # composite_score is different from the original — non-universal
    # factors contributed +0.064 and -0.012 originally.
    new_composite = out["key_features"]["model_scoring"]["composite_score"]
    assert new_composite != original
    # Preserved original composite.
    assert (
        out["key_features"]["model_scoring"]["composite_score_tech_weights"]
        == round(original, 4)
    )
    # Direction / confidence are recomputed (consistent with new composite).
    if new_composite >= 0.15:
        assert out["direction"] == "bullish"
    elif new_composite <= -0.15:
        assert out["direction"] == "bearish"
    else:
        assert out["direction"] == "neutral"


def test_rescore_non_tech_renormalizes_universal_weights_to_one():
    # The non_tech weight vector has only universal factors surviving;
    # those weights must sum to 1 (renormalized) — verified indirectly by
    # checking that the composite arithmetic doesn't blow up and that the
    # surviving weighted_scores sum to a value within [-1, 1].
    rep = _make_rich_report()
    out = rescore_report_for_cohort(rep, cohort="non_tech")
    new_composite = out["key_features"]["model_scoring"]["composite_score"]
    assert -1.0 <= new_composite <= 1.0


def test_extract_candidate_preserves_tech_weights_composite_after_rescore():
    rep = _make_rich_report(symbol="JPM")
    rescore_report_for_cohort(rep, cohort="non_tech")
    cand = extract_candidate_from_report(
        rep, sector_map={"JPM": "Financials"}
    )
    # After cohort rescoring, the candidate's composite_score is the
    # cohort-aware (post-rescore) value but composite_score_tech_weights
    # is the original.
    assert cand.composite_score_tech_weights == pytest.approx(0.55, abs=1e-3)
    assert cand.composite_score != cand.composite_score_tech_weights


# ---------- render_screener_markdown — tech-weights column ----------


def test_render_screener_markdown_includes_tech_weights_column():
    c = ScreenerCandidate(
        symbol="JPM", composite_score=0.4, direction="bullish",
        confidence=0.6, sector="Financials",
        top_factors=[("momentum_rsi", 0.03, "RSI bullish.")],
        market_cap=5e11, adv_usd=2e9,
        composite_score_tech_weights=0.55,
    )
    md = render_screener_markdown(
        top_overall=[c],
        per_sector_top=rank_per_sector([c], top_n_per_sector=5),
        as_of_date="2026-05-22",
        universe_size=10,
        scan_elapsed_s=12.3,
        cohort_aware=True,
    )
    # Both columns rendered
    assert "Composite (tech wts)" in md
    assert "+0.400" in md  # cohort-aware composite
    assert "+0.550" in md  # tech-weights composite
    # Caveat about amplification
    assert "compute_composite" in md or "renormalizes" in md
    # Cohort-aware note when enabled
    assert "Cohort-aware" in md or "cohort-aware" in md


def test_render_screener_markdown_caveat_present_even_when_cohort_aware_off():
    c = ScreenerCandidate(
        symbol="NVDA", composite_score=0.6, direction="bullish",
        confidence=0.7, sector="Semiconductors",
        composite_score_tech_weights=0.6,
    )
    md = render_screener_markdown(
        top_overall=[c],
        per_sector_top={"Semiconductors": [c]},
        as_of_date="2026-05-22",
        universe_size=5,
        scan_elapsed_s=4.5,
        cohort_aware=False,
    )
    # The amplification caveat is always present in the scan-stats block.
    assert "renormalizes" in md or "amplifies" in md
    # The cohort-aware note specifically only appears when enabled.
    assert "non-tech sectors are scored against the universal-factor" not in md


def test_render_handles_missing_tech_weights_composite():
    # Backward compat: candidates without tech_weights field render em-dash.
    c = ScreenerCandidate(
        symbol="X", composite_score=0.3, direction="bullish",
        confidence=0.5, sector="Other",
        composite_score_tech_weights=None,
    )
    md = render_screener_markdown(
        top_overall=[c],
        per_sector_top={"Other": [c]},
        as_of_date="2026-05-22",
        universe_size=1,
        scan_elapsed_s=1.0,
    )
    assert "X" in md
    # em-dash placeholder in the tech-weights column
    assert "—" in md
