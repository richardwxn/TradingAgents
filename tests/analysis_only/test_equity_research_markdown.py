from __future__ import annotations

from tradingagents.analysis_only import render_equity_research_markdown


def _sample_report() -> dict:
    return {
        "symbol": "NVDA",
        "as_of_date": "2026-06-01",
        "horizon": "swing_1_4_weeks",
        "generated_at_utc": "2026-06-01T20:00:00Z",
        "thesis": "Bullish bias with strong AI demand and manageable valuation risk.",
        "direction": "bullish",
        "confidence": 0.82,
        "bull_case": ["Revenue growth remains above peers."],
        "bear_case": ["Multiple compression could offset earnings growth."],
        "risk_flags": ["Crowded positioning."],
        "invalidation_conditions": ["Break below the 50-day moving average."],
        "data_quality": {
            "data_provider_requested": "polygon",
            "data_provider_resolved": "polygon",
            "scoring_coverage": 0.91,
            "price_rows": 252,
            "has_fundamentals": True,
            "pit_warnings": ["analyst consensus is live"],
        },
        "key_features": {
            "narrative_source": "llm",
            "technical": {"close": 125.0, "sma_20": 120.0, "atr_14": 4.5},
            "fundamental": {
                "market_cap": 3_000_000_000_000,
                "enterprise_value": 2_950_000_000_000,
                "trailing_pe": 45.0,
                "forward_pe": 34.0,
                "price_to_sales": 20.0,
                "revenue_growth": 0.28,
                "profit_margins": 0.55,
                "return_on_equity": 0.70,
            },
            "model_scoring": {
                "composite_score": 0.54,
                "pillar_scores": {"valuation": 0.1, "momentum": 0.6},
                "factor_scores": [
                    {
                        "factor": "peer_relative_valuation",
                        "pillar": "valuation",
                        "score": 0.4,
                        "weight": 0.06,
                        "weighted_score": 0.024,
                        "bucket": "positive",
                        "rationale": "Trades cheaper than peer growth-adjusted median.",
                    },
                    {
                        "factor": "market_vix_regime",
                        "pillar": "sentiment",
                        "score": -0.3,
                        "weight": 0.04,
                        "weighted_score": -0.012,
                        "bucket": "negative",
                        "rationale": "Volatility is elevated.",
                    },
                ],
            },
            "decision_summary": {
                "status": "ok",
                "action": "buy",
                "label": "Buy on pullbacks",
                "current_price": 125.0,
                "estimated_win_probability": 0.63,
                "confidence": 0.78,
                "base_target": 145.0,
                "base_upside_pct": 0.16,
                "summary": "Upside remains attractive versus risk.",
                "entry": {
                    "starter_buy_at_or_below": 124.0,
                    "preferred_buy_zone": {"low": 118.0, "high": 123.0},
                    "add_below": 116.0,
                },
                "exit": {
                    "take_profit_1": 140.0,
                    "take_profit_2": 152.0,
                    "stop_loss": 112.0,
                },
            },
            "price_target": {
                "status": "ok",
                "bear": 105.0,
                "base": 145.0,
                "bull": 170.0,
                "base_upside_pct": 0.16,
                "confidence": 0.72,
                "coverage": 0.88,
                "drivers": ["Forward EPS revisions remain positive."],
            },
            "competitor_analysis": {
                "summary": {
                    "peer_count": 2,
                    "return_20d_vs_peers": 0.03,
                    "trailing_pe_vs_peers": -0.15,
                    "peer_ev_to_revenue_median": 18.0,
                },
                "peer_metrics": [
                    {
                        "ticker": "AMD",
                        "return_20d": -0.02,
                        "return_60d": 0.05,
                        "trailing_pe": 40.0,
                        "revenue_growth": 0.12,
                        "market_cap": 300_000_000_000,
                    }
                ],
            },
            "news_sentiment": {
                "status": "ok",
                "net_sentiment": 0.4,
                "n_articles": 8,
            },
            "industry_news_context": {
                "status": "ok",
                "semantic_summary": "AI infrastructure spending remains firm.",
                "ranked_themes": [
                    {"theme": "AI capex", "count": 3, "examples": ["cloud demand"]}
                ],
            },
            "filings_context": {"status": "ok", "latest_form": "10-Q"},
            "earnings_calendar": {
                "status": "ok",
                "next_earnings_date": "2026-08-25",
                "next_earnings_in_calendar_days": 85,
            },
            "event_timeline": [
                {
                    "source": "earnings",
                    "event_type": "earnings",
                    "likely_direction": "volatile",
                    "relevance": 0.8,
                    "time_horizon": "3m",
                    "title": "Next earnings report",
                }
            ],
            "options_flow": {
                "scan_status": "ok",
                "unusual_count": 4,
                "net_call_put_notional": 2_000_000,
                "atm_iv_30d": 0.42,
            },
            "options_iv": {
                "status": "ok",
                "atm_iv_30d": 0.42,
                "iv_rank_252d": 0.67,
            },
            "llm_narrative": {"enabled": True, "status": "ok"},
            "llm_insights": {
                "enabled": True,
                "status": "ok",
                "analysis": {
                    "summary": "The setup depends on sustained AI server demand.",
                    "industry_angle": "The industry cycle still supports leaders.",
                    "competitor_angle": "Peers lag on margins.",
                    "company_event_angle": "Earnings remain the next major check.",
                },
            },
        },
    }


def test_equity_research_markdown_renders_full_report_sections():
    md = render_equity_research_markdown(_sample_report())

    assert "# NVDA Equity Research Report" in md
    assert "## Executive summary" in md
    assert "## Investment thesis" in md
    assert "## Valuation and peer read" in md
    assert "## Catalysts and sentiment" in md
    assert "## Risk assessment" in md
    assert "## Trading plan appendix" in md
    assert "peer_relative_valuation" in md
    assert "AI infrastructure spending remains firm." in md
    assert "Option strategy candidates" not in md


def test_equity_research_markdown_handles_sparse_payload():
    md = render_equity_research_markdown(
        {
            "symbol": "CASH",
            "as_of_date": "2026-06-01",
            "direction": "neutral",
            "confidence": 0.5,
            "key_features": {},
            "data_quality": {},
        }
    )

    assert "# CASH Equity Research Report" in md
    assert "## Executive summary" in md
    assert "## Valuation and peer read" not in md
    assert md.endswith("\n")
