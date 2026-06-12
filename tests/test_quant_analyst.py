"""Tests for the Quant Analyst node that grounds the debate in the quant signal."""

import json

import pytest

from tradingagents.agents.analysts.quant_analyst import (
    create_quant_analyst,
    format_quant_report,
    _load_latest_report,
)


def _report(symbol="NVDA", as_of="2025-08-22", c20=-0.182, c60=0.316, primary=0.316):
    return {
        "symbol": symbol,
        "as_of_date": as_of,
        "direction": "bullish",
        "confidence": 0.71,
        "risk_flags": ["elevated IV"],
        "key_features": {
            "model_scoring": {
                "composite_score": primary,
                "pillar_scores": {"technical": 0.3, "fundamental": -0.1},
                "per_horizon_composites": {
                    "ret_5d": {"composite_score": primary, "weight_source": "global"},
                    "ret_20d": {"composite_score": c20, "weight_source": "per_horizon"},
                    "ret_60d": {"composite_score": c60, "weight_source": "global"},
                },
                "factor_scores": [
                    {"factor": "market_vix_regime", "pillar": "market",
                     "score": -0.5, "weight": 0.03, "weighted_score": -0.18,
                     "data_available": True, "rationale": "VIX elevated."},
                    {"factor": "peer_relative_valuation", "pillar": "valuation",
                     "score": 0.8, "weight": 0.06, "weighted_score": 0.048,
                     "data_available": True, "rationale": "Cheap vs peers."},
                ],
            },
            "options_flow": {"scan_status": "ok", "unusual_count": 4,
                             "net_call_put_notional": 1.2e6, "iv_rank": 0.6,
                             "iv_skew": 0.05},
            "filings_context": {
                "filing_analysis": {
                    "filing": {"form": "10-Q", "filing_date": "2025-08-01"},
                    "analysis": {"output": {
                        "tone": "cautious",
                        "summary": "Margins compressed on input costs.",
                        "key_risks": ["supply concentration", "fx exposure"],
                    }},
                }
            },
        },
        "data_quality": {"pit_warnings": [{"x": 1}]},
    }


def test_format_includes_signal_and_factors():
    md = format_quant_report(_report())
    assert "QUANTITATIVE MODEL SIGNAL" in md
    assert "NVDA" in md
    assert "market_vix_regime" in md
    assert "peer_relative_valuation" in md
    assert "Calibrated confidence" in md


def test_format_flags_horizon_divergence():
    # 20d bearish (-0.18) vs 60d bullish (+0.32) must be called out.
    md = format_quant_report(_report(c20=-0.182, c60=0.316))
    assert "HORIZON DIVERGENCE" in md
    assert "20d view is **bearish**" in md
    assert "60d view is **bullish**" in md


def test_format_no_divergence_when_aligned():
    md = format_quant_report(_report(c20=0.30, c60=0.32))
    assert "HORIZON DIVERGENCE" not in md


def test_format_includes_sec_digest_and_options():
    md = format_quant_report(_report())
    assert "SEC filing" in md and "cautious" in md
    assert "supply concentration" in md
    assert "Options flow" in md


def test_format_handles_empty_payload():
    assert "unavailable" in format_quant_report({}).lower()


def test_loader_picks_latest_pit(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    for date in ("2025-07-01", "2025-08-01", "2025-09-01"):
        (d / f"NVDA_{date}.json").write_text(json.dumps(_report(as_of=date)))
    # As-of 2025-08-15 must pick the 2025-08-01 report, not the future one.
    rep = _load_latest_report(str(d), "NVDA", "2025-08-15")
    assert rep["as_of_date"] == "2025-08-01"


def test_loader_returns_none_when_missing(tmp_path):
    assert _load_latest_report(str(tmp_path), "ZZZZ", "2025-08-15") is None


def test_node_loads_and_formats(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "NVDA_2025-08-22.json").write_text(json.dumps(_report()))
    node = create_quant_analyst({"analysis_only_reports_dir": str(d)})
    out = node({"company_of_interest": "NVDA", "trade_date": "2025-08-22"})
    assert "quant_report" in out
    assert "HORIZON DIVERGENCE" in out["quant_report"]


def test_node_graceful_when_no_report(tmp_path):
    node = create_quant_analyst({"analysis_only_reports_dir": str(tmp_path)})
    out = node({"company_of_interest": "ZZZZ", "trade_date": "2025-08-22"})
    assert "unavailable" in out["quant_report"].lower()
