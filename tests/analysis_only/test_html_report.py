"""Tests for chart rendering and the HTML equity-research report."""

import base64

import pytest

from tradingagents.analysis_only.reporting import charts, html


def _sample_payload():
    return {
        "symbol": "TST",
        "as_of_date": "2026-05-26",
        "horizon": "swing_1_4_weeks",
        "direction": "bullish",
        "confidence": 0.62,
        "thesis": "A constructive setup.",
        "bull_case": ["catalyst A", "catalyst B"],
        "bear_case": ["risk A"],
        "generated_at_utc": "2026-05-26T12:00:00Z",
        "risk_flags": ["elevated IV"],
        "invalidation_conditions": ["close below 90"],
        "key_features": {
            "model_scoring": {
                "composite_score": 0.21,
                "pillar_scores": {
                    "technical": 0.3,
                    "fundamental": -0.1,
                    "sentiment": 0.05,
                    "options": 0.12,
                },
                "factor_scores": [
                    {"factor": "trend_a", "weighted_score": 0.18,
                     "data_available": True, "score": 1.0},
                    {"factor": "valuation_b", "weighted_score": -0.12,
                     "data_available": True, "score": -0.8},
                    {"factor": "missing_c", "weighted_score": None,
                     "data_available": False},
                ],
            },
            "price_target": {
                "status": "ok", "time_horizon": "3m", "spot": 100.0,
                "base": 110.0, "bull": 130.0, "bear": 80.0,
            },
            "price_range_forecast": {
                "1w": {"center_price": 101.0, "lower_80": 92.0, "upper_80": 110.0},
                "1m": {"center_price": 105.0, "lower_80": 85.0, "upper_80": 125.0},
            },
        },
    }


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def _is_png_data_uri(uri: str) -> bool:
    if not uri.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(uri.split(",", 1)[1])
    return raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_factor_scorecard_chart():
    uri = charts.factor_scorecard_chart(_sample_payload())
    assert _is_png_data_uri(uri)


def test_pillar_scores_chart():
    uri = charts.pillar_scores_chart(_sample_payload())
    assert _is_png_data_uri(uri)


def test_price_target_chart():
    uri = charts.price_target_chart(_sample_payload())
    assert _is_png_data_uri(uri)


def test_forecast_fan_chart():
    uri = charts.forecast_fan_chart(_sample_payload())
    assert _is_png_data_uri(uri)


def test_build_all_charts_returns_all_when_data_present():
    out = charts.build_all_charts(_sample_payload())
    assert set(out) == {"factor_scorecard", "pillar_scores", "price_target", "forecast_fan"}
    assert all(_is_png_data_uri(v) for v in out.values())


def test_charts_return_none_on_empty_payload():
    empty = {"key_features": {}}
    assert charts.factor_scorecard_chart(empty) is None
    assert charts.pillar_scores_chart(empty) is None
    assert charts.price_target_chart(empty) is None
    assert charts.forecast_fan_chart(empty) is None
    assert charts.build_all_charts(empty) == {}


def test_price_target_chart_skips_when_status_not_ok():
    payload = _sample_payload()
    payload["key_features"]["price_target"]["status"] = "unavailable"
    assert charts.price_target_chart(payload) is None


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def test_render_html_is_standalone_with_charts():
    doc = html.render_html(_sample_payload())
    assert doc.startswith("<!DOCTYPE html>")
    assert "<style>" in doc  # inline CSS, no external assets
    assert doc.count("data:image/png;base64,") == 4
    assert "TST" in doc
    assert "not investment advice" in doc.lower()


def test_render_html_without_charts():
    doc = html.render_html(_sample_payload(), embed_charts=False)
    assert "data:image/png;base64," not in doc
    assert "<article" in doc


def test_render_html_full_layout():
    # Full technical layout should still produce a valid standalone doc.
    doc = html.render_html(_sample_payload(), equity_research=False)
    assert doc.startswith("<!DOCTYPE html>")
    assert "<h1" in doc.lower() or "<h1>" in doc


def test_render_html_converts_markdown_tables():
    doc = html.render_html(_sample_payload())
    # The markdown renderers emit pipe tables; the converter must turn at
    # least one into an HTML table.
    assert "<table>" in doc


def test_render_html_degrades_on_sparse_payload():
    doc = html.render_html({"symbol": "X", "as_of_date": "2026-01-01"})
    assert doc.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," not in doc  # no chartable data


def test_render_html_file_writes(tmp_path):
    import json

    src = tmp_path / "rep.json"
    src.write_text(json.dumps(_sample_payload()))
    out = html.render_html_file(src)
    assert out == src.with_suffix(".html")
    assert out.exists()
    assert out.read_text().startswith("<!DOCTYPE html>")


def test_render_html_file_custom_output(tmp_path):
    import json

    src = tmp_path / "rep.json"
    src.write_text(json.dumps(_sample_payload()))
    target = tmp_path / "out" / "custom.html"
    out = html.render_html_file(src, output_path=target)
    assert out == target
    assert target.exists()
