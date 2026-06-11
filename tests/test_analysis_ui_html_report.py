"""Tests for serving the chart-embedded HTML report from the analysis UI."""

import io
import json

import pytest

import analysis_ui


def _sample_report():
    return {
        "symbol": "TST",
        "as_of_date": "2026-05-26",
        "horizon": "swing_1_4_weeks",
        "direction": "bullish",
        "confidence": 0.6,
        "thesis": "Constructive.",
        "bull_case": ["a"],
        "bear_case": ["b"],
        "generated_at_utc": "2026-05-26T12:00:00Z",
        "key_features": {
            "model_scoring": {
                "composite_score": 0.2,
                "pillar_scores": {"technical": 0.3, "fundamental": -0.1},
                "factor_scores": [
                    {"factor": "trend", "weighted_score": 0.18,
                     "data_available": True, "score": 1.0},
                ],
            },
            "price_target": {
                "status": "ok", "time_horizon": "3m", "spot": 100.0,
                "base": 110.0, "bull": 130.0, "bear": 80.0,
            },
        },
    }


class _StubHandler:
    """Drives AnalysisUIHandler._send_report_file without a real socket."""

    def __init__(self, handler_cls):
        self._impl = handler_cls.__new__(handler_cls)
        self._impl.send_response = self._send_response
        self._impl.send_header = self._send_header
        self._impl.end_headers = lambda: None
        self._impl.wfile = io.BytesIO()
        self.status = None
        self.headers = {}

    def _send_response(self, status):
        self.status = status

    def _send_header(self, key, value):
        self.headers[key] = value

    def send(self, *args, **kwargs):
        self._impl._send_report_file(*args, **kwargs)
        return self._impl.wfile.getvalue().decode("utf-8")


@pytest.fixture
def report_path(tmp_path, monkeypatch):
    # _send_report_file requires the path to live inside the working dir.
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "reports" / "TST_2026-05-26.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_sample_report()))
    return p


def test_send_report_html_renders_standalone_with_charts(report_path):
    handler = _StubHandler(analysis_ui.make_handler("reports"))
    body = handler.send(str(report_path), "html", "equity_research")
    assert handler.status == analysis_ui.HTTPStatus.OK
    assert handler.headers["Content-Type"] == "text/html; charset=utf-8"
    assert body.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in body
    assert "TST" in body


def test_send_report_html_standard_style(report_path):
    handler = _StubHandler(analysis_ui.make_handler("reports"))
    body = handler.send(str(report_path), "html", "standard")
    # Full technical layout still produces a valid standalone doc.
    assert body.startswith("<!DOCTYPE html>")


def test_send_report_html_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handler = _StubHandler(analysis_ui.make_handler("reports"))
    with pytest.raises(ValueError, match="not found"):
        handler.send(str(tmp_path / "nope.json"), "html", "equity_research")


def test_send_report_rejects_path_outside_workspace(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    monkeypatch.chdir(work)
    handler = _StubHandler(analysis_ui.make_handler("reports"))
    with pytest.raises(ValueError, match="inside the workspace"):
        handler.send(str(outside), "html", "equity_research")


def test_send_report_unknown_fmt_raises(report_path):
    handler = _StubHandler(analysis_ui.make_handler("reports"))
    with pytest.raises(ValueError, match="md, json, or html"):
        handler.send(str(report_path), "pdf", "equity_research")
