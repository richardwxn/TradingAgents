import json

import pytest

import analysis_ui
from portfolio.paper_trading import RecommendationRecord, write_recommendations_for_day


def _rec(symbol: str, action: str, score: float | None) -> dict:
    row = {
        "symbol": symbol,
        "action": action,
        "direction": "bullish" if action in {"BUY", "ADD", "SKIP"} else "bearish",
        "composite": 0.12,
        "confidence": 0.7,
        "target_weight": 0.03,
        "delta_pp": 0.01,
        "notes": [],
    }
    if score is not None:
        row["ml_shadow"] = {
            "models": {
                "ridge_return": {
                    "ret_60d": {
                        "score": score,
                        "raw_score": score - 0.5,
                        "calibrated": True,
                    }
                }
            }
        }
    return row


def _paper_rec(symbol: str, action: str, score: float | None) -> RecommendationRecord:
    shadow = {}
    if score is not None:
        shadow = {
            "models": {
                "ridge_return": {
                    "ret_60d": {
                        "score": score,
                        "raw_score": score - 0.5,
                        "calibrated": True,
                    }
                }
            }
        }
    return RecommendationRecord(
        as_of_date="2026-06-09",
        symbol=symbol,
        action=action,
        direction="bullish" if action in {"BUY", "ADD", "SKIP"} else "bearish",
        composite=0.12,
        confidence=0.7,
        target_weight=0.03 if action in {"BUY", "ADD"} else 0.0,
        current_weight=0.0,
        delta_pp=0.03 if action in {"BUY", "ADD"} else 0.0,
        target_shares=1,
        current_shares=0,
        delta_shares=1,
        limit_price=100.0,
        stop_loss=90.0,
        last_close=100.0,
        sma20=98.0,
        atr14=3.0,
        signal_age_days=0,
        price_source="fixture",
        notes=[],
        ml_shadow=shadow,
    )


def test_ml_gate_snapshot_compares_factor_trigger_to_shadow_trigger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec_dir = tmp_path / "paper" / "recommendations"
    rec_dir.mkdir(parents=True)
    path = rec_dir / "2026-06-09.jsonl"
    rows = [
        _rec("ALLOW", "BUY", 0.61),
        _rec("BLOCK", "BUY", 0.44),
        _rec("NEW", "SKIP", 0.72),
        _rec("QUIET", "SKIP", 0.33),
        _rec("SELL", "TRIM", 0.88),
        _rec("MISS", "ADD", None),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    snapshot = analysis_ui._ml_gate_snapshot(
        as_of="2026-06-09",
        model="ridge_return",
        horizon="ret_60d",
        threshold=0.55,
        base_dir="paper",
    )

    by_symbol = {row["symbol"]: row for row in snapshot["rows"]}
    assert by_symbol["ALLOW"]["ml_gate"] == "allow"
    assert by_symbol["ALLOW"]["disagreement"] is False
    assert by_symbol["BLOCK"]["ml_gate"] == "block"
    assert by_symbol["BLOCK"]["disagreement"] is True
    assert by_symbol["NEW"]["ml_gate"] == "would_trigger"
    assert by_symbol["NEW"]["disagreement"] is True
    assert by_symbol["QUIET"]["ml_gate"] == "not_triggered"
    assert by_symbol["QUIET"]["disagreement"] is False
    assert by_symbol["SELL"]["ml_gate"] == "not_applicable"
    assert by_symbol["SELL"]["disagreement"] is False
    assert by_symbol["MISS"]["ml_gate"] == "missing"

    assert snapshot["summary"]["production_buy_triggers"] == 3
    assert snapshot["summary"]["ml_buy_triggers"] == 2
    assert snapshot["summary"]["ml_allowed"] == 1
    assert snapshot["summary"]["ml_blocked"] == 1
    assert snapshot["summary"]["ml_new_triggers"] == 1
    assert snapshot["summary"]["disagreements"] == 2
    assert snapshot["summary"]["missing_ml"] == 1


def test_analysis_ui_html_exposes_ml_gate_tab():
    html = analysis_ui._html_page()

    assert 'data-tab="ml-gate"' in html
    assert 'id="ml-gate"' in html
    assert "/api/ml-gate" in html


def test_analysis_ui_html_includes_multi_regime_warning_banner():
    """The ML Gate panel must surface the multi-regime findings prominently
    so users do not mistake ML disagreement for ML being right."""
    html = analysis_ui._html_page()

    assert "multi-regime walk-forward gates" in html
    assert "ml_shadow_multi_regime_findings.md" in html
    assert "hint warn" in html


def test_analysis_ui_html_includes_ml_gate_refresh_progress_state():
    html = analysis_ui._html_page()

    assert "ml-gate-progress" in html
    assert "ML gate loading" in html
    assert "ML gate refresh in progress" in html
    assert "run.disabled = true;" in html


def test_analysis_ui_html_includes_ml_gate_shadow_performance_controls():
    html = analysis_ui._html_page()

    assert "/api/ml-gate-shadow-report" in html
    assert "Shadow performance" in html
    assert "Run Shadow Audit" in html
    assert "factor production, ML-gated factor, ML-only, and hybrid proxy" in html


def test_ml_gate_shadow_performance_summarizes_strategies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path / "paper"
    write_recommendations_for_day(
        [
            _paper_rec("ALLOW", "BUY", 0.61),
            _paper_rec("BLOCK", "BUY", 0.44),
            _paper_rec("NEW", "SKIP", 0.72),
            _paper_rec("QUIET", "SKIP", 0.33),
        ],
        base_dir=base_dir,
        as_of_date="2026-06-09",
    )

    report = analysis_ui._ml_gate_shadow_performance(
        {
            "from_date": "2026-06-09",
            "to_date": "2026-06-09",
            "base_dir": "paper",
            "model": "ridge_return",
            "horizon": "ret_60d",
            "threshold": "0.55",
            "fetch_prices": "false",
        }
    )

    by_strategy = {row["strategy"]: row for row in report["summaries"]}
    assert report["ok"] is True
    assert report["coverage"]["recommendation_rows"] == 4
    assert report["coverage"]["rows_with_ml_score"] == 4
    assert report["coverage"]["outcome_status"] == "prices_skipped"
    assert by_strategy["factor_production"]["selected"] == 2
    assert by_strategy["ml_gated_factor"]["selected"] == 1
    assert by_strategy["ml_only"]["selected"] == 2
    assert by_strategy["hybrid"]["selected"] >= 1
    assert [(row["symbol"], row["reason"]) for row in report["disagreements"]] == [
        ("BLOCK", "ML blocks production BUY/ADD"),
        ("NEW", "ML would trigger where factor did not"),
    ]
    assert "ML Gate shadow report" in report["markdown"]


def test_ml_gate_snapshot_rejects_unknown_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper" / "recommendations").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown ML model"):
        analysis_ui._ml_gate_snapshot(
            as_of="2026-06-09",
            model="totally_made_up",
            horizon="ret_60d",
            threshold=0.55,
            base_dir="paper",
        )


def test_ml_gate_snapshot_rejects_unknown_horizon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper" / "recommendations").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown ML horizon"):
        analysis_ui._ml_gate_snapshot(
            as_of="2026-06-09",
            model="ridge_return",
            horizon="ret_1y",
            threshold=0.55,
            base_dir="paper",
        )
