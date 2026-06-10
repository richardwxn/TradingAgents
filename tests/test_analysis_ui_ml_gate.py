import json

import analysis_ui


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
