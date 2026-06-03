from __future__ import annotations

import json
from pathlib import Path

import analysis_ui
from analysis_ui import _run_robinhood_sync_request, _run_trade_tickets


def test_analysis_ui_trade_tickets_writes_handoff_files(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "NVDA.json").write_text(
        json.dumps(
            {
                "symbol": "NVDA",
                "as_of_date": "2026-05-31",
                "direction": "bullish",
                "confidence": 0.8,
                "key_features": {
                    "model_scoring": {"composite_score": 0.5},
                    "technical": {"close": 150.0, "sma_20": 145.0, "atr_14": 5.0},
                    "decision_summary": {"current_price": 150.0},
                    "option_strategies": {
                        "strategies": [
                            {
                                "type": "sell_put",
                                "verdict": "consider",
                                "reason": "Pays premium to enter lower.",
                                "expiry": "2026-06-19",
                                "strike": 120.0,
                                "premium": 2.5,
                            }
                        ]
                    },
                },
            }
        )
    )
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"cash": 10_000, "positions": {}}))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text(
        "\n".join(
            [
                "policy: equal_weight_bullish",
                "max_per_name: 0.10",
                "max_long_exposure: 0.50",
                "min_position_weight: 0.01",
                "universe:",
                "  - NVDA",
            ]
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text(
        "\n".join(
            [
                "enabled_asset_types: [equity, option_intent]",
                "default_order_type: limit",
                "time_in_force: gfd",
                "market_hours: regular_hours",
                "require_fresh_signal: true",
                "max_signal_age_days: 7",
                "min_abs_delta_shares: 1",
                "allow_fractional: false",
                "options_mode: intent_only",
            ]
        )
    )

    out = _run_trade_tickets(
        {
            "date": "2026-05-31",
            "reports_glob": str(report_dir / "*.json"),
            "positions_path": str(positions),
            "sizing_config": str(sizing),
            "execution_config": str(execution),
            "tickets_dir": str(tmp_path / "tickets"),
            "workflow_dir": str(tmp_path / "workflow"),
        }
    )

    assert out["ok"] is True
    assert out["positions_source"]["source_type"] == "ledger"
    assert out["positions_source"]["cash"] == 10_000
    assert out["report_coverage"]["missing_count"] == 0
    assert out["report_coverage"]["resolved_count"] == 1
    assert out["batch"]["summary"]["ready_count"] == 1
    assert out["batch"]["summary"]["option_intent_count"] == 1
    assert Path(out["paths"]["ticket_json"]).exists()
    assert Path(out["paths"]["ticket_markdown"]).exists()
    assert Path(out["paths"]["codex_review"]).exists()
    assert Path(out["paths"]["fills_template"]).exists()
    assert "Do not place any order" in out["codex_review"]


def test_analysis_ui_trade_tickets_reports_missing_analysis(tmp_path):
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"cash": 10_000, "positions": {}}))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text(
        "\n".join(
            [
                "policy: equal_weight_bullish",
                "max_per_name: 0.10",
                "max_long_exposure: 0.50",
                "min_position_weight: 0.01",
                "universe:",
                "  - NVDA",
                "  - AMD",
            ]
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text("enabled_asset_types: [equity, option_intent]\n")

    out = _run_trade_tickets(
        {
            "date": "2026-05-31",
            "reports_glob": str(tmp_path / "missing" / "*.json"),
            "positions_path": str(positions),
            "sizing_config": str(sizing),
            "execution_config": str(execution),
            "tickets_dir": str(tmp_path / "tickets"),
            "workflow_dir": str(tmp_path / "workflow"),
        }
    )

    assert out["report_coverage"]["report_files_found"] == 0
    assert out["report_coverage"]["missing_symbols"] == ["NVDA", "AMD"]
    assert any("Missing analysis reports" in note for note in out["notes"])


def test_trade_tickets_refreshes_stale_reports_when_requested(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "NVDA_2026-05-20.json").write_text(
        json.dumps(
            {
                "symbol": "NVDA",
                "as_of_date": "2026-05-20",
                "direction": "bullish",
                "confidence": 0.8,
                "key_features": {
                    "model_scoring": {"composite_score": 0.5},
                    "technical": {"close": 100.0, "sma_20": 98.0, "atr_14": 2.0},
                    "decision_summary": {"current_price": 100.0},
                },
            }
        )
    )
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"cash": 10_000, "positions": {}}))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text(
        "\n".join(
            [
                "policy: equal_weight_bullish",
                "max_per_name: 0.10",
                "max_long_exposure: 0.50",
                "min_position_weight: 0.01",
                "universe:",
                "  - NVDA",
            ]
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text("enabled_asset_types: [equity]\nmax_signal_age_days: 7\n")

    def fake_run_analysis(payload, output_dir):
        fresh_path = Path(output_dir) / "NVDA_2026-06-01.json"
        fresh_path.write_text(
            json.dumps(
                {
                    "symbol": "NVDA",
                    "as_of_date": "2026-06-01",
                    "direction": "bullish",
                    "confidence": 0.8,
                    "key_features": {
                        "model_scoring": {"composite_score": 0.5},
                        "technical": {"close": 100.0, "sma_20": 98.0, "atr_14": 2.0},
                        "decision_summary": {"current_price": 100.0},
                    },
                }
            )
        )
        return {"json_path": str(fresh_path), "cache_hit": False}

    monkeypatch.setattr(analysis_ui, "_run_analysis", fake_run_analysis)

    out = _run_trade_tickets(
        {
            "date": "2026-06-01",
            "reports_glob": str(report_dir / "*.json"),
            "positions_path": str(positions),
            "sizing_config": str(sizing),
            "execution_config": str(execution),
            "tickets_dir": str(tmp_path / "tickets"),
            "workflow_dir": str(tmp_path / "workflow"),
            "refresh_stale_reports": True,
        }
    )

    assert out["generated_reports"] == [
        {
            "symbol": "NVDA",
            "reason": "stale",
            "json_path": str(report_dir / "NVDA_2026-06-01.json"),
            "cache_hit": False,
        }
    ]
    assert out["report_coverage"]["stale_count"] == 0
    assert out["report_coverage"]["report_age_rows"][0]["as_of_date"] == "2026-06-01"


def test_analysis_ui_robinhood_sync_request_writes_prompt(tmp_path):
    output = tmp_path / "robinhood_snapshot.json"
    out = _run_robinhood_sync_request(
        {
            "workflow_dir": str(tmp_path / "workflow"),
            "output_path": str(output),
            "account_hint": "****9844",
        }
    )

    assert out["ok"] is True
    assert "rh.get_portfolio" in out["codex_prompt"]
    assert "Do not review, place, or cancel any orders." in out["codex_prompt"]
    assert out["output_path"] == str(output)
    assert Path(out["prompt_path"]).exists()


def test_trade_tickets_align_ready_buys_to_best_buy_rank(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for symbol, composite, price in [
        ("NVDA", 0.50, 100.0),
        ("AMD", 0.20, 20.0),
    ]:
        (report_dir / f"{symbol}.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "as_of_date": "2026-05-31",
                    "direction": "bullish",
                    "confidence": 0.8,
                    "key_features": {
                        "model_scoring": {"composite_score": composite},
                        "technical": {"close": price, "sma_20": price, "atr_14": 1.0},
                        "decision_summary": {"current_price": price},
                        "price_target": {"base_upside_pct": 0.20},
                    },
                }
            )
        )
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"cash": 10_000, "positions": {}}))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text(
        "\n".join(
            [
                "policy: equal_weight_bullish",
                "max_per_name: 0.10",
                "max_long_exposure: 0.50",
                "min_position_weight: 0.01",
                "universe:",
                "  - NVDA",
                "  - AMD",
            ]
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text("enabled_asset_types: [equity, option_intent]\n")

    out = _run_trade_tickets(
        {
            "date": "2026-05-31",
            "reports_glob": str(report_dir / "*.json"),
            "positions_path": str(positions),
            "sizing_config": str(sizing),
            "execution_config": str(execution),
            "tickets_dir": str(tmp_path / "tickets"),
            "workflow_dir": str(tmp_path / "workflow"),
        }
    )

    ready_symbols = [ticket["symbol"] for ticket in out["batch"]["tickets"]]
    blocked_reasons = [
        ticket["blocked_reason"] or ""
        for ticket in out["batch"]["blocked_tickets"]
    ]
    assert ready_symbols == ["NVDA"]
    assert any("Best Buy alignment" in reason for reason in blocked_reasons)


def test_trade_tickets_alignment_skips_non_executable_top_rank(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for symbol, composite, price in [
        ("NVDA", 0.50, 10_000.0),  # top ranked, but target rounds to zero shares
        ("AMD", 0.20, 20.0),
    ]:
        (report_dir / f"{symbol}.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "as_of_date": "2026-05-31",
                    "direction": "bullish",
                    "confidence": 0.8,
                    "key_features": {
                        "model_scoring": {"composite_score": composite},
                        "technical": {"close": price, "sma_20": price, "atr_14": 1.0},
                        "decision_summary": {"current_price": price},
                        "price_target": {"base_upside_pct": 0.20},
                    },
                }
            )
        )
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"cash": 1_000, "positions": {}}))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text(
        "\n".join(
            [
                "policy: equal_weight_bullish",
                "max_per_name: 0.10",
                "max_long_exposure: 0.50",
                "min_position_weight: 0.01",
                "universe:",
                "  - NVDA",
                "  - AMD",
            ]
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text("enabled_asset_types: [equity, option_intent]\n")

    out = _run_trade_tickets(
        {
            "date": "2026-05-31",
            "reports_glob": str(report_dir / "*.json"),
            "positions_path": str(positions),
            "sizing_config": str(sizing),
            "execution_config": str(execution),
            "tickets_dir": str(tmp_path / "tickets"),
            "workflow_dir": str(tmp_path / "workflow"),
        }
    )

    ready_symbols = [ticket["symbol"] for ticket in out["batch"]["tickets"]]
    assert ready_symbols == ["AMD"]
