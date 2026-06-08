from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import trade_tickets
from portfolio.execution import ExecutionConfig
from portfolio.signals import Action, Signal


def _action(symbol: str = "NVDA", **overrides) -> Action:
    base = {
        "symbol": symbol,
        "action": "BUY",
        "direction": "bullish",
        "composite": 0.4,
        "confidence": 0.7,
        "target_weight": 0.10,
        "current_weight": 0.0,
        "delta_pp": 0.10,
        "target_shares": 5,
        "current_shares": 0,
        "delta_shares": 5,
        "limit_price": 100.0,
        "stop_loss": 95.0,
        "last_close": 101.0,
        "sma20": 100.0,
        "atr14": 4.0,
        "signal_age_days": 1,
    }
    base.update(overrides)
    return Action(**base)


def _signal(path: Path, **gate) -> Signal:
    return Signal(
        symbol="NVDA",
        as_of_date="2026-05-31",
        direction="bullish",
        composite=0.4,
        confidence=0.7,
        source_path=str(path),
        tradingagents_review_gate=gate,
    )


def test_select_review_candidates_only_buy_add_missing_valid_gate(tmp_path):
    report = tmp_path / "NVDA.json"
    report.write_text("{}")
    candidates = trade_tickets._select_review_candidates(
        [
            _action("NVDA", action="BUY", delta_pp=0.10),
            _action("AMD", action="TRIM", delta_pp=-0.20),
            _action("AVGO", action="ADD", delta_pp=0.05),
        ],
        {
            "NVDA": _signal(report),
            "AMD": _signal(report),
            "AVGO": _signal(report, status="ok", ticket_gate="allow"),
        },
        top_n=5,
    )
    assert [a.symbol for a in candidates] == ["NVDA"]


def test_ensure_reviews_injects_shadow_gate_without_enforcing(tmp_path, monkeypatch):
    report = tmp_path / "NVDA.json"
    report.write_text(json.dumps({
        "symbol": "NVDA",
        "as_of_date": "2026-05-31",
        "direction": "bullish",
        "confidence": 0.7,
        "key_features": {"model_scoring": {"composite_score": 0.4}},
    }))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text("tradingagents_review_top_screener_n: 3\n")

    def _fake_review(**_kwargs):
        return {
            "status": "ok",
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "review_type": "report",
            "gate": {
                "status": "ok",
                "ticket_gate": "manual_review",
                "reason": "Needs human confirmation.",
                "execution_caveats": ["Risk critique."],
            },
        }

    monkeypatch.setattr(
        "tradingagents.analysis_only.agent_review.run_report_agent_review",
        _fake_review,
    )
    args = Namespace(
        ensure_tradingagents_review=True,
        review_mode="shadow",
        sizing_config=str(sizing),
        tradingagents_review_top_n=None,
        tradingagents_review_provider="openai",
        tradingagents_review_model="gpt-5.4-mini",
        tradingagents_review_base_url=None,
    )
    actions, summary = trade_tickets._ensure_tradingagents_reviews(
        [_action()],
        {"NVDA": _signal(report)},
        args=args,
        execution_config=ExecutionConfig(),
    )
    assert summary["mode"] == "shadow"
    assert summary["manual_review_symbols"] == ["NVDA"]
    assert summary["enforced"] is False
    assert actions[0].review_gate_status == "manual_review"
    assert actions[0].tradingagents_review["mode"] == "shadow"


def test_ensure_reviews_marks_enforced_when_config_allows(tmp_path, monkeypatch):
    report = tmp_path / "NVDA.json"
    report.write_text(json.dumps({
        "symbol": "NVDA",
        "as_of_date": "2026-05-31",
        "direction": "bullish",
        "key_features": {"model_scoring": {"composite_score": 0.4}},
    }))
    sizing = tmp_path / "sizing.yaml"
    sizing.write_text("tradingagents_review_top_screener_n: 1\n")

    monkeypatch.setattr(
        "tradingagents.analysis_only.agent_review.run_report_agent_review",
        lambda **_kwargs: {
            "status": "ok",
            "gate": {
                "status": "ok",
                "ticket_gate": "block_buy_add",
                "reason": "Risk review vetoes new long exposure.",
                "execution_caveats": [],
            },
        },
    )
    args = Namespace(
        ensure_tradingagents_review=True,
        review_mode="enforce",
        sizing_config=str(sizing),
        tradingagents_review_top_n=None,
        tradingagents_review_provider="openai",
        tradingagents_review_model="gpt-5.4-mini",
        tradingagents_review_base_url=None,
    )
    actions, summary = trade_tickets._ensure_tradingagents_reviews(
        [_action()],
        {"NVDA": _signal(report)},
        args=args,
        execution_config=ExecutionConfig(tradingagents_review_apply_to_tickets=True),
    )
    assert summary["enforced"] is True
    assert summary["blocked_symbols"] == ["NVDA"]
    assert actions[0].tradingagents_review["enforced"] is True

def test_trade_tickets_no_prices_writes_blocked_artifacts(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report = {
        "symbol": "NVDA",
        "as_of_date": "2026-05-31",
        "direction": "bullish",
        "confidence": 0.8,
        "key_features": {
            "model_scoring": {"composite_score": 0.5},
            "option_strategies": {
                "strategies": [
                    {
                        "type": "buy_call_spread",
                        "verdict": "consider",
                        "reason": "Defines bullish upside risk.",
                        "expiry": "2026-06-19",
                        "dte": 19,
                        "long_strike": 120.0,
                        "short_strike": 130.0,
                        "debit": 2.5,
                    }
                ]
            },
        },
    }
    (report_dir / "NVDA.json").write_text(json.dumps(report))

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

    out_dir = tmp_path / "tickets"
    result = subprocess.run(
        [
            sys.executable,
            "trade_tickets.py",
            "--reports-glob",
            str(report_dir / "*.json"),
            "--positions",
            str(positions),
            "--sizing-config",
            str(sizing),
            "--execution-config",
            str(execution),
            "--output-dir",
            str(out_dir),
            "--as-of",
            "2026-05-31",
            "--no-prices",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Summary:" in result.stdout
    payload = json.loads((out_dir / "2026-05-31.json").read_text())
    md = (out_dir / "2026-05-31.md").read_text()
    assert payload["summary"]["ready_count"] == 0
    assert payload["summary"]["blocked_count"] == 2
    assert any(t["asset_type"] == "equity" for t in payload["blocked_tickets"])
    option_ticket = next(
        t for t in payload["blocked_tickets"] if t["asset_type"] == "option_intent"
    )
    assert option_ticket["details"]["option_intent_rank"] == 1
    assert option_ticket["details"]["option_intent_score"] == 86.2
    assert "Codex Robinhood MCP Steps" not in md
    assert "Blocked Or Intent Only" in md
    assert "Option Score" in md
    assert "Contract" in md
    assert "$120.00-$130.00 call spread" in md
    assert "- Option contract: 2026-06-19 / 19 DTE / $120.00-$130.00 call spread" in md
