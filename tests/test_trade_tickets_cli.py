from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
