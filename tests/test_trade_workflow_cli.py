from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_trade_workflow_writes_operator_packet(tmp_path):
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

    tickets_dir = tmp_path / "tickets"
    workflow_dir = tmp_path / "workflow"
    result = subprocess.run(
        [
            sys.executable,
            "trade_workflow.py",
            "--reports-glob",
            str(report_dir / "*.json"),
            "--positions",
            str(positions),
            "--sizing-config",
            str(sizing),
            "--execution-config",
            str(execution),
            "--tickets-dir",
            str(tickets_dir),
            "--workflow-dir",
            str(workflow_dir),
            "--as-of",
            "2026-05-31",
            "--no-prices",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote Codex review prompt" in result.stdout
    assert (tickets_dir / "2026-05-31.json").exists()
    assert (tickets_dir / "2026-05-31.md").exists()
    prompt = (workflow_dir / "2026-05-31_codex_review.md").read_text()
    fills_template = json.loads((workflow_dir / "2026-05-31_fills_template.json").read_text())
    assert "Do not place any order until I explicitly confirm." in prompt
    assert fills_template == {"orders": []}
