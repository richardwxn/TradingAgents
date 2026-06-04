from __future__ import annotations

import json
from pathlib import Path

from trade_reconcile import build_reconciliation, normalize_fills


def test_reconcile_builds_ledger_patch_without_editing_positions(tmp_path):
    positions_path = tmp_path / "positions.json"
    original = {
        "cash": 10_000,
        "positions": {
            "NVDA": {"shares": 10, "avg_cost": 100.0},
        },
    }
    positions_path.write_text(json.dumps(original))
    fills = normalize_fills(
        {
            "orders": [
                {
                    "ticket_id": "abc",
                    "order_id": "order-1",
                    "symbol": "NVDA",
                    "side": "buy",
                    "filled_quantity": "2",
                    "average_price": "120",
                    "state": "filled",
                },
                {
                    "ticket_id": "def",
                    "order_id": "order-2",
                    "symbol": "AMD",
                    "side": "sell",
                    "filled_quantity": "1",
                    "average_price": "200",
                    "state": "filled",
                },
            ]
        }
    )
    out = build_reconciliation(
        ticket_batch={
            "as_of": "2026-05-31",
            "summary": {"ready_count": 2},
            "tickets": [{"ticket_id": "abc", "symbol": "NVDA"}],
        },
        fills=fills,
        positions_path=positions_path,
        as_of="2026-05-31",
    )
    assert out["ledger_patch"]["cash_delta"] == -40.0
    assert out["fills"][0]["matched_ticket_id"] == "abc"
    assert out["fills"][1]["matched_ticket_id"] is None
    assert out["requires_manual_position_update"] is True
    assert out["position_update_preview"]["cash"] == 9960.0
    assert out["ledger_patch"]["positions_to_update"]["NVDA"]["shares"] == 12
    assert out["ledger_patch"]["positions_to_update"]["NVDA"]["avg_cost"] == 103.3333
    assert "AMD" in out["ledger_patch"]["positions_to_update"]
    assert positions_path.read_text() == json.dumps(original)


def test_reconcile_apply_position_update_requires_explicit_helper(tmp_path):
    from trade_reconcile import apply_position_update

    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({
        "cash": 1_000,
        "positions": {"NVDA": {"shares": 1, "avg_cost": 100.0}},
    }))
    reconciliation = {
        "ledger_patch": {
            "cash_after_estimate": 850.0,
            "positions_to_update": {
                "NVDA": {"shares": 2.0, "avg_cost": 125.0},
            },
        }
    }
    apply_position_update(
        positions_path=positions_path,
        reconciliation=reconciliation,
    )
    updated = json.loads(positions_path.read_text())
    assert updated["cash"] == 850.0
    assert updated["positions"]["NVDA"]["shares"] == 2.0
    assert updated["positions"]["NVDA"]["avg_cost"] == 125.0
