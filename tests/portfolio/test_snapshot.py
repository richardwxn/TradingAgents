from __future__ import annotations

from portfolio.snapshot import load_positions_payload


def test_load_positions_payload_supports_legacy_ledger():
    loaded = load_positions_payload(
        {
            "cash": 1000,
            "positions": {
                "nvda": {"shares": 2, "avg_cost": 100.0},
            },
        }
    )
    assert loaded.cash == 1000
    assert loaded.positions["NVDA"].shares == 2
    assert loaded.positions["NVDA"].avg_cost == 100.0
    assert loaded.metadata["source_type"] == "ledger"


def test_load_positions_payload_supports_snapshot_list():
    loaded = load_positions_payload(
        {
            "as_of": "2026-05-31",
            "source": "robinhood_mcp_read_only",
            "account": {"cash": 5000, "total_equity": 25_000},
            "positions": [
                {
                    "symbol": "AMD",
                    "shares": 10,
                    "average_cost": 120.0,
                    "equity": 1500,
                },
            ],
        }
    )
    assert loaded.cash == 5000
    assert loaded.positions["AMD"].shares == 10
    assert loaded.positions["AMD"].avg_cost == 120.0
    assert loaded.metadata["source_type"] == "snapshot"
    assert loaded.metadata["total_equity"] == 25_000


def test_load_positions_payload_keeps_empty_snapshot_as_snapshot():
    loaded = load_positions_payload(
        {
            "as_of": "2026-05-31",
            "source": "robinhood_mcp_read_only",
            "account": {"cash": 5000, "total_equity": 5000},
            "positions": [],
        }
    )
    assert loaded.cash == 5000
    assert loaded.positions == {}
    assert loaded.metadata["source_type"] == "snapshot"
