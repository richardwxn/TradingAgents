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


# ---------- malformed / defensive input handling ----------


def test_load_positions_payload_non_dict_is_invalid():
    loaded = load_positions_payload("not-a-dict")  # type: ignore[arg-type]
    assert loaded.cash == 0.0
    assert loaded.positions == {}
    assert loaded.metadata["source_type"] == "invalid"


def test_load_positions_payload_unknown_positions_shape():
    # positions present but neither list nor dict → "unknown", no crash.
    loaded = load_positions_payload({"positions": "garbage"})
    assert loaded.positions == {}
    assert loaded.metadata["source_type"] == "unknown"


def test_load_positions_payload_missing_positions_key_is_ledger():
    # No "positions" key → defaults to empty dict → ledger shape.
    loaded = load_positions_payload({"cash": 100})
    assert loaded.cash == 100
    assert loaded.positions == {}
    assert loaded.metadata["source_type"] == "ledger"


def test_ledger_skips_non_dict_entries():
    loaded = load_positions_payload(
        {"positions": {"NVDA": {"shares": 1, "avg_cost": 10.0}, "BAD": "oops"}}
    )
    assert "NVDA" in loaded.positions
    assert "BAD" not in loaded.positions


def test_snapshot_skips_non_dict_and_symbol_less_entries():
    loaded = load_positions_payload(
        {
            "positions": [
                "not-a-dict",
                {"shares": 5, "avg_cost": 10.0},  # no symbol
                {"symbol": "AMD", "shares": 3, "average_cost": 90.0},
            ]
        }
    )
    assert list(loaded.positions) == ["AMD"]
    assert loaded.positions["AMD"].shares == 3


def test_snapshot_cash_falls_back_to_buying_power():
    loaded = load_positions_payload(
        {"account": {"buying_power": 250.0}, "positions": [{"symbol": "X", "shares": 1}]}
    )
    assert loaded.cash == 250.0


def test_safe_float_falls_through_non_numeric_to_next_candidate():
    # shares="abc" is non-numeric → falls back to quantity=4.
    loaded = load_positions_payload(
        {"positions": {"NVDA": {"shares": "abc", "quantity": 4, "avg_cost": 10.0}}}
    )
    assert loaded.positions["NVDA"].shares == 4.0
