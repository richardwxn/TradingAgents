from __future__ import annotations

from robinhood_snapshot import build_snapshot


def test_build_snapshot_normalizes_raw_robinhood_like_payload():
    snapshot = build_snapshot(
        {
            "account_number": "abc",
            "portfolio": {"buying_power": "1000", "total_value": "2500"},
            "positions": {
                "results": [
                    {
                        "symbol": "NVDA",
                        "quantity": "3",
                        "average_buy_price": "120.50",
                        "market_value": "450",
                    }
                ]
            },
        }
    )
    assert snapshot["source"] == "robinhood_mcp_read_only"
    assert snapshot["account_number_masked"] == "****"
    assert snapshot["account"]["cash"] == 1000
    assert snapshot["account"]["total_equity"] == 2500
    assert snapshot["positions"][0]["symbol"] == "NVDA"
    assert snapshot["positions"][0]["average_cost"] == 120.5
