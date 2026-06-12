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


def test_build_snapshot_handles_nested_robinhood_buying_power_and_prices():
    snapshot = build_snapshot(
        {
            "account_number": "852349844",
            "portfolio": {
                "total_value": "10046.66",
                "cash": "7380.55",
                "buying_power": {"buying_power": "3845.2600"},
            },
            "positions": [
                {
                    "symbol": "MU",
                    "quantity": "1.000000",
                    "average_buy_price": "859.020000",
                    "price": "943.98",
                }
            ],
        }
    )

    assert snapshot["account_number_masked"] == "****9844"
    assert snapshot["account"]["cash"] == 7380.55
    assert snapshot["account"]["buying_power"] == 3845.26
    assert snapshot["account"]["total_equity"] == 10046.66
    assert snapshot["positions"][0]["equity"] == 943.98
