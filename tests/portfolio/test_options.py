from __future__ import annotations

import pytest

from portfolio.options import (
    OptionBookSummary,
    OptionPosition,
    load_option_positions,
    summarize_option_book,
)


# ---------- OptionPosition validation ----------


def test_option_position_valid_long_call():
    p = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=5.50,
    )
    assert p.is_long
    assert not p.is_short
    assert p.contract_basis == 550.0  # 5.50 × 1 × 100


def test_option_position_valid_short_put():
    p = OptionPosition(
        symbol="AAPL", right="put", strike=170.0,
        expiry="2026-08-15", quantity=-2, avg_cost=3.10,
    )
    assert p.is_short
    assert p.contract_basis == 620.0  # 3.10 × 2 × 100


def test_option_position_rejects_bad_right():
    with pytest.raises(ValueError, match="right"):
        OptionPosition(
            symbol="NVDA", right="straddle", strike=200.0,
            expiry="2026-06-19", quantity=1, avg_cost=5.50,
        )


def test_option_position_rejects_zero_quantity():
    with pytest.raises(ValueError, match="quantity"):
        OptionPosition(
            symbol="NVDA", right="call", strike=200.0,
            expiry="2026-06-19", quantity=0, avg_cost=5.50,
        )


def test_option_position_rejects_non_int_quantity():
    with pytest.raises(ValueError, match="quantity"):
        OptionPosition(
            symbol="NVDA", right="call", strike=200.0,
            expiry="2026-06-19", quantity=1.5, avg_cost=5.50,
        )


def test_option_position_rejects_negative_strike():
    with pytest.raises(ValueError, match="strike"):
        OptionPosition(
            symbol="NVDA", right="call", strike=-1.0,
            expiry="2026-06-19", quantity=1, avg_cost=5.50,
        )


def test_option_position_rejects_negative_avg_cost():
    with pytest.raises(ValueError, match="avg_cost"):
        OptionPosition(
            symbol="NVDA", right="call", strike=200.0,
            expiry="2026-06-19", quantity=1, avg_cost=-0.5,
        )


def test_option_position_rejects_bad_expiry():
    with pytest.raises(ValueError, match="expiry"):
        OptionPosition(
            symbol="NVDA", right="call", strike=200.0,
            expiry="not-a-date", quantity=1, avg_cost=5.50,
        )


# ---------- DTE ----------


def test_option_position_dte_positive():
    p = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=5.50,
    )
    assert p.dte("2026-06-01") == 18


def test_option_position_dte_negative_when_expired():
    p = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-05-01", quantity=1, avg_cost=5.50,
    )
    assert p.dte("2026-06-01") == -31


def test_option_position_dte_handles_bad_input():
    p = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=5.50,
    )
    assert p.dte("not-a-date") is None


# ---------- load_option_positions ----------


def test_load_option_positions_legacy_shares_only_yields_empty():
    """Legacy schema (shares + avg_cost, no options) must not break."""
    payload = {
        "cash": 100000,
        "positions": {
            "NVDA": {"shares": 100, "avg_cost": 150.0},
        },
    }
    assert load_option_positions(payload) == {}


def test_load_option_positions_mixed_legacy_and_options():
    payload = {
        "cash": 100000,
        "positions": {
            "NVDA": {
                "shares": 100, "avg_cost": 150.0,
                "options": [
                    {"right": "call", "strike": 200, "expiry": "2026-06-19",
                     "quantity": 1, "avg_cost": 5.50},
                ],
            },
            "AAPL": {"shares": 50, "avg_cost": 175.0},  # legacy, no options
        },
    }
    out = load_option_positions(payload)
    assert set(out.keys()) == {"NVDA"}
    assert len(out["NVDA"]) == 1


def test_load_option_positions_multiple_contracts_per_symbol():
    payload = {
        "positions": {
            "NVDA": {
                "options": [
                    {"right": "call", "strike": 200, "expiry": "2026-06-19",
                     "quantity": 1, "avg_cost": 5.50},
                    {"right": "put", "strike": 180, "expiry": "2026-06-19",
                     "quantity": -1, "avg_cost": 3.20},
                ],
            },
        },
    }
    contracts = load_option_positions(payload)["NVDA"]
    assert len(contracts) == 2
    assert contracts[0].right == "call"
    assert contracts[1].right == "put"
    assert contracts[1].is_short


def test_load_option_positions_normalizes_symbol_uppercase():
    payload = {
        "positions": {
            "nvda": {
                "options": [
                    {"right": "call", "strike": 200, "expiry": "2026-06-19",
                     "quantity": 1, "avg_cost": 5.50},
                ],
            },
        },
    }
    out = load_option_positions(payload)
    assert "NVDA" in out
    assert out["NVDA"][0].symbol == "NVDA"


def test_load_option_positions_raises_on_missing_field():
    payload = {
        "positions": {
            "NVDA": {
                "options": [
                    {"right": "call", "strike": 200, "expiry": "2026-06-19"},
                    # missing quantity + avg_cost
                ],
            },
        },
    }
    with pytest.raises(ValueError, match="quantity"):
        load_option_positions(payload)


def test_load_option_positions_raises_on_bad_options_type():
    payload = {
        "positions": {
            "NVDA": {"options": "not-a-list"},
        },
    }
    with pytest.raises(ValueError, match="must be a list"):
        load_option_positions(payload)


def test_load_option_positions_forward_compatible_unknown_keys():
    """Unknown keys on a contract entry are silently ignored."""
    payload = {
        "positions": {
            "NVDA": {
                "options": [
                    {"right": "call", "strike": 200, "expiry": "2026-06-19",
                     "quantity": 1, "avg_cost": 5.50,
                     "trade_id": "abc123", "notes": "earnings hedge"},
                ],
            },
        },
    }
    contracts = load_option_positions(payload)["NVDA"]
    assert len(contracts) == 1
    assert contracts[0].strike == 200


# ---------- summarize_option_book ----------


def test_summarize_option_book_empty_returns_none():
    assert summarize_option_book([]) is None


def test_summarize_option_book_long_call_only():
    p = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=2, avg_cost=5.50,
    )
    s = summarize_option_book([p])
    assert s.long_call_contracts == 2
    assert s.short_call_contracts == 0
    assert s.long_put_contracts == 0
    assert s.short_put_contracts == 0
    assert s.total_premium_paid == 1100.0  # 5.50 × 2 × 100
    assert s.total_premium_collected == 0.0
    assert s.net_premium_basis == 1100.0
    assert s.total_contracts == 2


def test_summarize_option_book_mixed_long_and_short():
    positions = [
        OptionPosition(symbol="NVDA", right="call", strike=200.0,
                       expiry="2026-06-19", quantity=1, avg_cost=5.50),
        OptionPosition(symbol="NVDA", right="put", strike=180.0,
                       expiry="2026-06-19", quantity=-1, avg_cost=3.20),
        OptionPosition(symbol="NVDA", right="call", strike=220.0,
                       expiry="2026-07-17", quantity=-2, avg_cost=2.00),
    ]
    s = summarize_option_book(positions)
    assert s.long_call_contracts == 1
    assert s.short_call_contracts == 2
    assert s.short_put_contracts == 1
    assert s.total_premium_paid == 550.0  # long: 5.50 × 1 × 100
    assert s.total_premium_collected == 320.0 + 400.0  # short: 3.20 + 2.00 × 2 × 100
    assert s.net_premium_basis == 550.0 - 720.0  # paid − collected (net credit)


# ---------- enrich_with_chain ----------


from portfolio.options import (
    BookGreeks,
    EnrichedOption,
    book_greeks,
    enrich_with_chain,
)


def _chain_row(**overrides):
    base = {
        "type": "call", "strike": 200.0, "expiry": "2026-06-19",
        "dte": 25, "bid": 5.40, "ask": 5.60, "mid": 5.50, "last": 5.50,
        "volume": 1234, "open_interest": 5678,
        "implied_volatility": 0.45, "delta": 0.52, "gamma": 0.018,
        "theta": -0.12, "vega": 0.085,
        "spot_distance_pct": 0.04,
    }
    base.update(overrides)
    return base


def test_enrich_matches_by_right_strike_expiry():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=4.00,
    )
    chain = [_chain_row(strike=180.0), _chain_row(strike=200.0)]
    enriched = enrich_with_chain([pos], chain)
    assert len(enriched) == 1
    assert enriched[0].mark == 5.50
    assert enriched[0].delta == 0.52


def test_enrich_returns_none_fields_when_no_match():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=300.0,
        expiry="2026-06-19", quantity=1, avg_cost=4.00,
    )
    enriched = enrich_with_chain([pos], [_chain_row(strike=200.0)])
    assert len(enriched) == 1
    assert enriched[0].mark is None
    assert enriched[0].delta is None


def test_enrich_doesnt_match_wrong_right():
    pos = OptionPosition(
        symbol="NVDA", right="put", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=4.00,
    )
    enriched = enrich_with_chain([pos], [_chain_row(type="call", strike=200.0)])
    assert enriched[0].mark is None


def test_enrich_doesnt_match_wrong_expiry():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=4.00,
    )
    enriched = enrich_with_chain(
        [pos], [_chain_row(strike=200.0, expiry="2026-07-17")],
    )
    assert enriched[0].mark is None


# ---------- signed_delta_shares ----------


def test_long_call_signed_delta_positive():
    pos = OptionPosition(symbol="NVDA", right="call", strike=200.0,
                         expiry="2026-06-19", quantity=2, avg_cost=4.00)
    enriched = enrich_with_chain([pos], [_chain_row(delta=0.5)])
    # 2 long calls × 0.5 delta × 100 shares/contract = +100
    assert enriched[0].signed_delta_shares == 100.0


def test_short_call_signed_delta_negative():
    pos = OptionPosition(symbol="NVDA", right="call", strike=200.0,
                         expiry="2026-06-19", quantity=-1, avg_cost=4.00)
    enriched = enrich_with_chain([pos], [_chain_row(delta=0.5)])
    # Short 1 call × 0.5 delta × 100 = -50
    assert enriched[0].signed_delta_shares == -50.0


def test_long_put_signed_delta_negative():
    pos = OptionPosition(symbol="NVDA", right="put", strike=180.0,
                         expiry="2026-06-19", quantity=1, avg_cost=2.50)
    enriched = enrich_with_chain(
        [pos],
        [_chain_row(type="put", strike=180.0, delta=-0.40)],
    )
    # 1 long put × -0.40 delta × 100 = -40
    assert enriched[0].signed_delta_shares == -40.0


def test_short_put_signed_delta_positive():
    pos = OptionPosition(symbol="NVDA", right="put", strike=180.0,
                         expiry="2026-06-19", quantity=-1, avg_cost=2.50)
    enriched = enrich_with_chain(
        [pos],
        [_chain_row(type="put", strike=180.0, delta=-0.40)],
    )
    # Short put: -1 × -0.40 × 100 = +40
    assert enriched[0].signed_delta_shares == 40.0


# ---------- book_greeks ----------


def test_book_greeks_shares_only_no_options():
    bg = book_greeks([], shares=100)
    assert bg.shares == 100
    # No options, but shares contribute directly to delta.
    assert bg.net_share_equivalent_delta == 100.0
    assert bg.options_count == 0
    assert bg.net_vega_dollars_per_vol_pt is None


def test_book_greeks_none_when_empty_and_no_shares():
    assert book_greeks([], shares=0) is None


def test_book_greeks_shares_plus_long_call():
    pos = OptionPosition(symbol="NVDA", right="call", strike=200.0,
                         expiry="2026-06-19", quantity=1, avg_cost=4.00)
    enriched = enrich_with_chain([pos], [_chain_row(delta=0.50, vega=0.085, theta=-0.12)])
    bg = book_greeks(enriched, shares=100)
    # 100 shares + 1 × 0.50 × 100 = 150 share-eq delta
    assert bg.net_share_equivalent_delta == 150.0
    # 1 × 0.085 × 100 = 8.5 vega
    assert bg.net_vega_dollars_per_vol_pt == 8.5
    # 1 × -0.12 × 100 = -12 theta
    assert bg.net_theta_dollars_per_day == -12.0
    assert bg.options_count == 1


def test_book_greeks_protective_collar_pattern():
    """Long stock + long put + short call = collar.

    100 shares + long 1 put (delta -0.4) + short 1 call (delta 0.5) →
    net delta = 100 + (-40) + (-50) = 10.
    Vega: long put +vega + short call −vega = small net.
    """
    long_put = OptionPosition(symbol="NVDA", right="put", strike=180.0,
                              expiry="2026-06-19", quantity=1, avg_cost=2.50)
    short_call = OptionPosition(symbol="NVDA", right="call", strike=220.0,
                                expiry="2026-06-19", quantity=-1, avg_cost=3.00)
    chain = [
        _chain_row(type="put", strike=180.0, delta=-0.40, vega=0.07, theta=-0.08),
        _chain_row(type="call", strike=220.0, delta=0.50, vega=0.08, theta=-0.10),
    ]
    enriched = enrich_with_chain([long_put, short_call], chain)
    bg = book_greeks(enriched, shares=100)
    assert bg.net_share_equivalent_delta == 100 + (-40) + (-50)  # = 10
    # Long put vega +7, short call vega -8 → net -1
    assert bg.net_vega_dollars_per_vol_pt == pytest.approx(-1.0, abs=1e-6)
    # Long put theta -8, short call theta +10 → net +2
    assert bg.net_theta_dollars_per_day == pytest.approx(2.0, abs=1e-6)


def test_book_greeks_aggregates_multiple_legs():
    p1 = OptionPosition(symbol="NVDA", right="call", strike=200.0,
                        expiry="2026-06-19", quantity=2, avg_cost=4.00)
    p2 = OptionPosition(symbol="NVDA", right="call", strike=220.0,
                        expiry="2026-06-19", quantity=-1, avg_cost=2.00)
    chain = [
        _chain_row(strike=200.0, delta=0.55),
        _chain_row(strike=220.0, delta=0.30),
    ]
    enriched = enrich_with_chain([p1, p2], chain)
    bg = book_greeks(enriched, shares=0)
    # 2 × 0.55 × 100 + (-1) × 0.30 × 100 = 110 - 30 = 80
    assert bg.net_share_equivalent_delta == 80.0
    assert bg.options_count == 2


def test_book_greeks_unrealized_pnl_long_position():
    pos = OptionPosition(symbol="NVDA", right="call", strike=200.0,
                         expiry="2026-06-19", quantity=1, avg_cost=4.00)
    # Bought at 4.00, now marked at 6.50 → profit $2.50 × 100 = $250.
    enriched = enrich_with_chain([pos], [_chain_row(mid=6.50)])
    bg = book_greeks(enriched, shares=0)
    assert bg.net_option_unrealized_pnl == 250.0


def test_book_greeks_unrealized_pnl_short_position():
    pos = OptionPosition(symbol="NVDA", right="put", strike=180.0,
                         expiry="2026-06-19", quantity=-1, avg_cost=3.00)
    # Sold at 3.00, mark dropped to 1.20 → profit $1.80 × 100 = $180.
    enriched = enrich_with_chain(
        [pos], [_chain_row(type="put", strike=180.0, mid=1.20)],
    )
    bg = book_greeks(enriched, shares=0)
    assert bg.net_option_unrealized_pnl == 180.0


def test_book_greeks_handles_missing_chain_data_gracefully():
    pos = OptionPosition(symbol="NVDA", right="call", strike=300.0,
                         expiry="2026-06-19", quantity=1, avg_cost=4.00)
    # No matching chain row → all enrichment fields are None.
    enriched = enrich_with_chain([pos], [])
    bg = book_greeks(enriched, shares=100)
    # Shares contribute; option delta is None so it doesn't enter the sum.
    assert bg.net_share_equivalent_delta == 100.0
    assert bg.net_vega_dollars_per_vol_pt is None
    assert bg.options_count == 1


# ---------- format_option_positions_section ----------


from datetime import date as _date

from portfolio.signals import format_option_positions_section


def test_render_empty_yields_empty_string():
    out = format_option_positions_section(
        options_by_symbol={},
        enriched_by_symbol={},
        book_greeks_by_symbol={},
        as_of=_date(2026, 5, 26),
    )
    assert out == ""


def test_render_single_long_call_shows_in_table():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-19", quantity=1, avg_cost=5.50,
    )
    enriched = enrich_with_chain(
        [pos],
        [{"type": "call", "strike": 200.0, "expiry": "2026-06-19",
          "mid": 6.20, "delta": 0.55, "gamma": 0.018,
          "theta": -0.10, "vega": 0.085, "implied_volatility": 0.45}],
    )
    bg = book_greeks(enriched, shares=100)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [pos]},
        enriched_by_symbol={"NVDA": enriched},
        book_greeks_by_symbol={"NVDA": bg},
        as_of=_date(2026, 5, 26),
    )
    assert "## Option positions" in out
    assert "### NVDA" in out
    assert "Long | call | $200" in out
    assert "100 shares" in out
    # Net delta = 100 + 1*0.55*100 = 155
    assert "+155 sh" in out


def test_render_warns_on_short_dte():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-06-01", quantity=1, avg_cost=5.50,
    )
    enriched = enrich_with_chain([pos], [])
    bg = book_greeks(enriched, shares=0)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [pos]},
        enriched_by_symbol={"NVDA": enriched},
        book_greeks_by_symbol={"NVDA": bg},
        as_of=_date(2026, 5, 28),  # 4 days to expiry
    )
    assert "⚠️" in out
    assert "Expiry within" in out


def test_render_warns_on_expired_position():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-04-19", quantity=1, avg_cost=5.50,
    )
    enriched = enrich_with_chain([pos], [])
    bg = book_greeks(enriched, shares=0)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [pos]},
        enriched_by_symbol={"NVDA": enriched},
        book_greeks_by_symbol={"NVDA": bg},
        as_of=_date(2026, 5, 26),
    )
    assert "Already expired" in out


def test_render_stop_out_warning_for_long_premium_decay():
    pos = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-08-15", quantity=1, avg_cost=10.00,
    )
    # Marked at 4.00 — 60% premium loss.
    enriched = enrich_with_chain(
        [pos],
        [{"type": "call", "strike": 200.0, "expiry": "2026-08-15",
          "mid": 4.00, "delta": 0.30}],
    )
    bg = book_greeks(enriched, shares=0)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [pos]},
        enriched_by_symbol={"NVDA": enriched},
        book_greeks_by_symbol={"NVDA": bg},
        as_of=_date(2026, 5, 26),
    )
    assert "premium loss" in out
    assert "consider stop" in out


def test_render_book_aggregate_section():
    p1 = OptionPosition(
        symbol="NVDA", right="call", strike=200.0,
        expiry="2026-08-15", quantity=1, avg_cost=5.00,
    )
    p2 = OptionPosition(
        symbol="AAPL", right="put", strike=180.0,
        expiry="2026-08-15", quantity=-1, avg_cost=2.50,
    )
    e1 = enrich_with_chain(
        [p1], [{"type": "call", "strike": 200.0, "expiry": "2026-08-15",
                "mid": 5.00, "delta": 0.50, "vega": 0.085, "theta": -0.05}],
    )
    e2 = enrich_with_chain(
        [p2], [{"type": "put", "strike": 180.0, "expiry": "2026-08-15",
                "mid": 2.50, "delta": -0.40, "vega": 0.07, "theta": 0.04}],
    )
    bg_nvda = book_greeks(e1, shares=100)
    bg_aapl = book_greeks(e2, shares=50)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [p1], "AAPL": [p2]},
        enriched_by_symbol={"NVDA": e1, "AAPL": e2},
        book_greeks_by_symbol={"NVDA": bg_nvda, "AAPL": bg_aapl},
        as_of=_date(2026, 5, 26),
    )
    assert "Book aggregate" in out
    # Net delta across both names: NVDA (100+50=150) + AAPL (50+40=90) = 240
    assert "+240 sh" in out


def test_render_handles_missing_chain_gracefully():
    """When fetch_current_chain returns [], enrichment is empty → render
    must still show the position with em-dashes for Greeks."""
    pos = OptionPosition(
        symbol="NVDA", right="put", strike=180.0,
        expiry="2026-06-19", quantity=-1, avg_cost=3.20,
    )
    enriched = enrich_with_chain([pos], [])  # no chain data
    bg = book_greeks(enriched, shares=0)
    out = format_option_positions_section(
        options_by_symbol={"NVDA": [pos]},
        enriched_by_symbol={"NVDA": enriched},
        book_greeks_by_symbol={"NVDA": bg},
        as_of=_date(2026, 5, 26),
    )
    assert "### NVDA" in out
    assert "Short | put" in out
    # Em-dashes for missing market data
    assert "—" in out
