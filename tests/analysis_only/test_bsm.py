from __future__ import annotations

import math

import pytest

from tradingagents.analysis_only.bsm import (
    bs_price,
    bs_vega,
    implied_vol,
)


# ---------- bs_price ----------


def test_atm_call_known_value():
    # Spot=100, K=100, T=0.25y, r=4.5%, sigma=30%, q=0
    # Expected ≈ 6.5 (computed once with reference implementation).
    px = bs_price(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    )
    assert 6.0 < px < 7.0


def test_atm_put_known_value():
    # Same setup as ATM call; put should be lower by S - K*exp(-r*T)*1 + 0 ≈ 1.12.
    call = bs_price(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    )
    put = bs_price(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30, kind="put",
    )
    # Put-call parity: C - P = S - K*exp(-rT)
    parity_rhs = 100 - 100 * math.exp(-0.045 * 0.25)
    assert math.isclose(call - put, parity_rhs, abs_tol=1e-6)


def test_deep_itm_call_near_intrinsic():
    # K=80, S=100, T tiny — intrinsic = 20.
    px = bs_price(
        spot=100, strike=80, time_to_expiry=0.01,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    )
    assert 19.5 < px < 20.5


def test_deep_otm_call_near_zero():
    px = bs_price(
        spot=100, strike=200, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    )
    assert px < 1e-3


def test_zero_or_negative_inputs_return_zero():
    assert bs_price(
        spot=0, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    ) == 0.0
    assert bs_price(
        spot=100, strike=100, time_to_expiry=0.0,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    ) == 0.0
    assert bs_price(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.0, kind="call",
    ) == 0.0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        bs_price(
            spot=100, strike=100, time_to_expiry=0.25,
            risk_free_rate=0.045, sigma=0.30, kind="straddle",
        )


# ---------- bs_vega ----------


def test_atm_vega_positive():
    v = bs_vega(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30,
    )
    assert v > 0


def test_deep_otm_vega_smaller_than_atm():
    atm = bs_vega(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30,
    )
    otm = bs_vega(
        spot=100, strike=140, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30,
    )
    assert 0 < otm < atm


# ---------- implied_vol round-trip ----------


@pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma_in", [0.15, 0.30, 0.55, 0.90])
def test_iv_round_trip_recovers_sigma(strike, kind, sigma_in):
    price = bs_price(
        spot=100, strike=strike, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=sigma_in, kind=kind,
    )
    # Tiny prices (deep OTM short-dated) below the no-arb intrinsic floor
    # cannot be inverted — skip those rather than asserting None for noise.
    if price < 1e-6:
        return
    sigma_out = implied_vol(
        price=price, spot=100, strike=strike, time_to_expiry=0.25,
        risk_free_rate=0.045, kind=kind,
    )
    assert sigma_out is not None
    assert math.isclose(sigma_out, sigma_in, abs_tol=1e-3)


def test_iv_round_trip_short_dated():
    # 7-day option, ATM, 40% IV — common stress case.
    price = bs_price(
        spot=100, strike=100, time_to_expiry=7 / 365.0,
        risk_free_rate=0.045, sigma=0.40, kind="call",
    )
    sigma_out = implied_vol(
        price=price, spot=100, strike=100, time_to_expiry=7 / 365.0,
        risk_free_rate=0.045, kind="call",
    )
    assert math.isclose(sigma_out, 0.40, abs_tol=1e-3)


def test_iv_round_trip_long_dated():
    # LEAPS-ish: 1 year, ATM, 30%.
    price = bs_price(
        spot=100, strike=100, time_to_expiry=1.0,
        risk_free_rate=0.045, sigma=0.30, kind="call",
    )
    sigma_out = implied_vol(
        price=price, spot=100, strike=100, time_to_expiry=1.0,
        risk_free_rate=0.045, kind="call",
    )
    assert math.isclose(sigma_out, 0.30, abs_tol=1e-3)


# ---------- implied_vol edge cases ----------


def test_iv_below_intrinsic_returns_none():
    # Asking for IV of a 50-dollar premium on an OTM call where intrinsic is 0
    # but the modeled price at any reasonable IV is < 50. This forces price >
    # all model prices, but that hits the hi-bracket. The clean "no answer"
    # case is price BELOW intrinsic, which is impossible by no-arb but
    # callers might pass garbage data.
    iv = implied_vol(
        price=-5.0, spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, kind="call",
    )
    assert iv is None


def test_iv_zero_price_returns_none():
    iv = implied_vol(
        price=0.0, spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, kind="call",
    )
    assert iv is None


def test_iv_invalid_inputs_return_none():
    # Zero time-to-expiry.
    assert implied_vol(
        price=1.0, spot=100, strike=100, time_to_expiry=0.0,
        risk_free_rate=0.045, kind="call",
    ) is None
    # Zero spot.
    assert implied_vol(
        price=1.0, spot=0, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, kind="call",
    ) is None


def test_iv_inverts_deep_itm_call():
    # 70-strike ATM 100 spot — call worth ~30 even at zero vol due to intrinsic.
    price = bs_price(
        spot=100, strike=70, time_to_expiry=0.5,
        risk_free_rate=0.045, sigma=0.25, kind="call",
    )
    iv = implied_vol(
        price=price, spot=100, strike=70, time_to_expiry=0.5,
        risk_free_rate=0.045, kind="call",
    )
    assert math.isclose(iv, 0.25, abs_tol=2e-3)


def test_iv_inverts_deep_itm_put():
    price = bs_price(
        spot=100, strike=130, time_to_expiry=0.5,
        risk_free_rate=0.045, sigma=0.25, kind="put",
    )
    iv = implied_vol(
        price=price, spot=100, strike=130, time_to_expiry=0.5,
        risk_free_rate=0.045, kind="put",
    )
    assert math.isclose(iv, 0.25, abs_tol=2e-3)


def test_iv_with_dividend_yield_round_trip():
    # Index option with q=2%.
    price = bs_price(
        spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, sigma=0.30,
        kind="call", dividend_yield=0.02,
    )
    iv = implied_vol(
        price=price, spot=100, strike=100, time_to_expiry=0.25,
        risk_free_rate=0.045, kind="call", dividend_yield=0.02,
    )
    assert math.isclose(iv, 0.30, abs_tol=1e-3)
