"""Black-Scholes pricer and implied-vol inverter (pure Python, no scipy).

Designed to reconstruct historical implied volatility from observed option
close prices when the Polygon Options Starter plan only exposes price aggs
(no historical IV endpoint). The IV inverter uses Newton-Raphson on vega
with a bisection fallback so it converges robustly for OTM/ITM/edge cases.

All functions accept and return regular Python floats. Conventions:
- T (time to expiry) in years (e.g. 30/365)
- sigma in absolute terms (0.30 = 30% annualized vol)
- r (risk-free) in absolute terms (0.045 = 4.5%)
- q (continuous dividend yield) in absolute terms; default 0
- kind ∈ {"call", "put"}
"""

from __future__ import annotations

import math

# Bracket [LO, HI] for IV. Below LO the model is essentially intrinsic-only;
# above HI is past the practical range of equity options (500% annualized).
_IV_LO = 1e-4
_IV_HI = 5.0
_PRICE_EPS = 1e-8


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    sigma: float,
    kind: str,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton price for a European call or put."""
    if not (spot > 0 and strike > 0 and time_to_expiry > 0 and sigma > 0):
        return 0.0
    d1, d2 = _d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield
    )
    discount = math.exp(-risk_free_rate * time_to_expiry)
    div_discount = math.exp(-dividend_yield * time_to_expiry)
    if kind == "call":
        return spot * div_discount * _Phi(d1) - strike * discount * _Phi(d2)
    if kind == "put":
        return strike * discount * _Phi(-d2) - spot * div_discount * _Phi(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_vega(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> float:
    """Vega (∂price/∂sigma), same for call and put."""
    if not (spot > 0 and strike > 0 and time_to_expiry > 0 and sigma > 0):
        return 0.0
    d1, _ = _d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield
    )
    div_discount = math.exp(-dividend_yield * time_to_expiry)
    return spot * div_discount * math.sqrt(time_to_expiry) * _phi(d1)


def _intrinsic(
    *, spot: float, strike: float, kind: str,
    time_to_expiry: float, risk_free_rate: float, dividend_yield: float,
) -> float:
    """Lower bound on European option price (no-arb intrinsic)."""
    disc = math.exp(-risk_free_rate * time_to_expiry)
    div = math.exp(-dividend_yield * time_to_expiry)
    if kind == "call":
        return max(0.0, spot * div - strike * disc)
    return max(0.0, strike * disc - spot * div)


def implied_vol(
    *,
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    kind: str,
    dividend_yield: float = 0.0,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> float | None:
    """Invert Black-Scholes to recover sigma from an observed option price.

    Returns `None` when:
    - inputs are non-positive,
    - the observed price violates the no-arb intrinsic bound,
    - or the solver can't bracket a root in [_IV_LO, _IV_HI].

    Strategy: Newton-Raphson on vega (typically 3-6 iterations) starting at a
    reasonable initial guess; falls back to bisection on the same bracket if
    Newton diverges or vega goes near zero.
    """
    if not (price > 0 and spot > 0 and strike > 0 and time_to_expiry > 0):
        return None
    intrinsic = _intrinsic(
        spot=spot, strike=strike, kind=kind,
        time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    if price < intrinsic - _PRICE_EPS:
        return None  # price below intrinsic ⇒ no real IV
    # An OK starting guess is sqrt(2π/T) * |price - intrinsic|/S
    # (Brenner-Subrahmanyam approximation, works for ATM near-the-money).
    try:
        sigma = math.sqrt(2.0 * math.pi / time_to_expiry) * (
            (price - intrinsic) / spot
        )
    except (ValueError, ZeroDivisionError):
        sigma = 0.3
    sigma = max(_IV_LO, min(_IV_HI, sigma if sigma > 0 else 0.3))

    # Newton-Raphson on vega.
    for _ in range(max_iter):
        try:
            modelled = bs_price(
                spot=spot, strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate, sigma=sigma,
                kind=kind, dividend_yield=dividend_yield,
            )
            diff = modelled - price
        except (ValueError, OverflowError):
            break
        if abs(diff) < tol:
            return sigma
        v = bs_vega(
            spot=spot, strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate, sigma=sigma,
            dividend_yield=dividend_yield,
        )
        if v < 1e-8:
            break  # vega vanished — drop to bisection
        step = diff / v
        new_sigma = sigma - step
        if not (_IV_LO < new_sigma < _IV_HI) or not math.isfinite(new_sigma):
            break  # stepped out of bracket — drop to bisection
        sigma = new_sigma
    return _bisect_iv(
        price=price, spot=spot, strike=strike,
        time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate,
        kind=kind, dividend_yield=dividend_yield, tol=tol,
    )


def _bisect_iv(
    *,
    price: float, spot: float, strike: float, time_to_expiry: float,
    risk_free_rate: float, kind: str, dividend_yield: float, tol: float,
) -> float | None:
    """Bisection on sigma over [_IV_LO, _IV_HI]. Last-resort solver."""
    def f(s: float) -> float:
        return bs_price(
            spot=spot, strike=strike, time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate, sigma=s,
            kind=kind, dividend_yield=dividend_yield,
        ) - price

    lo, hi = _IV_LO, _IV_HI
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        # Price not bracketable — outside the [LO, HI] vol range.
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if fmid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
