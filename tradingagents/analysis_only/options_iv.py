"""Single-snapshot IV-surface features derived from a normalized option chain.

Inputs are the same `contracts` list produced by
`AnalysisOnlyMVP._load_option_chain_for_strategies` — each item carries
`type`, `expiry`, `dte`, `strike`, `bid/ask/mid`, `implied_volatility`,
`delta`, `spot_distance_pct`.

Outputs are pure-Python primitives ready to drop into
`report.key_features.options_iv`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


TRADING_DAYS_PER_YEAR = 252

# ATM bucket = contracts within this fraction of spot.
_ATM_STRIKE_BAND = 0.05

# Tenor windows for the IV term structure (DTE inclusive).
_TENOR_WINDOWS: dict[str, tuple[int, int]] = {
    "atm_iv_30d": (14, 45),
    "atm_iv_60d": (46, 80),
    "atm_iv_90d": (81, 120),
}

# 25-delta skew window.
_SKEW_DTE_WINDOW = (20, 45)
_SKEW_TARGET_DELTA = 0.25
_SKEW_OTM_FALLBACK_PCT = 0.05  # used when Greeks missing

# Implied / realized regime thresholds.
_IV_RICH_THRESHOLD = 1.30
_IV_CHEAP_THRESHOLD = 0.90


def compute_iv_surface(
    *,
    contracts: list[dict[str, Any]] | None,
    spot: float | None,
    realized_vol_daily_20d: float | None,
    earnings_in_30_days: bool = False,
    next_earnings_dte: int | None = None,
) -> dict[str, Any]:
    """Return a dict of IV-surface features. Always returns a `status` key.

    `status` is `"ok"` when at least one tenor's ATM IV was computable,
    otherwise `"unavailable"` with a `reason`. Missing sub-features are
    `None` rather than absent so consumers can iterate fields uniformly.
    """
    base: dict[str, Any] = {
        "status": "unavailable",
        "atm_iv_30d": None,
        "atm_iv_60d": None,
        "atm_iv_90d": None,
        "term_structure_slope_30_to_60": None,
        "term_structure_is_backwardation": None,
        "skew_25d_30d": None,
        "skew_25d_30d_put_iv": None,
        "skew_25d_30d_call_iv": None,
        "realized_vol_annual_20d": None,
        "implied_realized_ratio": None,
        "implied_realized_signal": None,
        "earnings_implied_move": None,
        "earnings_implied_move_expiry": None,
    }
    if not _finite(spot):
        base["reason"] = "Spot price unavailable."
        return base
    if not contracts:
        base["reason"] = "Option chain empty."
        return base

    spot_f = float(spot or 0.0)

    atm_30 = _atm_iv_for_window(contracts, spot_f, _TENOR_WINDOWS["atm_iv_30d"])
    atm_60 = _atm_iv_for_window(contracts, spot_f, _TENOR_WINDOWS["atm_iv_60d"])
    atm_90 = _atm_iv_for_window(contracts, spot_f, _TENOR_WINDOWS["atm_iv_90d"])

    if not any(_finite(v) for v in (atm_30, atm_60, atm_90)):
        base["reason"] = "No ATM IV computable in any tenor window."
        return base

    base["atm_iv_30d"] = _round_finite(atm_30)
    base["atm_iv_60d"] = _round_finite(atm_60)
    base["atm_iv_90d"] = _round_finite(atm_90)

    if _finite(atm_30) and _finite(atm_60) and atm_30 > 0:
        slope = (atm_60 - atm_30) / atm_30
        base["term_structure_slope_30_to_60"] = _round_finite(slope)
        base["term_structure_is_backwardation"] = bool(atm_60 < atm_30)

    skew = _compute_25d_skew(contracts, spot_f)
    base["skew_25d_30d"] = _round_finite(skew["skew"])
    base["skew_25d_30d_put_iv"] = _round_finite(skew["put_iv"])
    base["skew_25d_30d_call_iv"] = _round_finite(skew["call_iv"])

    if _finite(realized_vol_daily_20d):
        rv_annual = float(realized_vol_daily_20d or 0.0) * math.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        base["realized_vol_annual_20d"] = _round_finite(rv_annual)
        if _finite(atm_30) and rv_annual > 0:
            ratio = float(atm_30 or 0.0) / rv_annual
            base["implied_realized_ratio"] = _round_finite(ratio)
            if ratio >= _IV_RICH_THRESHOLD:
                base["implied_realized_signal"] = "iv_rich"
            elif ratio <= _IV_CHEAP_THRESHOLD:
                base["implied_realized_signal"] = "iv_cheap"
            else:
                base["implied_realized_signal"] = "neutral"

    if earnings_in_30_days:
        ed_move = _earnings_implied_move(
            contracts,
            spot_f,
            next_earnings_dte=next_earnings_dte,
        )
        if ed_move:
            base["earnings_implied_move"] = _round_finite(ed_move["move"])
            base["earnings_implied_move_expiry"] = ed_move["expiry"]

    base["status"] = "ok"
    return base


def compute_iv_history_features(
    *,
    current_atm_iv_30d: float | None,
    history: list[dict[str, Any]] | None,
    min_observations: int = 20,
) -> dict[str, Any]:
    """Derive `iv_rank` and `iv_percentile` from a trailing IV history.

    `iv_rank` is the position of `current_atm_iv_30d` within the trailing
    min/max range, in [0, 1] (rank-of-extremes view).
    `iv_percentile` is the fraction of historical observations ≤ current
    value (rank-of-distribution view). Robust to outliers.

    Returns `iv_history_status: "ok" | "insufficient_history" | "unavailable"`
    plus the metric pair when computable.
    """
    out: dict[str, Any] = {
        "iv_rank_252d": None,
        "iv_percentile_252d": None,
        "iv_history_observations": 0,
        "iv_history_status": "unavailable",
    }
    if not _finite(current_atm_iv_30d):
        out["iv_history_status"] = "unavailable"
        return out
    series = [
        float(r.get("atm_iv_30d"))
        for r in (history or [])
        if _finite(r.get("atm_iv_30d"))
    ]
    out["iv_history_observations"] = len(series)
    if len(series) < min_observations:
        out["iv_history_status"] = "insufficient_history"
        return out
    lo, hi = min(series), max(series)
    current = float(current_atm_iv_30d or 0.0)
    if hi > lo:
        rank = (current - lo) / (hi - lo)
        out["iv_rank_252d"] = _round_finite(max(0.0, min(1.0, rank)))
    else:
        out["iv_rank_252d"] = 0.5
    le_count = sum(1 for v in series if v <= current)
    out["iv_percentile_252d"] = _round_finite(le_count / len(series))
    out["iv_history_status"] = "ok"
    return out


# ---------- internals ----------


def _atm_iv_for_window(
    contracts: Iterable[dict[str, Any]],
    spot: float,
    dte_window: tuple[int, int],
) -> float | None:
    """Mean of put+call IV at the strike closest to spot, for the expiry
    inside the window with the most ATM coverage. Skips contracts with
    `|spot_distance_pct| > _ATM_STRIKE_BAND`."""
    by_expiry: dict[str, list[dict[str, Any]]] = {}
    lo, hi = dte_window
    for c in contracts:
        dte = c.get("dte")
        iv = c.get("implied_volatility")
        if not isinstance(dte, (int, float)) or not _finite(iv):
            continue
        if dte < lo or dte > hi:
            continue
        dist = _abs_strike_distance(c, spot)
        if dist is None or dist > _ATM_STRIKE_BAND:
            continue
        expiry = str(c.get("expiry") or "")
        if not expiry:
            continue
        by_expiry.setdefault(expiry, []).append(c)

    if not by_expiry:
        return None

    # Prefer the expiry with both a call and a put present (more reliable
    # ATM); break ties by greatest coverage.
    def _expiry_score(items: list[dict[str, Any]]) -> tuple[int, int]:
        types = {str(i.get("type")) for i in items}
        both_sides = 1 if {"call", "put"}.issubset(types) else 0
        return (both_sides, len(items))

    best_expiry = max(by_expiry.items(), key=lambda kv: _expiry_score(kv[1]))[0]
    rows = by_expiry[best_expiry]

    calls = [r for r in rows if r.get("type") == "call"]
    puts = [r for r in rows if r.get("type") == "put"]
    call_iv = _nearest_strike_iv(calls, spot)
    put_iv = _nearest_strike_iv(puts, spot)
    candidates = [v for v in (call_iv, put_iv) if _finite(v)]
    if not candidates:
        return None
    return sum(candidates) / len(candidates)


def _nearest_strike_iv(
    contracts: list[dict[str, Any]], spot: float
) -> float | None:
    best: tuple[float, float] | None = None
    for c in contracts:
        iv = c.get("implied_volatility")
        if not _finite(iv):
            continue
        dist = _abs_strike_distance(c, spot)
        if dist is None:
            continue
        if best is None or dist < best[0]:
            best = (dist, float(iv or 0.0))
    return best[1] if best else None


def _abs_strike_distance(contract: dict[str, Any], spot: float) -> float | None:
    pct = contract.get("spot_distance_pct")
    if _finite(pct):
        return abs(float(pct or 0.0))
    strike = contract.get("strike")
    if _finite(strike) and spot > 0:
        return abs((float(strike or 0.0) - spot) / spot)
    return None


def _compute_25d_skew(
    contracts: Iterable[dict[str, Any]],
    spot: float,
) -> dict[str, float | None]:
    lo, hi = _SKEW_DTE_WINDOW
    pool = [
        c for c in contracts
        if isinstance(c.get("dte"), (int, float))
        and lo <= int(c["dte"]) <= hi
        and _finite(c.get("implied_volatility"))
    ]
    call_iv = _select_delta_iv(
        [c for c in pool if c.get("type") == "call"],
        target_delta=_SKEW_TARGET_DELTA,
        otm_fallback_strike_mult=1.0 + _SKEW_OTM_FALLBACK_PCT,
        spot=spot,
    )
    put_iv = _select_delta_iv(
        [c for c in pool if c.get("type") == "put"],
        target_delta=-_SKEW_TARGET_DELTA,
        otm_fallback_strike_mult=1.0 - _SKEW_OTM_FALLBACK_PCT,
        spot=spot,
    )
    skew = (
        (put_iv - call_iv)
        if _finite(call_iv) and _finite(put_iv)
        else None
    )
    return {"skew": skew, "call_iv": call_iv, "put_iv": put_iv}


def _select_delta_iv(
    contracts: list[dict[str, Any]],
    *,
    target_delta: float,
    otm_fallback_strike_mult: float,
    spot: float,
) -> float | None:
    with_delta = [c for c in contracts if _finite(c.get("delta"))]
    if with_delta:
        best = min(
            with_delta,
            key=lambda c: abs(float(c.get("delta") or 0.0) - target_delta),
        )
        return float(best.get("implied_volatility") or 0.0)
    if not contracts or spot <= 0:
        return None
    target_strike = spot * otm_fallback_strike_mult
    best = min(
        contracts,
        key=lambda c: abs(float(c.get("strike") or 0.0) - target_strike),
    )
    return float(best.get("implied_volatility") or 0.0)


def _earnings_implied_move(
    contracts: list[dict[str, Any]],
    spot: float,
    *,
    next_earnings_dte: int | None,
) -> dict[str, Any] | None:
    """ATM straddle / spot at the first expiry on/after earnings."""
    if spot <= 0:
        return None
    earnings_dte = next_earnings_dte if isinstance(next_earnings_dte, int) else 0
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for c in contracts:
        dte = c.get("dte")
        if not isinstance(dte, (int, float)):
            continue
        # Want expiries on/just-after earnings, capped at ~45d post-earnings.
        if dte < max(0, earnings_dte) or dte > earnings_dte + 45:
            continue
        expiry = str(c.get("expiry") or "")
        if not expiry:
            continue
        dist = _abs_strike_distance(c, spot)
        if dist is None or dist > _ATM_STRIKE_BAND:
            continue
        side = str(c.get("type") or "")
        if side not in {"call", "put"}:
            continue
        slot = candidates.setdefault(expiry, {})
        existing = slot.get(side)
        if existing is None or (
            (_abs_strike_distance(existing, spot) or 1e9) > dist
        ):
            slot[side] = c

    paired = [
        (exp, sides)
        for exp, sides in candidates.items()
        if "call" in sides and "put" in sides
    ]
    if not paired:
        return None
    paired.sort(key=lambda kv: int(kv[1]["call"].get("dte") or 0))
    expiry, sides = paired[0]
    call_mid = _mid_or_iv_value(sides["call"])
    put_mid = _mid_or_iv_value(sides["put"])
    if not (_finite(call_mid) and _finite(put_mid)):
        return None
    straddle = float(call_mid or 0.0) + float(put_mid or 0.0)
    if straddle <= 0:
        return None
    return {"move": straddle / spot, "expiry": expiry}


def _mid_or_iv_value(contract: dict[str, Any]) -> float | None:
    for key in ("mid", "mark", "last"):
        v = contract.get(key)
        if _finite(v) and float(v or 0.0) > 0:
            return float(v or 0.0)
    return None


def _finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _round_finite(v: Any, ndigits: int = 6) -> float | None:
    if not _finite(v):
        return None
    return round(float(v or 0.0), ndigits)
