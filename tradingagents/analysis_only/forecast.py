"""Pure forecast + trade-plan helpers.

Like `scoring.py`, these are deterministic and free of I/O. The pipeline
orchestrator passes already-extracted features in and gets back JSON-
serializable dicts.
"""

from __future__ import annotations

from typing import Any
import math


# Standard horizons in trading days.
DEFAULT_HORIZONS: dict[str, int] = {"1w": 5, "1m": 21, "3m": 63}


def estimate_price_ranges(
    spot: float | None,
    vol_daily: float | None,
    atr_14: float | None,
    ret_20d: float | None,
    composite_score: float,
    *,
    implied_vol_annual: float | None = None,
    event_risk_multiplier: float = 1.0,
    horizons: dict[str, int] | None = None,
    drift_clip: float = 0.006,
    composite_drift_clip: float = 0.001,
) -> dict[str, Any]:
    """Blend realized vol, ATR, and implied vol into per-horizon price bands."""
    if spot is None or not _finite(spot):
        return {}
    horizons = horizons or DEFAULT_HORIZONS

    base_daily_sigma = vol_daily if _finite(vol_daily) else 0.02
    atr_daily_pct: float | None = None
    if _finite(atr_14) and spot > 0:
        atr_daily_pct = atr_14 / spot

    daily_drift = 0.0
    if _finite(ret_20d):
        daily_drift = max(-drift_clip, min(drift_clip, ret_20d / 20.0))
    daily_drift += max(
        -composite_drift_clip,
        min(composite_drift_clip, composite_score * 0.0008),
    )

    z60, z80, z95 = 0.842, 1.282, 1.960
    results: dict[str, Any] = {}
    for label, days in horizons.items():
        move_vol = spot * base_daily_sigma * math.sqrt(days)
        if atr_daily_pct is not None:
            move_atr = spot * atr_daily_pct * math.sqrt(days)
        else:
            move_atr = move_vol

        if _finite(implied_vol_annual):
            move_iv = spot * implied_vol_annual * math.sqrt(days / 252.0)
            blended = (0.45 * move_vol) + (0.30 * move_atr) + (0.25 * move_iv)
        else:
            blended = (0.65 * move_vol) + (0.35 * move_atr)
        blended *= event_risk_multiplier

        center = spot * (1.0 + daily_drift * days)
        lower_60 = max(0.01, center - (z60 * blended))
        upper_60 = center + (z60 * blended)
        lower_80 = max(0.01, center - (z80 * blended))
        upper_80 = center + (z80 * blended)
        lower_95 = max(0.01, center - (z95 * blended))
        upper_95 = center + (z95 * blended)
        width_80 = (upper_80 - lower_80) / spot

        results[label] = {
            "trading_days": days,
            "center_price": _round(center),
            "lower_60": _round(lower_60),
            "upper_60": _round(upper_60),
            "lower_80": _round(lower_80),
            "upper_80": _round(upper_80),
            "lower_95": _round(lower_95),
            "upper_95": _round(upper_95),
            "range_width_pct_80": _round(width_80),
            "assumptions": {
                "realized_vol_daily": _round(base_daily_sigma),
                "atr_daily_pct": _round(atr_daily_pct),
                "implied_vol_annual": _round(implied_vol_annual),
                "event_risk_multiplier": _round(event_risk_multiplier),
            },
        }
    return results


def estimate_price_target_scenarios(
    *,
    spot: float | None,
    direction: str,
    confidence: float,
    composite_score: float,
    fundamentals: dict[str, Any],
    technicals: dict[str, Any],
    price_range_forecast: dict[str, Any],
    analyst_consensus: dict[str, Any],
    competitor_analysis: dict[str, Any],
    earnings_calendar: dict[str, Any],
    market_context: dict[str, Any],
    industry_news_context: dict[str, Any],
    options_flow: dict[str, Any],
) -> dict[str, Any]:
    """Build base/bull/bear scenario targets from available structured data.

    This is not a point forecast. It blends valuation anchors, sell-side
    targets, volatility ranges, and regime context into a transparent scenario
    block that can be rendered and audited.
    """
    if not _finite(spot) or float(spot) <= 0:
        return {
            "status": "unavailable",
            "reason": "Spot price unavailable.",
            "method": "hybrid_relative_valuation_momentum",
        }

    spot_f = float(spot)
    horizon_label, horizon_block = _select_target_horizon(price_range_forecast)
    sources: list[dict[str, Any]] = []
    missing: list[str] = []

    targets = (analyst_consensus or {}).get("price_targets") or {}
    analyst_mean = _first_finite(targets.get("mean"), targets.get("median"))
    analyst_low = _first_finite(targets.get("low"))
    analyst_high = _first_finite(targets.get("high"))
    analyst_status = analyst_consensus.get("consensus_pit_status")
    if _finite(analyst_mean):
        sources.append(
            {
                "name": "analyst_consensus",
                "target": _bounded_price(analyst_mean, spot_f),
                "weight": 0.40,
                "pit_status": analyst_status,
                "detail": "Mean/median sell-side price target.",
            }
        )
    else:
        missing.append("Analyst price target distribution.")

    peer_summary = (competitor_analysis or {}).get("summary") or {}
    peer_forward_pe = _first_finite(peer_summary.get("peer_forward_pe_median"))
    peer_trailing_pe = _first_finite(peer_summary.get("peer_trailing_pe_median"))
    forward_pe = _first_finite((fundamentals or {}).get("forward_pe"))
    trailing_pe = _first_finite((fundamentals or {}).get("trailing_pe"))
    valuation_targets: list[float] = []
    if (
        _finite(peer_forward_pe)
        and _finite(forward_pe)
        and float(forward_pe) > 0
    ):
        valuation_targets.append(spot_f * float(peer_forward_pe) / float(forward_pe))
    if (
        _finite(peer_trailing_pe)
        and _finite(trailing_pe)
        and float(trailing_pe) > 0
    ):
        valuation_targets.append(
            spot_f * float(peer_trailing_pe) / float(trailing_pe)
        )
    if valuation_targets:
        sources.append(
            {
                "name": "peer_relative_valuation",
                "target": _bounded_price(
                    sum(valuation_targets) / len(valuation_targets), spot_f
                ),
                "weight": 0.25,
                "pit_status": competitor_analysis.get(
                    "peer_fundamentals_pit_status"
                ),
                "detail": "Peer multiple normalization using forward/trailing P/E.",
            }
        )
    else:
        missing.append("Peer forward/trailing valuation multiples.")

    forward_eps = _next_forward_eps_estimate(earnings_calendar)
    annualized_forward_eps = (
        float(forward_eps) * 4.0 if _finite(forward_eps) else None
    )
    if _finite(annualized_forward_eps) and _finite(forward_pe):
        eps_target = float(annualized_forward_eps) * float(forward_pe)
        if eps_target > 0:
            sources.append(
                {
                    "name": "forward_eps_estimate",
                    "target": _bounded_price(eps_target, spot_f),
                    "weight": 0.12,
                    "pit_status": earnings_calendar.get(
                        "forward_eps_estimates_pit_status"
                    ),
                    "detail": "Annualized upcoming EPS estimate multiplied by current forward P/E.",
                }
            )
    else:
        missing.append("Forward EPS estimate usable for valuation.")

    forecast_center = _first_finite(horizon_block.get("center_price"))
    if _finite(forecast_center):
        sources.append(
            {
                "name": "volatility_adjusted_range_center",
                "target": _bounded_price(forecast_center, spot_f),
                "weight": 0.18,
                "pit_status": "pit",
                "detail": f"{horizon_label} forecast center from price/volatility model.",
            }
        )
    else:
        missing.append("Volatility-adjusted forecast center.")

    growth_adjusted = _growth_adjusted_target(
        spot=spot_f,
        fundamentals=fundamentals,
        technicals=technicals,
        market_context=market_context,
        industry_news_context=industry_news_context,
        composite_score=composite_score,
    )
    sources.append(
        {
            "name": "growth_momentum_regime",
            "target": growth_adjusted,
            "weight": 0.15,
            "pit_status": "mixed",
            "detail": "Growth, momentum, market regime, and industry-news adjustment.",
        }
    )

    base = _weighted_target(sources, spot_f)
    vol_spread = _scenario_spread(
        spot=spot_f,
        horizon_block=horizon_block,
        options_flow=options_flow,
    )
    skew = max(-0.10, min(0.10, float(composite_score) * 0.08))
    if direction == "bullish":
        bull_mult = 1.0 + vol_spread + max(0.02, skew)
        bear_mult = 1.0 - (vol_spread * 0.75)
    elif direction == "bearish":
        bull_mult = 1.0 + (vol_spread * 0.75)
        bear_mult = 1.0 - vol_spread - max(0.02, abs(skew))
    else:
        bull_mult = 1.0 + (vol_spread * 0.85)
        bear_mult = 1.0 - (vol_spread * 0.85)

    bull = base * bull_mult
    bear = base * bear_mult
    if _finite(analyst_high):
        bull = (0.65 * bull) + (0.35 * _bounded_price(analyst_high, spot_f, 2.5))
    if _finite(analyst_low):
        bear = (0.65 * bear) + (0.35 * _bounded_price(analyst_low, spot_f, 2.5))
    bull = max(base, bull)
    bear = min(base, max(0.01, bear))

    coverage = _target_coverage(sources, analyst_consensus, competitor_analysis)
    target_confidence = max(
        0.15,
        min(
            0.90,
            (0.25 * coverage)
            + (0.45 * max(0.0, min(1.0, float(confidence))))
            + (0.20 * min(1.0, abs(float(composite_score))))
            + 0.10,
        ),
    )
    if len(sources) <= 2:
        target_confidence = min(target_confidence, 0.45)

    news_signals = _industry_news_signals(industry_news_context)
    drivers = _target_drivers(
        sources=sources,
        market_context=market_context,
        industry_news_signals=news_signals,
        earnings_calendar=earnings_calendar,
        options_flow=options_flow,
    )

    return {
        "status": "ok",
        "method": "hybrid_relative_valuation_momentum",
        "time_horizon": horizon_label,
        "spot": _round(spot_f),
        "base": _round(base),
        "bull": _round(bull),
        "bear": _round(bear),
        "base_upside_pct": _round((base / spot_f) - 1.0),
        "bull_upside_pct": _round((bull / spot_f) - 1.0),
        "bear_upside_pct": _round((bear / spot_f) - 1.0),
        "confidence": _round(target_confidence),
        "coverage": _round(coverage),
        "scenario_spread_pct": _round(vol_spread),
        "source_weights": [
            {
                "name": s.get("name"),
                "target": _round(s.get("target")),
                "weight": _round(s.get("weight")),
                "pit_status": s.get("pit_status"),
                "detail": s.get("detail"),
            }
            for s in sources
        ],
        "drivers": drivers,
        "risks": _target_risks(
            market_context=market_context,
            industry_news_signals=news_signals,
            earnings_calendar=earnings_calendar,
            options_flow=options_flow,
        ),
        "missing_data": missing,
        "extra_data": {
            "analyst_targets": targets,
            "next_quarter_eps_estimate": _round(forward_eps),
            "annualized_forward_eps_estimate": _round(
                annualized_forward_eps
            ),
            "forward_eps_pit_status": earnings_calendar.get(
                "forward_eps_estimates_pit_status"
            ),
            "peer_forward_pe_median": _round(peer_forward_pe),
            "peer_trailing_pe_median": _round(peer_trailing_pe),
            "options_atm_iv_30d": _round((options_flow or {}).get("atm_iv_30d")),
            "market_fear_greed": {
                "score": market_context.get("fear_greed_score"),
                "rating": market_context.get("fear_greed_rating"),
                "pit_status": market_context.get("fear_greed_pit_status"),
            },
            "industry_news_signals": news_signals,
        },
    }


def build_trade_plan(
    direction: str,
    confidence: float,
    composite_score: float,
    spot: float | None,
    price_range_forecast: dict[str, Any],
) -> dict[str, Any]:
    """Translate a directional call into staged entry / exit levels.

    Mirrors the legacy `_build_trade_plan` behavior. Kept here so the
    pipeline orchestrator just hands off direction + forecast and gets a
    JSON-serializable plan back.
    """
    if not _finite(spot):
        return {
            "strategy_profile": "insufficient_data",
            "position_sizing": {},
            "entry_strategy": [],
            "exit_strategy": [],
            "notes": ["Spot price unavailable."],
        }

    weekly = price_range_forecast.get("1w", {}) or {}
    monthly = price_range_forecast.get("1m", {}) or {}

    w_l60 = _round(weekly.get("lower_60")) or spot
    w_l80 = _round(weekly.get("lower_80")) or spot
    w_l95 = _round(weekly.get("lower_95")) or spot
    w_u60 = _round(weekly.get("upper_60")) or spot
    m_u60 = _round(monthly.get("upper_60")) or w_u60
    m_u80 = _round(monthly.get("upper_80")) or m_u60
    width_1w = _round(weekly.get("range_width_pct_80")) or 0.08

    target_position_pct = 8.0 + (confidence * 18.0)
    target_position_pct *= max(0.6, 1.0 - (width_1w * 2.0))
    target_position_pct = max(4.0, min(30.0, target_position_pct))

    entry_strategy: list[dict[str, Any]] = []
    exit_strategy: list[dict[str, Any]] = []
    notes: list[str] = []

    if direction == "bullish":
        strategy_profile = "long_accumulation"
        entries = [0.35, 0.35, 0.30] if confidence >= 0.7 else [0.20, 0.35, 0.45]
        entry_strategy = [
            {
                "allocation_pct_of_target_position": entries[0] * 100.0,
                "trigger_price_lte": _round(spot),
                "label": "starter",
            },
            {
                "allocation_pct_of_target_position": entries[1] * 100.0,
                "trigger_price_lte": w_l60,
                "label": "pullback_add",
            },
            {
                "allocation_pct_of_target_position": entries[2] * 100.0,
                "trigger_price_lte": w_l80,
                "label": "deep_pullback_add",
            },
        ]
        exit_strategy = [
            {
                "sell_pct_of_position": 25.0,
                "trigger_price_gte": w_u60,
                "label": "take_profit_1",
            },
            {
                "sell_pct_of_position": 35.0,
                "trigger_price_gte": m_u60,
                "label": "take_profit_2",
            },
            {
                "sell_pct_of_position": 40.0,
                "trigger_price_gte": m_u80,
                "label": "take_profit_3",
            },
            {
                "sell_pct_of_position": 100.0,
                "trigger_price_lte": w_l95,
                "label": "stop_loss",
            },
        ]
        notes.append("Scale in on weakness, scale out into strength.")
    elif direction == "bearish":
        strategy_profile = "defensive_reduce"
        entry_strategy = [
            {
                "allocation_pct_of_target_position": 0.0,
                "trigger_price_lte": _round(spot),
                "label": "no_new_longs",
            }
        ]
        exit_strategy = [
            {
                "sell_pct_of_position": 35.0,
                "trigger_price_gte": w_u60,
                "label": "sell_rallies_1",
            },
            {
                "sell_pct_of_position": 35.0,
                "trigger_price_lte": w_l80,
                "label": "risk_off_on_breakdown",
            },
            {
                "sell_pct_of_position": 30.0,
                "trigger_price_lte": w_l95,
                "label": "full_exit_protection",
            },
        ]
        notes.append("Bias is defensive; prioritize capital protection.")
    else:
        strategy_profile = "range_trade_small_size"
        entry_strategy = [
            {
                "allocation_pct_of_target_position": 30.0,
                "trigger_price_lte": w_l60,
                "label": "small_probe",
            },
            {
                "allocation_pct_of_target_position": 40.0,
                "trigger_price_lte": w_l80,
                "label": "range_low_add",
            },
            {
                "allocation_pct_of_target_position": 30.0,
                "trigger_price_lte": w_l95,
                "label": "extreme_range_add",
            },
        ]
        exit_strategy = [
            {
                "sell_pct_of_position": 40.0,
                "trigger_price_gte": w_u60,
                "label": "range_take_profit_1",
            },
            {
                "sell_pct_of_position": 60.0,
                "trigger_price_gte": m_u60,
                "label": "range_take_profit_2",
            },
        ]
        notes.append("Keep exposure smaller in neutral regime.")

    notes.append("Use staged orders and revisit plan after major events.")
    notes.append(
        f"Composite score={round(composite_score, 4)}, confidence={confidence}."
    )
    return {
        "strategy_profile": strategy_profile,
        "position_sizing": {
            "target_position_pct_of_portfolio": _round(target_position_pct),
            "max_single_entry_slippage_bps": 30,
        },
        "entry_strategy": entry_strategy,
        "exit_strategy": exit_strategy,
        "notes": notes,
    }


def build_decision_summary(
    *,
    direction: str,
    confidence: float,
    composite_score: float,
    coverage: float,
    spot: float | None,
    price_target: dict[str, Any],
    price_range_forecast: dict[str, Any],
    trade_plan: dict[str, Any],
    risk_flags: list[str],
    portfolio_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plain-English action layer for quick human scanning."""
    if not _finite(spot):
        return {
            "status": "unavailable",
            "action": "watch",
            "summary": "No action: spot price unavailable.",
        }

    spot_f = float(spot)
    base_upside = _first_finite((price_target or {}).get("base_upside_pct"))
    target_confidence = _first_finite((price_target or {}).get("confidence"))
    decision_confidence = max(
        0.0,
        min(
            1.0,
            (0.65 * float(confidence))
            + (0.20 * float(coverage))
            + (0.15 * (target_confidence if target_confidence is not None else 0.5)),
        ),
    )
    risk_penalty = min(0.12, len(risk_flags) * 0.025)
    upside_bonus = 0.0
    if base_upside is not None:
        upside_bonus = max(-0.08, min(0.12, base_upside * 0.35))
    win_probability = max(
        0.30,
        min(
            0.78,
            0.50
            + (abs(float(composite_score)) * 0.18)
            + ((decision_confidence - 0.50) * 0.22)
            + upside_bonus
            - risk_penalty,
        ),
    )

    weekly = (price_range_forecast or {}).get("1w") or {}
    monthly = (price_range_forecast or {}).get("1m") or {}
    lower_60 = _first_finite(weekly.get("lower_60"))
    lower_80 = _first_finite(weekly.get("lower_80"))
    lower_95 = _first_finite(weekly.get("lower_95"))
    upper_60 = _first_finite(weekly.get("upper_60"))
    monthly_upper_60 = _first_finite(monthly.get("upper_60"))
    buy_zone_low = lower_80 if lower_80 is not None else lower_60
    buy_zone_high = lower_60 if lower_60 is not None else spot_f
    add_below = lower_60 if lower_60 is not None else spot_f
    stop_loss = lower_95 if lower_95 is not None else None
    take_profit_1 = upper_60
    take_profit_2 = monthly_upper_60

    if (
        direction == "bullish"
        and decision_confidence >= 0.62
        and (base_upside is None or base_upside >= 0.08)
    ):
        action = "buy"
        label = "Buy / accumulate"
        summary = "Bullish setup with enough upside to justify staged buying."
    elif direction == "bearish" and decision_confidence >= 0.60:
        action = "sell"
        label = "Sell / reduce"
        summary = "Bearish setup: prioritize risk reduction over new exposure."
    elif direction == "neutral" or (base_upside is not None and base_upside < 0.05):
        action = "hold"
        label = "Hold / wait"
        summary = "Signal is not strong enough for a fresh aggressive entry."
    else:
        action = "watch"
        label = "Watch"
        summary = "Setup is mixed; wait for a better price or cleaner signal."

    portfolio_context = portfolio_context or {}
    portfolio_notes: list[str] = []
    holding = portfolio_context.get("holding") or {}
    account = portfolio_context.get("account") or {}
    portfolio_weight = _first_finite(holding.get("portfolio_weight"))
    margin_remaining = _first_finite(account.get("margin_remaining"))
    total_equity = _first_finite(account.get("total_equity"))
    short_put_util = _first_finite(account.get("short_put_margin_utilization"))
    has_position = bool(holding.get("has_position"))
    if has_position:
        portfolio_notes.append(
            "Existing position: "
            f"{_round(holding.get('shares'))} shares, "
            f"{_round((portfolio_weight or 0.0) * 100.0)}% of account equity."
        )
    if short_put_util is not None and short_put_util >= 0.75:
        portfolio_notes.append(
            f"Short puts reserve about {short_put_util * 100.0:.0f}% of margin capacity."
        )
    if margin_remaining is not None and total_equity is not None:
        portfolio_notes.append(
            f"Approx. remaining margin capacity: {_money(margin_remaining)}."
        )

    capital_constrained = (
        short_put_util is not None
        and short_put_util >= 0.75
    ) or (
        margin_remaining is not None
        and total_equity is not None
        and margin_remaining / max(total_equity, 1.0) < 0.10
    )
    concentrated = portfolio_weight is not None and portfolio_weight >= 0.15
    meaningful_existing = portfolio_weight is not None and portfolio_weight >= 0.04
    if action == "buy" and concentrated:
        action = "hold"
        label = "Hold / trim strength"
        summary = (
            "Signal is constructive, but this is already a concentrated "
            "holding; prefer holding or trimming into strength."
        )
    elif action == "buy" and capital_constrained and has_position:
        action = "hold"
        label = "Hold / no new capital"
        summary = (
            "Setup is bullish, but existing exposure and short-put margin use "
            "argue against adding unless price pulls into the buy zone."
        )
    elif action == "buy" and meaningful_existing:
        label = "Add selectively"
        summary = (
            "Bullish setup, but you already have meaningful exposure; add only "
            "on pullbacks instead of chasing."
        )
    elif action == "buy" and capital_constrained:
        action = "watch"
        label = "Watch / preserve buying power"
        summary = (
            "Setup is bullish, but margin is constrained by short puts; wait "
            "for a cleaner entry or free capital first."
        )

    rationale: list[str] = [
        f"Composite score {composite_score:+.2f}; model confidence {confidence:.2f}.",
    ]
    if base_upside is not None:
        rationale.append(f"Base target upside is {base_upside * 100.0:.1f}%.")
    if price_target.get("base") is not None:
        rationale.append(
            "Scenario target base/bull/bear: "
            f"{_money(price_target.get('base'))} / "
            f"{_money(price_target.get('bull'))} / "
            f"{_money(price_target.get('bear'))}."
        )
    if risk_flags:
        rationale.append(f"Risk flags: {', '.join(risk_flags[:3])}.")
    rationale.extend(portfolio_notes)

    return {
        "status": "ok",
        "action": action,
        "label": label,
        "summary": summary,
        "current_price": _round(spot_f),
        "estimated_win_probability": _round(win_probability),
        "confidence": _round(decision_confidence),
        "base_target": _round((price_target or {}).get("base")),
        "bull_target": _round((price_target or {}).get("bull")),
        "bear_target": _round((price_target or {}).get("bear")),
        "base_upside_pct": _round(base_upside),
        "portfolio_context": portfolio_context,
        "entry": {
            "starter_buy_at_or_below": _round(spot_f if action == "buy" else add_below),
            "preferred_buy_zone": {
                "low": _round(buy_zone_low),
                "high": _round(buy_zone_high),
            },
            "add_below": _round(add_below),
        },
        "exit": {
            "take_profit_1": _round(take_profit_1),
            "take_profit_2": _round(take_profit_2),
            "stop_loss": _round(stop_loss),
        },
        "rationale": rationale,
        "caveats": [
            "Win probability is a heuristic, not a calibrated backtest probability.",
            "Use staged entries; revisit after earnings, guidance, or major news.",
        ],
        "source": "deterministic_scorecard_price_target_blend",
    }


def build_option_strategies(
    *,
    spot: float | None,
    direction: str,
    decision_summary: dict[str, Any],
    price_target: dict[str, Any],
    contracts: list[dict[str, Any]],
    portfolio_context: dict[str, Any],
    earnings_calendar: dict[str, Any],
) -> dict[str, Any]:
    """Build first-pass options strategy cards from a normalized chain."""
    if not _finite(spot):
        return {"status": "unavailable", "reason": "Spot price unavailable."}
    if not contracts:
        return {"status": "unavailable", "reason": "Option chain unavailable."}

    spot_f = float(spot)
    account = (portfolio_context or {}).get("account") or {}
    holding = (portfolio_context or {}).get("holding") or {}
    margin_remaining = _first_finite(account.get("margin_remaining"))
    margin_util = _first_finite(account.get("short_put_margin_utilization"))
    cash = _first_finite(account.get("cash"))
    shares = _first_finite(holding.get("shares")) or 0.0
    average_cost = _first_finite(holding.get("average_cost"))
    has_position = bool(holding.get("has_position"))
    earnings_soon = bool((earnings_calendar or {}).get("earnings_in_30_days"))
    entry = (decision_summary or {}).get("entry") or {}
    exit_plan = (decision_summary or {}).get("exit") or {}
    buy_zone = entry.get("preferred_buy_zone") or {}
    target_put_strike = _first_finite(buy_zone.get("high"), entry.get("add_below"))
    target_call_strike = _first_finite(exit_plan.get("take_profit_1"))
    base_target = _first_finite((price_target or {}).get("base"))

    puts = [c for c in contracts if c.get("type") == "put"]
    calls = [c for c in contracts if c.get("type") == "call"]
    short_put = _build_sell_put_card(
        puts=puts,
        spot=spot_f,
        target_strike=target_put_strike,
        margin_remaining=margin_remaining,
        margin_utilization=margin_util,
        cash=cash,
        earnings_soon=earnings_soon,
    )
    covered_call = _build_sell_call_card(
        calls=calls,
        spot=spot_f,
        target_strike=target_call_strike,
        shares=shares,
        average_cost=average_cost,
        has_position=has_position,
        earnings_soon=earnings_soon,
    )
    call_spread = _build_call_spread_card(
        calls=calls,
        spot=spot_f,
        direction=direction,
        target_price=base_target,
        earnings_soon=earnings_soon,
    )
    leap_call = _build_leap_call_card(
        calls=calls,
        spot=spot_f,
        direction=direction,
    )

    strategies = [short_put, covered_call, call_spread, leap_call]
    ranked = sorted(
        strategies,
        key=lambda x: _strategy_rank(x.get("verdict")),
        reverse=True,
    )
    recommended = next(
        (s.get("type") for s in ranked if s.get("verdict") == "consider"),
        ranked[0].get("type") if ranked else None,
    )
    capital_warning = bool(
        (margin_util is not None and margin_util >= 0.75)
        or (
            margin_remaining is not None
            and cash is not None
            and margin_remaining <= max(50_000.0, cash)
        )
    )
    return {
        "status": "ok",
        "source": "normalized_option_chain",
        "capital_warning": capital_warning,
        "earnings_warning": earnings_soon,
        "recommended": recommended,
        "strategies": strategies,
        "contracts_considered": len(contracts),
    }


def _select_target_horizon(
    forecasts: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not forecasts:
        return "unavailable", {}
    best_label = ""
    best_days = -1
    best_block: dict[str, Any] = {}
    for label, block in forecasts.items():
        if not isinstance(block, dict):
            continue
        days = block.get("trading_days")
        try:
            days_i = int(days)
        except (TypeError, ValueError):
            days_i = -1
        if days_i > best_days:
            best_label = str(label)
            best_days = days_i
            best_block = block
    return best_label or "unavailable", best_block


def _build_sell_put_card(
    *,
    puts: list[dict[str, Any]],
    spot: float,
    target_strike: float | None,
    margin_remaining: float | None,
    margin_utilization: float | None,
    cash: float | None,
    earnings_soon: bool,
) -> dict[str, Any]:
    target = target_strike or (spot * 0.92)
    candidates = [
        c for c in puts
        if _dte(c) is not None
        and 14 <= (_dte(c) or 0) <= 75
        and _finite(c.get("strike"))
        and float(c["strike"]) < spot
        and _mid(c) is not None
    ]
    contract = _nearest_contract(candidates, target)
    card = _empty_strategy("sell_put", "Sell put")
    if not contract:
        card.update({"verdict": "unavailable", "reason": "No suitable 14-75 DTE put found."})
        return card
    strike = float(contract["strike"])
    premium = _mid(contract) or 0.0
    assignment = strike * 100.0
    breakeven = strike - premium
    pop = _probability_otm(contract, fallback=0.65)
    verdict = "consider"
    reason = "Pays premium to enter lower near the preferred buy zone."
    if margin_utilization is not None and margin_utilization >= 0.75:
        verdict = "conditional"
        reason = (
            "Candidate is reasonable only after closing existing short puts "
            "or otherwise freeing buying power first."
        )
    elif margin_remaining is not None and assignment > margin_remaining:
        verdict = "conditional"
        reason = "Assignment notional exceeds current margin capacity; free buying power first."
    elif cash is not None and assignment > (cash * 2.0):
        verdict = "conditional"
        reason = "Assignment notional is large relative to cash; use only after freeing buying power."
    if earnings_soon and verdict == "consider":
        verdict = "wait"
        reason = "Earnings are close; short premium has event gap risk."
    card.update(_strategy_contract_fields(contract))
    card.update({
        "verdict": verdict,
        "reason": reason,
        "premium": _round(premium),
        "breakeven": _round(breakeven),
        "max_loss": _round(max(0.0, breakeven * 100.0)),
        "assignment_notional": _round(assignment),
        "estimated_pop": _round(pop),
        "requires_capital_action": verdict == "conditional",
        "required_assignment_capacity": _round(assignment),
        "margin_shortfall": _round(
            max(0.0, assignment - margin_remaining)
            if margin_remaining is not None
            else None
        ),
    })
    return card


def _build_sell_call_card(
    *,
    calls: list[dict[str, Any]],
    spot: float,
    target_strike: float | None,
    shares: float,
    average_cost: float | None,
    has_position: bool,
    earnings_soon: bool,
) -> dict[str, Any]:
    target = max(target_strike or (spot * 1.08), spot * 1.03)
    candidates = [
        c for c in calls
        if _dte(c) is not None
        and 14 <= (_dte(c) or 0) <= 75
        and _finite(c.get("strike"))
        and float(c["strike"]) > spot
        and _mid(c) is not None
    ]
    contract = _nearest_contract(candidates, target)
    card = _empty_strategy("sell_call", "Sell covered call")
    if not has_position or shares < 100:
        card.update({
            "verdict": "unavailable",
            "reason": "Covered call needs at least 100 shares; avoid naked calls.",
        })
        return card
    if not contract:
        card.update({"verdict": "unavailable", "reason": "No suitable 14-75 DTE call found."})
        return card
    premium = _mid(contract) or 0.0
    strike = float(contract["strike"])
    verdict = "consider"
    reason = "Can monetize existing shares and define a trim level."
    if earnings_soon:
        verdict = "wait"
        reason = "Earnings are close; avoid capping upside before event risk is clear."
    premium_adjusted_spot = spot - premium
    cost_basis_breakeven = (
        average_cost - premium
        if average_cost is not None and average_cost > 0
        else None
    )
    max_profit = (
        ((strike - average_cost) + premium) * 100.0
        if average_cost is not None and average_cost > 0
        else ((strike - spot) + premium) * 100.0
    )
    card.update(_strategy_contract_fields(contract))
    card.update({
        "verdict": verdict,
        "reason": reason,
        "premium": _round(premium),
        "breakeven": _round(cost_basis_breakeven),
        "cost_basis_breakeven": _round(cost_basis_breakeven),
        "premium_adjusted_reference_price": _round(premium_adjusted_spot),
        "current_price_reference": _round(spot),
        "average_cost_reference": _round(average_cost),
        "max_profit": _round(max_profit),
        "max_profit_basis": (
            "vs_cost_basis"
            if average_cost is not None and average_cost > 0
            else "mark_to_market"
        ),
        "assignment_notional": _round(strike * 100.0),
        "estimated_pop": _round(_probability_otm(contract, fallback=0.62)),
    })
    return card


def _build_call_spread_card(
    *,
    calls: list[dict[str, Any]],
    spot: float,
    direction: str,
    target_price: float | None,
    earnings_soon: bool,
) -> dict[str, Any]:
    card = _empty_strategy("buy_call_spread", "Buy call spread")
    candidates = [
        c for c in calls
        if _dte(c) is not None
        and 21 <= (_dte(c) or 0) <= 120
        and _finite(c.get("strike"))
        and _mid(c) is not None
    ]
    long_call = _nearest_contract(
        [c for c in candidates if float(c["strike"]) >= spot * 0.95],
        spot,
    )
    if not long_call:
        card.update({"verdict": "unavailable", "reason": "No suitable long call found."})
        return card
    expiry = long_call.get("expiry")
    upper_target = max(target_price or (spot * 1.12), spot * 1.08)
    short_call = _nearest_contract(
        [
            c for c in candidates
            if c.get("expiry") == expiry
            and float(c["strike"]) > float(long_call["strike"])
        ],
        upper_target,
    )
    if not short_call:
        card.update({"verdict": "unavailable", "reason": "No matching short call found."})
        return card
    debit = (_mid(long_call) or 0.0) - (_mid(short_call) or 0.0)
    width = float(short_call["strike"]) - float(long_call["strike"])
    if debit <= 0 or width <= 0:
        card.update({"verdict": "unavailable", "reason": "Spread pricing unusable."})
        return card
    verdict = "consider" if direction == "bullish" else "wait"
    reason = (
        "Defined-risk bullish exposure without using short-put margin."
        if verdict == "consider"
        else "Use only if the underlying thesis turns bullish."
    )
    if earnings_soon and verdict == "consider":
        reason += " Earnings are close, so size smaller."
    card.update({
        "type": "buy_call_spread",
        "label": "Buy call spread",
        "verdict": verdict,
        "reason": reason,
        "expiry": expiry,
        "dte": _dte(long_call),
        "contract_symbol": long_call.get("contract_symbol"),
        "short_contract_symbol": short_call.get("contract_symbol"),
        "long_strike": _round(long_call.get("strike")),
        "short_strike": _round(short_call.get("strike")),
        "debit": _round(debit),
        "max_loss": _round(debit * 100.0),
        "max_profit": _round((width - debit) * 100.0),
        "breakeven": _round(float(long_call["strike"]) + debit),
        "estimated_pop": _round(_probability_itm(short_call, fallback=0.35)),
    })
    return card


def _build_leap_call_card(
    *,
    calls: list[dict[str, Any]],
    spot: float,
    direction: str,
) -> dict[str, Any]:
    card = _empty_strategy("leap_call", "Buy LEAP call")
    candidates = [
        c for c in calls
        if _dte(c) is not None
        and (_dte(c) or 0) >= 180
        and _finite(c.get("strike"))
        and spot * 0.75 <= float(c["strike"]) <= spot * 1.15
        and _mid(c) is not None
    ]
    target = spot * 0.95
    contract = _nearest_contract(candidates, target)
    if not contract:
        card.update({"verdict": "unavailable", "reason": "No suitable 6+ month call found."})
        return card
    spread = _bid_ask_spread_pct(contract)
    verdict = "consider" if direction == "bullish" else "wait"
    reason = "Long-dated convex bullish exposure with defined premium risk."
    if spread is not None and spread > 0.18:
        verdict = "wait"
        reason = "LEAP bid/ask spread is wide; wait for better liquidity."
    premium = _mid(contract) or 0.0
    card.update(_strategy_contract_fields(contract))
    card.update({
        "verdict": verdict,
        "reason": reason,
        "premium": _round(premium),
        "max_loss": _round(premium * 100.0),
        "breakeven": _round(float(contract["strike"]) + premium),
        "estimated_pop": _round(_probability_itm(contract, fallback=0.45)),
    })
    return card


def _empty_strategy(strategy_type: str, label: str) -> dict[str, Any]:
    return {
        "type": strategy_type,
        "label": label,
        "verdict": "unavailable",
        "reason": "No suitable contract found.",
    }


def _dte(contract: dict[str, Any]) -> int | None:
    value = contract.get("dte")
    if _finite(value):
        return int(float(value))
    expiry = contract.get("expiry")
    if not expiry:
        return None
    try:
        from datetime import date, datetime

        exp_date = datetime.strptime(str(expiry), "%Y-%m-%d").date()
        return (exp_date - date.today()).days
    except Exception:
        return None


def _mid(contract: dict[str, Any]) -> float | None:
    explicit = _first_finite(contract.get("mid"), contract.get("mark"))
    if explicit is not None and explicit > 0:
        return explicit
    bid = _first_finite(contract.get("bid"))
    ask = _first_finite(contract.get("ask"))
    if bid is not None and ask is not None and bid >= 0 and ask > 0:
        return (bid + ask) / 2.0
    last = _first_finite(contract.get("last"), contract.get("last_price"))
    if last is not None and last > 0:
        return last
    return None


def _nearest_contract(
    candidates: list[dict[str, Any]],
    target_strike: float,
) -> dict[str, Any] | None:
    usable = [
        c for c in candidates
        if _finite(c.get("strike")) and _mid(c) is not None
    ]
    if not usable:
        return None

    def sort_key(contract: dict[str, Any]) -> tuple[float, float, int]:
        strike = float(contract.get("strike") or 0.0)
        spread = _bid_ask_spread_pct(contract)
        liquidity = int(contract.get("open_interest") or 0) + int(
            contract.get("volume") or 0
        )
        return (
            abs(strike - target_strike),
            spread if spread is not None else 9.99,
            -liquidity,
        )

    return sorted(usable, key=sort_key)[0]


def _probability_otm(
    contract: dict[str, Any],
    *,
    fallback: float,
) -> float:
    delta = _first_finite(contract.get("delta"))
    if delta is not None:
        return max(0.05, min(0.95, 1.0 - abs(float(delta))))
    return max(0.05, min(0.95, fallback))


def _probability_itm(
    contract: dict[str, Any],
    *,
    fallback: float,
) -> float:
    delta = _first_finite(contract.get("delta"))
    if delta is not None:
        return max(0.05, min(0.95, abs(float(delta))))
    return max(0.05, min(0.95, fallback))


def _strategy_contract_fields(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_symbol": contract.get("contract_symbol"),
        "expiry": contract.get("expiry"),
        "dte": _dte(contract),
        "strike": _round(contract.get("strike")),
        "option_type": contract.get("type"),
        "bid": _round(contract.get("bid")),
        "ask": _round(contract.get("ask")),
        "mid": _round(_mid(contract)),
        "last": _round(
            _first_finite(contract.get("last"), contract.get("last_price"))
        ),
        "delta": _round(contract.get("delta")),
        "implied_volatility": _round(contract.get("implied_volatility")),
        "open_interest": contract.get("open_interest"),
        "volume": contract.get("volume"),
        "bid_ask_spread_pct": _round(_bid_ask_spread_pct(contract)),
    }


def _strategy_rank(verdict: Any) -> int:
    return {
        "consider": 3,
        "conditional": 2,
        "wait": 2,
        "avoid_now": 1,
        "unavailable": 0,
    }.get(str(verdict or "unavailable"), 0)


def _bid_ask_spread_pct(contract: dict[str, Any]) -> float | None:
    bid = _first_finite(contract.get("bid"))
    ask = _first_finite(contract.get("ask"))
    mid = _mid(contract)
    if bid is None or ask is None or mid is None or mid <= 0 or ask < bid:
        return None
    return (ask - bid) / mid


def _next_forward_eps_estimate(earnings_calendar: dict[str, Any]) -> float | None:
    for row in earnings_calendar.get("upcoming_earnings") or []:
        if not isinstance(row, dict):
            continue
        for key in ("eps_estimate", "estimated_eps", "epsestimate"):
            value = row.get(key)
            if _finite(value):
                return float(value)
    return None


def _growth_adjusted_target(
    *,
    spot: float,
    fundamentals: dict[str, Any],
    technicals: dict[str, Any],
    market_context: dict[str, Any],
    industry_news_context: dict[str, Any],
    composite_score: float,
) -> float:
    revenue_growth = _first_finite(fundamentals.get("revenue_growth"))
    earnings_growth = _first_finite(fundamentals.get("earnings_growth"))
    margin = _first_finite(fundamentals.get("profit_margins"))
    ret_20d = _first_finite(technicals.get("return_20d"))
    spy_20d = _first_finite(market_context.get("spy_return_20d"))
    news_signals = _industry_news_signals(industry_news_context)

    adjustment = max(-0.12, min(0.12, float(composite_score) * 0.10))
    if _finite(revenue_growth):
        adjustment += max(-0.08, min(0.12, float(revenue_growth) * 0.35))
    if _finite(earnings_growth):
        adjustment += max(-0.08, min(0.12, float(earnings_growth) * 0.25))
    if _finite(margin):
        adjustment += max(-0.04, min(0.06, (float(margin) - 0.15) * 0.20))
    if _finite(ret_20d):
        adjustment += max(-0.06, min(0.06, float(ret_20d) * 0.35))
    if _finite(spy_20d):
        adjustment += max(-0.04, min(0.04, float(spy_20d) * 0.25))
    adjustment += news_signals.get("ai_capex_score", 0.0) * 0.025
    adjustment += news_signals.get("supply_chain_risk_score", 0.0) * -0.025
    adjustment = max(-0.35, min(0.45, adjustment))
    return _bounded_price(spot * (1.0 + adjustment), spot)


def _scenario_spread(
    *,
    spot: float,
    horizon_block: dict[str, Any],
    options_flow: dict[str, Any],
) -> float:
    lower_80 = _first_finite(horizon_block.get("lower_80"))
    upper_80 = _first_finite(horizon_block.get("upper_80"))
    if _finite(lower_80) and _finite(upper_80):
        width = (float(upper_80) - float(lower_80)) / max(spot, 0.01)
        return max(0.06, min(0.45, width / 2.0))
    iv = _first_finite((options_flow or {}).get("atm_iv_30d"))
    if _finite(iv):
        return max(0.06, min(0.45, float(iv) * 0.35))
    return 0.12


def _industry_news_signals(
    industry_news_context: dict[str, Any],
) -> dict[str, Any]:
    counts = {
        str(item.get("theme")): int(item.get("count") or 0)
        for item in (industry_news_context or {}).get("ranked_themes") or []
        if isinstance(item, dict)
    }
    ai_count = counts.get("ai_accelerator_demand", 0) + counts.get(
        "capex_infrastructure", 0
    )
    cycle_count = counts.get("semiconductor_cycle", 0)
    supply_count = counts.get("supply_chain_geopolitics", 0)
    competition_count = counts.get("competition_share", 0)
    total = max(1, sum(counts.values()))
    return {
        "status": industry_news_context.get("status"),
        "pit_status": industry_news_context.get("pit_status"),
        "top_themes": counts,
        "ai_capex_score": round(min(1.0, ai_count / total), 4),
        "semiconductor_cycle_score": round(min(1.0, cycle_count / total), 4),
        "supply_chain_risk_score": round(min(1.0, supply_count / total), 4),
        "competition_risk_score": round(min(1.0, competition_count / total), 4),
    }


def _target_coverage(
    sources: list[dict[str, Any]],
    analyst_consensus: dict[str, Any],
    competitor_analysis: dict[str, Any],
) -> float:
    source_weight = min(1.0, sum(float(s.get("weight") or 0.0) for s in sources))
    analyst_count = analyst_consensus.get("analyst_count")
    peer_count = ((competitor_analysis or {}).get("summary") or {}).get("peer_count")
    if _finite(analyst_count):
        source_weight += min(0.10, float(analyst_count) / 100.0)
    if _finite(peer_count):
        source_weight += min(0.10, float(peer_count) / 80.0)
    return min(1.0, source_weight)


def _target_drivers(
    *,
    sources: list[dict[str, Any]],
    market_context: dict[str, Any],
    industry_news_signals: dict[str, Any],
    earnings_calendar: dict[str, Any],
    options_flow: dict[str, Any],
) -> list[str]:
    drivers = [str(s.get("detail")) for s in sources[:4] if s.get("detail")]
    fg_rating = market_context.get("fear_greed_rating")
    if fg_rating:
        drivers.append(f"Market sentiment regime: CNN Fear & Greed {fg_rating}.")
    if industry_news_signals.get("ai_capex_score", 0.0) > 0.10:
        drivers.append("AI/capex related industry headlines are present.")
    if earnings_calendar.get("earnings_in_90_days"):
        drivers.append("Upcoming earnings may re-anchor the scenario target.")
    if _finite((options_flow or {}).get("atm_iv_30d")):
        drivers.append("Options implied volatility informs scenario width.")
    return drivers[:8]


def _target_risks(
    *,
    market_context: dict[str, Any],
    industry_news_signals: dict[str, Any],
    earnings_calendar: dict[str, Any],
    options_flow: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    if _finite(market_context.get("vix_level")) and float(market_context["vix_level"]) > 25:
        risks.append("Elevated VIX can compress multiples and widen downside.")
    if industry_news_signals.get("supply_chain_risk_score", 0.0) > 0.05:
        risks.append("Supply-chain/geopolitical headlines may pressure semi multiples.")
    if industry_news_signals.get("competition_risk_score", 0.0) > 0.05:
        risks.append("Competition/share headlines can challenge the bull case.")
    if earnings_calendar.get("earnings_in_30_days"):
        risks.append("Near-term earnings event can invalidate the target quickly.")
    if int((options_flow or {}).get("unusual_count") or 0) > 0:
        risks.append("Unusual options activity may reflect event risk or crowded positioning.")
    return risks[:8]


def _weighted_target(sources: list[dict[str, Any]], fallback: float) -> float:
    total_weight = 0.0
    total = 0.0
    for source in sources:
        target = source.get("target")
        weight = source.get("weight")
        if not _finite(target) or not _finite(weight) or float(weight) <= 0:
            continue
        total += float(target) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return fallback
    return total / total_weight


def _bounded_price(v: Any, spot: float, max_multiple: float = 2.0) -> float:
    if not _finite(v):
        return spot
    return max(0.01, min(float(v), spot * max_multiple))


def _first_finite(*values: Any) -> float | None:
    for value in values:
        if _finite(value):
            return float(value)
    return None


def _finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _round(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 6)


def _money(v: Any) -> str:
    rounded = _round(v)
    return "n/a" if rounded is None else f"${rounded:.2f}"
