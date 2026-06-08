"""Execution-ticket helpers for broker-gated trading.

This module is deliberately pure and broker-agnostic. It converts the
daily signal `Action` records into auditable tickets that Codex can review
and execute through the Robinhood MCP tools after explicit confirmation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from portfolio.signals import Action


EXECUTABLE_ACTIONS = ("BUY", "ADD", "TRIM", "EXIT")
BUY_ACTIONS = ("BUY", "ADD")
SELL_ACTIONS = ("TRIM", "EXIT")
SUPPORTED_ASSET_TYPES = ("equity", "option_intent")
OPTION_VERDICT_WEIGHTS = {
    "consider": 1.0,
    "conditional": 0.72,
    "wait": 0.45,
    "avoid": 0.15,
}


@dataclass(frozen=True)
class ExecutionConfig:
    enabled_asset_types: tuple[str, ...] = ("equity", "option_intent")
    default_order_type: str = "limit"
    time_in_force: str = "gtc"
    market_hours: str = "regular_hours"
    require_fresh_signal: bool = True
    max_signal_age_days: int = 7
    min_abs_delta_shares: int = 1
    allow_fractional: bool = False
    options_mode: str = "intent_only"
    tradingagents_review_apply_to_tickets: bool = True

    def __post_init__(self) -> None:
        unsupported = set(self.enabled_asset_types) - set(SUPPORTED_ASSET_TYPES)
        if unsupported:
            raise ValueError(f"unsupported asset type(s): {sorted(unsupported)}")
        if self.default_order_type != "limit":
            raise ValueError("v1 execution tickets only support limit orders")
        if self.time_in_force not in {"gfd", "gtc"}:
            raise ValueError("time_in_force must be 'gfd' or 'gtc'")
        if self.market_hours not in {"regular_hours", "extended_hours", "all_day_hours"}:
            raise ValueError("market_hours is not supported")
        if self.max_signal_age_days < 0:
            raise ValueError("max_signal_age_days must be >= 0")
        if self.min_abs_delta_shares < 1:
            raise ValueError("min_abs_delta_shares must be >= 1")
        if self.options_mode != "intent_only":
            raise ValueError("v1 options_mode must be 'intent_only'")


@dataclass(frozen=True)
class ExecutionTicket:
    ticket_id: str
    asset_type: str
    symbol: str
    side: str
    order_type: str
    quantity: str | None
    limit_price: float | None
    time_in_force: str
    market_hours: str
    source_action: str
    rationale: str
    risk_notes: list[str] = field(default_factory=list)
    status: str = "ready"
    blocked_reason: str | None = None
    review_gate_status: str | None = None
    review_gate_reason: str | None = None
    review_execution_caveats: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
            "market_hours": self.market_hours,
            "source_action": self.source_action,
            "rationale": self.rationale,
            "risk_notes": list(self.risk_notes),
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "review_gate_status": self.review_gate_status,
            "review_gate_reason": self.review_gate_reason,
            "review_execution_caveats": list(self.review_execution_caveats),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExecutionBatch:
    as_of: str
    source_daily_signals_path: str | None
    account_hint: str | None
    tickets: list[ExecutionTicket]
    blocked_tickets: list[ExecutionTicket]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "source_daily_signals_path": self.source_daily_signals_path,
            "account_hint": self.account_hint,
            "summary": dict(self.summary),
            "tickets": [t.to_dict() for t in self.tickets],
            "blocked_tickets": [t.to_dict() for t in self.blocked_tickets],
        }


def execution_config_from_dict(data: dict[str, Any]) -> ExecutionConfig:
    allowed = {
        "enabled_asset_types",
        "default_order_type",
        "time_in_force",
        "market_hours",
        "require_fresh_signal",
        "max_signal_age_days",
        "min_abs_delta_shares",
        "allow_fractional",
        "options_mode",
        "tradingagents_review_apply_to_tickets",
    }
    kwargs = {k: v for k, v in (data or {}).items() if k in allowed}
    if "enabled_asset_types" in kwargs:
        kwargs["enabled_asset_types"] = tuple(str(v) for v in kwargs["enabled_asset_types"])
    return ExecutionConfig(**kwargs)


def build_execution_batch(
    *,
    actions: Iterable[Action],
    as_of: str,
    config: ExecutionConfig,
    source_daily_signals_path: str | None = None,
    account_hint: str | None = None,
    option_strategy_reports: Iterable[dict[str, Any]] | None = None,
) -> ExecutionBatch:
    ready: list[ExecutionTicket] = []
    blocked: list[ExecutionTicket] = []

    for action in actions:
        ticket = ticket_from_action(action, as_of=as_of, config=config)
        if ticket is None:
            continue
        if ticket.status == "ready":
            ready.append(ticket)
        else:
            blocked.append(ticket)

    for report in option_strategy_reports or []:
        for ticket in option_intent_tickets_from_report(report, as_of=as_of, config=config):
            blocked.append(ticket)
    blocked = _rank_option_intents(blocked)

    summary = {
        "ready_count": len(ready),
        "blocked_count": len(blocked),
        "equity_ready_count": sum(1 for t in ready if t.asset_type == "equity"),
        "option_intent_count": sum(1 for t in blocked if t.asset_type == "option_intent"),
        "requires_robinhood_mcp": bool(ready),
        "live_execution_policy": "review_required_before_place",
    }
    return ExecutionBatch(
        as_of=as_of,
        source_daily_signals_path=source_daily_signals_path,
        account_hint=account_hint,
        tickets=ready,
        blocked_tickets=blocked,
        summary=summary,
    )


def ticket_from_action(
    action: Action,
    *,
    as_of: str,
    config: ExecutionConfig,
) -> ExecutionTicket | None:
    if action.action not in EXECUTABLE_ACTIONS:
        return None

    side = "buy" if action.action in BUY_ACTIONS else "sell"
    raw_delta_shares = _raw_delta_shares(action)
    quantity = abs(float(action.delta_shares or 0))
    blocked_reasons: list[str] = []

    if "equity" not in config.enabled_asset_types:
        blocked_reasons.append("equity tickets are disabled")
    if quantity < config.min_abs_delta_shares:
        blocked_reasons.append(
            f"absolute delta shares {quantity:g} < minimum {config.min_abs_delta_shares}"
        )
    if not config.allow_fractional and not quantity.is_integer():
        blocked_reasons.append("fractional shares are disabled")
    if action.limit_price is None:
        blocked_reasons.append("missing limit price")
    if action.last_close is None:
        blocked_reasons.append("missing price context")
    if (
        config.require_fresh_signal
        and action.signal_age_days is not None
        and action.signal_age_days > config.max_signal_age_days
    ):
        blocked_reasons.append(
            f"stale signal ({action.signal_age_days}d > {config.max_signal_age_days}d)"
        )
    review_gate_status = getattr(action, "review_gate_status", None)
    review_gate_reason = getattr(action, "review_gate_reason", None)
    review_meta = dict(getattr(action, "tradingagents_review", None) or {})
    review_caveats = [
        str(x)
        for x in (getattr(action, "review_execution_caveats", None) or [])
        if str(x).strip()
    ]
    if (
        config.tradingagents_review_apply_to_tickets
        and action.action in BUY_ACTIONS
        and review_gate_status in {"block_buy_add", "manual_review"}
    ):
        label = (
            "TradingAgents risk veto"
            if review_gate_status == "block_buy_add"
            else "TradingAgents manual-review gate"
        )
        blocked_reasons.append(
            f"{label}: {review_gate_reason or 'review gate requires confirmation'}"
        )

    status = "blocked" if blocked_reasons else "ready"
    qty_text = _format_quantity(quantity, allow_fractional=config.allow_fractional)
    ticket_id = stable_ticket_id(
        as_of=as_of,
        asset_type="equity",
        symbol=action.symbol,
        side=side,
        source_action=action.action,
        quantity=qty_text,
        limit_price=action.limit_price,
        order_type=config.default_order_type,
        time_in_force=config.time_in_force,
    )
    return ExecutionTicket(
        ticket_id=ticket_id,
        asset_type="equity",
        symbol=action.symbol.upper(),
        side=side,
        order_type=config.default_order_type,
        quantity=qty_text,
        limit_price=action.limit_price,
        time_in_force=config.time_in_force,
        market_hours=config.market_hours,
        source_action=action.action,
        rationale=_action_rationale(action),
        risk_notes=list(action.notes),
        status=status,
        blocked_reason="; ".join(blocked_reasons) if blocked_reasons else None,
        review_gate_status=review_gate_status,
        review_gate_reason=review_gate_reason,
        review_execution_caveats=review_caveats,
        details={
            "target_weight": action.target_weight,
            "current_weight": action.current_weight,
            "delta_pp": action.delta_pp,
            "target_shares": action.target_shares,
            "current_shares": action.current_shares,
            "delta_shares": action.delta_shares,
            "raw_delta_shares": raw_delta_shares,
            "raw_target_shares": getattr(action, "target_shares_exact", None),
            "signal_age_days": action.signal_age_days,
            "stop_loss": action.stop_loss,
            "last_close": action.last_close,
            "sma20": action.sma20,
            "atr14": action.atr14,
            "price_source": action.price_source,
            "tradingagents_review": review_meta,
        },
    )


def option_intent_tickets_from_report(
    report: dict[str, Any],
    *,
    as_of: str,
    config: ExecutionConfig,
) -> list[ExecutionTicket]:
    if "option_intent" not in config.enabled_asset_types:
        return []
    symbol = str(report.get("symbol") or "").upper()
    if not symbol:
        return []
    strategies_block = (report.get("key_features") or {}).get("option_strategies") or {}
    strategies = strategies_block.get("strategies") or []
    out: list[ExecutionTicket] = []
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        strategy_type = str(strategy.get("type") or "")
        if strategy_type not in {"sell_put", "sell_call", "buy_call_spread"}:
            continue
        verdict = str(strategy.get("verdict") or "").lower()
        if verdict in {"unavailable"}:
            continue
        score_details = option_intent_score_details(report, strategy)
        ticket_id = stable_ticket_id(
            as_of=as_of,
            asset_type="option_intent",
            symbol=symbol,
            side=_option_strategy_side(strategy_type),
            source_action=strategy_type,
            quantity=str(strategy.get("quantity") or 1),
            limit_price=_option_limit_price(strategy),
            order_type="intent_only",
            time_in_force=config.time_in_force,
        )
        out.append(
            ExecutionTicket(
                ticket_id=ticket_id,
                asset_type="option_intent",
                symbol=symbol,
                side=_option_strategy_side(strategy_type),
                order_type="intent_only",
                quantity=str(strategy.get("quantity") or 1),
                limit_price=_option_limit_price(strategy),
                time_in_force=config.time_in_force,
                market_hours=config.market_hours,
                source_action=strategy_type,
                rationale=strategy.get("reason") or f"{strategy_type} option intent",
                risk_notes=[
                    "Robinhood MCP options placement is not available yet.",
                    f"strategy verdict: {verdict or 'unknown'}",
                    (
                        "option intent rank score: "
                        f"{score_details['option_intent_score']:.0f}/100 "
                        f"({score_details['option_intent_score_label']})"
                    ),
                ],
                status="blocked",
                blocked_reason="Robinhood MCP options placement is not available yet",
                details={
                    **{k: v for k, v in strategy.items() if k != "reason"},
                    **score_details,
                },
            )
        )
    return out


def _rank_option_intents(tickets: list[ExecutionTicket]) -> list[ExecutionTicket]:
    non_options = [t for t in tickets if t.asset_type != "option_intent"]
    options = [t for t in tickets if t.asset_type == "option_intent"]
    ranked_options = sorted(
        options,
        key=lambda t: (
            -(_safe_float((t.details or {}).get("option_intent_score")) or 0.0),
            t.symbol,
            t.source_action,
            t.ticket_id,
        ),
    )
    with_rank: list[ExecutionTicket] = []
    for idx, ticket in enumerate(ranked_options, start=1):
        details = dict(ticket.details)
        details["option_intent_rank"] = idx
        with_rank.append(replace(ticket, details=details))
    return [*non_options, *with_rank]


def option_intent_score_details(
    report: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    key_features = report.get("key_features") or {}
    scoring = key_features.get("model_scoring") or {}
    price_target = key_features.get("price_target") or {}
    direction = str(report.get("direction") or "neutral").lower()
    strategy_type = str(strategy.get("type") or "")
    verdict = str(strategy.get("verdict") or "").lower()
    confidence = _safe_float(report.get("confidence"))
    if confidence is None:
        confidence = _safe_float(price_target.get("confidence"))
    if confidence is None:
        confidence = 0.5
    confidence = _clamp(confidence, 0.0, 1.0)
    composite = _safe_float(scoring.get("composite_score"))
    if composite is None:
        composite = 0.0
    composite = _clamp(composite, -1.0, 1.0)
    verdict_weight = OPTION_VERDICT_WEIGHTS.get(verdict, 0.35)
    alignment = _option_direction_alignment(strategy_type, direction)
    signal_component = _option_signal_component(strategy_type, composite)
    score = 100.0 * verdict_weight * (
        (0.50 * confidence) + (0.30 * signal_component) + (0.20 * alignment)
    )
    score = _clamp(score, 0.0, 100.0)
    return {
        "option_intent_score": round(score, 1),
        "option_intent_score_label": _option_score_label(score),
        "report_direction": direction,
        "report_confidence": round(confidence, 4),
        "report_composite": round(composite, 4),
        "strategy_verdict_weight": verdict_weight,
        "strategy_direction_alignment": round(alignment, 4),
        "strategy_signal_component": round(signal_component, 4),
    }


def _option_intent_score_details(
    report: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    return option_intent_score_details(report, strategy)


def _option_direction_alignment(strategy_type: str, direction: str) -> float:
    table = {
        "sell_put": {"bullish": 1.0, "neutral": 0.55, "bearish": 0.20},
        "buy_call_spread": {"bullish": 1.0, "neutral": 0.35, "bearish": 0.10},
        "leap_call": {"bullish": 0.90, "neutral": 0.35, "bearish": 0.10},
        "sell_call": {"bullish": 0.45, "neutral": 0.85, "bearish": 0.75},
    }
    return table.get(strategy_type, {}).get(direction, 0.40)


def _option_signal_component(strategy_type: str, composite: float) -> float:
    if strategy_type == "sell_call":
        return _clamp(0.70 - max(0.0, composite) + max(0.0, -composite) * 0.30, 0.0, 1.0)
    return _clamp((composite + 0.20) / 0.80, 0.0, 1.0)


def _option_score_label(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    if score >= 35:
        return "low"
    return "watch"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stable_ticket_id(
    *,
    as_of: str,
    asset_type: str,
    symbol: str,
    side: str,
    source_action: str,
    quantity: str | None,
    limit_price: float | None,
    order_type: str,
    time_in_force: str,
) -> str:
    raw = "|".join(
        [
            as_of,
            asset_type,
            symbol.upper(),
            side,
            source_action,
            quantity or "",
            "" if limit_price is None else f"{float(limit_price):.4f}",
            order_type,
            time_in_force,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _format_quantity(quantity: float, *, allow_fractional: bool) -> str:
    if allow_fractional:
        return f"{quantity:.6f}".rstrip("0").rstrip(".")
    return str(int(quantity))


def _raw_delta_shares(action: Action) -> float | None:
    exact = getattr(action, "delta_shares_exact", None)
    if exact is not None:
        return float(exact)
    if action.limit_price is None or action.limit_price <= 0:
        price = action.last_close
    else:
        price = action.limit_price
    if price is None or price <= 0:
        return None
    # delta_pp is target_weight - current_weight. For new positions this
    # recovers the fractional share intent behind a floor-rounded 0-share
    # target. For existing positions it is still a useful approximation.
    current_value = 0.0
    if action.current_weight > 0:
        current_value = (action.current_shares * price) / action.current_weight
    elif action.target_weight > 0:
        current_value = (action.target_shares * price) / action.target_weight
        if current_value <= 0 and action.delta_pp > 0:
            # Whole-share target rounded to zero; infer account value from
            # target dollars implied by target weight is unavailable here.
            return None
    if current_value <= 0:
        return None
    return (action.delta_pp * current_value) / price


def _action_rationale(action: Action) -> str:
    bits = [
        f"{action.action} {action.symbol}",
        f"target {action.target_weight * 100:.1f}%",
        f"current {action.current_weight * 100:.1f}%",
        f"delta {action.delta_pp * 100:+.1f}pp",
    ]
    if action.direction:
        bits.append(f"direction {action.direction}")
    if action.composite is not None:
        bits.append(f"composite {action.composite:+.3f}")
    if action.confidence is not None:
        bits.append(f"confidence {action.confidence:.2f}")
    return "; ".join(bits)


def _option_strategy_side(strategy_type: str) -> str:
    if strategy_type in {"sell_put", "sell_call"}:
        return "sell"
    return "buy"


def _option_limit_price(strategy: dict[str, Any]) -> float | None:
    for key in ("premium", "debit", "mid", "last"):
        value = strategy.get(key)
        if value is None:
            continue
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            continue
    return None
