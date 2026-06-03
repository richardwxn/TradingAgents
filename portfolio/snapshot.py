"""Portfolio/position snapshot loading.

Supports both the original manual ledger shape:

    {"cash": 1000, "positions": {"NVDA": {"shares": 2, "avg_cost": 100}}}

and the richer account snapshot shape used by analysis reports:

    {"account": {"cash": 1000}, "positions": [{"symbol": "NVDA", ...}]}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from portfolio.signals import Position


@dataclass(frozen=True)
class LoadedPositions:
    cash: float
    positions: dict[str, Position]
    metadata: dict[str, Any]


def load_positions_payload(payload: dict[str, Any]) -> LoadedPositions:
    if not isinstance(payload, dict):
        return LoadedPositions(0.0, {}, {"source_type": "invalid"})

    raw_positions = payload.get("positions") if "positions" in payload else {}
    if isinstance(raw_positions, list):
        return _load_snapshot_list(payload, raw_positions)
    if isinstance(raw_positions, dict):
        return _load_ledger_dict(payload, raw_positions)
    return LoadedPositions(0.0, {}, {"source_type": "unknown"})


def _load_ledger_dict(
    payload: dict[str, Any],
    raw_positions: dict[str, Any],
) -> LoadedPositions:
    account = payload.get("account") or {}
    cash = _safe_float(payload.get("cash"), account.get("cash")) or 0.0
    positions: dict[str, Position] = {}
    for sym, raw in raw_positions.items():
        if not isinstance(raw, dict):
            continue
        symbol = str(sym).upper()
        shares = _safe_float(raw.get("shares"), raw.get("quantity")) or 0.0
        avg_cost = _safe_float(
            raw.get("avg_cost"),
            raw.get("average_cost"),
            raw.get("average_buy_price"),
        ) or 0.0
        positions[symbol] = Position(shares=shares, avg_cost=avg_cost)
    return LoadedPositions(
        cash=cash,
        positions=positions,
        metadata={
            "source_type": "ledger",
            "source": payload.get("source"),
            "as_of": payload.get("as_of"),
            "account": account,
            "position_count": len(positions),
        },
    )


def _load_snapshot_list(
    payload: dict[str, Any],
    raw_positions: list[Any],
) -> LoadedPositions:
    account = payload.get("account") or {}
    cash = _safe_float(payload.get("cash"), account.get("cash"), account.get("buying_power")) or 0.0
    positions: dict[str, Position] = {}
    visible_equity = 0.0
    for raw in raw_positions:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or raw.get("instrument_symbol") or "").upper()
        if not symbol:
            continue
        shares = _safe_float(raw.get("shares"), raw.get("quantity")) or 0.0
        avg_cost = _safe_float(
            raw.get("avg_cost"),
            raw.get("average_cost"),
            raw.get("average_buy_price"),
            raw.get("average_price"),
        ) or 0.0
        equity = _safe_float(raw.get("equity"), raw.get("market_value"))
        visible_equity += equity or 0.0
        positions[symbol] = Position(shares=shares, avg_cost=avg_cost)
    return LoadedPositions(
        cash=cash,
        positions=positions,
        metadata={
            "source_type": "snapshot",
            "source": payload.get("source"),
            "as_of": payload.get("as_of"),
            "account": account,
            "position_count": len(positions),
            "visible_positions_equity": visible_equity,
            "total_equity": _safe_float(account.get("total_equity"), account.get("equity")),
        },
    )


def _safe_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
