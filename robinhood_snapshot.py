"""Normalize read-only Robinhood MCP data into a local portfolio snapshot.

This script does not call Robinhood. Codex gathers account/portfolio/position
data through the `rh` MCP tools, writes a raw JSON file, then this script
normalizes it for the UI and ticket generator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, help="Raw JSON captured from read-only rh MCP calls.")
    p.add_argument("--output", default="configs/robinhood_snapshot.json")
    p.add_argument("--account-number", default=None)
    return p.parse_args()


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _items(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = _items(value, *keys)
            if nested:
                return nested
    return []


def build_snapshot(raw: dict[str, Any], *, account_number: str | None = None) -> dict[str, Any]:
    portfolio = raw.get("portfolio") if isinstance(raw.get("portfolio"), dict) else raw
    account = raw.get("account") if isinstance(raw.get("account"), dict) else {}
    positions_raw = _items(raw.get("positions"), "positions", "results")
    if not positions_raw:
        positions_raw = _items(raw, "positions", "results")

    cash = _safe_float(
        _first(
            portfolio,
            "buying_power",
            "cash",
            "withdrawable_cash",
            "cash_available_for_withdrawal",
        )
    )
    total_equity = _safe_float(
        _first(
            portfolio,
            "total_equity",
            "total_value",
            "equity",
            "market_value",
            "portfolio_value",
            "total_market_value",
        )
    )

    positions: list[dict[str, Any]] = []
    for item in positions_raw:
        symbol = str(_first(item, "symbol", "instrument_symbol", "ticker") or "").upper()
        if not symbol:
            continue
        shares = _safe_float(_first(item, "quantity", "shares", "shares_held")) or 0.0
        avg_cost = _safe_float(
            _first(item, "average_buy_price", "average_cost", "avg_cost", "avg_price")
        ) or 0.0
        price = _safe_float(_first(item, "price", "last_price", "mark_price"))
        equity = _safe_float(_first(item, "equity", "market_value"))
        if equity is None and price is not None:
            equity = shares * price
        positions.append(
            {
                "symbol": symbol,
                "name": _first(item, "name", "simple_name", "instrument_name"),
                "shares": shares,
                "price": price,
                "average_cost": avg_cost,
                "equity": equity,
            }
        )

    if total_equity is None:
        total_equity = (cash or 0.0) + sum(
            p.get("equity") or 0.0 for p in positions
        )

    return {
        "as_of": raw.get("as_of") or datetime.now().date().isoformat(),
        "source": "robinhood_mcp_read_only",
        "account_number_masked": _mask_account_number(
            account_number or raw.get("account_number") or account.get("account_number")
        ),
        "account": {
            "total_equity": total_equity,
            "cash": cash or 0.0,
            "buying_power": _safe_float(_first(portfolio, "buying_power")),
        },
        "positions": positions,
        "raw_meta": {
            "portfolio_keys": sorted(portfolio.keys()) if isinstance(portfolio, dict) else [],
            "position_count": len(positions),
        },
    }


def main() -> None:
    args = _parse_args()
    raw = json.loads(Path(args.raw).read_text())
    if not isinstance(raw, dict):
        raise SystemExit("Raw snapshot JSON must be an object.")
    snapshot = build_snapshot(raw, account_number=args.account_number)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Wrote: {out_path}")
    print(
        "Summary: "
        f"cash={snapshot['account']['cash']:,.2f} "
        f"total_equity={snapshot['account']['total_equity']:,.2f} "
        f"positions={len(snapshot['positions'])}"
    )


def _mask_account_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 4:
        return "****"
    return "****" + text[-4:]


if __name__ == "__main__":
    main()
