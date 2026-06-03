"""Build a reconciliation artifact from Robinhood fills.

The fills file is supplied by Codex after Robinhood MCP order review and
placement. By default this command emits an audit artifact only; pass
`--apply-position-update` to explicitly update `portfolio/positions.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from daily_signals import _load_positions


@dataclass(frozen=True)
class NormalizedFill:
    ticket_id: str | None
    order_id: str | None
    symbol: str
    side: str
    quantity: float
    average_price: float
    state: str | None

    @property
    def notional(self) -> float:
        return self.quantity * self.average_price


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticket-batch", required=True, help="Path to reports/trade_tickets/*.json")
    p.add_argument("--fills", required=True, help="JSON file containing Robinhood order/fill summaries")
    p.add_argument("--positions", default="portfolio/positions.json")
    p.add_argument("--output-dir", default="reports/trade_reconcile")
    p.add_argument("--as-of", default=None, help="Reconcile date, YYYY-MM-DD; defaults to today.")
    p.add_argument(
        "--apply-position-update",
        action="store_true",
        help="Explicitly apply the previewed cash/position update to --positions.",
    )
    return p.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_as_of(raw: str | None) -> str:
    if raw is None:
        return date.today().isoformat()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise SystemExit(f"--as-of must be YYYY-MM-DD; got {raw!r}") from exc


def _fill_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("fills", "executions", "orders", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def normalize_fills(payload: Any) -> list[NormalizedFill]:
    fills: list[NormalizedFill] = []
    for item in _fill_items(payload):
        symbol = str(_first(item, ("symbol", "instrument_symbol")) or "").upper()
        side = str(_first(item, ("side", "order_side")) or "").lower()
        raw_qty = _first(item, ("filled_quantity", "quantity", "cumulative_quantity"))
        raw_price = _first(
            item,
            ("average_price", "filled_avg_price", "avg_price", "price", "executed_price"),
        )
        if not symbol or side not in {"buy", "sell"} or raw_qty is None or raw_price is None:
            continue
        try:
            qty = float(raw_qty)
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0:
            continue
        fills.append(
            NormalizedFill(
                ticket_id=_first(item, ("ticket_id", "client_ticket_id")),
                order_id=_first(item, ("order_id", "id")),
                symbol=symbol,
                side=side,
                quantity=qty,
                average_price=price,
                state=_first(item, ("state", "status")),
            )
        )
    return fills


def build_reconciliation(
    *,
    ticket_batch: dict[str, Any],
    fills: list[NormalizedFill],
    positions_path: Path,
    as_of: str,
) -> dict[str, Any]:
    cash, positions = _load_positions(positions_path)
    updated: dict[str, dict[str, float]] = {
        sym: {"shares": float(pos.shares), "avg_cost": float(pos.avg_cost)}
        for sym, pos in positions.items()
    }
    cash_delta = 0.0
    warnings: list[str] = []
    ticket_by_id = _ticket_index(ticket_batch)
    normalized_fill_rows: list[dict[str, Any]] = []

    for fill in fills:
        matched_ticket = ticket_by_id.get(str(fill.ticket_id)) if fill.ticket_id else None
        matched_ticket_id = (
            matched_ticket.get("ticket_id")
            if isinstance(matched_ticket, dict)
            else None
        )
        if fill.ticket_id and matched_ticket_id is None:
            warnings.append(
                f"Fill ticket_id={fill.ticket_id} did not match any ticket in the batch."
            )
        normalized_fill_rows.append(
            asdict(fill) | {
                "notional": fill.notional,
                "matched_ticket_id": matched_ticket_id,
            }
        )
        current = updated.setdefault(fill.symbol, {"shares": 0.0, "avg_cost": 0.0})
        old_shares = float(current["shares"])
        old_cost = float(current["avg_cost"])
        if fill.side == "buy":
            new_shares = old_shares + fill.quantity
            new_cost = (
                ((old_shares * old_cost) + fill.notional) / new_shares
                if new_shares > 0
                else 0.0
            )
            current["shares"] = new_shares
            current["avg_cost"] = round(new_cost, 4)
            cash_delta -= fill.notional
        else:
            if fill.quantity > old_shares:
                warnings.append(
                    f"{fill.symbol} sell fill {fill.quantity:g} exceeds ledger shares {old_shares:g}."
                )
            current["shares"] = max(0.0, old_shares - fill.quantity)
            current["avg_cost"] = old_cost if current["shares"] > 0 else 0.0
            cash_delta += fill.notional

    changed_positions = {
        sym: body
        for sym, body in updated.items()
        if sym not in positions
        or body["shares"] != float(positions[sym].shares)
        or body["avg_cost"] != float(positions[sym].avg_cost)
    }

    return {
        "as_of": as_of,
        "ticket_batch_as_of": ticket_batch.get("as_of"),
        "ticket_batch_summary": ticket_batch.get("summary") or {},
        "source_ticket_batch": ticket_batch.get("source_daily_signals_path"),
        "positions_path": str(positions_path),
        "fills": normalized_fill_rows,
        "ledger_patch": {
            "cash_before": cash,
            "cash_delta": round(cash_delta, 2),
            "cash_after_estimate": round(cash + cash_delta, 2),
            "positions_to_update": changed_positions,
        },
        "position_update_preview": {
            "cash": round(cash + cash_delta, 2),
            "positions": changed_positions,
        },
        "requires_manual_position_update": bool(changed_positions or cash_delta),
        "warnings": warnings,
        "note": (
            "This artifact is advisory unless trade_reconcile.py is run with "
            "--apply-position-update."
        ),
    }


def _ticket_index(ticket_batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in ("tickets", "blocked_tickets"):
        for ticket in ticket_batch.get(key) or []:
            if not isinstance(ticket, dict):
                continue
            ticket_id = ticket.get("ticket_id")
            if ticket_id:
                out[str(ticket_id)] = ticket
    return out


def apply_position_update(
    *,
    positions_path: Path,
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    payload = _read_json(positions_path)
    if not isinstance(payload, dict):
        raise ValueError("positions file must contain a JSON object")
    patch = reconciliation.get("ledger_patch") or {}
    positions_to_update = patch.get("positions_to_update") or {}
    if "positions" not in payload or not isinstance(payload["positions"], dict):
        payload["positions"] = {}
    if "cash" in payload:
        payload["cash"] = patch.get("cash_after_estimate", payload.get("cash"))
    else:
        account = payload.setdefault("account", {})
        if isinstance(account, dict):
            account["cash"] = patch.get("cash_after_estimate", account.get("cash"))
    for sym, body in positions_to_update.items():
        if not isinstance(body, dict):
            continue
        shares = float(body.get("shares") or 0.0)
        avg_cost = float(body.get("avg_cost") or 0.0)
        if shares <= 0:
            payload["positions"].pop(sym, None)
        else:
            payload["positions"][sym] = {
                "shares": shares,
                "avg_cost": avg_cost,
            }
    positions_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    args = _parse_args()
    as_of = _resolve_as_of(args.as_of)
    ticket_batch = _read_json(Path(args.ticket_batch))
    fills = normalize_fills(_read_json(Path(args.fills)))
    reconciliation = build_reconciliation(
        ticket_batch=ticket_batch if isinstance(ticket_batch, dict) else {},
        fills=fills,
        positions_path=Path(args.positions),
        as_of=as_of,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{as_of}.json"
    out_path.write_text(json.dumps(reconciliation, indent=2, default=str))
    if args.apply_position_update:
        apply_position_update(
            positions_path=Path(args.positions),
            reconciliation=reconciliation,
        )
    print(f"Wrote: {out_path}")
    print(
        "Summary: "
        f"fills={len(fills)} "
        f"cash_delta={reconciliation['ledger_patch']['cash_delta']:+,.2f} "
        f"positions={len(reconciliation['ledger_patch']['positions_to_update'])}"
    )


if __name__ == "__main__":
    main()
