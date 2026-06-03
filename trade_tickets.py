"""Generate broker-gated trade tickets from daily signal actions.

The output JSON is the source of truth for Codex/Robinhood MCP execution.
This CLI never calls Robinhood and never places orders.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from daily_signals import (
    _load_positions,
    _load_risk_limits,
    _load_sector_map,
    _load_sizing_config,
    fetch_betas_and_correlations,
    fetch_prices_for_universe,
)
from portfolio.execution import (
    ExecutionBatch,
    execution_config_from_dict,
    build_execution_batch,
)
from portfolio.signals import PriceContext, compute_actions, load_latest_signals


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    p.add_argument("--positions", default="portfolio/positions.json")
    p.add_argument("--sizing-config", default="configs/sizing.yaml")
    p.add_argument("--execution-config", default="configs/execution.yaml")
    p.add_argument("--output-dir", default="reports/trade_tickets")
    p.add_argument("--as-of", default=None, help="Run date, YYYY-MM-DD; defaults to today.")
    p.add_argument("--no-prices", action="store_true", help="Skip price fetch; executable tickets will be blocked.")
    p.add_argument("--account-hint", default=None, help="Optional Robinhood account number/nickname for the operator.")
    p.add_argument("--source-daily-signals-path", default=None)
    return p.parse_args()


def _load_execution_config(path: Path):
    data = yaml.safe_load(path.read_text()) or {}
    return execution_config_from_dict(data)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _option_strategy_reports(signals: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sig in signals.values():
        if sig is None or not getattr(sig, "source_path", None):
            continue
        path = str(sig.source_path)
        if path in seen:
            continue
        seen.add(path)
        payload = _read_json(Path(path))
        if payload:
            reports.append(payload)
    return reports


def _resolve_as_of(raw: str | None) -> date:
    if raw is None:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--as-of must be YYYY-MM-DD; got {raw!r}") from exc


def _load_actions(args: argparse.Namespace, as_of: date):
    sizing_config = _load_sizing_config(Path(args.sizing_config))
    cash, positions = _load_positions(Path(args.positions))
    universe = list(sizing_config.universe) or sorted(positions.keys())
    if not universe:
        raise SystemExit("No universe configured and no positions held; nothing to ticket.")

    report_paths = [Path(p) for p in glob.glob(args.reports_glob)]
    signals = load_latest_signals(report_paths, universe=universe, as_of=as_of)

    sector_map = _load_sector_map()
    if sector_map:
        from dataclasses import replace

        for sym, sig in list(signals.items()):
            if sig is None or sig.sector is not None:
                continue
            fallback = sector_map.get(sym.upper())
            if fallback:
                signals[sym] = replace(sig, sector=fallback)

    fetch_symbols = sorted(set(universe) | set(positions.keys()))
    if args.no_prices:
        prices = {sym: PriceContext(None, None, None) for sym in fetch_symbols}
    else:
        prices = fetch_prices_for_universe(fetch_symbols)

    risk_limits = _load_risk_limits(Path(args.sizing_config))
    beta_map: dict[str, float | None] = {}
    corr_matrix: dict[str, dict[str, float]] = {}
    if risk_limits is not None and not args.no_prices:
        beta_map, corr_matrix = fetch_betas_and_correlations(
            universe, benchmark="SPY", lookback_days=90,
        )

    actions, summary = compute_actions(
        signals=signals,
        positions=positions,
        prices=prices,
        config=sizing_config,
        cash=cash,
        as_of=as_of,
        risk_limits=risk_limits,
        beta_map=beta_map,
        correlation_matrix=corr_matrix,
    )
    return actions, summary, signals


def _format_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _format_batch_markdown(batch: ExecutionBatch) -> str:
    lines = [
        f"# Trade Tickets - {batch.as_of}",
        "",
        "These tickets are broker-gated. The repo does not call Robinhood or place orders.",
        "",
        "## Summary",
        "",
        f"- Ready equity tickets: {batch.summary.get('ready_count', 0)}",
        f"- Blocked tickets: {batch.summary.get('blocked_count', 0)}",
        f"- Option intent tickets: {batch.summary.get('option_intent_count', 0)}",
        f"- Execution policy: review with Robinhood MCP, then explicit confirmation before placement",
        "",
    ]

    if batch.tickets:
        lines.extend([
            "## Ready For Robinhood Review",
            "",
            "| Ticket | Symbol | Side | Qty | Limit | TIF | Source |",
            "|---|---|---|---:|---:|---|---|",
        ])
        for t in batch.tickets:
            lines.append(
                f"| `{t.ticket_id}` | {t.symbol} | {t.side} | {t.quantity} | "
                f"{_format_money(t.limit_price)} | {t.time_in_force} | {t.source_action} |"
            )
        lines.append("")
        lines.extend([
            "### Codex Robinhood MCP Steps",
            "",
            "1. Call `rh.get_accounts` and choose an `agentic_allowed=true` account.",
            "2. For each ready ticket, call `rh.get_portfolio`, `rh.get_equity_positions`, `rh.get_equity_quotes`, and `rh.get_equity_tradability`.",
            "3. Call `rh.review_equity_order` using `symbol`, `side`, `type=limit`, `quantity`, `limit_price`, `time_in_force`, and `market_hours`; use the ticket's `time_in_force` exactly.",
            "4. Present review output and warnings to the user.",
            "5. Call `rh.place_equity_order` only after explicit user confirmation.",
            "6. Record order IDs and fills for `trade_reconcile.py`.",
            "7. Run `trade_reconcile.py` after fills settle; only use `--apply-position-update` after reviewing the audit artifact.",
            "",
        ])
    else:
        lines.extend(["## Ready For Robinhood Review", "", "_No ready tickets._", ""])

    if batch.blocked_tickets:
        lines.extend([
            "## Blocked Or Intent Only",
            "",
            "| Ticket | Asset | Symbol | Side | Source | Reason |",
            "|---|---|---|---|---|---|",
        ])
        for t in batch.blocked_tickets:
            reason = (t.blocked_reason or "").replace("|", "\\|")
            lines.append(
                f"| `{t.ticket_id}` | {t.asset_type} | {t.symbol} | {t.side} | "
                f"{t.source_action} | {reason} |"
            )
        lines.append("")

    for t in [*batch.tickets, *batch.blocked_tickets]:
        lines.append(f"### {t.ticket_id} - {t.symbol}")
        lines.append(f"- Status: `{t.status}`")
        lines.append(f"- Rationale: {t.rationale}")
        if t.risk_notes:
            for note in t.risk_notes:
                lines.append(f"- Note: {note}")
        if t.review_gate_status:
            lines.append(f"- TradingAgents gate: `{t.review_gate_status}`")
        if t.review_gate_reason:
            lines.append(f"- Gate reason: {t.review_gate_reason}")
        for caveat in t.review_execution_caveats:
            lines.append(f"- Gate caveat: {caveat}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    as_of = _resolve_as_of(args.as_of)
    execution_config = _load_execution_config(Path(args.execution_config))
    actions, _summary, signals = _load_actions(args, as_of)
    batch = build_execution_batch(
        actions=actions,
        as_of=as_of.isoformat(),
        config=execution_config,
        source_daily_signals_path=args.source_daily_signals_path,
        account_hint=args.account_hint,
        option_strategy_reports=_option_strategy_reports(signals),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{as_of.isoformat()}.json"
    md_path = output_dir / f"{as_of.isoformat()}.md"
    json_path.write_text(json.dumps(batch.to_dict(), indent=2, default=str))
    md_path.write_text(_format_batch_markdown(batch))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(
        "Summary: "
        f"ready={batch.summary['ready_count']} "
        f"blocked={batch.summary['blocked_count']} "
        f"option_intents={batch.summary['option_intent_count']}"
    )


if __name__ == "__main__":
    main()
