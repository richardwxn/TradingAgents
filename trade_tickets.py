"""Generate broker-gated trade tickets from daily signal actions.

The output JSON is the source of truth for Codex/Robinhood MCP execution.
This CLI never calls Robinhood and never places orders.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import replace
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
    fetch_sector_shocks,
)
from portfolio.execution import (
    ExecutionBatch,
    ExecutionConfig,
    execution_config_from_dict,
    build_execution_batch,
)
from portfolio.signals import Action, PriceContext, Signal, compute_actions, load_latest_signals


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
    p.add_argument(
        "--ensure-tradingagents-review",
        action="store_true",
        help="Run TradingAgents review for BUY/ADD candidates missing a valid gate.",
    )
    p.add_argument(
        "--review-mode",
        choices=("shadow", "enforce"),
        default="shadow",
        help="shadow records gates without blocking; enforce applies ticket blocking.",
    )
    p.add_argument(
        "--tradingagents-review-top-n",
        type=int,
        default=None,
        help="Max BUY/ADD candidates to review; defaults to sizing config.",
    )
    p.add_argument("--tradingagents-review-provider", default="openai")
    p.add_argument("--tradingagents-review-model", default="gpt-5.4-mini")
    p.add_argument("--tradingagents-review-base-url", default=None)
    return p.parse_args()


def _load_execution_config(path: Path):
    data = yaml.safe_load(path.read_text()) or {}
    return execution_config_from_dict(data)


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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


def _valid_review_gate(gate: dict[str, Any] | None) -> bool:
    return isinstance(gate, dict) and gate.get("status") == "ok"


def _review_top_n(args: argparse.Namespace, sizing_path: Path) -> int:
    if args.tradingagents_review_top_n is not None:
        return max(0, int(args.tradingagents_review_top_n))
    data = _load_yaml_dict(sizing_path)
    return max(0, int(data.get("tradingagents_review_top_screener_n", 5)))


def _select_review_candidates(
    actions: list[Action],
    signals: dict[str, Signal | None],
    *,
    top_n: int,
) -> list[Action]:
    if top_n <= 0:
        return []
    candidates: list[Action] = []
    for action in actions:
        if action.action not in {"BUY", "ADD"}:
            continue
        sig = signals.get(action.symbol.upper())
        if sig is None or not sig.source_path:
            continue
        if _valid_review_gate(sig.tradingagents_review_gate):
            continue
        candidates.append(action)
    candidates.sort(
        key=lambda a: (
            abs(float(a.delta_pp or 0.0)),
            float(a.confidence or 0.0),
            abs(float(a.composite or 0.0)),
            a.symbol,
        ),
        reverse=True,
    )
    return candidates[:top_n]


def _review_metadata(
    *,
    block: dict[str, Any],
    mode: str,
    enforced: bool,
    source_path: str,
) -> dict[str, Any]:
    gate = block.get("gate") if isinstance(block.get("gate"), dict) else {}
    return {
        "enabled": True,
        "mode": mode,
        "enforced": enforced,
        "source_path": source_path,
        "status": block.get("status"),
        "provider": block.get("provider"),
        "model": block.get("model"),
        "review_type": block.get("review_type"),
        "gate": dict(gate),
        "reason": gate.get("reason"),
        "execution_caveats": list(gate.get("execution_caveats") or []),
        "error": block.get("error"),
    }


def _failed_review_metadata(
    *,
    symbol: str,
    mode: str,
    enforced: bool,
    source_path: str,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": mode,
        "enforced": enforced,
        "source_path": source_path,
        "symbol": symbol,
        "status": status,
        "gate": {},
        "reason": error,
        "execution_caveats": [],
        "error": error,
    }


def _ensure_tradingagents_reviews(
    actions: list[Action],
    signals: dict[str, Signal | None],
    *,
    args: argparse.Namespace,
    execution_config: ExecutionConfig,
) -> tuple[list[Action], dict[str, Any]]:
    top_n = _review_top_n(args, Path(args.sizing_config))
    candidates = _select_review_candidates(actions, signals, top_n=top_n)
    if not args.ensure_tradingagents_review:
        return actions, {
            "enabled": False,
            "mode": args.review_mode,
            "enforced": False,
            "top_n": top_n,
            "reviewed_symbols": [],
            "manual_review_symbols": [],
            "blocked_symbols": [],
            "failed_symbols": [],
        }

    from tradingagents.analysis_only.agent_review import (
        report_context_from_payload,
        run_report_agent_review,
    )

    enforced = (
        args.review_mode == "enforce"
        and execution_config.tradingagents_review_apply_to_tickets
    )
    review_by_symbol: dict[str, dict[str, Any]] = {}
    failed_symbols: list[str] = []

    for action in candidates:
        symbol = action.symbol.upper()
        sig = signals.get(symbol)
        source_path = str(sig.source_path) if sig is not None else ""
        payload = _read_json(Path(source_path)) if source_path else None
        if not payload:
            failed_symbols.append(symbol)
            review_by_symbol[symbol] = _failed_review_metadata(
                symbol=symbol,
                mode=args.review_mode,
                enforced=enforced,
                source_path=source_path,
                status="report_read_error",
                error="Could not read source report for TradingAgents review.",
            )
            continue
        try:
            context = report_context_from_payload(payload)
            block = run_report_agent_review(
                symbol=symbol,
                as_of_date=str(context.get("as_of_date") or ""),
                report_context=context,
                provider=args.tradingagents_review_provider,
                quick_model=args.tradingagents_review_model,
                deep_model=args.tradingagents_review_model,
                base_url=args.tradingagents_review_base_url,
            )
            review_by_symbol[symbol] = _review_metadata(
                block=block,
                mode=args.review_mode,
                enforced=enforced,
                source_path=source_path,
            )
            if block.get("status") != "ok":
                failed_symbols.append(symbol)
        except Exception as exc:
            failed_symbols.append(symbol)
            review_by_symbol[symbol] = _failed_review_metadata(
                symbol=symbol,
                mode=args.review_mode,
                enforced=enforced,
                source_path=source_path,
                status="review_runtime_error",
                error=str(exc),
            )

    updated: list[Action] = []
    for action in actions:
        meta = review_by_symbol.get(action.symbol.upper())
        if not meta:
            updated.append(action)
            continue
        gate = meta.get("gate") or {}
        updated.append(
            replace(
                action,
                review_gate_status=(
                    str(gate.get("ticket_gate"))
                    if gate.get("ticket_gate")
                    else action.review_gate_status
                ),
                review_gate_reason=(
                    str(gate.get("reason"))
                    if gate.get("reason")
                    else action.review_gate_reason
                ),
                review_execution_caveats=[
                    str(x)
                    for x in (gate.get("execution_caveats") or [])
                    if str(x).strip()
                ],
                tradingagents_review=meta,
            )
        )

    blocked_symbols = sorted(
        sym
        for sym, meta in review_by_symbol.items()
        if (meta.get("gate") or {}).get("ticket_gate") == "block_buy_add"
    )
    manual_symbols = sorted(
        sym
        for sym, meta in review_by_symbol.items()
        if (meta.get("gate") or {}).get("ticket_gate") == "manual_review"
    )
    summary = {
        "enabled": True,
        "mode": args.review_mode,
        "enforced": enforced,
        "top_n": top_n,
        "candidate_symbols": [a.symbol.upper() for a in candidates],
        "reviewed_symbols": sorted(review_by_symbol),
        "manual_review_symbols": manual_symbols,
        "blocked_symbols": blocked_symbols,
        "failed_symbols": sorted(set(failed_symbols)),
        "provider": args.tradingagents_review_provider,
        "model": args.tradingagents_review_model,
    }
    return updated, summary


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

    sector_shocks = {}
    if not args.no_prices and sizing_config.sector_shock_guard_enabled:
        sector_shocks = fetch_sector_shocks(
            signals,
            config=sizing_config,
            as_of=as_of,
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
        sector_shocks=sector_shocks,
    )
    return actions, summary, signals


def _format_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _option_rank(ticket) -> str:
    rank = (ticket.details or {}).get("option_intent_rank")
    return "" if rank is None else f"#{rank}"


def _option_score(ticket) -> str:
    details = ticket.details or {}
    score = details.get("option_intent_score")
    label = details.get("option_intent_score_label")
    if score is None:
        return ""
    suffix = f" {label}" if label else ""
    return f"{float(score):.0f}/100{suffix}"


def _option_contract_label(ticket) -> str:
    if getattr(ticket, "asset_type", "") != "option_intent":
        return ""
    details = ticket.details or {}
    expiry = details.get("expiry") or "—"
    dte = details.get("dte")
    dte_label = "—" if dte is None else str(dte)
    long_strike = _safe_float(details.get("long_strike"))
    short_strike = _safe_float(details.get("short_strike"))
    strike = _safe_float(details.get("strike"))
    source_action = str(getattr(ticket, "source_action", "") or details.get("type") or "")

    if long_strike is not None or short_strike is not None:
        return (
            f"{expiry} / {dte_label} DTE / "
            f"{_format_money(long_strike)}-{_format_money(short_strike)} call spread"
        )

    if strike is None:
        return ""
    option_type = details.get("option_type")
    if not option_type:
        if source_action == "sell_put":
            option_type = "put"
        elif source_action in {"sell_call", "buy_call_spread"}:
            option_type = "call"
    suffix = f" {option_type}" if option_type else ""
    return f"{expiry} / {dte_label} DTE / {_format_money(strike)}{suffix}"


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


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

    review_summary = (batch.summary or {}).get("tradingagents_review") or {}
    if review_summary.get("enabled"):
        mode = review_summary.get("mode", "shadow")
        enforced = review_summary.get("enforced", False)
        lines.extend([
            "## TradingAgents Review Gate",
            "",
            f"- Mode: `{mode}` ({'enforced' if enforced else 'shadow only'})",
            f"- Reviewed: {', '.join(review_summary.get('reviewed_symbols') or []) or 'none'}",
            f"- Manual review: {', '.join(review_summary.get('manual_review_symbols') or []) or 'none'}",
            f"- Block buy/add: {', '.join(review_summary.get('blocked_symbols') or []) or 'none'}",
            f"- Failures: {', '.join(review_summary.get('failed_symbols') or []) or 'none'}",
            "",
        ])

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
            "| Ticket | Asset | Symbol | Side | Contract | Source | Option Rank | Option Score | Reason |",
            "|---|---|---|---|---|---|---:|---:|---|",
        ])
        for t in batch.blocked_tickets:
            reason = _markdown_cell(t.blocked_reason)
            contract = _markdown_cell(_option_contract_label(t))
            rank = _option_rank(t)
            score = _option_score(t)
            lines.append(
                f"| `{t.ticket_id}` | {t.asset_type} | {t.symbol} | {t.side} | "
                f"{contract} | {t.source_action} | {rank} | {score} | {reason} |"
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
        option_rank = _option_rank(t)
        option_score = _option_score(t)
        if option_score:
            option_contract = _option_contract_label(t)
            if option_contract:
                lines.append(f"- Option contract: {option_contract}")
            lines.append(f"- Option intent rank: {option_rank or 'n/a'}")
            lines.append(f"- Option intent score: {option_score}")
            lines.append(
                "- Option signal context: "
                f"direction `{(t.details or {}).get('report_direction')}`, "
                f"composite `{(t.details or {}).get('report_composite')}`, "
                f"confidence `{(t.details or {}).get('report_confidence')}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    as_of = _resolve_as_of(args.as_of)
    execution_config = _load_execution_config(Path(args.execution_config))
    actions, _summary, signals = _load_actions(args, as_of)
    actions, review_summary = _ensure_tradingagents_reviews(
        actions,
        signals,
        args=args,
        execution_config=execution_config,
    )
    batch_execution_config = execution_config
    if args.ensure_tradingagents_review and args.review_mode == "shadow":
        batch_execution_config = replace(
            execution_config,
            tradingagents_review_apply_to_tickets=False,
        )
    batch = build_execution_batch(
        actions=actions,
        as_of=as_of.isoformat(),
        config=batch_execution_config,
        source_daily_signals_path=args.source_daily_signals_path,
        account_hint=args.account_hint,
        option_strategy_reports=_option_strategy_reports(signals),
    )
    batch = replace(
        batch,
        summary={
            **batch.summary,
            "tradingagents_review": review_summary,
        },
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
