"""One-command broker-gated trading workflow.

This wraps ticket generation and writes an operator packet for Codex/RH MCP
review. It still does not call Robinhood or place orders from inside the repo.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from portfolio.execution import build_execution_batch, execution_config_from_dict
from trade_tickets import (
    _format_batch_markdown,
    _load_actions,
    _option_strategy_reports,
)
import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    p.add_argument("--positions", default="portfolio/positions.json")
    p.add_argument("--sizing-config", default="configs/sizing.yaml")
    p.add_argument("--execution-config", default="configs/execution.yaml")
    p.add_argument("--tickets-dir", default="reports/trade_tickets")
    p.add_argument("--workflow-dir", default="reports/trade_workflow")
    p.add_argument("--as-of", default=None, help="Run date, YYYY-MM-DD; defaults to today.")
    p.add_argument("--no-prices", action="store_true", help="Skip price fetch; executable tickets will be blocked.")
    p.add_argument("--account-hint", default=None)
    return p.parse_args()


def _resolve_as_of(raw: str | None) -> date:
    if raw is None:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--as-of must be YYYY-MM-DD; got {raw!r}") from exc


def _load_execution_config(path: Path):
    data = yaml.safe_load(path.read_text()) or {}
    return execution_config_from_dict(data)


def _review_prompt(*, ticket_path: Path, fills_template_path: Path, account_hint: str | None) -> str:
    account_text = (
        f"Use Robinhood account {account_hint}."
        if account_hint
        else "Call `rh.get_accounts`; if there are multiple accounts, ask me which one to use."
    )
    return "\n".join(
        [
            "# Codex Robinhood Review Request",
            "",
            f"Review ready equity tickets in `{ticket_path}`.",
            account_text,
            "",
            "Do not place any order until I explicitly confirm.",
            "",
            "Workflow:",
            "",
            "1. Read the ticket batch JSON.",
            "2. Use `rh.get_accounts` if the account is not already specified.",
            "3. For each ready equity ticket, call `rh.get_portfolio`, `rh.get_equity_positions`, `rh.get_equity_quotes`, and `rh.get_equity_tradability`.",
            "4. Call `rh.review_equity_order` for each ready ticket using the ticket's `time_in_force` exactly.",
            "5. Present the review output, estimated cost/proceeds, and alerts.",
            "6. Place only the orders I explicitly approve.",
            f"7. Record placed/fill details into `{fills_template_path}` for `trade_reconcile.py`.",
            "8. Run `trade_reconcile.py` after fills settle; only use `--apply-position-update` after reviewing the audit artifact.",
            "",
        ]
    )


def _fills_template(batch) -> dict:
    return {
        "orders": [
            {
                "ticket_id": ticket.ticket_id,
                "order_id": "",
                "symbol": ticket.symbol,
                "side": ticket.side,
                "filled_quantity": ticket.quantity or "",
                "average_price": "",
                "state": "",
            }
            for ticket in batch.tickets
        ]
    }


def main() -> None:
    args = _parse_args()
    as_of = _resolve_as_of(args.as_of)
    execution_config = _load_execution_config(Path(args.execution_config))
    actions, _summary, signals = _load_actions(args, as_of)
    ticket_path_for_prompt = Path(args.tickets_dir) / f"{as_of.isoformat()}.json"
    batch = build_execution_batch(
        actions=actions,
        as_of=as_of.isoformat(),
        config=execution_config,
        source_daily_signals_path=None,
        account_hint=args.account_hint,
        option_strategy_reports=_option_strategy_reports(signals),
    )

    tickets_dir = Path(args.tickets_dir)
    workflow_dir = Path(args.workflow_dir)
    tickets_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)

    ticket_json_path = tickets_dir / f"{as_of.isoformat()}.json"
    ticket_md_path = tickets_dir / f"{as_of.isoformat()}.md"
    fills_template_path = workflow_dir / f"{as_of.isoformat()}_fills_template.json"
    review_prompt_path = workflow_dir / f"{as_of.isoformat()}_codex_review.md"

    ticket_json_path.write_text(json.dumps(batch.to_dict(), indent=2, default=str))
    ticket_md_path.write_text(_format_batch_markdown(batch))
    fills_template_path.write_text(json.dumps(_fills_template(batch), indent=2))
    review_prompt_path.write_text(
        _review_prompt(
            ticket_path=ticket_path_for_prompt,
            fills_template_path=fills_template_path,
            account_hint=args.account_hint,
        )
    )

    print(f"Wrote tickets: {ticket_json_path}")
    print(f"Wrote ticket summary: {ticket_md_path}")
    print(f"Wrote Codex review prompt: {review_prompt_path}")
    print(f"Wrote fills template: {fills_template_path}")
    print(
        "Summary: "
        f"ready={batch.summary['ready_count']} "
        f"blocked={batch.summary['blocked_count']} "
        f"option_intents={batch.summary['option_intent_count']}"
    )
    if batch.tickets:
        print("")
        print("Next message to Codex:")
        print(f"Review the ready equity tickets in {ticket_json_path}. Do not place anything until I confirm.")


if __name__ == "__main__":
    main()
