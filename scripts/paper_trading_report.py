"""Weekly paper-trading dry-run report.

Joins per-day recommendations (logged by daily_signals.py via
portfolio.paper_trading) and per-day executions (logged by
scripts/log_execution.py) over a date range and produces a markdown
report covering:

  - Follow-rate: of N actionable recommendations, how many had a matching
    execution? Split by action (BUY / ADD / TRIM / EXIT) and by
    direction (bullish / bearish / neutral).
  - Divergence breakdown: followed / partial / overridden / ignored,
    per action.
  - Per-symbol attribution: for each followed BUY/ADD, compute the
    forward return from fill_price to a chosen evaluation date (default:
    same horizon as the model's primary, 60d) via yfinance, and
    aggregate as "if you had followed every BUY at the fill price, your
    P&L over the window was …".
  - Unattributed executions: trades that don't match any recommendation
    — your pure-discretionary book.
  - Per-day divergence notes if `override_reason` is filled in.

Usage:

    .venv/bin/python scripts/paper_trading_report.py \\
        --from-date 2026-06-01 --to-date 2026-06-07 \\
        --output reports/paper_trading/weekly_2026-06-01_to_2026-06-07.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.paper_trading import (  # noqa: E402
    ExecutionRecord,
    RecommendationRecord,
    join_recommendations_to_executions,
    load_executions_range,
    load_recommendations_range,
    unattributed_executions,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", required=True, help="ISO start date (inclusive).")
    p.add_argument("--to-date", required=True, help="ISO end date (inclusive).")
    p.add_argument("--base-dir", default="reports/paper_trading",
                   help="Paper-trading log directory.")
    p.add_argument("--output", default=None,
                   help=(
                       "Markdown output path. Defaults to "
                       "<base-dir>/weekly_<from>_to_<to>.md."
                   ))
    p.add_argument("--eval-horizon-days", type=int, default=60,
                   help=(
                       "Forward-return horizon to evaluate followed BUYs. "
                       "Default 60 (matches the model's primary horizon)."
                   ))
    p.add_argument("--no-prices", action="store_true",
                   help="Skip yfinance price fetching (offline / smoke).")
    return p.parse_args()


def _validate_date(s: str, label: str) -> str:
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        print(f"{label} must be YYYY-MM-DD; got {s!r}", file=sys.stderr)
        sys.exit(2)
    return s


def _fetch_realized_return(
    symbol: str, *, fill_date: str, fill_price: float, eval_horizon_days: int,
) -> float | None:
    """Pull yfinance closes; compute (price_at_fill+H − fill_price)/fill_price.

    Returns None on any failure (offline / delisted / etc.). The reporter
    surfaces these as "n/a" rather than crashing.
    """
    try:
        import yfinance as yf  # local import — runtime only
    except Exception:
        return None
    try:
        start = datetime.strptime(fill_date, "%Y-%m-%d").date()
        end = start + timedelta(days=int(eval_horizon_days * 1.7) + 5)  # extra calendar slack
        raw = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if raw is None or getattr(raw, "empty", True):
        return None
    try:
        import pandas as pd
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if len(closes) <= eval_horizon_days:
            return None
        end_close = float(closes.iloc[eval_horizon_days])
    except Exception:
        return None
    if fill_price <= 0:
        return None
    return (end_close / fill_price) - 1.0


def _summarize_follow_rate(joined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bucket joined rows by action × divergence_type and return counts."""
    by_action: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    for row in joined_rows:
        rec: RecommendationRecord = row["recommendation"]
        dt = row["divergence_type"]
        by_action[rec.action][dt] += 1
        overall[dt] += 1
    return {"by_action": dict(by_action), "overall": overall}


def _render_report(
    *,
    from_date: str,
    to_date: str,
    recs_by_date: dict[str, list[RecommendationRecord]],
    execs_by_date: dict[str, list[ExecutionRecord]],
    joined: list[dict[str, Any]],
    unattributed: list[ExecutionRecord],
    eval_horizon_days: int,
    no_prices: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# Paper-trading report — {from_date} → {to_date}")
    lines.append("")
    n_days_recs = len(recs_by_date)
    n_recs = sum(len(v) for v in recs_by_date.values())
    n_actionable = sum(
        1 for v in recs_by_date.values() for r in v
        if r.action not in ("HOLD", "SKIP", "REVIEW")
    )
    n_days_execs = len(execs_by_date)
    n_execs = sum(len(v) for v in execs_by_date.values())
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Days with recommendations on file: **{n_days_recs}**")
    lines.append(f"- Total recommendations emitted: **{n_recs}** "
                 f"(actionable BUY/ADD/TRIM/EXIT: **{n_actionable}**)")
    lines.append(f"- Days with executions on file: **{n_days_execs}**")
    lines.append(f"- Total executions logged: **{n_execs}**")
    lines.append(f"- Unattributed executions (pure discretionary): "
                 f"**{len(unattributed)}**")
    lines.append("")

    # Follow-rate.
    summary = _summarize_follow_rate(joined)
    overall = summary["overall"]
    lines.append("## Follow-rate")
    lines.append("")
    lines.append(
        "| Divergence | Count | Share |"
    )
    lines.append("|---|---:|---:|")
    total = sum(overall.values())
    for d in ("followed", "partial", "overridden", "ignored", "n_a"):
        c = overall.get(d, 0)
        share = (c / total * 100) if total else 0
        lines.append(f"| {d} | {c} | {share:.1f}% |")
    actionable_total = total - overall.get("n_a", 0)
    if actionable_total > 0:
        followed_share = overall.get("followed", 0) / actionable_total * 100
        lines.append("")
        lines.append(
            f"**Actionable follow-rate: {overall.get('followed', 0)}/"
            f"{actionable_total} = {followed_share:.1f}%**"
        )
    lines.append("")

    # By-action breakdown.
    lines.append("### By action")
    lines.append("")
    lines.append("| Action | followed | partial | overridden | ignored |")
    lines.append("|---|---:|---:|---:|---:|")
    for action in ("BUY", "ADD", "TRIM", "EXIT"):
        c = summary["by_action"].get(action, Counter())
        if not c:
            continue
        lines.append(
            f"| {action} | {c.get('followed', 0)} | {c.get('partial', 0)} | "
            f"{c.get('overridden', 0)} | {c.get('ignored', 0)} |"
        )
    lines.append("")

    # Per-symbol attribution for followed BUY/ADDs.
    followed_buys = [
        r for r in joined
        if r["divergence_type"] == "followed"
        and r["recommendation"].action in ("BUY", "ADD")
        and r["executions"]
    ]
    if followed_buys:
        lines.append("## Per-symbol attribution — followed BUY/ADD")
        lines.append("")
        if no_prices:
            lines.append(f"_Prices skipped (`--no-prices`). Showing fills only._")
            lines.append("")
            lines.append("| Symbol | Rec date | Action | Fill date | Fill price | Shares |")
            lines.append("|---|---|---|---|---:|---:|")
            for row in followed_buys:
                rec = row["recommendation"]
                e = row["executions"][0]
                lines.append(
                    f"| {rec.symbol} | {rec.as_of_date} | {rec.action} | "
                    f"{e.trade_date} | {e.fill_price:.2f} | {e.shares:.0f} |"
                )
        else:
            lines.append(
                f"_Realized returns at {eval_horizon_days}d horizon from "
                f"fill date. yfinance failures show as n/a._"
            )
            lines.append("")
            lines.append(
                f"| Symbol | Rec date | Fill date | Fill price | "
                f"Shares | Ret_{eval_horizon_days}d |"
            )
            lines.append("|---|---|---|---:|---:|---:|")
            total_pnl_usd = 0.0
            n_evaluated = 0
            for row in followed_buys:
                rec = row["recommendation"]
                e = row["executions"][0]
                ret = _fetch_realized_return(
                    e.symbol, fill_date=e.trade_date,
                    fill_price=e.fill_price,
                    eval_horizon_days=eval_horizon_days,
                )
                ret_str = f"{ret*100:+.2f}%" if ret is not None else "n/a"
                pnl = (ret * e.fill_price * e.shares) if ret is not None else 0.0
                if ret is not None:
                    total_pnl_usd += pnl
                    n_evaluated += 1
                lines.append(
                    f"| {rec.symbol} | {rec.as_of_date} | {e.trade_date} | "
                    f"{e.fill_price:.2f} | {e.shares:.0f} | {ret_str} |"
                )
            if n_evaluated > 0:
                lines.append("")
                lines.append(
                    f"**Aggregate realized P&L on {n_evaluated} followed BUY/ADDs at "
                    f"{eval_horizon_days}d: ${total_pnl_usd:,.0f}**"
                )
        lines.append("")

    # Override reasons.
    overrides_with_reason = [
        e for execs in execs_by_date.values() for e in execs
        if e.override_reason
    ]
    if overrides_with_reason:
        lines.append("## Override reasons")
        lines.append("")
        lines.append("| Trade date | Symbol | Side | Shares | Reason |")
        lines.append("|---|---|---|---:|---|")
        for e in overrides_with_reason:
            lines.append(
                f"| {e.trade_date} | {e.symbol} | {e.side} | "
                f"{e.shares:.0f} | {e.override_reason} |"
            )
        lines.append("")

    # Unattributed executions.
    if unattributed:
        lines.append("## Unattributed executions (pure discretionary)")
        lines.append("")
        lines.append(
            "_Trades that don't match any recommendation. Either pure "
            "discretionary decisions or trades on names outside the universe._"
        )
        lines.append("")
        lines.append("| Trade date | Symbol | Side | Shares | Fill price | Reason |")
        lines.append("|---|---|---|---:|---:|---|")
        for e in unattributed:
            reason = e.override_reason or "—"
            lines.append(
                f"| {e.trade_date} | {e.symbol} | {e.side} | "
                f"{e.shares:.0f} | {e.fill_price:.2f} | {reason} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    from_date = _validate_date(args.from_date, "--from-date")
    to_date = _validate_date(args.to_date, "--to-date")
    if from_date > to_date:
        print("--from-date must be ≤ --to-date", file=sys.stderr)
        return 2
    base_dir = Path(args.base_dir)
    recs_by_date = load_recommendations_range(base_dir, from_date=from_date, to_date=to_date)
    execs_by_date = load_executions_range(base_dir, from_date=from_date, to_date=to_date)
    joined = join_recommendations_to_executions(recs_by_date, execs_by_date)
    unattributed = unattributed_executions(recs_by_date, execs_by_date)

    output_path = Path(args.output) if args.output else (
        base_dir / f"weekly_{from_date}_to_{to_date}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md = _render_report(
        from_date=from_date,
        to_date=to_date,
        recs_by_date=recs_by_date,
        execs_by_date=execs_by_date,
        joined=joined,
        unattributed=unattributed,
        eval_horizon_days=int(args.eval_horizon_days),
        no_prices=bool(args.no_prices),
    )
    output_path.write_text(md)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
