"""Weekly ML-gate shadow performance report.

Builds a side-by-side shadow report from paper-trading recommendation logs:

  - factor_production: current factor BUY/ADD recommendations.
  - ml_gated_factor: factor BUY/ADD recommendations allowed by the ML gate.
  - ml_only: ML BUY triggers regardless of factor production action.
  - hybrid: simple factor-probability + ML-score shadow proxy.

The report is intentionally read-only. It never changes production
recommendations or trade tickets.

Usage:

    .venv/bin/python scripts/ml_gate_shadow_report.py \\
        --from-date 2026-06-09 --to-date 2026-06-09 \\
        --no-prices
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.paper_trading import (  # noqa: E402
    RecommendationRecord,
    load_recommendations_range,
)


BUY_ACTIONS = {"BUY", "ADD"}
SELL_ACTIONS = {"TRIM", "EXIT"}
STRATEGIES = ("factor_production", "ml_gated_factor", "ml_only", "hybrid")


@dataclass(frozen=True)
class Outcome:
    raw_return: float | None = None
    benchmark_return: float | None = None

    @property
    def alpha_return(self) -> float | None:
        if self.raw_return is None or self.benchmark_return is None:
            return None
        return self.raw_return - self.benchmark_return


@dataclass(frozen=True)
class ShadowRow:
    as_of_date: str
    symbol: str
    action: str
    target_weight: float
    factor_score: float
    ml_score: float | None
    hybrid_score: float | None
    last_close: float | None
    outcome: Outcome
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    row: ShadowRow
    strategy: str
    selected: bool
    score: float | None
    weight: float


OutcomeProvider = Callable[[RecommendationRecord], Outcome]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="ISO start date, inclusive.")
    parser.add_argument("--to-date", required=True, help="ISO end date, inclusive.")
    parser.add_argument("--base-dir", default="reports/paper_trading", help="Paper-trading log directory.")
    parser.add_argument("--output", default=None, help="Markdown output path.")
    parser.add_argument("--model", default="ridge_return", help="ML shadow model name.")
    parser.add_argument("--horizon", default="ret_60d", help="ML shadow horizon key, e.g. ret_60d.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker for alpha outcomes.")
    parser.add_argument("--threshold", type=float, default=0.55, help="ML trigger threshold.")
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.55,
        help="Trigger threshold for the hybrid factor/ML proxy.",
    )
    parser.add_argument(
        "--ml-only-weight",
        type=float,
        default=0.05,
        help="Default shadow target weight for ML-only rows with no factor target.",
    )
    parser.add_argument(
        "--top-k-per-date",
        type=int,
        default=10,
        help="Max BUY candidates per strategy per recommendation date.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="One-way turnover cost in basis points for proxy net return.",
    )
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help="Skip yfinance outcomes; useful for current/pending reports and CI smoke runs.",
    )
    return parser.parse_args()


def _validate_date(value: str, flag: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        print(f"{flag} must be YYYY-MM-DD; got {value!r}", file=sys.stderr)
        sys.exit(2)
    return value


def _horizon_days(horizon: str) -> int:
    digits = "".join(ch for ch in str(horizon) if ch.isdigit())
    return int(digits or "60")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp_probability(value: float | None, default: float = 0.5) -> float:
    if value is None:
        return default
    return min(1.0, max(0.0, float(value)))


def _factor_score(rec: RecommendationRecord) -> float:
    confidence = _clamp_probability(_safe_float(rec.confidence), default=0.5)
    direction = (rec.direction or "").lower()
    if direction == "bullish":
        return confidence
    if direction == "bearish":
        return 1.0 - confidence
    return 0.5


def _nested_score(payload: Any) -> float | None:
    if isinstance(payload, dict):
        return _safe_float(payload.get("score"))
    return _safe_float(payload)


def ml_score(rec: RecommendationRecord, *, model: str, horizon: str) -> float | None:
    """Return the calibrated ML score for a recommendation, when present."""

    shadow = rec.ml_shadow or {}
    models = shadow.get("models") if isinstance(shadow, dict) else None
    if isinstance(models, dict):
        return _nested_score((models.get(model) or {}).get(horizon))
    if isinstance(shadow, dict):
        return _nested_score((shadow.get(model) or {}).get(horizon))
    return None


def _hybrid_score(factor_score: float, ml_score_value: float | None) -> float | None:
    if ml_score_value is None:
        return None
    return _clamp_probability((factor_score + ml_score_value) / 2.0)


def _weight_for_row(row: ShadowRow, *, default_ml_weight: float) -> float:
    if row.target_weight > 0:
        return row.target_weight
    return max(0.0, float(default_ml_weight))


def build_shadow_rows(
    recs_by_date: dict[str, list[RecommendationRecord]],
    *,
    model: str,
    horizon: str,
    outcome_provider: OutcomeProvider | None = None,
) -> list[ShadowRow]:
    outcome_provider = outcome_provider or (lambda _rec: Outcome())
    rows: list[ShadowRow] = []
    for date_key in sorted(recs_by_date):
        for rec in recs_by_date[date_key]:
            factor = _factor_score(rec)
            ml = ml_score(rec, model=model, horizon=horizon)
            target_weight = max(0.0, _safe_float(rec.target_weight) or 0.0)
            rows.append(
                ShadowRow(
                    as_of_date=rec.as_of_date,
                    symbol=rec.symbol.upper(),
                    action=(rec.action or "").upper(),
                    target_weight=target_weight,
                    factor_score=factor,
                    ml_score=ml,
                    hybrid_score=_hybrid_score(factor, ml),
                    last_close=_safe_float(rec.last_close),
                    outcome=outcome_provider(rec),
                    notes=tuple(rec.notes or ()),
                )
            )
    return rows


def _raw_decision(
    row: ShadowRow,
    *,
    strategy: str,
    threshold: float,
    hybrid_threshold: float,
    default_ml_weight: float,
) -> Decision:
    sell_like = row.action in SELL_ACTIONS
    factor_selected = row.action in BUY_ACTIONS
    if strategy == "factor_production":
        return Decision(row, strategy, factor_selected, row.factor_score, row.target_weight)
    if strategy == "ml_gated_factor":
        selected = factor_selected and row.ml_score is not None and row.ml_score >= threshold
        return Decision(row, strategy, selected, row.ml_score, row.target_weight if selected else 0.0)
    if strategy == "ml_only":
        selected = (not sell_like) and row.ml_score is not None and row.ml_score >= threshold
        return Decision(
            row,
            strategy,
            selected,
            row.ml_score,
            _weight_for_row(row, default_ml_weight=default_ml_weight) if selected else 0.0,
        )
    if strategy == "hybrid":
        selected = (
            (not sell_like)
            and row.hybrid_score is not None
            and row.hybrid_score >= hybrid_threshold
        )
        return Decision(
            row,
            strategy,
            selected,
            row.hybrid_score,
            _weight_for_row(row, default_ml_weight=default_ml_weight) if selected else 0.0,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def strategy_decisions(
    rows: Iterable[ShadowRow],
    *,
    strategy: str,
    threshold: float,
    hybrid_threshold: float,
    default_ml_weight: float,
    top_k_per_date: int,
) -> list[Decision]:
    """Return selected and non-selected decisions, with top-k caps applied."""

    raw = [
        _raw_decision(
            row,
            strategy=strategy,
            threshold=threshold,
            hybrid_threshold=hybrid_threshold,
            default_ml_weight=default_ml_weight,
        )
        for row in rows
    ]
    if strategy == "factor_production" or top_k_per_date <= 0:
        return raw

    selected_by_date: dict[str, list[Decision]] = defaultdict(list)
    for decision in raw:
        if decision.selected:
            selected_by_date[decision.row.as_of_date].append(decision)

    keep: set[tuple[str, str, str]] = set()
    for date_key, decisions in selected_by_date.items():
        ranked = sorted(
            decisions,
            key=lambda d: (float(d.score or -1.0), d.row.factor_score, d.row.symbol),
            reverse=True,
        )
        for decision in ranked[:top_k_per_date]:
            keep.add((decision.strategy, date_key, decision.row.symbol))

    capped: list[Decision] = []
    for decision in raw:
        key = (decision.strategy, decision.row.as_of_date, decision.row.symbol)
        if decision.selected and key not in keep:
            capped.append(Decision(decision.row, decision.strategy, False, decision.score, 0.0))
        else:
            capped.append(decision)
    return capped


def _weighted_average(pairs: Iterable[tuple[float, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in pairs:
        if weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def _rate(values: Iterable[bool]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _max_drawdown(equity: list[float]) -> float | None:
    if len(equity) < 2:
        return None
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd


def _turnover_by_date(selected: list[Decision]) -> dict[str, float]:
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for decision in selected:
        if decision.selected and decision.weight > 0:
            by_date[decision.row.as_of_date][decision.row.symbol] = decision.weight
    prev: dict[str, float] = {}
    out: dict[str, float] = {}
    for date_key in sorted(by_date):
        current = by_date[date_key]
        symbols = set(prev) | set(current)
        turnover = sum(abs(current.get(sym, 0.0) - prev.get(sym, 0.0)) for sym in symbols)
        out[date_key] = turnover
        prev = current
    return out


def summarize_strategy(
    decisions: list[Decision],
    *,
    strategy: str,
    cost_bps: float,
    top_k_per_date: int,
) -> dict[str, Any]:
    selected = [d for d in decisions if d.selected and d.weight > 0]
    outcome_rows = [
        d for d in selected
        if d.row.outcome.raw_return is not None
    ]
    alpha_rows = [
        d for d in selected
        if d.row.outcome.alpha_return is not None
    ]
    turnover = _turnover_by_date(selected)
    cost_rate = max(0.0, float(cost_bps)) / 10000.0

    returns_by_date: dict[str, list[Decision]] = defaultdict(list)
    for decision in selected:
        returns_by_date[decision.row.as_of_date].append(decision)

    equity = [1.0]
    evaluated_dates = 0
    for date_key in sorted(returns_by_date):
        date_decisions = returns_by_date[date_key]
        if not date_decisions:
            continue
        if any(d.row.outcome.raw_return is None for d in date_decisions):
            continue
        gross = sum(d.weight * float(d.row.outcome.raw_return) for d in date_decisions)
        net = gross - (turnover.get(date_key, 0.0) * cost_rate)
        equity.append(equity[-1] * (1.0 + net))
        evaluated_dates += 1

    return {
        "strategy": strategy,
        "selected": len(selected),
        "dates": len({d.row.as_of_date for d in selected}),
        "avg_exposure": _weighted_average(
            (sum(d.weight for d in selected if d.row.as_of_date == date_key), 1.0)
            for date_key in sorted({d.row.as_of_date for d in selected})
        ),
        "turnover": sum(turnover.values()),
        "cost_drag": sum(turnover.values()) * cost_rate,
        "outcome_rows": len(outcome_rows),
        "pending_rows": len(selected) - len(outcome_rows),
        "hit_rate": _rate(d.row.outcome.raw_return > 0 for d in outcome_rows),
        "alpha_hit_rate": _rate(d.row.outcome.alpha_return > 0 for d in alpha_rows),
        "avg_return": _weighted_average(
            (float(d.row.outcome.raw_return), d.weight) for d in outcome_rows
        ),
        "avg_alpha_return": _weighted_average(
            (float(d.row.outcome.alpha_return), d.weight) for d in alpha_rows
        ),
        "proxy_net_return": equity[-1] - 1.0 if evaluated_dates else None,
        "max_drawdown": _max_drawdown(equity),
        "evaluated_dates": evaluated_dates,
        "top_k_per_date": top_k_per_date,
    }


def summarize_all_strategies(
    rows: list[ShadowRow],
    *,
    threshold: float,
    hybrid_threshold: float,
    default_ml_weight: float,
    top_k_per_date: int,
    cost_bps: float,
) -> tuple[list[dict[str, Any]], dict[str, list[Decision]]]:
    decisions_by_strategy: dict[str, list[Decision]] = {}
    summaries: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        decisions = strategy_decisions(
            rows,
            strategy=strategy,
            threshold=threshold,
            hybrid_threshold=hybrid_threshold,
            default_ml_weight=default_ml_weight,
            top_k_per_date=top_k_per_date,
        )
        decisions_by_strategy[strategy] = decisions
        summaries.append(
            summarize_strategy(
                decisions,
                strategy=strategy,
                cost_bps=cost_bps,
                top_k_per_date=top_k_per_date,
            )
        )
    return summaries, decisions_by_strategy


def disagreement_rows(
    rows: Iterable[ShadowRow],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        factor_buy = row.action in BUY_ACTIONS
        ml_buy = row.action not in SELL_ACTIONS and row.ml_score is not None and row.ml_score >= threshold
        if factor_buy == ml_buy:
            continue
        if row.ml_score is None:
            continue
        reason = "ML blocks production BUY/ADD" if factor_buy else "ML would trigger where factor did not"
        out.append(
            {
                "date": row.as_of_date,
                "symbol": row.symbol,
                "action": row.action,
                "factor_score": row.factor_score,
                "ml_score": row.ml_score,
                "reason": reason,
                "raw_return": row.outcome.raw_return,
                "alpha_return": row.outcome.alpha_return,
            }
        )
    return sorted(out, key=lambda r: (r["date"], r["reason"], r["symbol"]))


def _format_pct(value: float | None) -> str:
    if value is None:
        return "pending"
    return f"{value * 100:+.2f}%"


def _format_num(value: float | None) -> str:
    if value is None:
        return "pending"
    return f"{value:.2f}"


def _strategy_label(strategy: str) -> str:
    labels = {
        "factor_production": "Factor production",
        "ml_gated_factor": "ML-gated factor",
        "ml_only": "ML-only",
        "hybrid": "Hybrid proxy",
    }
    return labels.get(strategy, strategy)


def render_markdown(
    *,
    from_date: str,
    to_date: str,
    model: str,
    horizon: str,
    threshold: float,
    hybrid_threshold: float,
    benchmark: str,
    cost_bps: float,
    no_prices: bool,
    rows: list[ShadowRow],
    summaries: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# ML Gate shadow report - {from_date} to {to_date}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Shadow model: `{model}`")
    lines.append(f"- Horizon: `{horizon}`")
    lines.append(f"- ML threshold: `{threshold:.2f}`")
    lines.append(f"- Hybrid threshold: `{hybrid_threshold:.2f}`")
    lines.append(f"- Benchmark: `{benchmark}`")
    lines.append(f"- Cost proxy: `{cost_bps:.1f}` bps per unit turnover")
    lines.append(f"- Prices: `{'skipped' if no_prices else 'attempted'}`")
    lines.append("")
    lines.append(
        "This report is shadow-only. It compares decisions that would have been "
        "made by alternative gates; it does not modify recommendations, tickets, "
        "or broker actions."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Recommendation rows: **{len(rows)}**")
    lines.append(f"- Dates with recommendations: **{len({r.as_of_date for r in rows})}**")
    lines.append(f"- Rows with ML score: **{sum(1 for r in rows if r.ml_score is not None)}**")
    lines.append(
        f"- Rows with matured price outcome: **{sum(1 for r in rows if r.outcome.raw_return is not None)}**"
    )
    lines.append("")
    lines.append("## Strategy comparison")
    lines.append("")
    lines.append(
        "| Strategy | Selected | Dates | Avg exposure | Turnover | Cost drag | "
        "Outcome rows | Pending | Hit rate | Alpha hit | Avg return | Avg alpha | "
        "Proxy net return | Max drawdown |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            "| {strategy} | {selected} | {dates} | {exposure} | {turnover} | {cost} | "
            "{outcomes} | {pending} | {hit} | {alpha_hit} | {avg_ret} | {avg_alpha} | "
            "{net} | {dd} |".format(
                strategy=_strategy_label(str(row["strategy"])),
                selected=row["selected"],
                dates=row["dates"],
                exposure=_format_pct(row["avg_exposure"]),
                turnover=_format_num(row["turnover"]),
                cost=_format_pct(row["cost_drag"]),
                outcomes=row["outcome_rows"],
                pending=row["pending_rows"],
                hit=_format_pct(row["hit_rate"]),
                alpha_hit=_format_pct(row["alpha_hit_rate"]),
                avg_ret=_format_pct(row["avg_return"]),
                avg_alpha=_format_pct(row["avg_alpha_return"]),
                net=_format_pct(row["proxy_net_return"]),
                dd=_format_pct(row["max_drawdown"]),
            )
        )
    lines.append("")
    lines.append(
        "_Proxy net return is computed only on dates where every selected row has "
        "a matured price outcome. Until the horizon matures, outcome fields remain pending._"
    )
    lines.append("")

    lines.append("## Disagreement ledger")
    lines.append("")
    if not disagreements:
        lines.append("_No factor-vs-ML buy-side disagreements in this window._")
        lines.append("")
    else:
        lines.append("| Date | Symbol | Production action | Factor score | ML score | Result | Return | Alpha |")
        lines.append("|---|---|---|---:|---:|---|---:|---:|")
        for row in disagreements:
            lines.append(
                f"| {row['date']} | {row['symbol']} | {row['action']} | "
                f"{row['factor_score']:.2f} | {row['ml_score']:.2f} | "
                f"{row['reason']} | {_format_pct(row['raw_return'])} | "
                f"{_format_pct(row['alpha_return'])} |"
            )
        lines.append("")

    lines.append("## How to read")
    lines.append("")
    lines.append("- `ML-gated factor` answers: keep production BUY/ADD only if ML agrees.")
    lines.append("- `ML-only` answers: what would the ML model buy on its own score.")
    lines.append("- `Hybrid proxy` averages the factor alpha probability and ML score as a shadow diagnostic.")
    lines.append("- Disagreement rows are the review queue: blocked production buys and ML-only triggers.")
    lines.append("")
    return "\n".join(lines)


def _close_series(symbol: str, *, start: str, end: str):
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
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

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame["Close"].dropna()
    except Exception:
        return None


def _return_from_series(series: Any, *, as_of_date: str, horizon_days: int) -> float | None:
    if series is None or len(series) <= horizon_days:
        return None
    try:
        dates = [idx.date() for idx in series.index]
        as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        start_idx = next((i for i, d in enumerate(dates) if d >= as_of), None)
        if start_idx is None:
            return None
        end_idx = start_idx + int(horizon_days)
        if end_idx >= len(series):
            return None
        start_price = float(series.iloc[start_idx])
        end_price = float(series.iloc[end_idx])
    except Exception:
        return None
    if start_price <= 0:
        return None
    return end_price / start_price - 1.0


def make_price_outcome_provider(
    recs_by_date: dict[str, list[RecommendationRecord]],
    *,
    horizon: str,
    benchmark: str,
) -> OutcomeProvider:
    horizon_days = _horizon_days(horizon)
    all_recs = [rec for recs in recs_by_date.values() for rec in recs]
    if not all_recs:
        return lambda _rec: Outcome()
    from_date = min(rec.as_of_date for rec in all_recs)
    to_date = max(rec.as_of_date for rec in all_recs)
    start = (
        datetime.strptime(from_date, "%Y-%m-%d").date() - timedelta(days=5)
    ).isoformat()
    end = (
        datetime.strptime(to_date, "%Y-%m-%d").date()
        + timedelta(days=int(horizon_days * 1.7) + 10)
    ).isoformat()
    symbols = sorted({rec.symbol.upper() for rec in all_recs} | {benchmark.upper()})
    closes = {symbol: _close_series(symbol, start=start, end=end) for symbol in symbols}

    def provider(rec: RecommendationRecord) -> Outcome:
        raw = _return_from_series(
            closes.get(rec.symbol.upper()),
            as_of_date=rec.as_of_date,
            horizon_days=horizon_days,
        )
        bench = _return_from_series(
            closes.get(benchmark.upper()),
            as_of_date=rec.as_of_date,
            horizon_days=horizon_days,
        )
        return Outcome(raw_return=raw, benchmark_return=bench)

    return provider


def main() -> int:
    args = _parse_args()
    from_date = _validate_date(args.from_date, "--from-date")
    to_date = _validate_date(args.to_date, "--to-date")
    if from_date > to_date:
        print("--from-date must be <= --to-date", file=sys.stderr)
        return 2

    base_dir = Path(args.base_dir)
    recs_by_date = load_recommendations_range(base_dir, from_date=from_date, to_date=to_date)
    provider = None if args.no_prices else make_price_outcome_provider(
        recs_by_date,
        horizon=args.horizon,
        benchmark=args.benchmark,
    )
    rows = build_shadow_rows(
        recs_by_date,
        model=args.model,
        horizon=args.horizon,
        outcome_provider=provider,
    )
    summaries, _decisions = summarize_all_strategies(
        rows,
        threshold=float(args.threshold),
        hybrid_threshold=float(args.hybrid_threshold),
        default_ml_weight=float(args.ml_only_weight),
        top_k_per_date=int(args.top_k_per_date),
        cost_bps=float(args.cost_bps),
    )
    disagreements = disagreement_rows(rows, threshold=float(args.threshold))

    output = Path(args.output) if args.output else (
        base_dir / f"ml_gate_weekly_{from_date}_to_{to_date}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown(
            from_date=from_date,
            to_date=to_date,
            model=args.model,
            horizon=args.horizon,
            threshold=float(args.threshold),
            hybrid_threshold=float(args.hybrid_threshold),
            benchmark=args.benchmark,
            cost_bps=float(args.cost_bps),
            no_prices=bool(args.no_prices),
            rows=rows,
            summaries=summaries,
            disagreements=disagreements,
        )
    )
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
