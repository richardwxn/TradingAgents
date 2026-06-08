"""Pure helpers that turn weekly analysis reports + a position ledger
into a list of actionable per-ticker `Action` records, then render that
list as a single markdown report.

All functions are deterministic and free of I/O so the CLI can be unit
tested without yfinance, the LLM, or file globbing. The CLI in
`daily_signals.py` loads JSONs, fetches yfinance prices, and feeds
plain dicts into these helpers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from portfolio.sizing import (
    SizingConfig,
    buy_limit_price,
    compute_target_weights,
    stop_loss_price,
    trim_limit_price,
)
from portfolio.risk import (
    RiskAdjustment,
    RiskLimits,
    SectorShock,
    apply_all_risk_caps,
    apply_sector_shock_guard,
    compute_portfolio_beta,
    compute_sector_exposure,
)


# ---------- data shapes ----------


@dataclass(frozen=True)
class Signal:
    """One weekly composite snapshot for one ticker."""

    symbol: str
    as_of_date: str
    direction: str
    composite: float | None
    confidence: float | None
    source_path: str
    # Section 26: surface earnings info + sector label so the daily
    # signals layer can apply earnings-aware sizing and portfolio-level
    # risk caps without re-reading the source report JSONs.
    next_earnings_in_calendar_days: int | None = None
    next_earnings_date: str | None = None
    sector: str | None = None
    industry: str | None = None
    tradingagents_review_gate: dict[str, Any] = field(default_factory=dict)

    def to_sizing_input(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "composite": self.composite,
            "confidence": self.confidence,
            "tradingagents_review_gate": dict(self.tradingagents_review_gate),
        }


@dataclass(frozen=True)
class PriceContext:
    """End-of-day price context for one ticker. Any field may be None
    when yfinance fails or history is too short."""

    last_close: float | None
    sma20: float | None
    atr14: float | None
    source: str | None = None


@dataclass(frozen=True)
class Position:
    shares: float
    avg_cost: float


@dataclass(frozen=True)
class Action:
    symbol: str
    action: str
    direction: str | None
    composite: float | None
    confidence: float | None
    target_weight: float
    current_weight: float
    delta_pp: float
    target_shares: int
    current_shares: int
    delta_shares: int
    limit_price: float | None
    stop_loss: float | None
    last_close: float | None
    sma20: float | None
    atr14: float | None
    signal_age_days: int | None
    price_source: str | None = None
    notes: list[str] = field(default_factory=list)
    target_shares_exact: float | None = None
    delta_shares_exact: float | None = None
    review_gate_status: str | None = None
    review_gate_reason: str | None = None
    review_execution_caveats: list[str] = field(default_factory=list)
    tradingagents_review: dict[str, Any] = field(default_factory=dict)


# Action precedence for the report ordering (BUY first, then EXIT, etc.).
_ACTION_ORDER = {
    "BUY": 0,
    "ADD": 1,
    "TRIM": 2,
    "EXIT": 3,
    "HOLD": 4,
    "SKIP": 5,
    "REVIEW": 6,  # held but no signal — manual review required
}


# ---------- signal loading ----------


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _load_signal_from_json(path: Path) -> Signal | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    symbol = (payload.get("symbol") or "").upper()
    as_of = payload.get("as_of_date")
    if not symbol or not as_of:
        return None
    kf = payload.get("key_features") or {}
    model_scoring = kf.get("model_scoring") or {}
    composite = model_scoring.get("composite_score")
    ec = kf.get("earnings_calendar") or {}
    ind = kf.get("industry_context") or {}
    review_gate = (kf.get("tradingagents_review") or {}).get("gate") or {}
    return Signal(
        symbol=symbol,
        as_of_date=as_of,
        direction=(payload.get("direction") or "neutral").lower(),
        composite=float(composite) if composite is not None else None,
        confidence=(
            float(payload["confidence"])
            if payload.get("confidence") is not None
            else None
        ),
        source_path=str(path),
        next_earnings_in_calendar_days=_safe_int(
            ec.get("next_earnings_in_calendar_days")
        ),
        next_earnings_date=ec.get("next_earnings_date"),
        sector=ind.get("sector"),
        industry=ind.get("industry"),
        tradingagents_review_gate=(
            dict(review_gate) if isinstance(review_gate, dict) else {}
        ),
    )


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _calibrated_confidence_from_composite(
    composite: float | None,
    *,
    calibration: dict[str, Any] | None,
) -> float | None:
    """Look up calibrated confidence from an isotonic-fit JSON.

    Reports generated BEFORE the calibration plumbing fix (Section 28
    `--confidence-calibration-path` not yet threaded through
    `generate_corpus.py`) have heuristic confidence baked in. The daily
    signals layer would then over-allocate to weakly-bullish names
    (heuristic predicts 50-70% confidence; calibrated says 17% at the
    corpus base rate). This helper re-derives confidence from the
    composite via the calibration curve at READ time, so the daily
    layer benefits without a corpus regen.

    Returns None when calibration is unavailable or composite is None —
    caller falls back to the baked-in heuristic value.
    """
    if composite is None or calibration is None:
        return None
    fits = calibration.get("fit") or []
    if not fits:
        return None
    # PAV isotonic: each segment is a flat plateau [x_lower, x_upper] →
    # hit_rate. Pick the segment whose range contains composite; if none
    # match exactly (gaps between segments), use the largest segment with
    # x_lower <= composite.
    best = None
    for seg in fits:
        lo, hi = seg.get("x_lower"), seg.get("x_upper")
        if lo is None or hi is None:
            continue
        if lo <= composite <= hi:
            best = seg
            break
        if lo <= composite:
            if best is None or lo > best["x_lower"]:
                best = seg
    if best is None:
        return None
    return float(best.get("hit_rate") or 0.0)


def load_latest_signals(
    report_paths: Iterable[Path | str],
    *,
    universe: Iterable[str],
    as_of: date | None = None,
    calibration_path: str | Path | None = None,
) -> dict[str, Signal | None]:
    """For each ticker in `universe`, return the most recent Signal whose
    `as_of_date <= as_of` (defaults to today). Missing tickers map to None.

    Reports past `as_of` are filtered out so historical replays don't
    leak future composites into a back-dated report.

    When `calibration_path` points to a valid `configs/
    confidence_calibration.json`, the heuristic confidence baked into
    each report is REPLACED with the calibrated value derived from the
    isotonic curve. This corrects for reports generated before the
    calibration plumbing fix (no corpus regen needed).
    """
    cutoff = as_of or date.today()
    calibration: dict[str, Any] | None = None
    if calibration_path:
        try:
            calibration = json.loads(Path(calibration_path).read_text())
        except Exception:
            calibration = None

    by_symbol: dict[str, Signal] = {}
    for raw in report_paths:
        path = Path(raw)
        sig = _load_signal_from_json(path)
        if sig is None:
            continue
        d = _parse_date(sig.as_of_date)
        if d is None or d > cutoff:
            continue
        existing = by_symbol.get(sig.symbol)
        if existing is None or sig.as_of_date > existing.as_of_date:
            by_symbol[sig.symbol] = sig

    if calibration is not None:
        # Override heuristic confidence in-place with the calibrated value.
        # Leave composite/direction alone — those are correct as emitted.
        for sym, sig in list(by_symbol.items()):
            cal = _calibrated_confidence_from_composite(
                sig.composite, calibration=calibration,
            )
            if cal is not None:
                by_symbol[sym] = replace(sig, confidence=cal)
    return {sym.upper(): by_symbol.get(sym.upper()) for sym in universe}


# ---------- action classification ----------


def classify_action(
    *,
    target_weight: float,
    current_weight: float,
    current_shares: int,
    rebalance_threshold_pp: float = 0.01,
) -> str:
    """Map (target, current) weights to a single action label.

    `rebalance_threshold_pp` (default 1%) is the minimum |delta| that
    triggers a TRIM/ADD; smaller deltas resolve to HOLD to avoid
    micro-rebalances driven by daily price drift alone.
    """
    delta = target_weight - current_weight
    if current_shares == 0 and target_weight <= 0:
        return "SKIP"
    if current_shares == 0 and target_weight > 0:
        return "BUY"
    if current_shares > 0 and target_weight <= 0:
        return "EXIT"
    if abs(delta) <= rebalance_threshold_pp:
        return "HOLD"
    return "ADD" if delta > 0 else "TRIM"


# ---------- portfolio math ----------


def compute_portfolio_value(
    cash: float,
    positions: dict[str, Position],
    prices: dict[str, PriceContext | None],
) -> tuple[float, dict[str, float]]:
    """Return `(total_value, current_weights)` keyed by symbol. Positions
    with no current price use the position's average cost as a fallback
    so the ledger never silently shrinks when yfinance fails."""
    total = float(cash)
    market_values: dict[str, float] = {}
    for sym, pos in positions.items():
        price_ctx = prices.get(sym)
        last = (
            price_ctx.last_close
            if price_ctx is not None and price_ctx.last_close is not None
            else pos.avg_cost
        )
        mv = float(pos.shares) * float(last or 0.0)
        market_values[sym] = mv
        total += mv
    weights = (
        {sym: (mv / total if total > 0 else 0.0) for sym, mv in market_values.items()}
        if total > 0
        else {sym: 0.0 for sym in market_values}
    )
    return total, weights


def compute_actions(
    *,
    signals: dict[str, Signal | None],
    positions: dict[str, Position],
    prices: dict[str, PriceContext | None],
    config: SizingConfig,
    cash: float,
    as_of: date | None = None,
    rebalance_threshold_pp: float = 0.01,
    risk_limits: RiskLimits | None = None,
    beta_map: dict[str, float | None] | None = None,
    correlation_matrix: dict[str, dict[str, float]] | None = None,
    sector_shocks: dict[str, SectorShock] | None = None,
) -> tuple[list[Action], dict[str, Any]]:
    """Compute per-ticker actions + a portfolio summary.

    `signals` covers the configured universe. `positions` may carry
    extra symbols not in the universe — they are appended as REVIEW
    actions so the user is reminded to handle them. `cash` is in dollars
    (kept separate from positions because the user maintains it
    separately in positions.json).
    """
    cutoff = as_of or date.today()

    # 1) Build the sizing input, suppressing bearish names per config.
    # Each entry carries `age_days` so sizing can apply stale-signal decay.
    sizing_input: dict[str, dict[str, Any]] = {}
    for sym, sig in signals.items():
        age_days: int | None = None
        if sig is not None:
            sig_date = _parse_date(sig.as_of_date)
            if sig_date is not None:
                age_days = (cutoff - sig_date).days
        if sig is None:
            sizing_input[sym] = {
                "direction": "neutral", "composite": None, "confidence": None,
                "age_days": age_days,
            }
        elif not config.enable_bearish and sig.direction == "bearish":
            sizing_input[sym] = {
                "direction": "neutral", "composite": None, "confidence": None,
                "age_days": age_days,
            }
        else:
            payload = sig.to_sizing_input()
            payload["age_days"] = age_days
            sizing_input[sym] = payload
    target_weights = compute_target_weights(sizing_input, config=config)

    # 1a) TradingAgents review gate. This is opt-in because the graph is a
    # governance layer around the calibrated composite, not the primary
    # signal. When enabled, it scales target exposure before earnings and
    # portfolio risk caps so all later constraints see the adjusted intent.
    review_gate_notes: dict[str, str] = {}
    if (
        getattr(config, "enable_tradingagents_review_gate", False)
        and getattr(config, "tradingagents_review_apply_to_sizing", True)
    ):
        for sym, sig in signals.items():
            if sig is None:
                continue
            gate = sig.tradingagents_review_gate or {}
            old_w = target_weights.get(sym, 0.0)
            if old_w <= 0:
                continue
            multiplier = _review_gate_sizing_multiplier(gate)
            if multiplier >= 1.0:
                continue
            target_weights[sym] = old_w * multiplier
            review_gate_notes[sym] = (
                f"TradingAgents gate {gate.get('ticket_gate', 'allow')} — "
                f"target size scaled to {multiplier*100:.0f}% "
                f"({old_w*100:.1f}% → {target_weights[sym]*100:.1f}%). "
                f"{gate.get('reason') or ''}"
            ).strip()

    # 1b) Earnings-aware sizing adjustment (Section 26). For each symbol
    # whose next earnings date is within config.pre_earnings_trim_days
    # of the cutoff (today), multiply target weight by
    # pre_earnings_size_factor. Skipped when the config sets the factor
    # to 1.0 or trim_days to 0.
    earnings_adjustment_notes: dict[str, str] = {}
    if (
        config.pre_earnings_size_factor < 1.0
        and config.pre_earnings_trim_days > 0
    ):
        for sym, sig in signals.items():
            if sig is None or sig.next_earnings_in_calendar_days is None:
                continue
            days = int(sig.next_earnings_in_calendar_days)
            if 0 <= days <= config.pre_earnings_trim_days:
                old_w = target_weights.get(sym, 0.0)
                if old_w <= 0:
                    continue
                target_weights[sym] = old_w * config.pre_earnings_size_factor
                earnings_adjustment_notes[sym] = (
                    f"Earnings in {days}d ({sig.next_earnings_date}) — "
                    f"target size reduced to "
                    f"{config.pre_earnings_size_factor*100:.0f}% "
                    f"({old_w*100:.1f}% → {target_weights[sym]*100:.1f}%)."
                )

    # 1c) Same-day sector shock guard. This is an execution-layer circuit
    # breaker: when a sector ETF drops through the configured threshold,
    # new buys can be stood down and existing-position targets can be cut.
    sector_shock_notes: dict[str, str] = {}
    sector_map_for_guards = {
        sym: (sig.sector if sig is not None else None)
        for sym, sig in signals.items()
    }
    if (
        getattr(config, "sector_shock_guard_enabled", True)
        and sector_shocks
    ):
        target_weights, sector_shock_notes = apply_sector_shock_guard(
            target_weights,
            sector_map_for_guards,
            position_shares={sym: pos.shares for sym, pos in positions.items()},
            shocks=sector_shocks,
            new_buy_size_factor=getattr(
                config, "sector_shock_new_buy_size_factor", 0.0
            ),
            existing_position_size_factor=getattr(
                config, "sector_shock_existing_position_size_factor", 0.5
            ),
        )

    # 1d) Portfolio-level risk caps (Section 26). Apply sector,
    # beta, and correlation caps to the post-earnings target weights.
    # Trimmed weight becomes uninvested (cash), not redistributed —
    # consistent with the long-or-cash mode.
    risk_notes: dict[str, str] = {}
    risk_diagnostic: dict[str, Any] = {}
    if risk_limits is not None:
        beta_map = beta_map or {}
        correlation_matrix = correlation_matrix or {}
        adj = apply_all_risk_caps(
            target_weights,
            sector_map=sector_map_for_guards,
            beta_map=beta_map,
            correlation_matrix=correlation_matrix,
            limits=risk_limits,
        )
        target_weights = adj.adjusted_weights
        risk_notes = adj.notes
        risk_diagnostic = {
            "sector_exposure": adj.sector_exposure,
            "portfolio_beta": adj.portfolio_beta,
            "limits": {
                "max_sector_exposure": risk_limits.max_sector_exposure,
                "max_portfolio_beta": risk_limits.max_portfolio_beta,
                "max_pair_correlation": risk_limits.max_pair_correlation,
            },
        }

    # 2) Compute current portfolio value + per-symbol weights.
    total_value, current_weights = compute_portfolio_value(
        cash=cash, positions=positions, prices=prices
    )

    # 3) Build actions for every universe symbol, then append REVIEWs for
    # extra held names that fell out of the universe.
    actions: list[Action] = []
    for sym, sig in signals.items():
        ctx = prices.get(sym) or PriceContext(None, None, None)
        position = positions.get(sym, Position(shares=0.0, avg_cost=0.0))
        target_w = float(target_weights.get(sym, 0.0))
        current_w = float(current_weights.get(sym, 0.0))
        target_dollars = target_w * total_value
        target_shares = (
            int(math.floor(target_dollars / ctx.last_close))
            if ctx.last_close and ctx.last_close > 0
            else 0
        )
        target_shares_exact = (
            target_dollars / ctx.last_close
            if ctx.last_close and ctx.last_close > 0
            else None
        )
        current_shares = int(position.shares)
        delta_shares = target_shares - current_shares
        delta_shares_exact = (
            target_shares_exact - float(position.shares)
            if target_shares_exact is not None
            else None
        )
        side = classify_action(
            target_weight=target_w,
            current_weight=current_w,
            current_shares=current_shares,
            rebalance_threshold_pp=rebalance_threshold_pp,
        )
        # Override BUY → ADD when we already hold the name (classify_action
        # treats current_shares==0 vs >0; this is just nomenclature).
        if side == "BUY" and current_shares > 0:
            side = "ADD"

        if side in ("BUY", "ADD"):
            limit = buy_limit_price(
                ctx.last_close, ctx.sma20,
                pullback_to=config.entry_pullback_to,
                max_pullback_pct=config.max_entry_pullback_pct,
            )
            stop = stop_loss_price(limit or ctx.last_close, ctx.atr14, atr_multiple=config.stop_loss_atr_multiple)
        elif side in ("TRIM", "EXIT"):
            limit = trim_limit_price(ctx.last_close, ctx.atr14, atrs=config.exit_patience_atrs)
            stop = stop_loss_price(position.avg_cost or ctx.last_close, ctx.atr14, atr_multiple=config.stop_loss_atr_multiple)
        else:
            limit = ctx.last_close
            stop = (
                stop_loss_price(position.avg_cost or ctx.last_close, ctx.atr14, atr_multiple=config.stop_loss_atr_multiple)
                if current_shares > 0
                else None
            )

        signal_age = None
        if sig is not None:
            sig_date = _parse_date(sig.as_of_date)
            if sig_date is not None:
                signal_age = (cutoff - sig_date).days

        notes: list[str] = []
        if sig is None:
            notes.append("No analysis report found for this ticker.")
        elif signal_age is not None and signal_age > config.stale_composite_days:
            notes.append(f"Stale composite ({signal_age}d old; threshold {config.stale_composite_days}d). Re-run analysis_mvp.py.")
        if sig is not None and sig.direction == "bearish" and not config.enable_bearish:
            notes.append("Bearish signal suppressed (long-or-cash mode, handoff Section 15).")
        if ctx.last_close is None:
            notes.append("Price unavailable from yfinance; targets/limits may be incomplete.")
        if sym in earnings_adjustment_notes:
            notes.append(earnings_adjustment_notes[sym])
        if sym in risk_notes:
            notes.append(risk_notes[sym])
        if sym in sector_shock_notes:
            notes.append(sector_shock_notes[sym])
        if sym in review_gate_notes:
            notes.append(review_gate_notes[sym])

        review_gate = sig.tradingagents_review_gate if sig is not None else {}

        actions.append(Action(
            symbol=sym,
            action=side,
            direction=sig.direction if sig else None,
            composite=sig.composite if sig else None,
            confidence=sig.confidence if sig else None,
            target_weight=target_w,
            current_weight=current_w,
            delta_pp=target_w - current_w,
            target_shares=target_shares,
            current_shares=current_shares,
            delta_shares=delta_shares,
            limit_price=limit,
            stop_loss=stop,
            last_close=ctx.last_close,
            sma20=ctx.sma20,
            atr14=ctx.atr14,
            price_source=ctx.source,
            signal_age_days=signal_age,
            notes=notes,
            target_shares_exact=target_shares_exact,
            delta_shares_exact=delta_shares_exact,
            review_gate_status=(
                str(review_gate.get("ticket_gate"))
                if review_gate.get("ticket_gate")
                else None
            ),
            review_gate_reason=(
                str(review_gate.get("reason"))
                if review_gate.get("reason")
                else None
            ),
            review_execution_caveats=[
                str(x)
                for x in (review_gate.get("execution_caveats") or [])
                if str(x).strip()
            ],
            tradingagents_review={},
        ))

    for sym in positions:
        if sym in signals:
            continue
        ctx = prices.get(sym) or PriceContext(None, None, None)
        position = positions[sym]
        current_w = current_weights.get(sym, 0.0)
        actions.append(Action(
            symbol=sym,
            action="REVIEW",
            direction=None,
            composite=None,
            confidence=None,
            target_weight=0.0,
            current_weight=current_w,
            delta_pp=-current_w,
            target_shares=0,
            current_shares=int(position.shares),
            delta_shares=-int(position.shares),
            limit_price=ctx.last_close,
            stop_loss=None,
            last_close=ctx.last_close,
            sma20=ctx.sma20,
            atr14=ctx.atr14,
            price_source=ctx.source,
            signal_age_days=None,
            notes=["Held but outside the configured universe — manual review required."],
        ))

    actions.sort(key=lambda a: (_ACTION_ORDER.get(a.action, 99), a.symbol))

    target_long = sum(max(0.0, a.target_weight) for a in actions if a.action != "REVIEW")
    current_long = sum(max(0.0, a.current_weight) for a in actions)
    summary = {
        "as_of": cutoff.isoformat(),
        "total_value": total_value,
        "cash": cash,
        "cash_weight": (cash / total_value) if total_value > 0 else 1.0,
        "current_long_exposure": current_long,
        "target_long_exposure": target_long,
        "n_bullish_signals": sum(1 for s in signals.values() if s and s.direction == "bullish"),
        "n_bearish_signals": sum(1 for s in signals.values() if s and s.direction == "bearish"),
        "n_neutral_signals": sum(1 for s in signals.values() if s and s.direction == "neutral"),
        "n_missing_signals": sum(1 for s in signals.values() if s is None),
        "n_stale_signals": sum(
            1 for a in actions
            if a.signal_age_days is not None and a.signal_age_days > config.stale_composite_days
        ),
        "risk_diagnostic": risk_diagnostic,
        "n_earnings_adjusted": len(earnings_adjustment_notes),
        "n_risk_capped": len(risk_notes),
        "n_review_gate_adjusted": len(review_gate_notes),
        "n_sector_shock_adjusted": len(sector_shock_notes),
        "sector_shocks": {
            sector: {
                "trigger_symbol": shock.trigger_symbol,
                "pct_change": shock.pct_change,
                "threshold": shock.threshold,
                "source": shock.source,
            }
            for sector, shock in (sector_shocks or {}).items()
        },
    }
    return actions, summary


def _review_gate_sizing_multiplier(gate: dict[str, Any]) -> float:
    if not isinstance(gate, dict) or gate.get("status") != "ok":
        return 1.0
    if gate.get("risk_veto"):
        return 0.0
    raw = gate.get("sizing_multiplier", 1.0)
    try:
        multiplier = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, multiplier))


# ---------- markdown rendering ----------


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.1f}%" if abs(x) >= 1e-9 else "0.0%"


def _fmt_weight(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_money(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:,.2f}"


def _fmt_int(x: int | None) -> str:
    return "—" if x is None else f"{x:,}"


def format_daily_report(
    actions: list[Action],
    summary: dict[str, Any],
    *,
    config: SizingConfig,
    as_of: date,
) -> str:
    """Render the action list + summary as a single markdown document
    suitable for daily manual review."""
    lines: list[str] = []
    lines.append(f"# Daily Signals — {as_of.isoformat()}")
    lines.append("")
    lines.append(
        f"**Universe:** {len(config.universe) if config.universe else len([a for a in actions if a.action != 'REVIEW'])} tickers "
        f"· **Policy:** `{config.policy}` "
        f"· **Mode:** {'long-or-short' if config.enable_bearish else 'long-or-cash (bearish suppressed)'}"
    )
    lines.append("")
    lines.append("## Portfolio snapshot")
    lines.append("")
    lines.append(f"- Total value: {_fmt_money(summary.get('total_value'))}")
    lines.append(f"- Cash: {_fmt_money(summary.get('cash'))} ({_fmt_weight(summary.get('cash_weight'))})")
    lines.append(
        f"- Long exposure: current {_fmt_weight(summary.get('current_long_exposure'))} "
        f"→ target {_fmt_weight(summary.get('target_long_exposure'))}"
    )
    lines.append(
        f"- Signals: {summary.get('n_bullish_signals', 0)} bullish · "
        f"{summary.get('n_neutral_signals', 0)} neutral · "
        f"{summary.get('n_bearish_signals', 0)} bearish · "
        f"{summary.get('n_missing_signals', 0)} missing · "
        f"{summary.get('n_stale_signals', 0)} stale"
    )
    lines.append("")
    lines.append(
        "_Limit prices and stop-losses are heuristic (entry = `min(last_close, SMA20)`, "
        "stop = `entry − 1.5 × ATR(14)`), not backtest-validated. See handoff §16._"
    )
    lines.append("")

    # Portfolio risk section (Section 26) — only when risk caps were active.
    risk_diag = (summary.get("risk_diagnostic") or {}) if summary else {}
    if risk_diag:
        lines.extend(_render_risk_section(risk_diag, summary))
        lines.append("")
    sector_shocks = (summary or {}).get("sector_shocks") or {}
    if sector_shocks:
        lines.extend(_render_sector_shock_section(sector_shocks, summary))
        lines.append("")
    if summary.get("n_review_gate_adjusted", 0):
        lines.append("## TradingAgents review gate")
        lines.append("")
        lines.append(
            f"- {summary.get('n_review_gate_adjusted', 0)} target(s) were reduced by the review gate."
        )
        lines.append("")

    grouped: dict[str, list[Action]] = {}
    for a in actions:
        grouped.setdefault(a.action, []).append(a)

    section_titles = {
        "BUY": "## BUY (new positions)",
        "ADD": "## ADD (increase existing)",
        "TRIM": "## TRIM (reduce existing)",
        "EXIT": "## EXIT (close existing)",
        "REVIEW": "## REVIEW (held, no signal)",
        "HOLD": "## HOLD (no action)",
        "SKIP": "## SKIP (not held, no bullish signal)",
    }
    for side in ("BUY", "ADD", "TRIM", "EXIT", "REVIEW", "HOLD", "SKIP"):
        bucket = grouped.get(side) or []
        if not bucket:
            continue
        lines.append(section_titles[side])
        lines.append("")
        for a in bucket:
            lines.extend(_render_action_block(a))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_action_block(a: Action) -> list[str]:
    pieces: list[str] = []
    delta_shares_str = (
        f"{a.delta_shares:+,d} sh" if a.delta_shares else "0 sh"
    )
    title_meta = (
        f"target {_fmt_weight(a.target_weight)} · "
        f"current {_fmt_weight(a.current_weight)} · "
        f"Δ {_fmt_pct(a.delta_pp)}"
    )
    pieces.append(f"### {a.action} {a.symbol}  ({title_meta})")
    sig_bits = []
    if a.direction:
        sig_bits.append(f"direction `{a.direction}`")
    if a.composite is not None:
        sig_bits.append(f"composite `{a.composite:+.3f}`")
    if a.confidence is not None:
        sig_bits.append(f"confidence `{a.confidence:.2f}`")
    if a.signal_age_days is not None:
        sig_bits.append(f"signal {a.signal_age_days}d old")
    if sig_bits:
        pieces.append("- Signal: " + " · ".join(sig_bits))
    if a.review_gate_status:
        pieces.append(
            "- TradingAgents gate: "
            f"`{a.review_gate_status}`"
            + (f" — {a.review_gate_reason}" if a.review_gate_reason else "")
        )
        for caveat in a.review_execution_caveats[:3]:
            pieces.append(f"- Gate caveat: {caveat}")

    if a.action in ("BUY", "ADD", "TRIM", "EXIT"):
        pieces.append(
            f"- Shares: {delta_shares_str} "
            f"(current {_fmt_int(a.current_shares)} → target {_fmt_int(a.target_shares)})"
        )
        if a.limit_price is not None:
            pieces.append(f"- Limit price: **{_fmt_money(a.limit_price)}**")
        if a.stop_loss is not None and a.action in ("BUY", "ADD"):
            pieces.append(f"- Stop-loss reminder: {_fmt_money(a.stop_loss)}")
    if a.last_close is not None:
        ctx_bits = [f"last close {_fmt_money(a.last_close)}"]
        if a.sma20 is not None:
            ctx_bits.append(f"SMA20 {_fmt_money(a.sma20)}")
        if a.atr14 is not None:
            ctx_bits.append(f"ATR(14) {_fmt_money(a.atr14)}")
        pieces.append("- Price context: " + " · ".join(ctx_bits))
    for note in a.notes:
        pieces.append(f"- _Note: {note}_")
    pieces.append("")
    return pieces


# ---------- option position rendering (Section 25) ----------


def format_option_positions_section(
    *,
    options_by_symbol: dict[str, list],
    enriched_by_symbol: dict[str, list],
    book_greeks_by_symbol: dict[str, Any],
    as_of: date,
) -> str:
    """Render the "Option positions" section as markdown.

    Returns an empty string when there are no option positions across any
    symbol — callers can append unconditionally without inserting an empty
    section.

    Inputs are keyed by uppercase symbol so they can be assembled
    incrementally as the daily-signals CLI fetches each symbol's chain.
    """
    if not options_by_symbol:
        return ""

    lines: list[str] = ["## Option positions", ""]

    # Top-of-section book aggregate across all symbols.
    book_lines = _format_book_aggregate(book_greeks_by_symbol)
    if book_lines:
        lines.extend(book_lines)
        lines.append("")

    for sym in sorted(options_by_symbol.keys()):
        positions = options_by_symbol[sym]
        enriched = enriched_by_symbol.get(sym) or []
        bg = book_greeks_by_symbol.get(sym)
        lines.extend(_format_symbol_options(sym, positions, enriched, bg, as_of))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_book_aggregate(book_greeks_by_symbol: dict[str, Any]) -> list[str]:
    if not book_greeks_by_symbol:
        return []
    total_delta = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_mv = 0.0
    total_pnl = 0.0
    have_data = False
    for bg in book_greeks_by_symbol.values():
        if bg is None:
            continue
        have_data = True
        if bg.net_share_equivalent_delta is not None:
            total_delta += bg.net_share_equivalent_delta
        if bg.net_vega_dollars_per_vol_pt is not None:
            total_vega += bg.net_vega_dollars_per_vol_pt
        if bg.net_theta_dollars_per_day is not None:
            total_theta += bg.net_theta_dollars_per_day
        if bg.net_option_market_value is not None:
            total_mv += bg.net_option_market_value
        if bg.net_option_unrealized_pnl is not None:
            total_pnl += bg.net_option_unrealized_pnl
    if not have_data:
        return []
    return [
        "**Book aggregate**",
        "",
        f"- Net share-equivalent delta (all symbols): **{total_delta:+,.0f} sh**",
        f"- Net vega: **{_fmt_money(total_vega)}** per 1% IV move",
        f"- Net theta: **{_fmt_money(total_theta)}** per day "
        f"({'time-decay credit' if total_theta > 0 else 'time-decay drag'})",
        f"- Net option market value: {_fmt_money(total_mv)}",
        f"- Net option unrealized P&L: **{_fmt_money(total_pnl)}**",
    ]


def _format_symbol_options(
    symbol: str,
    positions: list,
    enriched: list,
    book_greeks_for_symbol,
    as_of: date,
) -> list[str]:
    lines = [f"### {symbol}"]
    if book_greeks_for_symbol is not None:
        bg = book_greeks_for_symbol
        bg_bits = []
        if bg.shares:
            bg_bits.append(f"{bg.shares:,d} shares")
        if bg.net_share_equivalent_delta is not None:
            bg_bits.append(f"net Δ {bg.net_share_equivalent_delta:+,.0f} sh")
        if bg.net_vega_dollars_per_vol_pt is not None:
            bg_bits.append(f"vega {_fmt_money(bg.net_vega_dollars_per_vol_pt)}/vol-pt")
        if bg.net_theta_dollars_per_day is not None:
            bg_bits.append(f"theta {_fmt_money(bg.net_theta_dollars_per_day)}/day")
        if bg_bits:
            lines.append("- " + " · ".join(bg_bits))

    if not positions:
        lines.append("- _(no option contracts)_")
        return lines

    lines.append("")
    lines.append(
        "| Side | Right | Strike | Expiry | DTE | Qty | Avg cost | Mark | Δ | "
        "Vega | Theta | $ M2M | Unreal P&L |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_key: dict[tuple, Any] = {
        (e.position.right, e.position.strike, e.position.expiry, e.position.quantity): e
        for e in enriched
    }
    for p in positions:
        key = (p.right, p.strike, p.expiry, p.quantity)
        e = by_key.get(key)
        side = "Long" if p.is_long else "Short"
        qty_disp = f"{abs(int(p.quantity))} contracts"
        dte = p.dte(as_of.isoformat())
        dte_disp = f"{dte}d" if dte is not None else "—"

        def fmt_g(v: float | None, suffix: str = "") -> str:
            return f"{v:+.3f}{suffix}" if v is not None else "—"

        mark = e.mark if e else None
        delta = e.delta if e else None
        vega = e.vega if e else None
        theta = e.theta if e else None
        mv = e.market_basis if e else None
        pnl = e.unrealized_pnl if e else None
        lines.append(
            "| {side} | {right} | {strike} | {expiry} | {dte} | {qty} | "
            "{cost} | {mark} | {delta} | {vega} | {theta} | {mv} | {pnl} |"
            .format(
                side=side, right=p.right, strike=f"${p.strike:g}",
                expiry=p.expiry, dte=dte_disp, qty=qty_disp,
                cost=_fmt_money(p.avg_cost),
                mark=_fmt_money(mark),
                delta=fmt_g(delta),
                vega=fmt_g(vega),
                theta=fmt_g(theta),
                mv=_fmt_money(mv),
                pnl=_fmt_money(pnl),
            )
        )

    # Warnings: short-dated contracts, ITM assignment risk.
    warnings: list[str] = []
    for p in positions:
        dte = p.dte(as_of.isoformat())
        if dte is None:
            continue
        if 0 <= dte <= 7:
            warnings.append(
                f"Expiry within {dte}d for {p.right} K=${p.strike:g} "
                f"{p.expiry} — close, roll, or let expire."
            )
        elif dte < 0:
            warnings.append(
                f"Already expired: {p.right} K=${p.strike:g} {p.expiry} — "
                f"remove from ledger."
            )

    # Premium-buyer stop-out hints.
    for e in enriched:
        if not e.position.is_long:
            continue
        if e.mark is None:
            continue
        if e.position.avg_cost <= 0:
            continue
        decay = (e.position.avg_cost - e.mark) / e.position.avg_cost
        if decay >= 0.50:
            warnings.append(
                f"Long {e.position.right} K=${e.position.strike:g} "
                f"{e.position.expiry} marked {_fmt_money(e.mark)} "
                f"vs cost {_fmt_money(e.position.avg_cost)} "
                f"({decay*100:.0f}% premium loss) — consider stop."
            )
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"- ⚠️ _{w}_")

    return lines


def _render_risk_section(
    risk_diag: dict[str, Any],
    summary: dict[str, Any] | None,
) -> list[str]:
    """Render the Portfolio risk section (Section 26)."""
    limits = risk_diag.get("limits") or {}
    sector_exposure = risk_diag.get("sector_exposure") or {}
    portfolio_beta = risk_diag.get("portfolio_beta")
    n_risk_capped = (summary or {}).get("n_risk_capped", 0)
    n_earnings = (summary or {}).get("n_earnings_adjusted", 0)

    lines = ["## Portfolio risk", ""]
    lines.append(
        f"- **Portfolio β (cash-drag included):** "
        f"{f"{portfolio_beta:.2f}" if portfolio_beta is not None else '—'} "
        f"(cap: {f"{limits.get('max_portfolio_beta', 0):.2f}"})"
    )
    lines.append(
        f"- **Sector concentration cap:** {_fmt_weight(limits.get('max_sector_exposure'))}; "
        f"correlation cap: ρ≥{limits.get('max_pair_correlation', 0):.2f}"
    )
    if n_earnings:
        lines.append(
            f"- **{n_earnings}** name(s) had target weight reduced for "
            f"approaching earnings (see notes on individual actions)."
        )
    if n_risk_capped:
        lines.append(
            f"- **{n_risk_capped}** name(s) had target weight reduced by "
            f"risk caps (see notes on individual actions)."
        )
    if sector_exposure:
        lines.append("")
        lines.append("**Sector exposure (post-caps):**")
        lines.append("")
        for sector, exposure in sorted(
            sector_exposure.items(), key=lambda kv: -kv[1]
        ):
            if exposure <= 0:
                continue
            cap = limits.get("max_sector_exposure")
            at_cap = (
                cap is not None and exposure >= cap - 1e-4
            )
            indicator = " ← at cap" if at_cap else ""
            lines.append(f"- {sector}: {_fmt_weight(exposure)}{indicator}")
    return lines


def _render_sector_shock_section(
    sector_shocks: dict[str, Any],
    summary: dict[str, Any] | None,
) -> list[str]:
    """Render same-day sector shock guard diagnostics."""
    n_adjusted = (summary or {}).get("n_sector_shock_adjusted", 0)
    lines = ["## Sector shock guard", ""]
    lines.append(
        f"- **{n_adjusted}** target(s) were reduced by same-day sector shock rules."
    )
    for sector, raw in sorted(sector_shocks.items()):
        shock = raw or {}
        pct_change = shock.get("pct_change")
        threshold = shock.get("threshold")
        pct_label = "n/a" if pct_change is None else f"{float(pct_change)*100:.1f}%"
        threshold_label = (
            "n/a" if threshold is None else f"-{float(threshold)*100:.1f}%"
        )
        lines.append(
            f"- {sector}: {shock.get('trigger_symbol', 'n/a')} "
            f"{pct_label} (trigger {threshold_label})"
        )
    return lines
