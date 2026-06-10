"""Paper-trading logging primitives.

The model emits per-day action recommendations via `daily_signals.py`. To
test whether the model's recommendations are worth following in live
conditions, we log every recommendation to a structured JSONL file. A
separate user-facing CLI records the actual trades made (manual entry,
no broker integration). A weekly reporter joins the two logs and
produces hit-rate + attribution stats.

This module owns the schema and the pure read/write helpers. Wall-clock
I/O (yfinance fills, file globs) lives in the CLIs that use them.

Schema versions follow the same discipline as the analysis reports:
breaking changes bump `RECOMMENDATIONS_SCHEMA_VERSION` /
`EXECUTIONS_SCHEMA_VERSION` so reports across schema changes can be
detected and handled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RECOMMENDATIONS_SCHEMA_VERSION = 1
EXECUTIONS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RecommendationRecord:
    """One model-emitted recommendation for one ticker on one date.

    Mirrors the fields of `portfolio.signals.Action` minus the rendered
    columns. Persisted to JSONL so a weekly reporter can aggregate
    without re-reading per-day markdown.
    """

    as_of_date: str  # ISO date the recommendation is for (today)
    symbol: str
    action: str  # BUY / ADD / TRIM / EXIT / HOLD / SKIP / REVIEW
    direction: str | None  # bullish / bearish / neutral / None
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
    review_gate_status: str | None = None
    review_gate_reason: str | None = None
    # Optional shadow-model outputs. This is intentionally non-actionable:
    # production recommendations remain driven by the factor model unless
    # a future config explicitly promotes an ML model.
    ml_shadow: dict[str, Any] = field(default_factory=dict)
    # ISO timestamp the recommendation was generated (the LOGGING moment,
    # which may differ from as_of_date when daily_signals.py runs after
    # market close).
    generated_at_utc: str = ""
    schema_version: int = RECOMMENDATIONS_SCHEMA_VERSION


@dataclass(frozen=True)
class ExecutionRecord:
    """One actual trade the user made. Manual entry via log_execution.py.

    `ref_recommendation_date` lets the weekly reporter join executions
    back to the recommendation that motivated them — None when the user
    traded outside the model's suggestions (override / discretionary).
    """

    trade_date: str  # ISO date the trade was executed
    symbol: str
    side: str  # BUY / SELL
    shares: float
    fill_price: float
    ref_recommendation_date: str | None = None
    # Free-text reason — useful for postmortem aggregation:
    # "model said buy, I bought" / "model said hold, I trimmed because earnings"
    override_reason: str | None = None
    logged_at_utc: str = ""
    schema_version: int = EXECUTIONS_SCHEMA_VERSION


# ---------- file paths ----------


def recommendations_path(base_dir: Path | str, as_of_date: str) -> Path:
    """Per-day recommendations JSONL: `<base>/recommendations/<date>.jsonl`."""
    return Path(base_dir) / "recommendations" / f"{as_of_date}.jsonl"


def executions_path(base_dir: Path | str, trade_date: str) -> Path:
    """Per-day executions JSONL: `<base>/executions/<date>.jsonl`."""
    return Path(base_dir) / "executions" / f"{trade_date}.jsonl"


# ---------- writers ----------


def append_recommendation(record: RecommendationRecord, *, base_dir: Path | str) -> Path:
    """Append a recommendation to the per-day JSONL. Returns the path written."""
    path = recommendations_path(base_dir, record.as_of_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return path


def write_recommendations_for_day(
    records: Iterable[RecommendationRecord],
    *,
    base_dir: Path | str,
    as_of_date: str,
) -> Path:
    """Overwrite a day's recommendations with the given batch.

    Intended use: daily_signals.py emits all actions in one pass; this
    is one atomic write for that day's run. Re-running the same day's
    daily_signals.py replaces the file (idempotent).
    """
    path = recommendations_path(base_dir, as_of_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
    return path


def append_execution(record: ExecutionRecord, *, base_dir: Path | str) -> Path:
    """Append an execution to the per-day JSONL. Returns the path written.

    Multiple trades on the same date append; a re-run of the daily
    recommendations doesn't touch executions.
    """
    path = executions_path(base_dir, record.trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return path


# ---------- readers ----------


def load_recommendations(base_dir: Path | str, as_of_date: str) -> list[RecommendationRecord]:
    """Read one day's recommendations. Returns [] when file is missing."""
    path = recommendations_path(base_dir, as_of_date)
    if not path.exists():
        return []
    out: list[RecommendationRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.append(RecommendationRecord(**{
                k: v for k, v in d.items()
                if k in RecommendationRecord.__dataclass_fields__
            }))
        except Exception:
            continue
    return out


def load_executions(base_dir: Path | str, trade_date: str) -> list[ExecutionRecord]:
    """Read one day's executions. Returns [] when file is missing."""
    path = executions_path(base_dir, trade_date)
    if not path.exists():
        return []
    out: list[ExecutionRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.append(ExecutionRecord(**{
                k: v for k, v in d.items()
                if k in ExecutionRecord.__dataclass_fields__
            }))
        except Exception:
            continue
    return out


def load_recommendations_range(
    base_dir: Path | str, *, from_date: str, to_date: str,
) -> dict[str, list[RecommendationRecord]]:
    """Load all per-day recommendation files whose date is in
    `[from_date, to_date]` inclusive. Returns `dict[date, records]` —
    missing days simply absent from the dict (NOT a key with empty list).
    """
    base = Path(base_dir) / "recommendations"
    out: dict[str, list[RecommendationRecord]] = {}
    if not base.exists():
        return out
    for path in sorted(base.glob("*.jsonl")):
        date_str = path.stem
        if not (from_date <= date_str <= to_date):
            continue
        recs = load_recommendations(base_dir, date_str)
        if recs:
            out[date_str] = recs
    return out


def load_executions_range(
    base_dir: Path | str, *, from_date: str, to_date: str,
) -> dict[str, list[ExecutionRecord]]:
    base = Path(base_dir) / "executions"
    out: dict[str, list[ExecutionRecord]] = {}
    if not base.exists():
        return out
    for path in sorted(base.glob("*.jsonl")):
        date_str = path.stem
        if not (from_date <= date_str <= to_date):
            continue
        execs = load_executions(base_dir, date_str)
        if execs:
            out[date_str] = execs
    return out


# ---------- joins + diff ----------


def join_recommendations_to_executions(
    recommendations_by_date: dict[str, list[RecommendationRecord]],
    executions_by_date: dict[str, list[ExecutionRecord]],
) -> list[dict[str, Any]]:
    """Join recommendations and executions into per-(date, symbol) rows.

    For each recommendation, find any matching execution that:
    - References the recommendation date via `ref_recommendation_date`
      (preferred — explicit user-asserted link)
    - OR matches symbol on the trade_date and shares-direction is
      consistent with the recommended action (fallback).

    Returns a list of dicts with `recommendation`, `executions` (list,
    possibly empty), `divergence_type` ∈ {"followed", "partial",
    "overridden", "ignored"}, and `realized_at_fill` set when there is
    at least one execution.

    Executions that match no recommendation (pure discretionary trades)
    are NOT returned — they're tracked separately via
    `unattributed_executions()`.
    """
    rows: list[dict[str, Any]] = []
    # Pre-index executions by (ref_recommendation_date, symbol) for the
    # explicit-link path, AND by (trade_date, symbol) for the fallback.
    by_ref: dict[tuple[str, str], list[ExecutionRecord]] = {}
    by_trade: dict[tuple[str, str], list[ExecutionRecord]] = {}
    for trade_date, execs in executions_by_date.items():
        for e in execs:
            by_trade.setdefault((trade_date, e.symbol.upper()), []).append(e)
            if e.ref_recommendation_date:
                by_ref.setdefault((e.ref_recommendation_date, e.symbol.upper()), []).append(e)

    for rec_date, recs in sorted(recommendations_by_date.items()):
        for rec in recs:
            sym = rec.symbol.upper()
            # 1. Explicit link.
            explicit = by_ref.get((rec_date, sym), [])
            # 2. Fallback: trades on the same date with consistent side.
            same_day = by_trade.get((rec_date, sym), [])
            matched: list[ExecutionRecord] = []
            seen_ids: set[int] = set()
            for e in explicit:
                matched.append(e)
                seen_ids.add(id(e))
            if not matched:
                for e in same_day:
                    if id(e) in seen_ids:
                        continue
                    if _execution_side_matches_action(e.side, rec.action):
                        matched.append(e)
            divergence_type = _classify_divergence(rec, matched)
            rows.append({
                "recommendation": rec,
                "executions": matched,
                "divergence_type": divergence_type,
            })
    return rows


def _execution_side_matches_action(side: str, action: str) -> bool:
    side_u = (side or "").upper()
    action_u = (action or "").upper()
    if action_u in ("BUY", "ADD"):
        return side_u == "BUY"
    if action_u in ("TRIM", "EXIT"):
        return side_u == "SELL"
    return False


def _classify_divergence(
    rec: RecommendationRecord, executions: list[ExecutionRecord],
) -> str:
    """Bucket a recommendation by its execution status.

    - `followed`: action is BUY/ADD/TRIM/EXIT AND at least one matching
      execution exists with shares ≥ ~90% of delta_shares.
    - `partial`: actionable recommendation has executions but shares
      filled < 90% of recommended delta.
    - `overridden`: actionable recommendation has zero matching
      executions but the user TRADED the same symbol on the same date
      in the OPPOSITE side.
    - `ignored`: actionable recommendation, zero executions, no opposite
      trade on the symbol.
    - `n_a`: action is HOLD/SKIP/REVIEW (nothing to compare).
    """
    if rec.action in ("HOLD", "SKIP", "REVIEW"):
        return "n_a"
    if not executions:
        return "ignored"
    abs_filled = sum(abs(e.shares) for e in executions)
    abs_target = abs(rec.delta_shares) if rec.delta_shares else 0
    if abs_target == 0:
        # Recommended action with zero delta — treat as fully-followed.
        return "followed"
    coverage = abs_filled / abs_target
    if coverage >= 0.9:
        return "followed"
    return "partial"


def unattributed_executions(
    recommendations_by_date: dict[str, list[RecommendationRecord]],
    executions_by_date: dict[str, list[ExecutionRecord]],
) -> list[ExecutionRecord]:
    """Executions that don't match any recommendation — pure discretionary."""
    rec_keys: set[tuple[str, str]] = set()
    for rec_date, recs in recommendations_by_date.items():
        for r in recs:
            if r.action not in ("HOLD", "SKIP", "REVIEW"):
                rec_keys.add((rec_date, r.symbol.upper()))
    out: list[ExecutionRecord] = []
    for trade_date, execs in executions_by_date.items():
        for e in execs:
            sym = e.symbol.upper()
            if e.ref_recommendation_date and (e.ref_recommendation_date, sym) in rec_keys:
                continue
            if (trade_date, sym) in rec_keys:
                continue
            out.append(e)
    return out


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
