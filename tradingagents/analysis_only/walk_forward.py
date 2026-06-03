"""Walk-forward out-of-sample evaluation harness.

Pure functions that take a corpus of `BacktestRecord`s plus a
weight-fitting `Callable[[list[BacktestRecord]], dict[str, float] | None]`
and produce rolling-window train/test stats. Lets you compare a fixed
weight vector (e.g. v1.4 / v1.5) vs a per-window IC refit against the
same set of windows, exposing how much of any single-split lift is
real vs overfit.

Cadence (defaults, configurable):
- 18-month train window
- 3-month test window (non-overlapping, immediately after train)
- 1-month step between consecutive windows

This file is intentionally distinct from the older
`walk_forward_backtest` in `backtest.py`, which (a) emits a single
concatenated rebuilt-record stream rather than per-window train/test
stats and (b) uses weekly anchors with a gap. The Unit-A harness here
exposes per-window stats so we can compute the median test hit-rate and
median overfit gap across a calendar grid of windows.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

from tradingagents.analysis_only.backtest import (
    BacktestRecord,
    is_hit,
    rebuild_records_with_weights,
)


WeightFn = Callable[[list[BacktestRecord]], "dict[str, float] | None"]


# ---------- windowing ----------


@dataclass(frozen=True)
class WalkForwardWindow:
    """One non-overlapping (train, test) span. Dates are ISO YYYY-MM-DD.

    Train spans `[train_start, train_end]` inclusive. Test spans
    `[test_start, test_end]` inclusive. By convention `test_start` is
    the day after `train_end` and the test slice is `test_months` long.
    """

    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _parse_iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _add_months(d: date, months: int) -> date:
    """Add `months` to `d` (calendar-month arithmetic, day clamps).

    Pure-Python so we don't pull in `dateutil`. Pinning to the last day
    of the month when overflowing is fine for window arithmetic.
    """
    total_months = d.month - 1 + months
    new_year = d.year + total_months // 12
    new_month = total_months % 12 + 1
    # Clamp day to the new month's max day.
    if new_month == 12:
        next_month_first = date(new_year + 1, 1, 1)
    else:
        next_month_first = date(new_year, new_month + 1, 1)
    from datetime import timedelta

    last_day_of_new_month = (next_month_first - timedelta(days=1)).day
    new_day = min(d.day, last_day_of_new_month)
    return date(new_year, new_month, new_day)


def _sub_days(d: date, days: int) -> date:
    from datetime import timedelta

    return d - timedelta(days=days)


def generate_windows(
    *,
    corpus_min_date: str,
    corpus_max_date: str,
    train_months: int = 18,
    test_months: int = 3,
    step_months: int = 1,
) -> list[WalkForwardWindow]:
    """Emit non-overlapping (train, test) windows that walk forward.

    Each window has `train_months` of training followed immediately by
    `test_months` of testing. Successive windows step by `step_months`.
    Windows whose `test_end > corpus_max_date` are skipped — incomplete
    forward-return data would corrupt the test hit-rate.
    """
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError(
            "train_months, test_months, step_months must all be positive"
        )
    start = _parse_iso(corpus_min_date)
    end_cap = _parse_iso(corpus_max_date)
    out: list[WalkForwardWindow] = []
    cursor_train_start = start
    while True:
        train_end_exclusive = _add_months(cursor_train_start, train_months)
        train_end = _sub_days(train_end_exclusive, 1)
        test_start = train_end_exclusive
        test_end_exclusive = _add_months(test_start, test_months)
        test_end = _sub_days(test_end_exclusive, 1)
        if test_end > end_cap:
            break
        out.append(
            WalkForwardWindow(
                train_start=cursor_train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
            )
        )
        cursor_train_start = _add_months(cursor_train_start, step_months)
    return out


# ---------- per-window evaluation ----------


def _slice_inclusive(
    records: Iterable[BacktestRecord], start: str, end: str
) -> list[BacktestRecord]:
    return [r for r in records if start <= r.as_of_date <= end]


def _hit_rate(records: Sequence[BacktestRecord], horizon: str) -> tuple[float | None, int]:
    """Direction-aware hit-rate aggregated across all directions.

    Each record contributes one boolean — True iff its direction call
    "hit" its own forward return at this horizon. Mixes bullish/bearish/
    neutral records into a single number. Useful for one-line summaries
    but tanks when the neutral bucket dominates (neutral misses unless
    |ret| < 2%, so high-vol horizons read low here).

    See `_hit_rate_by_direction` for the per-bucket form that matches
    handoff Section 27's bullish-only hit-rate convention.
    """
    hits: list[bool] = []
    for r in records:
        v = r.forward_returns.get(horizon)
        if v is None:
            continue
        h = is_hit(r.direction, v)
        if h is None:
            continue
        hits.append(h)
    if not hits:
        return None, 0
    return round(sum(1 for h in hits if h) / len(hits), 4), len(hits)


def _hit_rate_by_direction(
    records: Sequence[BacktestRecord], horizon: str
) -> dict[str, dict[str, Any]]:
    """Per-direction hit-rate. Matches Section 27's bullish-only headline.

    Returns `{direction: {"hit_rate": float|None, "n": int}}` for
    bullish, bearish, neutral. Records missing a forward return are
    dropped from the count for that horizon.
    """
    by_dir: dict[str, list[bool]] = {"bullish": [], "bearish": [], "neutral": []}
    for r in records:
        if r.direction not in by_dir:
            continue
        v = r.forward_returns.get(horizon)
        if v is None:
            continue
        h = is_hit(r.direction, v)
        if h is None:
            continue
        by_dir[r.direction].append(h)
    out: dict[str, dict[str, Any]] = {}
    for d, hits in by_dir.items():
        if hits:
            out[d] = {
                "hit_rate": round(sum(1 for h in hits if h) / len(hits), 4),
                "n": len(hits),
            }
        else:
            out[d] = {"hit_rate": None, "n": 0}
    return out


def evaluate_window(
    records: list[BacktestRecord],
    window: WalkForwardWindow,
    *,
    weight_fn: WeightFn,
    horizons: Sequence[str] = ("ret_5d", "ret_20d", "ret_60d"),
) -> dict[str, Any]:
    """Train weights on the train slice, evaluate on the test slice.

    `weight_fn` is called with the train slice and must return either a
    weight vector (dict[str, float]) or `None`. When it returns `None`
    the records' as-emitted composite/direction are used — useful for
    measuring a fixed (already-baked) weight vector against the rolling
    cadence.

    When `weight_fn` returns a weight vector, both train and test slices
    are rebuilt with that vector before hit-rates are computed. This
    keeps the train_hit (in-sample) and test_hit (out-of-sample)
    comparable: both are direction-conditional hit-rates under the same
    weight vector — just on different time slices.

    Returns a dict shaped like:
        {
            "window": {train_start, train_end, test_start, test_end},
            "train_n": int,
            "test_n": int,
            "weights_used": dict or None,
            "per_horizon": {
                horizon: {
                    "train_hit": float | None,
                    "test_hit": float | None,
                    "overfit_gap": float | None,  # train_hit - test_hit
                    "n_train_with_return": int,
                    "n_test_with_return": int,
                },
            },
        }
    """
    train = _slice_inclusive(records, window.train_start, window.train_end)
    test = _slice_inclusive(records, window.test_start, window.test_end)
    weights = weight_fn(train) if weight_fn is not None else None
    if weights:
        train_eval = rebuild_records_with_weights(train, weights=weights)
        test_eval = rebuild_records_with_weights(test, weights=weights)
    else:
        train_eval = train
        test_eval = test
    per_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        train_hit, n_train = _hit_rate(train_eval, horizon)
        test_hit, n_test = _hit_rate(test_eval, horizon)
        gap: float | None
        if train_hit is None or test_hit is None:
            gap = None
        else:
            gap = round(train_hit - test_hit, 4)
        # Per-direction hit-rates: lets us track the bullish-only number
        # that's apples-to-apples vs handoff Section 27's headline.
        train_by_dir = _hit_rate_by_direction(train_eval, horizon)
        test_by_dir = _hit_rate_by_direction(test_eval, horizon)
        bull_train = train_by_dir.get("bullish", {}).get("hit_rate")
        bull_test = test_by_dir.get("bullish", {}).get("hit_rate")
        bull_gap: float | None
        if bull_train is None or bull_test is None:
            bull_gap = None
        else:
            bull_gap = round(bull_train - bull_test, 4)
        per_horizon[horizon] = {
            "train_hit": train_hit,
            "test_hit": test_hit,
            "overfit_gap": gap,
            "n_train_with_return": n_train,
            "n_test_with_return": n_test,
            "bullish_train_hit": bull_train,
            "bullish_test_hit": bull_test,
            "bullish_overfit_gap": bull_gap,
            "n_bullish_test": test_by_dir.get("bullish", {}).get("n", 0),
            "bearish_test_hit": test_by_dir.get("bearish", {}).get("hit_rate"),
            "n_bearish_test": test_by_dir.get("bearish", {}).get("n", 0),
            "neutral_test_hit": test_by_dir.get("neutral", {}).get("hit_rate"),
            "n_neutral_test": test_by_dir.get("neutral", {}).get("n", 0),
        }
    return {
        "window": window.as_dict(),
        "train_n": len(train),
        "test_n": len(test),
        "weights_used": (dict(weights) if weights else None),
        "per_horizon": per_horizon,
    }


# ---------- aggregation ----------


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile. `pct` in [0, 100]."""
    if not values:
        return None
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    if pct <= 0:
        return vs[0]
    if pct >= 100:
        return vs[-1]
    k = (len(vs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def summarize_walk_forward(
    per_window_stats: list[dict[str, Any]],
    horizons: Sequence[str] = ("ret_5d", "ret_20d", "ret_60d"),
    *,
    baseline_hit_rate: float = 0.5,
) -> dict[str, Any]:
    """Aggregate per-window stats into median/mean/p25/p75 of test_hit etc.

    `fraction_windows_beat_baseline` is the fraction of windows where
    `test_hit > baseline_hit_rate` at that horizon (None windows skipped).
    """
    per_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        test_hits: list[float] = []
        train_hits: list[float] = []
        gaps: list[float] = []
        bull_test_hits: list[float] = []
        bull_train_hits: list[float] = []
        bull_gaps: list[float] = []
        n_beat = 0
        n_eval = 0
        n_bull_beat = 0
        n_bull_eval = 0
        for ws in per_window_stats:
            stats = (ws.get("per_horizon") or {}).get(horizon) or {}
            th = stats.get("test_hit")
            trh = stats.get("train_hit")
            g = stats.get("overfit_gap")
            if th is not None:
                test_hits.append(float(th))
                n_eval += 1
                if th > baseline_hit_rate:
                    n_beat += 1
            if trh is not None:
                train_hits.append(float(trh))
            if g is not None:
                gaps.append(float(g))
            bth = stats.get("bullish_test_hit")
            btrh = stats.get("bullish_train_hit")
            bg = stats.get("bullish_overfit_gap")
            if bth is not None:
                bull_test_hits.append(float(bth))
                n_bull_eval += 1
                if bth > baseline_hit_rate:
                    n_bull_beat += 1
            if btrh is not None:
                bull_train_hits.append(float(btrh))
            if bg is not None:
                bull_gaps.append(float(bg))
        per_horizon[horizon] = {
            "n_windows_with_test_hit": n_eval,
            "median_test_hit": (
                round(statistics.median(test_hits), 4) if test_hits else None
            ),
            "mean_test_hit": (
                round(statistics.fmean(test_hits), 4) if test_hits else None
            ),
            "p25_test_hit": (
                round(_percentile(test_hits, 25), 4) if test_hits else None
            ),
            "p75_test_hit": (
                round(_percentile(test_hits, 75), 4) if test_hits else None
            ),
            "median_train_hit": (
                round(statistics.median(train_hits), 4)
                if train_hits else None
            ),
            "median_overfit_gap": (
                round(statistics.median(gaps), 4) if gaps else None
            ),
            "mean_overfit_gap": (
                round(statistics.fmean(gaps), 4) if gaps else None
            ),
            "fraction_windows_beat_baseline": (
                round(n_beat / n_eval, 4) if n_eval else None
            ),
            # Bullish-only block. Apples-to-apples vs handoff Section 27.
            "n_windows_with_bullish_test_hit": n_bull_eval,
            "median_bullish_test_hit": (
                round(statistics.median(bull_test_hits), 4)
                if bull_test_hits else None
            ),
            "mean_bullish_test_hit": (
                round(statistics.fmean(bull_test_hits), 4)
                if bull_test_hits else None
            ),
            "p25_bullish_test_hit": (
                round(_percentile(bull_test_hits, 25), 4)
                if bull_test_hits else None
            ),
            "p75_bullish_test_hit": (
                round(_percentile(bull_test_hits, 75), 4)
                if bull_test_hits else None
            ),
            "median_bullish_train_hit": (
                round(statistics.median(bull_train_hits), 4)
                if bull_train_hits else None
            ),
            "median_bullish_overfit_gap": (
                round(statistics.median(bull_gaps), 4)
                if bull_gaps else None
            ),
            "fraction_windows_bullish_beat_baseline": (
                round(n_bull_beat / n_bull_eval, 4) if n_bull_eval else None
            ),
        }
    return {
        "n_windows": len(per_window_stats),
        "baseline_hit_rate": baseline_hit_rate,
        "per_horizon": per_horizon,
    }


# ---------- rendering ----------


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:,.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _pp(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+,.2f}pp"
    except (TypeError, ValueError):
        return str(v)


def render_walk_forward_markdown(
    summary: dict[str, Any],
    *,
    baseline_hit_rate: float = 0.5,
    title: str = "Walk-forward OOS summary",
) -> str:
    """One table per horizon. Compact median / quartile / gap layout."""
    lines = [f"# {title}", ""]
    n_windows = summary.get("n_windows", 0)
    lines.append(f"- Windows evaluated: **{n_windows}**")
    lines.append(
        f"- Baseline (passive `test_hit > X` threshold): **{baseline_hit_rate:.2f}**"
    )
    lines.append("")
    per_horizon = summary.get("per_horizon") or {}
    if not per_horizon:
        lines.append("_No per-horizon stats — empty input._")
        return "\n".join(lines).rstrip() + "\n"
    for horizon, stats in per_horizon.items():
        lines.append(f"## Horizon `{horizon}`")
        lines.append("")
        lines.append(
            "_All-direction hit-rate mixes bullish/bearish/neutral. "
            "Bullish-only block below is apples-to-apples vs the "
            "handoff Section 27 headline._"
        )
        lines.append("")
        lines.append(
            "| Metric | Value |"
        )
        lines.append("|---|---:|")
        lines.append(
            f"| N windows (with test_hit) | {stats.get('n_windows_with_test_hit') or 0} |"
        )
        lines.append(
            f"| Median test_hit (all-direction) | {_pct(stats.get('median_test_hit'))} |"
        )
        lines.append(
            f"| Mean test_hit (all-direction) | {_pct(stats.get('mean_test_hit'))} |"
        )
        lines.append(
            f"| P25 / P75 test_hit | {_pct(stats.get('p25_test_hit'))} / {_pct(stats.get('p75_test_hit'))} |"
        )
        lines.append(
            f"| Median train_hit (all-direction) | {_pct(stats.get('median_train_hit'))} |"
        )
        lines.append(
            f"| Median overfit gap (train - test) | {_pp(stats.get('median_overfit_gap'))} |"
        )
        lines.append(
            f"| Mean overfit gap (train - test) | {_pp(stats.get('mean_overfit_gap'))} |"
        )
        lines.append(
            f"| Fraction windows beat baseline | {_pct(stats.get('fraction_windows_beat_baseline'))} |"
        )
        lines.append(
            f"| **Median BULLISH test_hit** | **{_pct(stats.get('median_bullish_test_hit'))}** |"
        )
        lines.append(
            f"| Mean BULLISH test_hit | {_pct(stats.get('mean_bullish_test_hit'))} |"
        )
        lines.append(
            f"| P25 / P75 BULLISH test_hit | {_pct(stats.get('p25_bullish_test_hit'))} / {_pct(stats.get('p75_bullish_test_hit'))} |"
        )
        lines.append(
            f"| Median BULLISH train_hit | {_pct(stats.get('median_bullish_train_hit'))} |"
        )
        lines.append(
            f"| Median BULLISH overfit gap (train - test) | {_pp(stats.get('median_bullish_overfit_gap'))} |"
        )
        lines.append(
            f"| Fraction windows BULLISH beat baseline | {_pct(stats.get('fraction_windows_bullish_beat_baseline'))} |"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
