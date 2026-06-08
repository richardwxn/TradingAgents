"""Pure backtest aggregation helpers.

Score past `AnalysisReport` JSON outputs against forward returns. All
functions in here are pure: callers pass already-collected
`BacktestRecord`s in and get summary stats out. The CLI in
`backtest.py` at the repo root handles I/O.

The "hit" definition is direction-aware:
- bullish  → hit if forward_return > 0
- bearish  → hit if forward_return < 0
- neutral  → hit if |forward_return| < `neutral_band` (default 2%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import math
import statistics


# ---------- record types ----------


@dataclass
class BacktestRecord:
    symbol: str
    as_of_date: str
    direction: str
    confidence: float | None
    composite_score: float | None
    forward_returns: dict[str, float | None] = field(default_factory=dict)
    # Same shape as `forward_returns`, but with the benchmark's forward
    # return over the matching trading-day horizon subtracted off. Populated
    # by the CLI only when a benchmark symbol is supplied.
    benchmark_adjusted_returns: dict[str, float | None] = field(default_factory=dict)
    # Raw per-factor rows pulled straight from
    # `key_features.model_scoring.factor_scores`. Only populated when the
    # `--by-factor` CLI path needs them; kept optional so the basic
    # `summarize_all` flow doesn't pay the memory cost.
    factor_scores: list[dict[str, Any]] | None = None
    # Phase 4: market_context snapshot from the report. Used to derive a
    # regime label (`scoring.regime_for_market_context`). Kept optional
    # so callers that don't need regime conditioning don't pay the cost.
    market_context: dict[str, Any] | None = None
    # Phase 6: validated llm_critic output (from `llm_critic.run_critic`).
    # Only the validated `output` subdict is stored here for backtest
    # arithmetic; the full block (status/provider/hash) lives on disk.
    llm_critic: dict[str, Any] | None = None
    # Phase 7: pre-computed `llm_critic_multi.disagreement` block. Captured
    # the same way as `llm_critic` — only populated when the caller passes
    # `capture_llm_critic=True` and the report actually has a disagreement
    # block (i.e. was backfilled across 2+ models).
    llm_disagreement: dict[str, Any] | None = None
    source_path: str | None = None


@dataclass
class FactorRecord:
    """One factor observation extracted from a `BacktestRecord`.

    `score` is the normalized -1..+1 factor score from `scoring.py`.
    `bucket` is `"bullish" | "bearish" | "neutral"` if present in the
    source report.
    """

    symbol: str
    as_of_date: str
    factor: str
    pillar: str
    score: float | None
    weighted_score: float | None
    bucket: str | None
    data_available: bool
    forward_returns: dict[str, float | None] = field(default_factory=dict)
    benchmark_adjusted_returns: dict[str, float | None] = field(default_factory=dict)


# ---------- bucketing ----------


def bucket_by_direction(
    records: Iterable[BacktestRecord],
) -> dict[str, list[BacktestRecord]]:
    out: dict[str, list[BacktestRecord]] = {
        "bullish": [],
        "bearish": [],
        "neutral": [],
    }
    for r in records:
        out.setdefault(r.direction, []).append(r)
    return out


def bucket_by_confidence(
    records: Iterable[BacktestRecord],
    bins: list[float] | None = None,
) -> dict[str, list[BacktestRecord]]:
    bins = bins or [0.5, 0.6, 0.7, 0.8, 1.0]
    return _bucket_numeric(
        records,
        bins=bins,
        get=lambda r: r.confidence,
        label_fmt="[{lo:.2f}, {hi:.2f})",
    )


def bucket_by_composite(
    records: Iterable[BacktestRecord],
    bins: list[float] | None = None,
) -> dict[str, list[BacktestRecord]]:
    bins = bins or [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0 + 1e-9]
    return _bucket_numeric(
        records,
        bins=bins,
        get=lambda r: r.composite_score,
        label_fmt="[{lo:+.2f}, {hi:+.2f})",
    )


def _bucket_numeric(
    records: Iterable[BacktestRecord],
    *,
    bins: list[float],
    get,
    label_fmt: str,
) -> dict[str, list[BacktestRecord]]:
    out: dict[str, list[BacktestRecord]] = {}
    for i in range(len(bins) - 1):
        out[label_fmt.format(lo=bins[i], hi=bins[i + 1])] = []
    out["unknown"] = []
    for r in records:
        value = get(r)
        if value is None:
            out["unknown"].append(r)
            continue
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                out[label_fmt.format(lo=bins[i], hi=bins[i + 1])].append(r)
                placed = True
                break
        if not placed:
            out["unknown"].append(r)
    return out


# ---------- hit rate + summary ----------


def is_hit(
    direction: str,
    forward_return: float | None,
    neutral_band: float = 0.02,
) -> bool | None:
    if forward_return is None:
        return None
    if direction == "bullish":
        return forward_return > 0
    if direction == "bearish":
        return forward_return < 0
    if direction == "neutral":
        return abs(forward_return) < neutral_band
    return None


def summarize_bucket(
    records: list[BacktestRecord],
    return_field: str = "ret_20d",
    neutral_band: float = 0.02,
) -> dict[str, Any]:
    rets: list[float] = []
    hits: list[bool] = []
    for r in records:
        v = r.forward_returns.get(return_field)
        if v is None:
            continue
        rets.append(float(v))
        hit = is_hit(r.direction, v, neutral_band=neutral_band)
        if hit is not None:
            hits.append(hit)
    if not rets:
        return {
            "count": len(records),
            "count_with_return": 0,
            "mean_forward_return": None,
            "median_forward_return": None,
            "p25_forward_return": None,
            "p75_forward_return": None,
            "hit_rate": None,
        }
    mean_r = round(statistics.fmean(rets), 6)
    median_r = round(statistics.median(rets), 6)
    if len(rets) >= 4:
        rets_sorted = sorted(rets)
        p25 = rets_sorted[len(rets_sorted) // 4]
        p75 = rets_sorted[(3 * len(rets_sorted)) // 4]
    else:
        p25 = min(rets)
        p75 = max(rets)
    hit_rate = (
        round(sum(1 for h in hits if h) / len(hits), 4) if hits else None
    )
    return {
        "count": len(records),
        "count_with_return": len(rets),
        "mean_forward_return": mean_r,
        "median_forward_return": median_r,
        "p25_forward_return": round(p25, 6),
        "p75_forward_return": round(p75, 6),
        "hit_rate": hit_rate,
    }


def summarize_all(
    records: list[BacktestRecord],
    return_fields: list[str] | None = None,
    neutral_band: float = 0.02,
) -> dict[str, Any]:
    """Top-level summary across all bucketing dimensions and horizons.

    Also adds a `return_metrics` block at top level keyed by horizon
    with risk-adjusted Sharpe/Sortino/MaxDD/ProfitFactor stats (overall
    plus per-direction). See `compute_return_metrics`.
    """
    return_fields = return_fields or ["ret_5d", "ret_20d", "ret_60d"]
    out: dict[str, Any] = {
        "total_records": len(records),
        "by_horizon": {},
        "return_metrics": {},
    }
    for field_name in return_fields:
        horizon_block: dict[str, Any] = {
            "overall": summarize_bucket(
                records,
                return_field=field_name,
                neutral_band=neutral_band,
            ),
            "by_direction": {},
            "by_confidence": {},
            "by_composite": {},
        }
        for bucket_name, bucket_records in bucket_by_direction(
            records
        ).items():
            horizon_block["by_direction"][bucket_name] = summarize_bucket(
                bucket_records,
                return_field=field_name,
                neutral_band=neutral_band,
            )
        for bucket_name, bucket_records in bucket_by_confidence(
            records
        ).items():
            if not bucket_records:
                continue
            horizon_block["by_confidence"][bucket_name] = summarize_bucket(
                bucket_records,
                return_field=field_name,
                neutral_band=neutral_band,
            )
        for bucket_name, bucket_records in bucket_by_composite(
            records
        ).items():
            if not bucket_records:
                continue
            horizon_block["by_composite"][bucket_name] = summarize_bucket(
                bucket_records,
                return_field=field_name,
                neutral_band=neutral_band,
            )
        out["by_horizon"][field_name] = horizon_block
        out["return_metrics"][field_name] = {
            "overall": compute_return_metrics(
                records,
                horizon=field_name,
                neutral_band=neutral_band,
            ),
            "by_direction": compute_return_metrics_by_direction(
                records,
                horizon=field_name,
                neutral_band=neutral_band,
            ),
        }
    return out


# ---------- volatility / P&L-based risk-adjusted metrics ----------


_TRADING_DAYS_PER_YEAR = 252


def _parse_horizon_days(horizon: str) -> int | None:
    """Extract trading-day count from a horizon string like 'ret_60d'.

    Returns `None` if the string doesn't match the expected `ret_<N>d`
    pattern. The Sharpe/Sortino annualization factor needs this; when
    `None`, the metric is reported unannualized.
    """
    if not isinstance(horizon, str):
        return None
    s = horizon.strip().lower()
    if s.startswith("ret_") and s.endswith("d"):
        body = s[4:-1]
        try:
            n = int(body)
            return n if n > 0 else None
        except ValueError:
            return None
    return None


def _winsorize(values: list[float], *, p: float = 0.05) -> list[float]:
    """Symmetric winsorize at `p` and `1-p` quantiles.

    Returns the values clipped to those quantile bounds. With fewer than
    `1/p` samples the bounds collapse to the min/max so the clip is a
    no-op — that's the intended behavior for tiny buckets.

    Specifically we clip the top `floor(p*n)` and bottom `floor(p*n)`
    values: the lower bound is the `lo_idx`-th sorted value (with
    `lo_idx = floor(p*n)`) and the upper bound is the
    `(n - 1 - lo_idx)`-th sorted value.
    """
    if not values:
        return []
    if not 0.0 < p < 0.5:
        return list(values)
    n = len(values)
    sorted_v = sorted(values)
    k = int(p * n)
    lo_idx = k
    hi_idx = max(lo_idx, n - 1 - k)
    lo = sorted_v[lo_idx]
    hi = sorted_v[hi_idx]
    return [min(hi, max(lo, v)) for v in values]


def _max_drawdown_from_returns(returns_in_order: list[float]) -> float:
    """Max drawdown on a cumulative-sum P&L curve.

    `returns_in_order` is the sequence of per-period strategy returns,
    treated as equal-sized bets so the cumulative P&L is just their
    running sum. Returns a non-positive number; 0.0 means no drawdown
    observed.

    Why sum-based instead of compounded (1+r)? The backtest records are
    many parallel bets per date, not one sequential portfolio. The
    compounded interpretation blows up when a single bearish record's
    P&L is below -100% (which is possible with sign-flipped raw returns
    above 100%). Equal-sized-bet cumulative P&L stays numerically
    well-defined for any return magnitude and aligns with
    `portfolio/simulator.py` Section 16's "equal-weight basket" model.
    """
    if not returns_in_order:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns_in_order:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak  # absolute drawdown in P&L units
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _strategy_pnl(direction: str, forward_return: float) -> float:
    """Convert a raw `forward_return` to the strategy's P&L for `direction`.

    - bullish: long-the-stock → P&L = forward_return.
    - bearish: short-the-stock → P&L = -forward_return.
    - neutral / anything else: treated as "no strategy bet"; we pass the
      raw return through so neutral-bucket stats describe the realized
      outcomes of those records (not a P&L, just the realized return).

    This direction-aware sign flip is what makes the bearish-bucket
    Sharpe a meaningful "strategy did/didn't work" signal — on a bull
    tape, calling bearish on stocks that actually rose is loss-making
    for the short, so bearish Sharpe should go negative.
    """
    if direction == "bearish":
        return -float(forward_return)
    return float(forward_return)


def compute_return_metrics(
    records: Iterable[BacktestRecord],
    *,
    horizon: str,
    direction_filter: str | None = None,
    neutral_band: float = 0.02,
    winsorize_p: float = 0.05,
) -> dict[str, Any]:
    """Risk-adjusted P&L metrics for `records` at one `horizon`.

    Returns are interpreted as **strategy P&L**, not raw stock returns:
    bearish records have their forward return sign-flipped (because the
    bearish "strategy" is short-the-stock). Bullish and neutral records
    pass the raw forward return through unchanged. This makes the Sharpe
    / Sortino / profit-factor numbers comparable across direction
    buckets and matches the convention used in `portfolio/simulator.py`.

    Keys returned:
    - `n_records`: total input count after `direction_filter` (if any).
    - `n_with_return`: count that actually had a forward-return at `horizon`.
    - `mean_return`, `median_return`: arithmetic stats on strategy P&L
      (raw, not annualized).
    - `sharpe`: annualized Sharpe = (mean / stdev) * sqrt(252 / horizon_days).
      `None` when stdev is zero or fewer than 2 returns are present, or
      when the horizon string isn't parseable as `ret_<N>d`.
    - `sortino`: annualized Sortino = (mean / downside_stdev) * sqrt(252 / horizon_days).
      Downside stdev uses returns < 0 with mean=0 (Sortino convention).
    - `max_drawdown`: peak-to-trough drawdown on the cumulative sum of
      strategy P&L with records ordered by `as_of_date` (then `symbol`
      for stability). Equal-sized-bet convention so the metric stays
      well-defined when individual records have extreme returns.
      Non-positive; 0.0 = no drawdown.
    - `winsorized_mean`: mean after 5/95-percentile winsorization.
    - `hit_rate`: direction-aware hit-rate using `is_hit` on the raw
      forward return. When `direction_filter` is set, all records share
      that direction so the hit-rate is well-defined; otherwise the
      per-record direction is used.
    - `profit_factor`: sum(positive P&L) / |sum(negative P&L)|.
      `None` when there are no losing trades (avoid div-by-zero); a
      very large but finite number when losses are tiny.
    - `n_positive`, `n_negative`, `n_zero`: count breakdown on the P&L.

    `winsorize_p` controls the symmetric trim used for `winsorized_mean`
    (default 0.05 = 5/95).

    Pure helper — no I/O, no yfinance, no dependence on `forward_returns`
    coming from any particular source.
    """
    horizon_days = _parse_horizon_days(horizon)
    annualization = (
        math.sqrt(_TRADING_DAYS_PER_YEAR / horizon_days)
        if horizon_days and horizon_days > 0
        else None
    )
    # Filter by direction first (so n_records reflects the filtered pool).
    if direction_filter is not None:
        records = [r for r in records if r.direction == direction_filter]
    else:
        records = list(records)

    # Sort by (as_of_date, symbol) so the equity curve for max_drawdown
    # is deterministic; this also matches the "time-series of bets"
    # interpretation used by the rest of the file.
    # `paired` carries (date, symbol, direction, raw_return, pnl_return).
    paired: list[tuple[str, str, str, float, float]] = []
    for r in records:
        v = r.forward_returns.get(horizon)
        if v is None:
            continue
        raw = float(v)
        paired.append((r.as_of_date, r.symbol, r.direction, raw, _strategy_pnl(r.direction, raw)))
    paired.sort(key=lambda t: (t[0], t[1]))

    rets_in_order = [t[4] for t in paired]
    n_records = len(records)
    n_with_return = len(rets_in_order)

    if n_with_return == 0:
        return {
            "n_records": n_records,
            "n_with_return": 0,
            "horizon_days": horizon_days,
            "annualization_factor": annualization,
            "mean_return": None,
            "median_return": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "winsorized_mean": None,
            "hit_rate": None,
            "profit_factor": None,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
        }

    mean_r = statistics.fmean(rets_in_order)
    median_r = statistics.median(rets_in_order)

    if n_with_return >= 2:
        stdev_r = statistics.stdev(rets_in_order)
    else:
        stdev_r = 0.0

    if stdev_r > 0 and annualization is not None:
        sharpe = (mean_r / stdev_r) * annualization
    elif stdev_r > 0:
        # No horizon-days parse → report unannualized so callers still
        # get a sensible signed ratio.
        sharpe = mean_r / stdev_r
    else:
        sharpe = None

    downside = [r for r in rets_in_order if r < 0]
    if len(downside) >= 2:
        # Sortino convention: RMS of negative returns relative to 0.
        downside_var = sum(r * r for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
    elif len(downside) == 1:
        downside_std = abs(downside[0])
    else:
        downside_std = 0.0

    if downside_std > 0 and annualization is not None:
        sortino = (mean_r / downside_std) * annualization
    elif downside_std > 0:
        sortino = mean_r / downside_std
    else:
        sortino = None

    max_dd = _max_drawdown_from_returns(rets_in_order)
    winsorized_mean = statistics.fmean(_winsorize(rets_in_order, p=winsorize_p))

    pos = [r for r in rets_in_order if r > 0]
    neg = [r for r in rets_in_order if r < 0]
    zero = [r for r in rets_in_order if r == 0]
    sum_neg = sum(neg)
    if neg and sum_neg < 0:
        profit_factor = sum(pos) / abs(sum_neg)
    else:
        profit_factor = None

    # Direction-aware hit-rate. When `direction_filter` is set, every
    # record shares that direction so the hit definition is fixed; when
    # not, we use the per-record direction stored on the record itself.
    # Hit-rate operates on the RAW forward return (not the sign-flipped
    # P&L) so the existing `is_hit` semantics stay intact.
    hits: list[bool] = []
    for _date, _sym, direction, raw, _pnl in paired:
        d = direction_filter or direction
        h = is_hit(d, raw, neutral_band=neutral_band)
        if h is not None:
            hits.append(h)
    hit_rate = (sum(1 for h in hits if h) / len(hits)) if hits else None

    return {
        "n_records": n_records,
        "n_with_return": n_with_return,
        "horizon_days": horizon_days,
        "annualization_factor": (
            round(annualization, 6) if annualization is not None else None
        ),
        "mean_return": round(mean_r, 6),
        "median_return": round(median_r, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6),
        "winsorized_mean": round(winsorized_mean, 6),
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "profit_factor": (
            round(profit_factor, 4) if profit_factor is not None else None
        ),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "n_zero": len(zero),
    }


def compute_return_metrics_by_direction(
    records: Iterable[BacktestRecord],
    *,
    horizon: str,
    neutral_band: float = 0.02,
    winsorize_p: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Per-direction risk-adjusted metrics at one horizon.

    Always returns a dict with three keys (`bullish`, `bearish`,
    `neutral`) so downstream renderers can iterate predictably even when
    a direction bucket is empty (the value will be a metrics dict with
    `n_records=0` / `n_with_return=0`).
    """
    records = list(records)
    out: dict[str, dict[str, Any]] = {}
    for direction in ("bullish", "bearish", "neutral"):
        out[direction] = compute_return_metrics(
            records,
            horizon=horizon,
            direction_filter=direction,
            neutral_band=neutral_band,
            winsorize_p=winsorize_p,
        )
    return out


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a `summarize_all` dict as a compact Markdown report."""
    lines = ["# Backtest summary", ""]
    lines.append(f"- Total records: **{summary.get('total_records', 0)}**")
    by_horizon = summary.get("by_horizon") or {}
    for horizon, block in by_horizon.items():
        lines.append(f"\n## Horizon: `{horizon}`")
        overall = block.get("overall") or {}
        lines.append("")
        lines.append("**Overall**\n")
        lines.append("| Count (with return) | Mean | Median | P25 | P75 | Hit rate |")
        lines.append("|---|---|---|---|---|---|")
        lines.append(
            "| {c} | {m} | {md} | {p25} | {p75} | {hr} |".format(
                c=overall.get("count_with_return"),
                m=_pct(overall.get("mean_forward_return")),
                md=_pct(overall.get("median_forward_return")),
                p25=_pct(overall.get("p25_forward_return")),
                p75=_pct(overall.get("p75_forward_return")),
                hr=_pct(overall.get("hit_rate")),
            )
        )
        lines += _render_bucket_table("By direction", block.get("by_direction") or {})
        lines += _render_bucket_table("By confidence bucket", block.get("by_confidence") or {})
        lines += _render_bucket_table("By composite score bucket", block.get("by_composite") or {})
        # Risk-adjusted (Sharpe/Sortino/MaxDD/ProfitFactor) per-horizon
        # section. Only emitted when the caller populated
        # `summary["return_metrics"][<horizon>]`. Schema matches the
        # output of `compute_return_metrics` / `_by_direction`.
        rm_for_horizon = (summary.get("return_metrics") or {}).get(horizon) or {}
        lines += _render_return_metrics_section(rm_for_horizon)
    return "\n".join(lines).rstrip() + "\n"


def _render_return_metrics_section(block: dict[str, Any]) -> list[str]:
    """One Risk-adjusted-metrics section for a single horizon.

    `block` shape (any subset of these may be missing — render what we
    have): {"overall": {...}, "by_direction": {bullish: {...}, ...}}.
    """
    if not block:
        return []
    out: list[str] = ["", "**Risk-adjusted metrics**", ""]
    overall = block.get("overall") or {}
    by_dir = block.get("by_direction") or {}
    rows = []
    if overall:
        rows.append(("overall", overall))
    for direction in ("bullish", "bearish", "neutral"):
        stats = by_dir.get(direction)
        if not stats:
            continue
        rows.append((direction, stats))
    if not rows:
        return []
    out.append(
        "| Bucket | N | Mean | Winsor. mean | Sharpe (ann.) | "
        "Sortino (ann.) | Max DD (R units) | Profit factor | Hit rate |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, stats in rows:
        out.append(
            "| {l} | {n} | {m} | {wm} | {sh} | {so} | {dd} | {pf} | {hr} |".format(
                l=label,
                n=stats.get("n_with_return") or 0,
                m=_pct(stats.get("mean_return")),
                wm=_pct(stats.get("winsorized_mean")),
                sh=_fmt_signed(stats.get("sharpe"), digits=2),
                so=_fmt_signed(stats.get("sortino"), digits=2),
                # Max DD is cumulative-sum P&L (R units = each record's
                # forward-return at 1R bet size), NOT a percentage.
                dd=_fmt_signed(stats.get("max_drawdown"), digits=2),
                pf=_fmt_signed(stats.get("profit_factor"), digits=2),
                hr=_pct(stats.get("hit_rate")),
            )
        )
    return out


def _render_bucket_table(title: str, buckets: dict[str, dict]) -> list[str]:
    if not buckets:
        return []
    lines = ["", f"**{title}**", ""]
    lines.append("| Bucket | Count (with return) | Mean | Median | Hit rate |")
    lines.append("|---|---|---|---|---|")
    for label, stats in buckets.items():
        lines.append(
            "| {l} | {c} | {m} | {md} | {hr} |".format(
                l=label,
                c=stats.get("count_with_return"),
                m=_pct(stats.get("mean_forward_return")),
                md=_pct(stats.get("median_forward_return")),
                hr=_pct(stats.get("hit_rate")),
            )
        )
    return lines


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:,.2f}%"
    except (TypeError, ValueError):
        return str(v)


# ---------- per-factor IC ----------


def _rank_with_ties(values: Sequence[float]) -> list[float]:
    """Average-rank tied values (1-indexed) for Spearman correlation."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_correlation(
    xs: Sequence[float], ys: Sequence[float]
) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return None
    return sxy / denom


def spearman_correlation(
    xs: Sequence[float], ys: Sequence[float]
) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx = _rank_with_ties(list(xs))
    ry = _rank_with_ties(list(ys))
    return pearson_correlation(rx, ry)


def explode_records_to_factors(
    records: Iterable[BacktestRecord],
) -> dict[str, list[FactorRecord]]:
    """Flatten reports into per-factor observations grouped by factor name."""
    by_factor: dict[str, list[FactorRecord]] = {}
    for r in records:
        if not r.factor_scores:
            continue
        for f in r.factor_scores:
            name = f.get("factor")
            if not name:
                continue
            fr = FactorRecord(
                symbol=r.symbol,
                as_of_date=r.as_of_date,
                factor=name,
                pillar=f.get("pillar") or "",
                score=_as_float(f.get("score")),
                weighted_score=_as_float(f.get("weighted_score")),
                bucket=f.get("bucket"),
                data_available=bool(f.get("data_available", True)),
                forward_returns=dict(r.forward_returns),
                benchmark_adjusted_returns=dict(r.benchmark_adjusted_returns),
            )
            by_factor.setdefault(name, []).append(fr)
    return by_factor


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize_factor(
    records: list[FactorRecord],
    return_field: str,
    *,
    bullish_threshold: float = 0.2,
    bearish_threshold: float = -0.2,
    use_benchmark_adjusted: bool = False,
) -> dict[str, Any]:
    """Per-factor IC + long-short + hit-rate stats for one horizon.

    The "long" side is records with `score >= bullish_threshold` and the
    "short" side is `score <= bearish_threshold`. `long_short_spread` is
    `mean_ret_when_bullish - mean_ret_when_bearish`. Higher is better.
    """
    return_dict_attr = (
        "benchmark_adjusted_returns" if use_benchmark_adjusted else "forward_returns"
    )
    scores: list[float] = []
    rets: list[float] = []
    bull_rets: list[float] = []
    bear_rets: list[float] = []
    neutral_rets: list[float] = []
    n_missing_score = 0
    n_missing_return = 0
    for r in records:
        if r.score is None or not r.data_available:
            n_missing_score += 1
            continue
        v = getattr(r, return_dict_attr).get(return_field)
        if v is None:
            n_missing_return += 1
            continue
        scores.append(float(r.score))
        rets.append(float(v))
        if r.score >= bullish_threshold:
            bull_rets.append(float(v))
        elif r.score <= bearish_threshold:
            bear_rets.append(float(v))
        else:
            neutral_rets.append(float(v))

    sp_ic = spearman_correlation(scores, rets)
    pe_ic = pearson_correlation(scores, rets)

    def _mean(seq: list[float]) -> float | None:
        return round(statistics.fmean(seq), 6) if seq else None

    bull_hit = (
        round(sum(1 for v in bull_rets if v > 0) / len(bull_rets), 4)
        if bull_rets else None
    )
    bear_hit = (
        round(sum(1 for v in bear_rets if v < 0) / len(bear_rets), 4)
        if bear_rets else None
    )
    long_short = None
    if bull_rets and bear_rets:
        long_short = round(_mean(bull_rets) - _mean(bear_rets), 6)

    return {
        "n_paired": len(scores),
        "n_missing_score": n_missing_score,
        "n_missing_return": n_missing_return,
        "spearman_ic": round(sp_ic, 4) if sp_ic is not None else None,
        "pearson_ic": round(pe_ic, 4) if pe_ic is not None else None,
        "n_bullish_score": len(bull_rets),
        "n_bearish_score": len(bear_rets),
        "n_neutral_score": len(neutral_rets),
        "mean_ret_when_bullish": _mean(bull_rets),
        "mean_ret_when_bearish": _mean(bear_rets),
        "mean_ret_when_neutral": _mean(neutral_rets),
        "hit_rate_when_bullish": bull_hit,
        "hit_rate_when_bearish": bear_hit,
        "long_short_spread": long_short,
    }


def summarize_factors(
    records_by_factor: dict[str, list[FactorRecord]],
    return_fields: list[str],
    *,
    use_benchmark_adjusted: bool = False,
    bullish_threshold: float = 0.2,
    bearish_threshold: float = -0.2,
) -> dict[str, Any]:
    """Compute per-factor stats across all horizons."""
    out: dict[str, Any] = {}
    for factor, recs in records_by_factor.items():
        if not recs:
            continue
        out[factor] = {
            "pillar": recs[0].pillar,
            "total_observations": len(recs),
            "by_horizon": {
                f: summarize_factor(
                    recs,
                    return_field=f,
                    use_benchmark_adjusted=use_benchmark_adjusted,
                    bullish_threshold=bullish_threshold,
                    bearish_threshold=bearish_threshold,
                )
                for f in return_fields
            },
        }
    return out


def summarize_factors_by_ticker(
    records_by_factor: dict[str, list[FactorRecord]],
    return_fields: list[str],
    *,
    use_benchmark_adjusted: bool = False,
    min_obs_per_ticker: int = 8,
) -> dict[str, Any]:
    """For each factor × ticker × horizon, compute IC.

    Then per-factor per-horizon, aggregate across tickers:
    - `n_tickers_evaluated`: number of tickers with ≥ `min_obs_per_ticker`
      paired observations.
    - `median_ic_across_tickers`
    - `mean_ic_across_tickers`
    - `consistency_pct`: fraction of tickers whose IC sign matches the sign
      of the median IC. 1.0 = unanimous, 0.5 = pure noise.
    - `per_ticker_ic`: ticker → IC at the headline horizon (for inspection).

    This is the cheap robustness check that exposes single-ticker-driven IC.
    """
    return_dict_attr = (
        "benchmark_adjusted_returns" if use_benchmark_adjusted else "forward_returns"
    )
    out: dict[str, Any] = {}
    for factor, recs in records_by_factor.items():
        if not recs:
            continue
        # Group records by ticker for this factor.
        by_ticker: dict[str, list[FactorRecord]] = {}
        for r in recs:
            by_ticker.setdefault(r.symbol, []).append(r)
        horizon_block: dict[str, Any] = {}
        for horizon in return_fields:
            per_ticker_ic: dict[str, float | None] = {}
            for ticker, t_recs in by_ticker.items():
                scores: list[float] = []
                rets: list[float] = []
                for r in t_recs:
                    if r.score is None or not r.data_available:
                        continue
                    v = getattr(r, return_dict_attr).get(horizon)
                    if v is None:
                        continue
                    scores.append(float(r.score))
                    rets.append(float(v))
                if len(scores) < min_obs_per_ticker:
                    continue
                per_ticker_ic[ticker] = spearman_correlation(scores, rets)

            non_null = [v for v in per_ticker_ic.values() if v is not None]
            if non_null:
                med = statistics.median(non_null)
                mean = statistics.fmean(non_null)
                # Consistency = fraction of tickers whose IC has the same sign
                # as the median. Ties (IC=0) count as inconsistent.
                if med > 0:
                    same_sign = sum(1 for v in non_null if v > 0)
                elif med < 0:
                    same_sign = sum(1 for v in non_null if v < 0)
                else:
                    same_sign = 0
                consistency = round(same_sign / len(non_null), 4)
            else:
                med = None
                mean = None
                consistency = None
            horizon_block[horizon] = {
                "n_tickers_evaluated": len(non_null),
                "median_ic_across_tickers": (
                    round(med, 4) if med is not None else None
                ),
                "mean_ic_across_tickers": (
                    round(mean, 4) if mean is not None else None
                ),
                "consistency_pct": consistency,
                "per_ticker_ic": {
                    k: (round(v, 4) if v is not None else None)
                    for k, v in sorted(per_ticker_ic.items())
                },
            }
        out[factor] = {
            "pillar": recs[0].pillar,
            "total_observations": len(recs),
            "by_horizon": horizon_block,
        }
    return out


def render_factor_summary_markdown(
    factor_summary: dict[str, Any],
    return_fields: list[str],
    *,
    title: str = "Per-factor IC summary",
    return_label: str = "raw",
) -> str:
    """Render factor stats as one Markdown table per horizon, sorted by |IC|."""
    lines = [f"# {title}", "", f"_Return type: **{return_label}**_", ""]
    if not factor_summary:
        return "\n".join(lines).rstrip() + "\n"
    for horizon in return_fields:
        lines.append(f"## Horizon `{horizon}`")
        lines.append("")
        lines.append(
            "| Factor | Pillar | N | Spearman IC | Pearson IC | "
            "N bull / N bear | Mean ret bull | Mean ret bear | "
            "Long-short | Hit bull | Hit bear |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = []
        for factor, block in factor_summary.items():
            stats = (block.get("by_horizon") or {}).get(horizon) or {}
            ic = stats.get("spearman_ic")
            rows.append((factor, block.get("pillar") or "", stats, ic))
        rows.sort(
            key=lambda t: (abs(t[3]) if t[3] is not None else -1.0),
            reverse=True,
        )
        for factor, pillar, stats, ic in rows:
            lines.append(
                "| {f} | {p} | {n} | {ic} | {pi} | {nb}/{ne} | "
                "{mb} | {me} | {ls} | {hb} | {he} |".format(
                    f=factor,
                    p=pillar,
                    n=stats.get("n_paired") or 0,
                    ic=_fmt_signed(ic, digits=4),
                    pi=_fmt_signed(stats.get("pearson_ic"), digits=4),
                    nb=stats.get("n_bullish_score") or 0,
                    ne=stats.get("n_bearish_score") or 0,
                    mb=_pct(stats.get("mean_ret_when_bullish")),
                    me=_pct(stats.get("mean_ret_when_bearish")),
                    ls=_pct(stats.get("long_short_spread")),
                    hb=_pct(stats.get("hit_rate_when_bullish")),
                    he=_pct(stats.get("hit_rate_when_bearish")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt_signed(v: Any, *, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):+.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def render_factor_by_ticker_markdown(
    by_ticker_summary: dict[str, Any],
    return_fields: list[str],
    *,
    title: str = "Per-factor IC stratified by ticker",
    return_label: str = "raw",
) -> str:
    """Render the per-ticker stratification as one table per horizon."""
    lines = [f"# {title}", "", f"_Return type: **{return_label}**_", ""]
    lines.append(
        "Consistency = fraction of tickers whose per-ticker IC has the "
        "same sign as the median across tickers. 1.00 = unanimous, "
        "0.50 = pure noise. Only tickers with ≥ 8 paired observations "
        "are counted."
    )
    lines.append("")
    if not by_ticker_summary:
        return "\n".join(lines).rstrip() + "\n"
    for horizon in return_fields:
        lines.append(f"## Horizon `{horizon}`")
        lines.append("")
        lines.append(
            "| Factor | Pillar | N tickers | Median IC | Mean IC | Consistency |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        rows = []
        for factor, block in by_ticker_summary.items():
            stats = (block.get("by_horizon") or {}).get(horizon) or {}
            med = stats.get("median_ic_across_tickers")
            rows.append((factor, block.get("pillar") or "", stats, med))
        rows.sort(
            key=lambda t: (abs(t[3]) if t[3] is not None else -1.0),
            reverse=True,
        )
        for factor, pillar, stats, med in rows:
            consistency = stats.get("consistency_pct")
            lines.append(
                "| {f} | {p} | {nt} | {med} | {mean} | {c} |".format(
                    f=factor,
                    p=pillar,
                    nt=stats.get("n_tickers_evaluated") or 0,
                    med=_fmt_signed(med),
                    mean=_fmt_signed(stats.get("mean_ic_across_tickers")),
                    c=_pct(consistency),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------- counterfactual composite rebuild ----------


def rebuild_records_with_weights(
    records: list[BacktestRecord],
    *,
    weights: dict[str, float],
    direction_threshold: float = 0.2,
    neutral_band: float = 0.02,
    bullish_factor_threshold: float = 0.0,
    bullish_threshold: float | None = None,
    bearish_threshold: float | None = None,
) -> list[BacktestRecord]:
    """Return a copy of `records` with composite/direction recomputed.

    Each record must already have `factor_scores` populated (i.e. loaded
    with `capture_factor_scores=True`). For each row, recompute
    `composite_score` and `direction` using the provided weights, leaving
    `forward_returns` and `benchmark_adjusted_returns` untouched.

    `weights` may contain negative values to model sign-inverted factors.
    Renormalization uses absolute values so the composite stays in [-1, 1].

    `bullish_threshold` / `bearish_threshold` (Section 15) override
    `direction_threshold` on the respective side; `bearish_threshold` is
    an absolute magnitude (negated internally). When both override
    kwargs are None, behavior is identical to the prior symmetric form.
    """
    bt = bullish_threshold if bullish_threshold is not None else direction_threshold
    br = bearish_threshold if bearish_threshold is not None else direction_threshold
    abs_total = sum(abs(v) for v in weights.values()) or 1.0
    rebuilt: list[BacktestRecord] = []
    for r in records:
        scores = r.factor_scores or []
        if not scores:
            rebuilt.append(r)
            continue
        composite_raw = 0.0
        active_weight = 0.0
        for f in scores:
            if not f.get("data_available"):
                continue
            name = f.get("factor")
            if name not in weights:
                continue
            w_signed = float(weights[name])
            if w_signed == 0.0:
                continue
            score = f.get("score")
            if score is None:
                continue
            w_norm = w_signed / abs_total
            composite_raw += float(score) * w_norm
            active_weight += abs(w_signed) / abs_total
        if active_weight > 0:
            composite_raw = composite_raw / active_weight
        composite = max(-1.0, min(1.0, composite_raw))
        if composite >= bt:
            direction = "bullish"
        elif composite <= -br:
            direction = "bearish"
        else:
            direction = "neutral"
        new_rec = BacktestRecord(
            symbol=r.symbol,
            as_of_date=r.as_of_date,
            direction=direction,
            confidence=r.confidence,
            composite_score=round(composite, 4),
            forward_returns=dict(r.forward_returns),
            benchmark_adjusted_returns=dict(r.benchmark_adjusted_returns),
            factor_scores=r.factor_scores,
            market_context=r.market_context,
            llm_critic=r.llm_critic,
            llm_disagreement=r.llm_disagreement,
            source_path=r.source_path,
        )
        rebuilt.append(new_rec)
    return rebuilt


def sweep_direction_threshold(
    records: list[BacktestRecord],
    *,
    weights: dict[str, float] | None,
    thresholds: list[float],
    return_fields: list[str] | None = None,
    neutral_band: float = 0.02,
    use_benchmark_adjusted: bool = False,
) -> dict[str, Any]:
    """Sweep `direction_for_composite` threshold across candidate values.

    For each threshold, recompute each record's `direction` from its
    `composite_score` and aggregate direction-conditional hit-rate +
    mean return per horizon. Result lets you trade off precision
    (hit-rate per call) vs coverage (n calls) when picking a threshold.

    If `weights` is provided, composites are first rebuilt under those
    weights via `rebuild_records_with_weights` so the sweep evaluates
    the threshold *under* a proposed weight vector. Pass `weights=None`
    to sweep against the as-emitted composites from the report JSONs.

    The "best" threshold is intentionally not chosen here — different
    objectives (max bullish hit-rate, max long-short spread, min
    neutral coverage) are valid. Caller picks from the table.
    """
    return_fields = return_fields or ["ret_5d", "ret_20d", "ret_60d"]
    out: dict[str, Any] = {
        "n_records": len(records),
        "weights_supplied": weights is not None,
        "thresholds": list(thresholds),
        "neutral_band": neutral_band,
        "use_benchmark_adjusted": use_benchmark_adjusted,
        "by_threshold": {},
    }
    for thr in thresholds:
        rebuilt = (
            rebuild_records_with_weights(
                records, weights=weights, direction_threshold=thr,
                neutral_band=neutral_band,
            )
            if weights is not None
            else _reclassify_direction(records, direction_threshold=thr)
        )
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for r in rebuilt:
            counts[r.direction] = counts.get(r.direction, 0) + 1
        horizon_block: dict[str, Any] = {}
        for horizon in return_fields:
            by_dir: dict[str, dict[str, Any]] = {}
            for direction in ("bullish", "bearish", "neutral"):
                bucket = [r for r in rebuilt if r.direction == direction]
                by_dir[direction] = summarize_bucket(
                    bucket,
                    return_field=horizon,
                    neutral_band=neutral_band,
                )
            horizon_block[horizon] = {"by_direction": by_dir}
        out["by_threshold"][f"{thr:.3f}"] = {
            "counts": counts,
            "by_horizon": horizon_block,
        }
    return out


def _reclassify_direction(
    records: list[BacktestRecord],
    *,
    direction_threshold: float | None = None,
    bullish_threshold: float | None = None,
    bearish_threshold: float | None = None,
) -> list[BacktestRecord]:
    """Reclassify direction from the existing composite_score, no rebuild.

    Pass `direction_threshold` for symmetric, or `bullish_threshold` /
    `bearish_threshold` (absolute magnitude on bearish side) for
    asymmetric. At least one of the three must be provided.
    """
    if (
        direction_threshold is None
        and bullish_threshold is None
        and bearish_threshold is None
    ):
        raise ValueError(
            "_reclassify_direction requires direction_threshold or "
            "bullish_threshold/bearish_threshold"
        )
    bt = bullish_threshold if bullish_threshold is not None else direction_threshold
    br = bearish_threshold if bearish_threshold is not None else direction_threshold
    out: list[BacktestRecord] = []
    for r in records:
        score = r.composite_score
        if score is None:
            direction = "neutral"
        elif score >= bt:
            direction = "bullish"
        elif score <= -br:
            direction = "bearish"
        else:
            direction = "neutral"
        out.append(
            BacktestRecord(
                symbol=r.symbol,
                as_of_date=r.as_of_date,
                direction=direction,
                confidence=r.confidence,
                composite_score=r.composite_score,
                forward_returns=dict(r.forward_returns),
                benchmark_adjusted_returns=dict(r.benchmark_adjusted_returns),
                factor_scores=r.factor_scores,
                source_path=r.source_path,
            )
        )
    return out


def render_threshold_sweep_markdown(
    sweep: dict[str, Any],
    *,
    title: str = "Direction-threshold sweep",
    horizons_to_show: list[str] | None = None,
) -> str:
    """One table per horizon. Rows = thresholds. Direction-conditional
    hit-rate + count + mean return so the precision/coverage trade-off
    is legible at a glance."""
    horizons_to_show = horizons_to_show or list(
        next(iter(sweep.get("by_threshold", {}).values()), {})
        .get("by_horizon", {}).keys()
    )
    lines = [f"# {title}", ""]
    lines.append(f"- Records: **{sweep.get('n_records', 0)}**")
    lines.append(
        f"- Composites: {'rebuilt under provided weights' if sweep.get('weights_supplied') else 'as emitted in JSON (no rebuild)'}"
    )
    lines.append(
        f"- Neutral band (for neutral-bucket hit-rate only): "
        f"{_pct(sweep.get('neutral_band'))}"
    )
    lines.append(
        f"- Return type: "
        f"**{'benchmark-adjusted' if sweep.get('use_benchmark_adjusted') else 'raw'}**"
    )
    lines.append("")
    by_thr = sweep.get("by_threshold") or {}
    if not by_thr:
        return "\n".join(lines).rstrip() + "\n"

    for horizon in horizons_to_show:
        lines.append(f"## Horizon `{horizon}`")
        lines.append("")
        lines.append(
            "| Threshold | N bull | N bear | N neu | "
            "Bull hit | Bull mean | Bear hit | Bear mean | Neu hit |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for thr_key, block in by_thr.items():
            counts = block.get("counts") or {}
            by_dir = (block.get("by_horizon") or {}).get(horizon, {}).get("by_direction") or {}
            bull = by_dir.get("bullish") or {}
            bear = by_dir.get("bearish") or {}
            neu = by_dir.get("neutral") or {}
            lines.append(
                "| {t} | {nb} | {ne} | {nu} | {bh} | {bm} | {eh} | {em} | {uh} |".format(
                    t=thr_key,
                    nb=counts.get("bullish") or 0,
                    ne=counts.get("bearish") or 0,
                    nu=counts.get("neutral") or 0,
                    bh=_pct(bull.get("hit_rate")),
                    bm=_pct(bull.get("mean_forward_return")),
                    eh=_pct(bear.get("hit_rate")),
                    em=_pct(bear.get("mean_forward_return")),
                    uh=_pct(neu.get("hit_rate")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def recommend_direction_threshold(
    sweep: dict[str, Any],
    *,
    horizon: str = "ret_20d",
    min_n_bullish: int = 30,
    min_n_bearish: int = 5,
) -> dict[str, Any]:
    """Pick the threshold with the highest **bullish** hit-rate at `horizon`
    while keeping `n_bullish >= min_n_bullish`. Also surface the
    threshold with the highest **bearish** hit-rate subject to its own
    minimum. The two may differ — this surfaces both so the caller can
    inspect a trade-off rather than blindly pick one.
    """
    by_thr = sweep.get("by_threshold") or {}
    if not by_thr:
        return {"horizon": horizon, "bullish_pick": None, "bearish_pick": None}

    def _eval(direction: str, min_n: int):
        candidates = []
        for thr_key, block in by_thr.items():
            counts = block.get("counts") or {}
            n = counts.get(direction) or 0
            if n < min_n:
                continue
            by_dir = (block.get("by_horizon") or {}).get(horizon, {}).get("by_direction") or {}
            hit = (by_dir.get(direction) or {}).get("hit_rate")
            if hit is None:
                continue
            candidates.append({"threshold": float(thr_key), "n": n, "hit_rate": hit})
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c["hit_rate"], c["n"]))

    return {
        "horizon": horizon,
        "bullish_pick": _eval("bullish", min_n_bullish),
        "bearish_pick": _eval("bearish", min_n_bearish),
    }


def sweep_direction_threshold_asymmetric(
    records: list[BacktestRecord],
    *,
    weights: dict[str, float] | None,
    bullish_thresholds: list[float],
    bearish_thresholds: list[float],
    return_fields: list[str] | None = None,
    neutral_band: float = 0.02,
    use_benchmark_adjusted: bool = False,
) -> dict[str, Any]:
    """2-D sweep over (bullish_threshold, bearish_threshold) pairs.

    Returns the same JSON-serializable shape as `sweep_direction_threshold`
    but keyed by ``f"{bt:.3f}|{br:.3f}"``. `bearish_thresholds` are
    absolute magnitudes (negated internally — pass positive numbers).

    If `weights` is provided, composites are first rebuilt under those
    weights via `rebuild_records_with_weights` (per cell). Otherwise,
    direction is reclassified directly from each record's saved
    `composite_score` via `_reclassify_direction`.
    """
    return_fields = return_fields or ["ret_5d", "ret_20d", "ret_60d"]
    out: dict[str, Any] = {
        "n_records": len(records),
        "weights_supplied": weights is not None,
        "bullish_thresholds": list(bullish_thresholds),
        "bearish_thresholds": list(bearish_thresholds),
        "neutral_band": neutral_band,
        "use_benchmark_adjusted": use_benchmark_adjusted,
        "by_cell": {},
    }
    for bt in bullish_thresholds:
        for br in bearish_thresholds:
            if weights is not None:
                rebuilt = rebuild_records_with_weights(
                    records,
                    weights=weights,
                    bullish_threshold=bt,
                    bearish_threshold=br,
                    neutral_band=neutral_band,
                )
            else:
                rebuilt = _reclassify_direction(
                    records,
                    bullish_threshold=bt,
                    bearish_threshold=br,
                )
            counts = {"bullish": 0, "bearish": 0, "neutral": 0}
            for r in rebuilt:
                counts[r.direction] = counts.get(r.direction, 0) + 1
            horizon_block: dict[str, Any] = {}
            for horizon in return_fields:
                by_dir: dict[str, dict[str, Any]] = {}
                for direction in ("bullish", "bearish", "neutral"):
                    bucket = [r for r in rebuilt if r.direction == direction]
                    by_dir[direction] = summarize_bucket(
                        bucket,
                        return_field=horizon,
                        neutral_band=neutral_band,
                    )
                horizon_block[horizon] = {"by_direction": by_dir}
            out["by_cell"][f"{bt:.3f}|{br:.3f}"] = {
                "bullish_threshold": float(bt),
                "bearish_threshold": float(br),
                "counts": counts,
                "by_horizon": horizon_block,
            }
    return out


def render_asymmetric_sweep_markdown(
    sweep: dict[str, Any],
    *,
    title: str = "Asymmetric direction-threshold sweep",
    horizons_to_show: list[str] | None = None,
) -> str:
    """One grid per horizon. Rows = bullish_threshold, cols = bearish_threshold.

    Each cell shows a compact ``bull_hit / bear_hit / nb / ne`` summary so
    you can read off both axes at once. The bullish hit-rate varies only
    across rows (the bearish threshold cannot change which records were
    classified bullish), and vice versa — the grid surfaces the
    trade-off explicitly.
    """
    by_cell = sweep.get("by_cell") or {}
    if not by_cell:
        return f"# {title}\n\n_No cells._\n"
    horizons_to_show = horizons_to_show or list(
        next(iter(by_cell.values()), {}).get("by_horizon", {}).keys()
    )
    bull_thrs = sorted({block["bullish_threshold"] for block in by_cell.values()})
    bear_thrs = sorted({block["bearish_threshold"] for block in by_cell.values()})
    lines = [f"# {title}", ""]
    lines.append(f"- Records: **{sweep.get('n_records', 0)}**")
    lines.append(
        f"- Composites: "
        f"{'rebuilt under provided weights' if sweep.get('weights_supplied') else 'as emitted in JSON (no rebuild)'}"
    )
    lines.append(
        f"- Neutral band: {_pct(sweep.get('neutral_band'))}"
    )
    lines.append(
        f"- Cell format: `bull_hit% / bear_hit% / nb / ne`"
    )
    lines.append("")

    for horizon in horizons_to_show:
        lines.append(f"## Horizon `{horizon}`")
        lines.append("")
        header_cols = " | ".join(f"bear={br:.3f}" for br in bear_thrs)
        lines.append(f"| bull \\\\ bear | {header_cols} |")
        lines.append("|---:|" + "---:|" * len(bear_thrs))
        for bt in bull_thrs:
            row = [f"**{bt:.3f}**"]
            for br in bear_thrs:
                key = f"{bt:.3f}|{br:.3f}"
                block = by_cell.get(key) or {}
                counts = block.get("counts") or {}
                by_dir = (
                    (block.get("by_horizon") or {}).get(horizon, {}).get("by_direction") or {}
                )
                bull = by_dir.get("bullish") or {}
                bear = by_dir.get("bearish") or {}
                row.append(
                    f"{_pct(bull.get('hit_rate'))} / "
                    f"{_pct(bear.get('hit_rate'))} / "
                    f"{counts.get('bullish') or 0} / "
                    f"{counts.get('bearish') or 0}"
                )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def recommend_asymmetric_thresholds(
    sweep: dict[str, Any],
    *,
    horizon: str = "ret_20d",
    min_n_bullish: int = 30,
    min_n_bearish: int = 5,
    bearish_precision_floor: float = 0.50,
) -> dict[str, Any]:
    """Pick the best (bullish_threshold, bearish_threshold) pair under
    asymmetric tuning rules.

    Bullish pick: max bullish hit-rate at `horizon` s.t.
    `n_bullish >= min_n_bullish`. Because bullish counts/hits depend
    only on `bullish_threshold`, the bullish dimension is effectively
    1-D — we collapse over `bearish_threshold` and return one bullish
    threshold value with its bullish stats.

    Bearish pick: max bearish hit-rate at `horizon` s.t.
    `n_bearish >= min_n_bearish` AND `hit_rate >= bearish_precision_floor`.
    Returns `None` (explicitly) when no cell qualifies — that is the
    headline expected outcome on the current corpus, and it signals
    that the bearish side cannot be fixed by threshold alone.
    """
    by_cell = sweep.get("by_cell") or {}
    if not by_cell:
        return {
            "horizon": horizon,
            "bullish_pick": None,
            "bearish_pick": None,
            "bearish_precision_floor": bearish_precision_floor,
        }

    bullish_by_bt: dict[float, dict[str, Any]] = {}
    for block in by_cell.values():
        bt = float(block["bullish_threshold"])
        if bt in bullish_by_bt:
            continue
        counts = block.get("counts") or {}
        n = counts.get("bullish") or 0
        if n < min_n_bullish:
            continue
        by_dir = (block.get("by_horizon") or {}).get(horizon, {}).get("by_direction") or {}
        hit = (by_dir.get("bullish") or {}).get("hit_rate")
        if hit is None:
            continue
        bullish_by_bt[bt] = {"threshold": bt, "n": n, "hit_rate": hit}
    bullish_pick = (
        max(bullish_by_bt.values(), key=lambda c: (c["hit_rate"], c["n"]))
        if bullish_by_bt
        else None
    )

    bearish_by_br: dict[float, dict[str, Any]] = {}
    for block in by_cell.values():
        br = float(block["bearish_threshold"])
        if br in bearish_by_br:
            continue
        counts = block.get("counts") or {}
        n = counts.get("bearish") or 0
        if n < min_n_bearish:
            continue
        by_dir = (block.get("by_horizon") or {}).get(horizon, {}).get("by_direction") or {}
        hit = (by_dir.get("bearish") or {}).get("hit_rate")
        if hit is None or hit < bearish_precision_floor:
            continue
        bearish_by_br[br] = {"threshold": br, "n": n, "hit_rate": hit}
    bearish_pick = (
        max(bearish_by_br.values(), key=lambda c: (c["hit_rate"], c["n"]))
        if bearish_by_br
        else None
    )

    return {
        "horizon": horizon,
        "bullish_pick": bullish_pick,
        "bearish_pick": bearish_pick,
        "bearish_precision_floor": bearish_precision_floor,
    }


def ic_signed_weights(
    factor_summary: dict[str, Any],
    *,
    horizon: str = "ret_20d",
    min_abs_ic: float = 0.05,
    min_n: int = 50,
) -> dict[str, float]:
    """Build a weight vector from a per-factor IC summary.

    Factors with |IC| >= `min_abs_ic` and `n_paired >= min_n` get
    `weight = IC` (signed, NOT normalized — caller can pass to
    `rebuild_records_with_weights` which handles abs-normalization).
    Other factors get weight 0 (dropped).
    """
    out: dict[str, float] = {}
    for factor, block in factor_summary.items():
        stats = (block.get("by_horizon") or {}).get(horizon) or {}
        ic = stats.get("spearman_ic")
        n = stats.get("n_paired") or 0
        if ic is None or abs(ic) < min_abs_ic or n < min_n:
            continue
        out[factor] = float(ic)
    return out


# ---------- walk-forward backtest (Phase 3) ----------


def _add_days(iso_date: str, days: int) -> str:
    from datetime import datetime, timedelta
    return (
        datetime.strptime(iso_date, "%Y-%m-%d").date()
        + timedelta(days=days)
    ).isoformat()


def walk_forward_refit_dates(
    records: Iterable[BacktestRecord],
    *,
    refit_freq_weeks: int = 4,
    first_refit_after_weeks: int = 26,
) -> list[str]:
    """Pick refit anchor dates (ISO strings) at fixed weekly intervals.

    Anchor is the SECOND record after at least `first_refit_after_weeks`
    have elapsed since the corpus start. Subsequent anchors step
    `refit_freq_weeks` apart. Returns sorted distinct anchor dates that
    actually appear in the corpus, so the caller can iterate them.

    Date-only arithmetic (no trading-day calendar) — adequate at weekly
    cadence and avoids a yfinance dependency in pure-logic code.
    """
    dates = sorted({r.as_of_date for r in records})
    if not dates:
        return []
    start = dates[0]
    first_anchor = _add_days(start, first_refit_after_weeks * 7)
    candidates = [d for d in dates if d >= first_anchor]
    if not candidates:
        return []
    anchors: list[str] = [candidates[0]]
    step_days = refit_freq_weeks * 7
    next_target = _add_days(anchors[0], step_days)
    for d in candidates[1:]:
        if d >= next_target:
            anchors.append(d)
            next_target = _add_days(d, step_days)
    return anchors


def split_train_score(
    records: Iterable[BacktestRecord],
    *,
    anchor_date: str,
    train_window_weeks: int,
    score_window_weeks: int,
    gap_weeks: int,
) -> tuple[list[BacktestRecord], list[BacktestRecord]]:
    """Walk-forward window split.

    Returns (train_records, score_records) for one refit step:
    - train: records with `as_of_date < anchor - gap_weeks` AND
      `as_of_date >= anchor - gap_weeks - train_window_weeks`
    - score: records with `anchor <= as_of_date < anchor + score_window_weeks`

    The gap is critical: it must cover the longest forward-return horizon
    plus settlement so the trainer never sees a label that overlaps the
    scoring window.
    """
    train_end_exclusive = _add_days(anchor_date, -gap_weeks * 7)
    train_start_inclusive = _add_days(
        train_end_exclusive, -train_window_weeks * 7
    )
    score_end_exclusive = _add_days(anchor_date, score_window_weeks * 7)
    train: list[BacktestRecord] = []
    score: list[BacktestRecord] = []
    for r in records:
        d = r.as_of_date
        if train_start_inclusive <= d < train_end_exclusive:
            train.append(r)
        elif anchor_date <= d < score_end_exclusive:
            score.append(r)
    return train, score


def fit_weights_from_records(
    train_records: list[BacktestRecord],
    *,
    horizon: str = "ret_20d",
    min_abs_ic: float = 0.05,
    min_n: int = 50,
) -> dict[str, float]:
    """Fit walk-forward factor weights from a training-window slice.

    Wraps the existing per-factor IC pipeline so walk-forward fits use
    the same logic as the global `--rebuild-with-ic-weights` flow.
    Returns a {factor -> signed weight} dict that can be fed directly
    into `rebuild_records_with_weights`.
    """
    by_factor = explode_records_to_factors(train_records)
    if not by_factor:
        return {}
    summary = summarize_factors(by_factor, return_fields=[horizon])
    return ic_signed_weights(
        summary,
        horizon=horizon,
        min_abs_ic=min_abs_ic,
        min_n=min_n,
    )


@dataclass
class WalkForwardStep:
    anchor_date: str
    train_start: str
    train_end_exclusive: str
    score_end_exclusive: str
    n_train: int
    n_score: int
    weights: dict[str, float]


def walk_forward_backtest(
    records: list[BacktestRecord],
    *,
    refit_freq_weeks: int = 4,
    train_window_weeks: int = 52,
    gap_weeks: int = 4,
    first_refit_after_weeks: int = 26,
    horizon: str = "ret_20d",
    min_abs_ic: float = 0.05,
    min_n: int = 50,
) -> tuple[list[BacktestRecord], list[WalkForwardStep]]:
    """Walk-forward backtest with rolling refit.

    At each refit anchor, fit IC weights on `train_window_weeks` of data
    ending `gap_weeks` before the anchor, then rebuild composites for
    every record in `[anchor, anchor + refit_freq_weeks)` using those
    frozen weights. Returns:
    - rebuilt_records: the concatenated out-of-sample scored slices (one
      record per source `as_of_date` falling in any scoring window).
    - timeline: per-step diagnostics with the weights that were live.

    Records outside any scoring window (before the first anchor, or
    inside the gap) are dropped. The caller can summarize the rebuilt
    records the same way as the in-sample backtest — every metric is
    now out-of-sample.

    `horizon` selects which forward-return column drives weight fitting.
    `min_abs_ic` / `min_n` are passed straight to `ic_signed_weights`.
    Same gap is reused for scoring window length (refit_freq_weeks).
    """
    anchors = walk_forward_refit_dates(
        records,
        refit_freq_weeks=refit_freq_weeks,
        first_refit_after_weeks=first_refit_after_weeks,
    )
    timeline: list[WalkForwardStep] = []
    rebuilt: list[BacktestRecord] = []
    for anchor in anchors:
        train, score = split_train_score(
            records,
            anchor_date=anchor,
            train_window_weeks=train_window_weeks,
            score_window_weeks=refit_freq_weeks,
            gap_weeks=gap_weeks,
        )
        weights = fit_weights_from_records(
            train,
            horizon=horizon,
            min_abs_ic=min_abs_ic,
            min_n=min_n,
        )
        train_end = _add_days(anchor, -gap_weeks * 7)
        step = WalkForwardStep(
            anchor_date=anchor,
            train_start=_add_days(train_end, -train_window_weeks * 7),
            train_end_exclusive=train_end,
            score_end_exclusive=_add_days(anchor, refit_freq_weeks * 7),
            n_train=len(train),
            n_score=len(score),
            weights=weights,
        )
        timeline.append(step)
        if not score or not weights:
            continue
        rebuilt_slice = rebuild_records_with_weights(
            score, weights=weights
        )
        rebuilt.extend(rebuilt_slice)
    return rebuilt, timeline


def walk_forward_timeline_to_dict(
    timeline: list[WalkForwardStep],
) -> list[dict[str, Any]]:
    """Serialize the per-step diagnostics for walk_forward_weights_timeline.json."""
    return [
        {
            "anchor_date": s.anchor_date,
            "train_start": s.train_start,
            "train_end_exclusive": s.train_end_exclusive,
            "score_end_exclusive": s.score_end_exclusive,
            "n_train": s.n_train,
            "n_score": s.n_score,
            "weights": s.weights,
        }
        for s in timeline
    ]


# ---------- Phase 4 regime-conditional walk-forward ----------


@dataclass
class RegimeWalkForwardStep:
    anchor_date: str
    train_start: str
    train_end_exclusive: str
    score_end_exclusive: str
    n_train_total: int
    n_train_by_regime: dict[str, int]
    n_score_total: int
    n_score_by_regime: dict[str, int]
    global_weights: dict[str, float]
    global_ic: float | None
    regime_weights: dict[str, dict[str, float]]
    regime_ics: dict[str, float | None]
    regime_ic_lifts: dict[str, float | None]
    regime_skip_reasons: dict[str, str]
    regimes_used: list[str]
    regimes_fellback_to_global: list[str]


def _record_regime(rec: BacktestRecord) -> str:
    from tradingagents.analysis_only.scoring import (
        REGIME_UNKNOWN,
        regime_for_market_context,
    )

    if rec.market_context is None:
        return REGIME_UNKNOWN
    return regime_for_market_context(rec.market_context)


def _composite_ic(records: list[BacktestRecord], horizon: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for r in records:
        if r.composite_score is None:
            continue
        ret = r.forward_returns.get(horizon)
        if ret is None:
            continue
        pairs.append((float(r.composite_score), float(ret)))
    if len(pairs) < 30:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return spearman_correlation(xs, ys)


def regime_walk_forward_backtest(
    records: list[BacktestRecord],
    *,
    refit_freq_weeks: int = 4,
    train_window_weeks: int = 52,
    gap_weeks: int = 4,
    first_refit_after_weeks: int = 26,
    horizon: str = "ret_20d",
    min_abs_ic: float = 0.05,
    min_n: int = 50,
    min_samples_per_regime: int = 250,
    min_regime_ic_lift: float = 0.02,
    eligible_regimes: Sequence[str] | None = None,
    require_regime_ic_ge_global: bool = True,
) -> tuple[list[BacktestRecord], list[RegimeWalkForwardStep]]:
    """Phase 4: walk-forward with per-regime weight fits.

    At each refit:
    1. Fit `global_weights` on the full training window (always present).
    2. For each eligible regime that has >= `min_samples_per_regime`
       training samples, fit `regime_weights[regime]` on the regime-only
       slice and ship it only when rebuilt regime IC beats global IC by
       `min_regime_ic_lift`.
    3. At scoring time, look up each record's regime and apply the
       matching regime weights if available; otherwise fall back to
       global weights (logged in `regimes_fellback_to_global`).

    Requires each record to carry a `market_context` dict (loaded with
    `capture_market_context=True` in `backtest.load_records`). Records
    without it get `REGIME_UNKNOWN` and use global weights.
    """
    from tradingagents.analysis_only.scoring import (
        REGIME_CHOP,
        REGIME_TREND_ON,
        REGIME_UNKNOWN,
    )

    eligible = set(eligible_regimes or (REGIME_CHOP, REGIME_TREND_ON))
    anchors = walk_forward_refit_dates(
        records,
        refit_freq_weeks=refit_freq_weeks,
        first_refit_after_weeks=first_refit_after_weeks,
    )
    timeline: list[RegimeWalkForwardStep] = []
    rebuilt: list[BacktestRecord] = []
    for anchor in anchors:
        train, score = split_train_score(
            records,
            anchor_date=anchor,
            train_window_weeks=train_window_weeks,
            score_window_weeks=refit_freq_weeks,
            gap_weeks=gap_weeks,
        )
        global_w = fit_weights_from_records(
            train,
            horizon=horizon,
            min_abs_ic=min_abs_ic,
            min_n=min_n,
        )
        global_ic = (
            _composite_ic(rebuild_records_with_weights(train, weights=global_w), horizon)
            if global_w
            else None
        )
        train_by_regime: dict[str, list[BacktestRecord]] = {}
        for r in train:
            train_by_regime.setdefault(_record_regime(r), []).append(r)
        score_by_regime: dict[str, list[BacktestRecord]] = {}
        for r in score:
            score_by_regime.setdefault(_record_regime(r), []).append(r)
        regime_weights: dict[str, dict[str, float]] = {}
        regime_ics: dict[str, float | None] = {}
        regime_ic_lifts: dict[str, float | None] = {}
        regime_skip_reasons: dict[str, str] = {}
        regimes_used: list[str] = []
        regimes_fellback: list[str] = []
        for regime, regime_train in train_by_regime.items():
            if regime == REGIME_UNKNOWN:
                continue
            if regime not in eligible:
                regimes_fellback.append(regime)
                regime_skip_reasons[regime] = "regime_not_eligible"
                continue
            if len(regime_train) < min_samples_per_regime:
                regimes_fellback.append(regime)
                regime_skip_reasons[regime] = (
                    f"n_train<{min_samples_per_regime}"
                )
                continue
            w = fit_weights_from_records(
                regime_train,
                horizon=horizon,
                min_abs_ic=min_abs_ic,
                min_n=min_n,
            )
            if not w:
                regimes_fellback.append(regime)
                regime_skip_reasons[regime] = "no_factors_passed_ic_threshold"
                continue
            rebuilt_train = rebuild_records_with_weights(regime_train, weights=w)
            regime_ic = _composite_ic(rebuilt_train, horizon)
            regime_ics[regime] = regime_ic
            if global_ic is None or regime_ic is None:
                lift = None
            else:
                lift = regime_ic - global_ic
            regime_ic_lifts[regime] = lift
            if require_regime_ic_ge_global:
                if global_ic is None:
                    regimes_fellback.append(regime)
                    regime_skip_reasons[regime] = "global_ic_unavailable"
                    continue
                if regime_ic is None:
                    regimes_fellback.append(regime)
                    regime_skip_reasons[regime] = "regime_ic_unavailable"
                    continue
                if lift is not None and lift < min_regime_ic_lift:
                    regimes_fellback.append(regime)
                    regime_skip_reasons[regime] = (
                        f"regime_ic_lift {lift:.4f} < required "
                        f"{min_regime_ic_lift:.4f}"
                    )
                    continue
            regime_weights[regime] = w
            regimes_used.append(regime)
        train_end = _add_days(anchor, -gap_weeks * 7)
        step = RegimeWalkForwardStep(
            anchor_date=anchor,
            train_start=_add_days(train_end, -train_window_weeks * 7),
            train_end_exclusive=train_end,
            score_end_exclusive=_add_days(anchor, refit_freq_weeks * 7),
            n_train_total=len(train),
            n_train_by_regime={k: len(v) for k, v in train_by_regime.items()},
            n_score_total=len(score),
            n_score_by_regime={k: len(v) for k, v in score_by_regime.items()},
            global_weights=global_w,
            global_ic=global_ic,
            regime_weights=regime_weights,
            regime_ics=regime_ics,
            regime_ic_lifts=regime_ic_lifts,
            regime_skip_reasons=regime_skip_reasons,
            regimes_used=sorted(regimes_used),
            regimes_fellback_to_global=sorted(regimes_fellback),
        )
        timeline.append(step)
        if not score:
            continue
        for regime, slice_recs in score_by_regime.items():
            weights = regime_weights.get(regime, global_w)
            if not weights:
                continue
            rebuilt_slice = rebuild_records_with_weights(
                slice_recs, weights=weights
            )
            rebuilt.extend(rebuilt_slice)
    return rebuilt, timeline


def regime_walk_forward_timeline_to_dict(
    timeline: list[RegimeWalkForwardStep],
) -> list[dict[str, Any]]:
    return [
        {
            "anchor_date": s.anchor_date,
            "train_start": s.train_start,
            "train_end_exclusive": s.train_end_exclusive,
            "score_end_exclusive": s.score_end_exclusive,
            "n_train_total": s.n_train_total,
            "n_train_by_regime": s.n_train_by_regime,
            "n_score_total": s.n_score_total,
            "n_score_by_regime": s.n_score_by_regime,
            "global_weights": s.global_weights,
            "global_ic": s.global_ic,
            "regime_weights": s.regime_weights,
            "regime_ics": s.regime_ics,
            "regime_ic_lifts": s.regime_ic_lifts,
            "regime_skip_reasons": s.regime_skip_reasons,
            "regimes_used": s.regimes_used,
            "regimes_fellback_to_global": s.regimes_fellback_to_global,
        }
        for s in timeline
    ]
