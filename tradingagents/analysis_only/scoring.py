"""Pure scoring helpers for the analysis-only pipeline.

All functions here are deterministic and free of I/O so they can be unit
tested without touching market data providers. The pipeline orchestrator
holds the per-factor evidence and rationale; this module owns the
weight resolution, composite arithmetic, direction thresholds, confidence
mapping, per-factor bucket label, and cross-sectional normalization.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


# Default factor weights v1 (handoff Section 12).
#
# Renormalized to sum to 1 at the end of `resolve_factor_weights`.
# v1 was set after the Tier-1 backtest in Section 11 + per-ticker IC
# stratification:
#   - dropped weight on factors with broken or sparse data
#     (`filings_recency_signal` n=0; `intraday_breakout_signal` n=10)
#   - cut `options_net_flow` from 0.15 → 0.05: was the heaviest single
#     factor but headline IC ≈ 0 / 60d ≈ -0.02; kept some weight to
#     preserve the signal for re-eval after the Polygon Options decision
#   - cut undersampled valuation factors (`valuation_forward_vs_trailing_pe`
#     n_paired=30; `valuation_sales_multiple_vs_growth` rarely fires)
#   - bumped `market_fear_greed_regime` (IC +0.32, 91% per-ticker
#     consistency across 11 tickers at 20d — the best validated signal)
#   - bumped `industry_relative_strength` and `peer_relative_valuation`
#     (both validated cross-ticker at 20d).
#
# Lift on the 306-record corpus, vs v0 weights, at 60d:
#   bullish hit-rate +2.10pp; bearish hit-rate +4.44pp; composite
#   [+0.5,+1.0) bucket mean return +5.40pp.
# 20d horizon is mixed (small regressions in the top composite bucket).
# Sign-inversion of the four contrarian factors (trend SMA crossovers,
# momentum_return_20d, market_vix_regime) is deferred until we have
# data from a non-bullish regime to confirm the inversion persists.
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "trend_price_vs_sma20": 0.08,
    "trend_sma20_vs_sma50": 0.08,
    "trend_sma50_vs_sma200": 0.08,
    "momentum_rsi": 0.06,
    "momentum_macd_hist": 0.05,
    "momentum_return_20d": 0.05,
    "breakout_60d": 0.05,
    "fund_revenue_growth": 0.08,
    "fund_earnings_growth": 0.08,
    "fund_profit_margins": 0.08,
    "fund_fcf_growth": 0.08,
    "valuation_forward_vs_trailing_pe": 0.04,
    "valuation_sales_multiple_vs_growth": 0.03,
    "industry_relative_strength": 0.08,
    "peer_relative_momentum": 0.05,
    "peer_relative_valuation": 0.06,
    "market_spy_trend": 0.04,
    "market_vix_regime": 0.03,
    # v1.4 (Section 24): restored to 0.05 after Phase 2 cohort IC analysis
    # confirmed the inversion is real and tech-specific. 24-tech core IC
    # was -0.36 at 60d with 70% sign consistency (23 evaluable tickers).
    # Sign of `score_fear_greed_regime` was flipped at the same time
    # (fear → bearish, greed → bullish — momentum-following for tech).
    # Weight 0.05 is conservative — lower than the v1's 0.08 since the
    # signal magnitude is also down from the Section 12 single-regime fit.
    # Inverts on canaries (+0.18 IC) — usable only because the trading
    # universe is tech.
    "market_fear_greed_regime": 0.05,
    "intraday_momentum_rsi": 0.04,
    "intraday_breakout_signal": 0.00,
    "filings_recency_signal": 0.00,
    "options_net_flow": 0.05,
    # IV-derived factors (added in Section 18). Two of three stay at 0:
    # `options_iv_skew` had ~zero IC across horizons; `options_iv_term_structure`
    # showed regime-dependent sign (+0.05 at 20d, -0.12 at 60d with 82%
    # per-ticker consistency on the 60d inversion) — too horizon-dependent
    # to commit either direction without more data.
    "options_iv_term_structure": 0.00,
    "options_iv_skew": 0.00,
    # v1.3 (Section 21): promoted from 0.00 to 0.04 after Phase 1 IC
    # validated the placeholder sign: low IV rank → bullish forward
    # returns. Headline IC +0.154 at 20d (n=597); per-ticker median
    # +0.107, 63.6% sign consistency across 11 tickers. Magnitude
    # comparable to peer_relative_momentum (weight 0.05).
    "options_iv_rank": 0.04,
}


def resolve_factor_weights(
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Merge user overrides on top of defaults and renormalize to sum=1."""
    weights = dict(DEFAULT_FACTOR_WEIGHTS)
    for key, value in (overrides or {}).items():
        if key in weights and value is not None and value >= 0:
            weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {key: round(value / total, 6) for key, value in weights.items()}


def bucket_for_score(score: float | None) -> str:
    """Return 'bullish' / 'bearish' / 'neutral' for a raw factor score."""
    if score is None:
        return "neutral"
    if score > 0:
        return "bullish"
    if score < 0:
        return "bearish"
    return "neutral"


def score_fear_greed_regime(
    score: float | None,
    rating: str | None = None,
) -> tuple[float, str, bool]:
    """Momentum-following market sentiment from CNN Fear & Greed.

    v1.4 (Section 24): inverted from the original contrarian mapping after
    Phase 2 cohort IC analysis showed -0.36 IC on the 24-ticker tech cohort
    at 60d (70% sign consistency across 23 evaluable tickers). For the tech /
    AI / semi universe the model trades, extreme fear coincides with
    breakdowns and extreme greed with continuation, not reversals. Note this
    finding inverts on cross-sector canaries (+0.18 IC at 60d) — the
    contrarian story still holds for defensives. The signed-for-tech mapping
    here is intentional given the trading universe.
    """
    if score is None:
        return 0.0, "Fear & Greed index unavailable.", False
    rating_text = (rating or "").strip().lower()
    if score <= 25 or rating_text == "extreme fear":
        return -0.4, "Extreme fear coincides with tech-sector breakdowns.", True
    if score < 45 or rating_text == "fear":
        return -0.2, "Fearful sentiment is a tech-momentum drag.", True
    if score >= 75 or rating_text == "extreme greed":
        return 0.4, "Extreme greed supports tech-momentum continuation.", True
    if score > 55 or rating_text == "greed":
        return 0.2, "Greedy sentiment supports tech upside.", True
    return 0.0, "Fear & Greed sentiment is neutral.", True


def score_sales_multiple_vs_growth(
    *,
    price_to_sales: float | None,
    ev_to_revenue: float | None = None,
    revenue_growth: float | None = None,
    peer_ev_to_revenue_median: float | None = None,
) -> tuple[float, str, bool, float | None]:
    """Score whether sales multiples look justified by growth.

    For high-growth AI/semi names, P/E can be noisy. This factor compares
    P/S or EV/revenue against revenue growth and, when available, peer
    EV/revenue. Lower growth-adjusted sales multiples score better.
    """
    multiple = _first_finite(ev_to_revenue, price_to_sales)
    if multiple is None or revenue_growth is None:
        return 0.0, "Sales multiple/growth comparison unavailable.", False, None
    if multiple <= 0:
        return 0.0, "Sales multiple is non-positive/unusable.", False, None

    growth = max(-0.5, min(2.0, float(revenue_growth)))
    growth_floor = max(0.05, growth)
    growth_adjusted_multiple = float(multiple) / (growth_floor * 100.0)
    score = 0.0
    if growth >= 0.30 and growth_adjusted_multiple <= 0.8:
        score = 0.8
        rationale = "Sales multiple appears reasonable versus high revenue growth."
    elif growth >= 0.15 and growth_adjusted_multiple <= 1.2:
        score = 0.4
        rationale = "Sales multiple is moderately supported by revenue growth."
    elif growth < 0.05 and float(multiple) > 8:
        score = -0.8
        rationale = "High sales multiple is weakly supported by low revenue growth."
    elif growth_adjusted_multiple > 1.8:
        score = -0.6
        rationale = "Sales multiple looks stretched versus revenue growth."
    else:
        rationale = "Sales multiple is broadly in line with revenue growth."

    if peer_ev_to_revenue_median is not None and peer_ev_to_revenue_median > 0:
        peer_gap = (float(multiple) / float(peer_ev_to_revenue_median)) - 1.0
        if peer_gap > 0.35:
            score -= 0.25
            rationale += " It also trades above peer EV/revenue."
        elif peer_gap < -0.25:
            score += 0.25
            rationale += " It also trades below peer EV/revenue."

    return (
        round(max(-1.0, min(1.0, score)), 4),
        rationale,
        True,
        round(growth_adjusted_multiple, 6),
    )


def score_iv_term_structure(
    slope: float | None,
) -> tuple[float, str, bool]:
    """Score the 30→60d ATM IV term-structure slope.

    Contango (positive slope) is the normal calm-regime shape. Backwardation
    (front IV > back IV) signals stress and historically aligns with risk-off
    moves. Returned score is placeholder-signed (positive when contango,
    negative when backwardated) at low magnitude; IC analysis on a regenerated
    corpus should confirm or invert the sign.
    """
    if slope is None:
        return 0.0, "IV term-structure slope unavailable.", False
    if slope > 0.05:
        return 0.5, "Steep IV contango — calm vol regime.", True
    if slope > 0.02:
        return 0.25, "Mild IV contango.", True
    if slope < -0.05:
        return -1.0, "Deep IV backwardation — stress signal.", True
    if slope < -0.02:
        return -0.5, "Mild IV backwardation.", True
    return 0.0, "IV term structure roughly flat.", True


def score_iv_skew(
    skew: float | None,
) -> tuple[float, str, bool]:
    """Score the 25Δ put−call IV skew at ~30d.

    Stocks normally show small positive skew (puts a few vol points richer
    than calls). Extreme positive skew = elevated tail-risk premium (mildly
    bearish, but can be contrarian-bullish at panic extremes; placeholder
    starts conservative bearish). Strong negative skew (calls > puts) is
    common in NVDA-style upside-chase regimes — placeholder treats this as
    mildly contrarian-bearish.
    """
    if skew is None:
        return 0.0, "25Δ IV skew unavailable.", False
    if skew > 0.15:
        return -0.5, "Extreme put-side IV skew — elevated tail-risk premium.", True
    if skew < -0.05:
        return -0.25, "Strong call-side IV skew — upside-chase regime.", True
    return 0.0, "IV skew in normal range.", True


def score_iv_rank(
    iv_rank: float | None,
) -> tuple[float, str, bool]:
    """Score IV rank (current ATM IV vs trailing 1Y min/max).

    High IV rank indicates elevated uncertainty premium. Low IV rank suggests
    complacency (which can precede regime expansion). Placeholder treats high
    rank as mildly bearish and low rank as mildly bullish; IC will validate.
    """
    if iv_rank is None:
        return 0.0, "IV rank unavailable (insufficient history).", False
    if iv_rank >= 0.80:
        return -0.5, "IV rank in top quintile — elevated uncertainty.", True
    if iv_rank < 0.20:
        return 0.5, "IV rank in bottom quintile — compressed vol regime.", True
    if iv_rank < 0.40:
        return 0.25, "IV rank below average.", True
    return 0.0, "IV rank in mid range.", True


def direction_for_composite(
    composite_score: float,
    threshold: float = 0.15,
    *,
    bullish_threshold: float | None = None,
    bearish_threshold: float | None = None,
) -> str:
    """Bucket a composite score into a directional call.

    Default symmetric threshold 0.15 chosen in Section 13 calibration
    sweep (re-verified by the systematic sweep in Section 14).

    For asymmetric calibration (Section 15), callers can pass
    `bullish_threshold` and/or `bearish_threshold` explicitly to use
    different cutoffs on each side. `bearish_threshold` is an absolute
    magnitude (negated internally). When omitted, each side falls back
    to the scalar `threshold`. This keeps the signature backward
    compatible with all existing scalar callers (pipeline.py, tests)
    while enabling per-direction tuning from the backtest harness.
    """
    bt = bullish_threshold if bullish_threshold is not None else threshold
    br = bearish_threshold if bearish_threshold is not None else threshold
    if composite_score >= bt:
        return "bullish"
    if composite_score <= -br:
        return "bearish"
    return "neutral"


def confidence_for(
    composite_score: float,
    coverage: float,
    *,
    floor: float = 0.5,
    slope: float = 0.45,
    cap: float = 0.95,
    calibration: dict[str, Any] | None = None,
) -> float:
    """Map composite + coverage to a calibrated confidence in [floor, cap].

    `composite_score` is expected to be in [-1, 1]; `coverage` in [0, 1].
    With defaults: confidence = clip(0.5 + |composite| * 0.45 * coverage, _, 0.95).

    Phase 5: when `calibration` is supplied (the loaded contents of
    `configs/confidence_calibration.json`), look up the empirical hit
    rate for this composite via `apply_isotonic_calibration` and use
    that instead of the heuristic formula. Coverage is still mixed in
    multiplicatively as a sanity weight: low coverage should pull the
    confidence toward 0.5 even when the calibrated hit-rate is high.
    """
    coverage = max(0.0, min(1.0, coverage))
    if calibration:
        calibrated = apply_isotonic_calibration(
            composite_score, calibration=calibration,
        )
        if calibrated is not None:
            adjusted = 0.5 + (calibrated - 0.5) * coverage
            return round(max(floor, min(cap, adjusted)), 2)
    raw = floor + abs(composite_score) * slope * coverage
    return round(min(cap, raw), 2)


# ---------- Phase 5 isotonic confidence calibration ----------


def fit_isotonic_calibration(
    composite_scores: Sequence[float],
    realized_hits: Sequence[int | bool],
    *,
    min_obs: int = 30,
) -> dict[str, Any]:
    """Fit a monotone non-decreasing isotonic regression of composite -> hit rate.

    Uses the pool-adjacent-violators (PAV) algorithm. The plan calls for
    bucketing by decile, but PAV is the standard implementation that
    handles ties and irregular distributions without arbitrary bucket
    choices. Returns a dict that `apply_isotonic_calibration` consumes
    and `save_isotonic_calibration` writes to JSON.

    Inputs:
    - composite_scores: clipped to [-1, 1].
    - realized_hits: 0/1 (or False/True). Pairs with NaN or None scores
      should be filtered upstream.
    """
    if len(composite_scores) != len(realized_hits):
        raise ValueError("composite_scores and realized_hits length mismatch")
    paired = [
        (float(s), 1.0 if h else 0.0)
        for s, h in zip(composite_scores, realized_hits)
    ]
    n = len(paired)
    if n < min_obs:
        return {
            "version": 1,
            "method": "isotonic_pav",
            "n_observations": n,
            "fit": [],
            "fallback": True,
            "fallback_reason": f"n_observations<{min_obs}",
        }
    paired.sort(key=lambda kv: kv[0])
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]
    weights = [1.0] * n

    # PAV: walk left-to-right, merging blocks that violate monotonicity.
    block_y = ys[:]
    block_w = weights[:]
    block_xs = [[x] for x in xs]
    i = 0
    while i + 1 < len(block_y):
        if block_y[i] > block_y[i + 1]:
            total_w = block_w[i] + block_w[i + 1]
            merged = (block_y[i] * block_w[i] + block_y[i + 1] * block_w[i + 1]) / total_w
            block_y[i] = merged
            block_w[i] = total_w
            block_xs[i] = block_xs[i] + block_xs[i + 1]
            del block_y[i + 1]
            del block_w[i + 1]
            del block_xs[i + 1]
            # Backtrack to fix any new violation.
            while i > 0 and block_y[i - 1] > block_y[i]:
                total_w = block_w[i - 1] + block_w[i]
                merged = (
                    block_y[i - 1] * block_w[i - 1]
                    + block_y[i] * block_w[i]
                ) / total_w
                block_y[i - 1] = merged
                block_w[i - 1] = total_w
                block_xs[i - 1] = block_xs[i - 1] + block_xs[i]
                del block_y[i]
                del block_w[i]
                del block_xs[i]
                i -= 1
        else:
            i += 1
    # Each block now has a constant fitted value. Persist as
    # (x_lower, x_upper, fitted_hit_rate) tuples, sorted by x_lower.
    fit_segments: list[dict[str, float]] = []
    for xs_block, y_fit, w_block in zip(block_xs, block_y, block_w):
        fit_segments.append(
            {
                "x_lower": round(min(xs_block), 6),
                "x_upper": round(max(xs_block), 6),
                "hit_rate": round(float(y_fit), 6),
                "n_obs": int(round(w_block)),
            }
        )
    return {
        "version": 1,
        "method": "isotonic_pav",
        "n_observations": n,
        "fit": fit_segments,
        "fallback": False,
    }


def apply_isotonic_calibration(
    composite_score: float,
    *,
    calibration: dict[str, Any],
) -> float | None:
    """Map a composite score to the calibrated hit rate.

    Returns ``None`` when calibration is in fallback mode (no fit
    available); the caller should then use the heuristic
    `confidence_for` path.
    """
    if not calibration or calibration.get("fallback"):
        return None
    segments = calibration.get("fit") or []
    if not segments:
        return None
    x = max(-1.0, min(1.0, float(composite_score)))
    if x <= segments[0]["x_upper"]:
        return float(segments[0]["hit_rate"])
    if x >= segments[-1]["x_lower"]:
        return float(segments[-1]["hit_rate"])
    for seg in segments:
        if seg["x_lower"] <= x <= seg["x_upper"]:
            return float(seg["hit_rate"])
    # Composite lies in a gap between segments; linearly interpolate
    # between the two surrounding fitted hit-rates.
    for prev_seg, next_seg in zip(segments, segments[1:]):
        if prev_seg["x_upper"] < x < next_seg["x_lower"]:
            span = next_seg["x_lower"] - prev_seg["x_upper"]
            if span <= 0:
                return float(prev_seg["hit_rate"])
            frac = (x - prev_seg["x_upper"]) / span
            return float(
                prev_seg["hit_rate"]
                + frac * (next_seg["hit_rate"] - prev_seg["hit_rate"])
            )
    return None


def brier_score(
    predicted_probs: Sequence[float],
    realized_hits: Sequence[int | bool],
) -> float | None:
    """Mean squared error of probability forecasts; lower is better.

    Returns None if inputs are empty or mismatched.
    """
    if not predicted_probs or len(predicted_probs) != len(realized_hits):
        return None
    return round(
        sum(
            (float(p) - (1.0 if h else 0.0)) ** 2
            for p, h in zip(predicted_probs, realized_hits)
        )
        / len(predicted_probs),
        6,
    )


def save_isotonic_calibration(
    calibration: dict[str, Any],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calibration, indent=2, sort_keys=True))


def load_isotonic_calibration(
    path: str | Path,
) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def reliability_diagram(
    predicted_probs: Sequence[float],
    realized_hits: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Per-bin observed vs. predicted hit-rate, useful for Phase 5's
    "reliability within +/-5pp of diagonal" gate.
    """
    if not predicted_probs or len(predicted_probs) != len(realized_hits):
        return []
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[dict[str, Any]] = [
        {
            "lower": round(bin_edges[i], 4),
            "upper": round(bin_edges[i + 1], 4),
            "n": 0,
            "mean_predicted": 0.0,
            "observed_hit_rate": None,
        }
        for i in range(n_bins)
    ]
    sums_p = [0.0] * n_bins
    sums_h = [0.0] * n_bins
    for p, h in zip(predicted_probs, realized_hits):
        idx = min(int(float(p) * n_bins), n_bins - 1)
        bins[idx]["n"] += 1
        sums_p[idx] += float(p)
        sums_h[idx] += 1.0 if h else 0.0
    for i, b in enumerate(bins):
        if b["n"] > 0:
            b["mean_predicted"] = round(sums_p[i] / b["n"], 4)
            b["observed_hit_rate"] = round(sums_h[i] / b["n"], 4)
    return bins


# ---------- Phase 4 regime labels ----------

# Regime A (trend_on): SPY above its 50d MA AND VIX < 20 (calm trend).
# Regime B (chop): everything else.
# The plan calls for SPY 200d slope, but `spy_above_50dma` is the field
# the pipeline persists today (avoids a corpus regen just for this).
# Two-bucket scheme until per-bucket sample counts justify a third.
REGIME_TREND_ON = "trend_on"
REGIME_CHOP = "chop"
REGIME_UNKNOWN = "unknown"
VIX_CALM_THRESHOLD = 20.0


def regime_for_market_context(
    market_context: dict[str, Any] | None,
    *,
    vix_threshold: float = VIX_CALM_THRESHOLD,
) -> str:
    """Bucket a record's market_context into a regime label.

    Returns ``"unknown"`` when either input is missing — the caller
    (walk-forward + scoring) should then fall back to the global weight
    vector rather than guess.
    """
    if not market_context:
        return REGIME_UNKNOWN
    spy_above_50 = market_context.get("spy_above_50dma")
    vix_level = market_context.get("vix_level")
    if spy_above_50 is None or vix_level is None:
        return REGIME_UNKNOWN
    try:
        vix_val = float(vix_level)
    except (TypeError, ValueError):
        return REGIME_UNKNOWN
    if bool(spy_above_50) and vix_val < vix_threshold:
        return REGIME_TREND_ON
    return REGIME_CHOP


# Factors that are inherently relative to the universe and benefit from
# cross-sectional normalization (rank within the cohort at each date).
# Factors that already self-normalize (RSI in [0, 100], IV rank, fear-greed
# regime) are intentionally absent — re-ranking them would discard the
# information their absolute level carries.
CROSS_SECTIONAL_FACTORS: frozenset[str] = frozenset({
    "trend_price_vs_sma20",
    "trend_sma20_vs_sma50",
    "trend_sma50_vs_sma200",
    "momentum_macd_hist",
    "momentum_return_20d",
    "breakout_60d",
    "fund_revenue_growth",
    "fund_earnings_growth",
    "fund_profit_margins",
    "fund_fcf_growth",
    "industry_relative_strength",
    "peer_relative_momentum",
    "peer_relative_valuation",
    "options_net_flow",
})


def normalize_cross_section(
    raw_values: dict[str, float | None],
    *,
    method: str = "rank",
    min_universe: int = 3,
    output_range: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, float | None]:
    """Map a {symbol -> raw factor value} dict to {symbol -> normalized score}.

    `method`:
    - ``"rank"`` (default): convert to fractional rank in [0, 1] with
      average-rank tied-value handling, then linearly rescale to
      ``output_range``. Highest raw value → top of range. Symbols with
      ``None`` raw values are returned as ``None``.
    - ``"zscore"``: standardize against the universe mean/stdev, then
      clip to ``output_range``. Falls back to ``"rank"`` if stdev is 0.

    The cross-sectional cohort is the set of non-None values in
    `raw_values`. With fewer than `min_universe` non-None entries the
    normalization is skipped and the function returns all ``None`` — the
    caller (pipeline) keeps the factor as data_available=False so the
    composite is not polluted by a one-stock "rank".

    Phase 2 caveat: with the 12-ticker tech-focused universe, the cohort
    is intra-sector. The lift is smaller than for a diversified universe;
    the decorrelation audit is the more valuable half of Phase 2.
    """
    finite = {sym: float(v) for sym, v in raw_values.items() if v is not None}
    if len(finite) < min_universe:
        return {sym: None for sym in raw_values}

    lo, hi = float(output_range[0]), float(output_range[1])
    width = hi - lo
    out: dict[str, float | None] = {sym: None for sym in raw_values}

    if method == "zscore":
        vals = list(finite.values())
        mu = statistics.fmean(vals)
        sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if sigma > 0:
            for sym, v in finite.items():
                z = (v - mu) / sigma
                z_clipped = max(-3.0, min(3.0, z))
                out[sym] = round(lo + width * (z_clipped + 3.0) / 6.0, 4)
            return out

    sorted_items = sorted(finite.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_items[j + 1][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_items[k][0]] = avg_rank
        i = j + 1
    if n == 1:
        out[next(iter(finite))] = round((lo + hi) / 2.0, 4)
        return out
    for sym, r in ranks.items():
        frac = r / (n - 1)
        out[sym] = round(lo + width * frac, 4)
    return out


def is_cross_sectional_factor(factor_name: str) -> bool:
    return factor_name in CROSS_SECTIONAL_FACTORS


def factor_correlation_matrix(
    factor_scores_by_record: Iterable[dict[str, float | None]],
    *,
    min_paired: int = 30,
) -> dict[str, dict[str, float | None]]:
    """Pairwise Spearman correlation across all factors.

    Input is an iterable of {factor_name -> score} dicts (one per record).
    Output is a {factor_a -> {factor_b -> rho}} nested dict. Pairs with
    fewer than `min_paired` non-None observations get ``None``.

    Used by `scripts/factor_correlation_audit.py` for the Phase 2 gate
    (no off-diagonal cell |rho| > 0.7 after decorrelation).
    """
    rows = list(factor_scores_by_record)
    if not rows:
        return {}
    all_factors = sorted({f for row in rows for f in row.keys()})
    matrix: dict[str, dict[str, float | None]] = {}
    for fa in all_factors:
        matrix[fa] = {}
        for fb in all_factors:
            if fa == fb:
                matrix[fa][fb] = 1.0
                continue
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                a = row.get(fa)
                b = row.get(fb)
                if a is None or b is None:
                    continue
                xs.append(float(a))
                ys.append(float(b))
            if len(xs) < min_paired:
                matrix[fa][fb] = None
                continue
            matrix[fa][fb] = _spearman_pairwise(xs, ys)
    return matrix


def _spearman_pairwise(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman correlation. Returns None for degenerate inputs."""
    if len(xs) < 3:
        return None
    rx = _avg_ranks(xs)
    ry = _avg_ranks(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    sxx = sum((x - mx) ** 2 for x in rx)
    syy = sum((y - my) ** 2 for y in ry)
    sxy = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    denom = (sxx * syy) ** 0.5
    if denom == 0:
        return None
    return round(sxy / denom, 4)


def _avg_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _first_finite(*values: float | None) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f == f and f not in (float("inf"), float("-inf")):
            return f
    return None


def compute_composite(
    factor_scores: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-factor scores into pillar + composite scores.

    Inputs are expected to look like the rows produced by `_build_report`:
    each dict has `factor`, `pillar`, `score`, `weight`, `weighted_score`,
    `data_available`. `weights` is optional; when provided it's used to
    compute `total_weight`, otherwise we fall back to summing per-factor
    weights from the rows themselves.

    Returns a dict with:
    - composite_score: clipped to [-1, 1]
    - pillar_scores: pillar -> mean weighted score (only counting
      data-available rows in that pillar)
    - coverage: fraction of total weight that had data available
    - active_weight, total_weight
    """
    total_weight = sum((weights or {}).values()) or sum(
        float(f.get("weight") or 0.0) for f in factor_scores
    )
    if total_weight <= 0:
        total_weight = 1.0

    active_weight = sum(
        float(f.get("weight") or 0.0)
        for f in factor_scores
        if f.get("data_available")
    )
    if active_weight <= 0:
        active_weight = total_weight

    # Only data-available rows contribute. The pipeline always writes
    # weighted_score=0 when data_available=False, so this filter is a
    # no-op in production today; it's here defensively so that
    # `compute_composite` and `backtest.rebuild_records_with_weights`
    # agree on edge-case inputs (see test_rebuild_matches_compute_composite).
    weighted_sum = sum(
        float(f.get("weighted_score") or 0.0)
        for f in factor_scores
        if f.get("data_available")
    )
    composite_raw = weighted_sum / active_weight
    composite = max(-1.0, min(1.0, composite_raw))

    pillar_weighted: dict[str, float] = {}
    pillar_weights: dict[str, float] = {}
    for f in factor_scores:
        if not f.get("data_available"):
            continue
        pillar = str(f.get("pillar") or "other")
        pillar_weighted[pillar] = (
            pillar_weighted.get(pillar, 0.0)
            + float(f.get("weighted_score") or 0.0)
        )
        pillar_weights[pillar] = (
            pillar_weights.get(pillar, 0.0)
            + float(f.get("weight") or 0.0)
        )

    pillar_scores: dict[str, float] = {}
    for pillar, p_weight in pillar_weights.items():
        if p_weight > 0:
            pillar_scores[pillar] = round(
                pillar_weighted.get(pillar, 0.0) / p_weight, 4
            )

    coverage = min(1.0, active_weight / total_weight) if total_weight else 0.0
    return {
        "composite_score": round(composite, 4),
        "pillar_scores": pillar_scores,
        "coverage": coverage,
        "active_weight": active_weight,
        "total_weight": total_weight,
    }
