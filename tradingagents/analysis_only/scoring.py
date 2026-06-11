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
    # IV-derived factors (added in Section 18). v1.5 (Section 27) commits
    # options_iv_term_structure with sign INVERTED and weight 0.04 after the
    # post-regen IC analysis (4863 obs): core 60d IC -0.109 / 80% sign-cons,
    # canary 60d IC -0.105 — cohorts agree, factor is UNIVERSAL not
    # tech-specific. The sign flip and weight bump are in lockstep (see
    # `score_iv_term_structure`); the score function emits negative for
    # contango and positive for backwardation now, so the weight stays
    # positive. options_iv_skew remains at 0 — never crossed the noise floor.
    "options_iv_term_structure": 0.04,
    "options_iv_skew": 0.00,
    # v1.3 (Section 21): promoted from 0.00 to 0.04 after Phase 1 IC
    # validated the placeholder sign: low IV rank → bullish forward
    # returns. Headline IC +0.154 at 20d (n=597); per-ticker median
    # +0.107, 63.6% sign consistency across 11 tickers.
    #
    # v1.8 (2026-06-06): reduced 0.04 → 0.02 after multi-regime IC
    # degradation. Combined corpus core IC = −0.017 with 54% sign-cons
    # (random); v1.7-only IC also ~0. The Phase 1 +0.107 signal didn't
    # generalize forward — likely a single-corpus regime artifact.
    # Cutting weight rather than zeroing pending further evidence;
    # the factor data path is healthy, just the signal has rotated out.
    "options_iv_rank": 0.02,
    # X1 (2026-06-04): Polygon news-sentiment factor. Per-ticker net
    # sentiment over a trailing 14d window, computed from Polygon's
    # `insights[].sentiment` classifications with a keyword-based
    # fallback. Ships at weight=0 pending IC validation on a regenerated
    # corpus (matches v1.5-v1.7 discipline).
    "news_sentiment": 0.00,
    # v1.7: new per-ticker fear/greed composite — aggregates 5 per-ticker
    # signals (IV rank, 25Δ IV skew, drawdown from 52w high, RSI(14), net
    # option flow) into a 0-100 score, analogous to CNN's market-wide F&G
    # but per-ticker. Shipped at weight=0 with momentum-following placeholder
    # sign pending IC validation.
    #
    # v1.8 (2026-06-06): promoted to weight 0.02 + sign INVERTED in
    # `score_ticker_fear_greed_regime`. Multi-regime cohort IC analysis on
    # the combined corpus (10,241 records, 4 regimes): core IC −0.095 with
    # 73% sign-consistency across 26 tickers; canary IC −0.117 (cross-sector
    # signs agree); v1.7-only matched at −0.102 / 76% — most stable signal
    # in the v1.8 candidate set. Negative IC + consistent sign + current
    # placeholder sign was momentum-following → flip to classical contrarian
    # (fear → bullish via mean reversion, greed → bearish via extension risk).
    # Conservative 0.02 weight relative to |IC|=0.095.
    "ticker_fear_greed_regime": 0.02,
}


# Universal factors — sign-agrees across the 24-tech core cohort and the
# 6-name cross-sector canary cohort in `backtest/results/phase2_cohort/`
# (handoff.md Section 22). These are the factors safe to score with
# their as-signed weights on a non-tech ticker. The screener's
# `--cohort-aware` mode uses this set when scoring non-tech sectors so
# the composite doesn't bake in tech-specific factor calibration.
#
# Membership rule (handoff Section 22 cohort table + the plan Unit 4):
#   1. Sign-agrees across core/canary at 20d AND 60d in the cohort IC
#      analysis: market_vix_regime, peer_relative_valuation,
#      options_iv_term_structure, momentum_rsi.
#   2. Always-mechanical-direction factors that aren't sector-conditional
#      (trend / valuation crossovers are mechanical, not sector-fitted):
#      trend_price_vs_sma20, trend_sma20_vs_sma50,
#      valuation_forward_vs_trailing_pe.
# Every name MUST be a key in `DEFAULT_FACTOR_WEIGHTS` — verified by
# `test_universal_factor_names_subset_of_default_weights`. Unit-1 corpus
# regen may shift the (1) list at 20d/60d; re-verify against
# `backtest/results/phase2_v1_4_cohort/cohort_20d.md` post-Unit-1.
# TODO(unit5): re-verify against post-Unit-1 cohort IC.
UNIVERSAL_FACTOR_NAMES: frozenset[str] = frozenset({
    # Cohort sign-agreed (Section 22):
    "market_vix_regime",
    "peer_relative_valuation",
    "options_iv_term_structure",
    "momentum_rsi",
    # Mechanical-direction (not sector-fitted):
    "trend_price_vs_sma20",
    "trend_sma20_vs_sma50",
    "valuation_forward_vs_trailing_pe",
})


def resolve_factor_weights(
    overrides: dict[str, float] | None = None,
    *,
    cohort: str | None = None,
) -> dict[str, float]:
    """Merge user overrides on top of defaults and renormalize to sum=1.

    `cohort`:
    - ``None`` / ``"tech"`` (default): no behavior change — every factor
      keeps its weight from `DEFAULT_FACTOR_WEIGHTS` (modulo overrides).
    - ``"non_tech"``: zero out every factor whose name is NOT in
      `UNIVERSAL_FACTOR_NAMES`, then renormalize the surviving weights
      so they sum to 1. Used by the screener's `--cohort-aware` mode
      to score non-tech sectors with only the cross-sector-validated
      factor set (handoff.md Section 22).
    """
    weights = dict(DEFAULT_FACTOR_WEIGHTS)
    for key, value in (overrides or {}).items():
        if key in weights and value is not None and value >= 0:
            weights[key] = float(value)
    if cohort == "non_tech":
        weights = {
            key: (value if key in UNIVERSAL_FACTOR_NAMES else 0.0)
            for key, value in weights.items()
        }
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

    **v1.5 (Section 27): sign inverted from placeholder, weight bumped 0.00 → 0.04.**

    Phase 2 post-regen IC analysis (Section 27, n=4863 obs) shows:
      - Core 60d IC = -0.109, 80% sign-consistency across 25 tech tickers.
      - Canary 60d IC = -0.105 → sign-AGREES across cohorts at 60d
        (this factor is UNIVERSAL, not tech-specific).
      - Core 20d IC = +0.025 (weak positive, near noise floor).
      - Core 5d IC ≈ +0.04.

    The original placeholder treated contango as bullish (positive score)
    and backwardation as bearish. The 60d/cohort-agree evidence inverts
    that: contango (high IV slope) → low forward returns; backwardation
    → high forward returns. Probable economic story: contango is the
    market pricing complacency that mean-reverts; backwardation marks
    short-term stress that resolves higher.

    Trade-off: small 20d/5d regression (sign was already near-zero there);
    clear 60d lift. 60d is the calibration anchor per Section 13.
    """
    if slope is None:
        return 0.0, "IV term-structure slope unavailable.", False
    if slope > 0.05:
        return -0.5, "Steep IV contango — complacency signal (v1.5 inverted).", True
    if slope > 0.02:
        return -0.25, "Mild IV contango — mildly bearish (v1.5 inverted).", True
    if slope < -0.05:
        return 1.0, "Deep IV backwardation — short-term stress, mean-reversion setup (v1.5 inverted).", True
    if slope < -0.02:
        return 0.5, "Mild IV backwardation — bullish reversion bias (v1.5 inverted).", True
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


# ---------- News sentiment factor (X1) ----------
#
# Keyword fallback lists used when an item lacks Polygon `insights`.
# Intentionally small + market-flavored — broader lexica add noise on the
# small-sample windows we score over (≤50 items per ticker).
_POSITIVE_NEWS_KEYWORDS: tuple[str, ...] = (
    "beat", "beats", "raise", "raised", "raises", "upgrade", "upgraded",
    "outperform", "buy rating", "record", "surge", "surged", "rally",
    "soared", "jumps", "gains", "growth", "profit", "profits",
    "strong", "wins", "win", "expand", "expansion", "partnership",
    "approval", "approved", "exceed", "exceeded", "boost",
    "bullish", "tops", "topped", "breakthrough", "innovation",
)
_NEGATIVE_NEWS_KEYWORDS: tuple[str, ...] = (
    "miss", "misses", "missed", "cut", "cuts", "downgrade", "downgraded",
    "underperform", "sell rating", "plunge", "plunged", "drop", "dropped",
    "fall", "fell", "decline", "declined", "loss", "losses",
    "weak", "lawsuit", "probe", "investigation", "fraud",
    "warn", "warning", "warned", "delay", "delayed", "halt", "halted",
    "bearish", "concerns", "fears", "recall", "risk", "risks",
    "disappointing", "shortfall",
)


def _keyword_sentiment_score(text: str) -> int:
    """Return positive_hits - negative_hits on a lowercased token-pass."""
    if not text:
        return 0
    lowered = text.lower()
    pos = sum(1 for kw in _POSITIVE_NEWS_KEYWORDS if kw in lowered)
    neg = sum(1 for kw in _NEGATIVE_NEWS_KEYWORDS if kw in lowered)
    return pos - neg


def _polygon_insight_sentiment(item: dict[str, Any]) -> int | None:
    """Map Polygon `insights[].sentiment` to {+1, 0, -1}.

    When an article has multiple insights, take the sign of the sum
    (positive minus negative count). Returns None when no insight has
    a usable sentiment string.
    """
    insights = item.get("insights")
    if not isinstance(insights, list) or not insights:
        return None
    pos = 0
    neg = 0
    seen_any = False
    for ins in insights:
        if not isinstance(ins, dict):
            continue
        sentiment = ins.get("sentiment")
        if not isinstance(sentiment, str):
            continue
        seen_any = True
        s = sentiment.strip().lower()
        if s == "positive":
            pos += 1
        elif s == "negative":
            neg += 1
        # "neutral" or any other value contributes 0
    if not seen_any:
        return None
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def _parse_news_timestamp(value: Any) -> str | None:
    """Polygon's `published_utc` is RFC3339, e.g. '2025-08-21T13:45:00Z'.
    Returns the ISO date portion (YYYY-MM-DD) or None on parse failure.
    """
    if not isinstance(value, str) or len(value) < 10:
        return None
    candidate = value[:10]
    # cheap shape check
    if candidate[4] != "-" or candidate[7] != "-":
        return None
    return candidate


def compute_news_sentiment(
    news_items: Sequence[dict[str, Any]],
    as_of_date: str,
    *,
    lookback_days: int = 14,
) -> dict[str, Any]:
    """Aggregate per-article sentiment into a net score over a trailing window.

    Strategy:
    - For each article published in (as_of_date - lookback_days, as_of_date],
      score in {+1, 0, -1}.
    - Prefer Polygon `insights[].sentiment` when present (publisher-supplied
      classification). Otherwise fall back to keyword counts on
      `title + ' ' + description`.
    - `net_sentiment` = mean of per-article signs, in [-1, 1].

    Returns a dict with `net_sentiment`, `n_articles`, `n_positive`,
    `n_negative`, `n_neutral`, `n_with_insights`, `n_keyword_fallback`,
    `window_start`, `window_end`. `net_sentiment` is None when no
    articles fall inside the window.
    """
    # Parse `as_of_date` with stdlib date math (no external imports needed).
    from datetime import date as _date, timedelta as _td

    try:
        y, m, d = (int(x) for x in as_of_date.split("-"))
        end = _date(y, m, d)
    except Exception:
        return {
            "net_sentiment": None,
            "n_articles": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_neutral": 0,
            "n_with_insights": 0,
            "n_keyword_fallback": 0,
            "window_start": None,
            "window_end": as_of_date,
        }
    start = end - _td(days=max(1, int(lookback_days)))
    window_start = start.isoformat()

    n_pos = 0
    n_neg = 0
    n_neu = 0
    n_insight = 0
    n_keyword = 0
    n_total = 0
    for item in news_items or []:
        if not isinstance(item, dict):
            continue
        published_date = _parse_news_timestamp(item.get("published_utc"))
        if not published_date:
            continue
        if published_date <= window_start or published_date > as_of_date:
            # Strictly inside (window_start, as_of_date] — `> as_of_date`
            # filter is defense in depth for callers that didn't pre-filter
            # via Polygon `published_utc.lte`.
            continue
        n_total += 1
        sign = _polygon_insight_sentiment(item)
        if sign is not None:
            n_insight += 1
        else:
            title = item.get("title") or ""
            desc = item.get("description") or ""
            text = f"{title} {desc}".strip()
            kw = _keyword_sentiment_score(text)
            if kw > 0:
                sign = 1
            elif kw < 0:
                sign = -1
            else:
                sign = 0
            n_keyword += 1
        if sign > 0:
            n_pos += 1
        elif sign < 0:
            n_neg += 1
        else:
            n_neu += 1

    if n_total == 0:
        net = None
    else:
        net = round((n_pos - n_neg) / n_total, 4)

    return {
        "net_sentiment": net,
        "n_articles": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_neutral": n_neu,
        "n_with_insights": n_insight,
        "n_keyword_fallback": n_keyword,
        "window_start": window_start,
        "window_end": as_of_date,
    }


def score_news_sentiment(
    net_score: float | None,
    n_articles: int,
    *,
    min_articles: int = 3,
) -> tuple[float, str, bool]:
    """Factor score function for the news_sentiment factor.

    Requires ≥`min_articles` (default 3) in the window to emit a non-zero
    score. Below that threshold the factor is `data_available=False`
    (score=0) so it doesn't pollute the composite.

    Score band mapping (chosen conservatively — IC will validate):
       net ≥ +0.50  →  +0.7  (strongly positive)
       net ≥ +0.20  →  +0.4  (mildly positive)
       net ≤ -0.50  →  -0.7  (strongly negative)
       net ≤ -0.20  →  -0.4  (mildly negative)
       otherwise    →   0.0  (mixed / neutral, but data_available=True)
    """
    if n_articles < int(min_articles) or net_score is None:
        return 0.0, (
            f"News sentiment unavailable (n={n_articles} < {int(min_articles)})."
        ), False
    if net_score >= 0.5:
        return 0.7, "Strongly positive news sentiment.", True
    if net_score >= 0.2:
        return 0.4, "Mildly positive news sentiment.", True
    if net_score <= -0.5:
        return -0.7, "Strongly negative news sentiment.", True
    if net_score <= -0.2:
        return -0.4, "Mildly negative news sentiment.", True
    return 0.0, "News sentiment is mixed / neutral.", True


# ---------- v1.7 per-ticker fear/greed composite ----------
#
# Component sub-scoring discretizes each input into a 0-100 fear/greed
# sub-score (0 = extreme fear, 100 = extreme greed) using a small number
# of buckets that mirror the discretization the existing single-input
# factor scorers use (e.g. `score_iv_rank`, `score_iv_skew`). Aggregating
# the available sub-scores by simple mean keeps the composite robust to
# missing components — when a ticker only has 3 of 5 inputs we still
# emit a meaningful score, but require ≥3 to avoid 1-component noise.
#
# Inputs and their fear→greed mapping:
#   iv_rank (0-1)           : high rank → fear (uncertainty premium up)
#   iv_skew (~ −0.05..0.20) : high put-skew → fear (tail-risk premium up)
#   drawdown_from_52w_high  : deep drawdown (negative) → fear
#                              (positive value means up from low base)
#   rsi_14 (0-100)          : low → fear, high → greed
#   net_flow_notional (USD) : call-heavy positive → greed, put-heavy → fear


def _bucket_iv_rank_fg(iv_rank: float | None) -> float | None:
    """Map IV rank ∈ [0, 1] to a 0-100 fear/greed sub-score.

    High IV rank means options are pricing elevated uncertainty -> fear.
    Buckets mirror `score_iv_rank` (top quintile, mid, bottom 20-40%,
    bottom quintile).
    """
    if iv_rank is None:
        return None
    r = float(iv_rank)
    if r >= 0.80:
        return 10.0   # extreme fear
    if r >= 0.60:
        return 30.0   # fear
    if r >= 0.40:
        return 50.0   # neutral
    if r >= 0.20:
        return 70.0   # greed
    return 90.0       # extreme greed (compressed vol)


def _bucket_iv_skew_fg(skew: float | None) -> float | None:
    """Map 25Δ put-call IV skew to a 0-100 fear/greed sub-score.

    Stocks normally show small positive skew (~0.00..0.05). Wide positive
    skew means puts are priced rich vs calls — tail-risk premium up —
    fear. Strong negative skew (calls > puts) is upside-chase greed.
    """
    if skew is None:
        return None
    s = float(skew)
    if s >= 0.15:
        return 10.0
    if s >= 0.08:
        return 30.0
    if s >= 0.00:
        return 50.0
    if s >= -0.05:
        return 70.0
    return 90.0


def _bucket_drawdown_fg(drawdown: float | None) -> float | None:
    """Map drawdown-from-52w-high to a 0-100 fear/greed sub-score.

    `drawdown` is signed and ≤ 0 by construction: 0 = at 52w high,
    -0.30 = 30% below 52w high. Deeper drawdown → fear. Anything within
    2% of the high is greed territory.
    """
    if drawdown is None:
        return None
    d = float(drawdown)
    if d <= -0.30:
        return 10.0
    if d <= -0.15:
        return 30.0
    if d <= -0.05:
        return 50.0
    if d <= -0.02:
        return 70.0
    return 90.0


def _bucket_rsi_fg(rsi_14: float | None) -> float | None:
    """Map RSI(14) ∈ [0, 100] to a 0-100 fear/greed sub-score.

    Bucket boundaries mirror the `momentum_rsi` factor's discretization
    in pipeline.py (oversold < 30, constructive 45-68, overbought > 75)
    but in fear/greed space (low RSI = fear, high RSI = greed).
    """
    if rsi_14 is None:
        return None
    r = float(rsi_14)
    if r < 30.0:
        return 10.0
    if r < 45.0:
        return 30.0
    if r <= 68.0:
        return 60.0
    if r <= 75.0:
        return 75.0
    return 90.0


def _bucket_net_flow_fg(net_flow_notional: float | None) -> float | None:
    """Map net call-put options notional ($) to a 0-100 fear/greed sub-score.

    Threshold ±1M USD mirrors the `options_net_flow` factor in
    pipeline.py (which fires call-heavy / put-heavy at ±$1M). Strong
    call-heavy → greed, put-heavy → fear, mixed → neutral.
    """
    if net_flow_notional is None:
        return None
    v = float(net_flow_notional)
    if v >= 2_000_000:
        return 90.0
    if v >= 1_000_000:
        return 75.0
    if v > -1_000_000:
        return 50.0
    if v > -2_000_000:
        return 25.0
    return 10.0


# Rating cutoffs match the CNN F&G convention used by `FearGreedProvider`
# (extreme fear ≤ 25, fear < 45, neutral 45-55, greed > 55, extreme greed
# ≥ 75) so downstream code reading the rating string can reuse existing
# patterns.
def _rating_for_fg_score(score: float) -> str:
    if score <= 25.0:
        return "extreme_fear"
    if score < 45.0:
        return "fear"
    if score <= 55.0:
        return "neutral"
    if score < 75.0:
        return "greed"
    return "extreme_greed"


def compute_ticker_fear_greed(
    *,
    iv_rank: float | None = None,
    iv_skew: float | None = None,
    drawdown_from_52w_high: float | None = None,
    rsi_14: float | None = None,
    net_flow_notional: float | None = None,
    min_components: int = 3,
) -> dict[str, Any]:
    """Aggregate per-ticker fear/greed sub-scores into a composite.

    Inputs are the already-available per-ticker signals from the
    pipeline:
      iv_rank                : `key_features.options_iv.iv_rank_252d`
      iv_skew                : `key_features.options_iv.skew_25d_30d`
      drawdown_from_52w_high : derived from price history (signed, ≤ 0)
      rsi_14                 : `key_features.technical.rsi_14`
      net_flow_notional      : `key_features.options_flow.net_call_put_notional`

    Returns one of:
      {
        "status": "ok",
        "score": <0-100 float>,
        "rating": "extreme_fear|fear|neutral|greed|extreme_greed",
        "components": {<name>: <0-100>, ...},  # only available ones
        "available_components": N,
      }
      {
        "status": "unavailable",
        "available_components": N,  # 0..(min_components-1)
        "components": {<name>: <0-100>, ...},
      }

    Components with `None` input are skipped (not 50.0-defaulted) so a
    ticker with no options data doesn't get a spurious neutral pull.
    Requires `>= min_components` available sub-scores to emit a score.
    """
    components: dict[str, float] = {}
    bucketed: list[tuple[str, float | None]] = [
        ("iv_rank", _bucket_iv_rank_fg(iv_rank)),
        ("iv_skew", _bucket_iv_skew_fg(iv_skew)),
        ("drawdown_from_52w_high", _bucket_drawdown_fg(drawdown_from_52w_high)),
        ("rsi_14", _bucket_rsi_fg(rsi_14)),
        ("net_flow_notional", _bucket_net_flow_fg(net_flow_notional)),
    ]
    for name, sub in bucketed:
        if sub is not None:
            components[name] = round(float(sub), 2)

    n_available = len(components)
    if n_available < min_components:
        return {
            "status": "unavailable",
            "available_components": n_available,
            "components": components,
        }

    score = sum(components.values()) / n_available
    score_clipped = max(0.0, min(100.0, score))
    return {
        "status": "ok",
        "score": round(score_clipped, 2),
        "rating": _rating_for_fg_score(score_clipped),
        "components": components,
        "available_components": n_available,
    }


def score_ticker_fear_greed_regime(
    score: float | None,
) -> tuple[float, str, bool]:
    """Score the per-ticker fear/greed composite into a factor signal.

    **v1.8 (2026-06-06): sign INVERTED from the v1.7 placeholder.** Phase 2
    multi-regime cohort IC analysis (4,931 v1.7 + 5,310 extended = 10,241
    records) shows:
      - Combined core IC = −0.095, 73% sign-cons across 26 tickers.
      - v1.7-only core IC = −0.102, 76% sign-cons.
      - Canary cohort IC = −0.117 (cross-sector signs agree).

    Cross-corpus + cross-cohort stable negative IC means the placeholder
    momentum-following sign (greed → +) was emitting the OPPOSITE of what
    predicts forward returns. v1.8 commits the classical contrarian
    mapping: ticker in extreme fear → +0.4 (mean-reversion bullish);
    extreme greed → -0.4 (extension bearish). Weight bumped from 0.00 to
    0.02 — small relative to the IC magnitude (0.095) so the bet is
    proportional to the evidence.

    `score` is the 0-100 ticker F&G composite from `compute_ticker_fear_greed`.
    """
    if score is None:
        return 0.0, "Per-ticker fear/greed composite unavailable.", False
    s = float(score)
    if s <= 25.0:
        return (
            0.4,
            "Ticker in extreme fear — contrarian mean-reversion bullish (v1.8 inverted).",
            True,
        )
    if s < 45.0:
        return 0.2, "Ticker in fear regime — mild contrarian bullish (v1.8 inverted).", True
    if s >= 75.0:
        return (
            -0.4,
            "Ticker in extreme greed — extension-risk bearish (v1.8 inverted).",
            True,
        )
    if s > 55.0:
        return -0.2, "Ticker in greed regime — mild extension-risk bearish (v1.8 inverted).", True
    return 0.0, "Ticker fear/greed composite neutral.", True


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
    direction: str | None = None,
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

    Section 29 (direction-conditional): when `calibration` contains a
    `by_direction` block and `direction` is provided, use the
    direction-specific curve. Bearish bucket is anti-predictive in this
    corpus (Section 14/15) — directional lookup exposes that asymmetry
    in the emitted confidence values.
    """
    coverage = max(0.0, min(1.0, coverage))
    if calibration:
        if direction is not None and (calibration.get("by_direction") or {}):
            calibrated = apply_isotonic_calibration_directional(
                composite_score, direction, calibration=calibration,
            )
        else:
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


def fit_isotonic_calibration_by_direction(
    composite_scores: Sequence[float],
    realized_hits: Sequence[int | bool],
    directions: Sequence[str],
    *,
    min_obs_per_direction: int = 30,
) -> dict[str, Any]:
    """Fit per-direction isotonic curves (bullish / bearish / neutral).

    Section 14/15 found that bearish calls are anti-predictive in this
    corpus (~33% hit vs 50% base rate). A single calibration curve
    averages over directions and obscures that asymmetry. Fitting
    separately exposes the structure: the bearish curve will likely be
    nearly flat (low confidence everywhere), the bullish curve will
    show the meaningful rising slope, and neutral lives at the base
    rate.

    Output extends the single-direction calibration JSON with a
    ``by_direction`` block. Apps that use `apply_isotonic_calibration`
    (single-curve) still work — the top-level ``fit`` field carries the
    all-directions curve as the fallback.
    """
    if not (len(composite_scores) == len(realized_hits) == len(directions)):
        raise ValueError("composite_scores / realized_hits / directions length mismatch")
    out: dict[str, Any] = fit_isotonic_calibration(
        composite_scores, realized_hits, min_obs=min_obs_per_direction,
    )
    out["method"] = "isotonic_pav_by_direction"
    out["version"] = 2
    by_dir: dict[str, Any] = {}
    for dirn in ("bullish", "bearish", "neutral"):
        sub_scores: list[float] = []
        sub_hits: list[int | bool] = []
        for s, h, d in zip(composite_scores, realized_hits, directions):
            if d == dirn:
                sub_scores.append(float(s))
                sub_hits.append(h)
        by_dir[dirn] = fit_isotonic_calibration(
            sub_scores, sub_hits, min_obs=min_obs_per_direction,
        )
    out["by_direction"] = by_dir
    return out


def apply_isotonic_calibration_directional(
    composite_score: float,
    direction: str | None,
    *,
    calibration: dict[str, Any],
) -> float | None:
    """Look up calibrated hit-rate using direction-conditional curve.

    Prefers `calibration["by_direction"][direction]` when populated and
    not fallback. Falls back to the top-level `calibration` if the
    direction-specific curve is missing / undersampled. Returns None on
    total cache miss.
    """
    if calibration is None:
        return None
    by_dir = calibration.get("by_direction") or {}
    sub = by_dir.get(direction) if direction else None
    if sub and sub.get("fit") and not sub.get("fallback"):
        v = apply_isotonic_calibration(composite_score, calibration=sub)
        if v is not None:
            return v
    return apply_isotonic_calibration(composite_score, calibration=calibration)


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


# Per-regime sign multipliers for factors where the chop-cohort IC fit
# (Priority #1, `backtest/results/v1_7_regime/`) disagreed in SIGN with the
# global v1.6 weights. Wholesale weight replacement failed the +0.5pp gate
# (the chop fit's magnitudes were noisy on a single corpus), but the SIGN
# disagreement on a handful of well-established factors is robust to that
# noise: in chop, these factors show clear OPPOSITE-direction predictive
# power vs trend_on. So the conservative v1.7 commit is sign-flip only,
# magnitudes unchanged.
#
# Selection criteria (all three must hold):
#   1. Chop-regime IC magnitude >= 0.13 (strong signal in chop).
#   2. Sign of chop IC DISAGREES with sign of current weight (in trend_on
#      this factor predicts +; in chop it predicts −).
#   3. NOT a trend factor — chop "trends fail" intuition is real but a
#      wholesale trend-factor flip is too aggressive on one corpus.
#
# Three factors qualify per `/tmp/regime_weights_v1_5.json`:
#   - `market_fear_greed_regime`: chop IC −0.17 vs trend_on commit +0.05
#     (v1.4 momentum-following inverted on trend_on; chop wants the
#      original contrarian story back — fear → bullish in chop).
#   - `fund_profit_margins`: chop IC −0.17 vs trend_on +0.08
#     (high-margin names mean-revert in chop).
#   - `options_iv_rank`: chop IC −0.14 vs trend_on +0.04
#     (low IV = complacency = bullish in trend_on; in chop, low IV =
#      regime expansion incoming = bearish).
#
# Applied AFTER weight resolution but BEFORE compute_composite by
# `apply_regime_to_factor_scores`. Walk-forward gated before commit.
REGIME_SIGN_FLIPS: dict[str, dict[str, int]] = {
    REGIME_CHOP: {
        "market_fear_greed_regime": -1,
        "fund_profit_margins": -1,
        "options_iv_rank": -1,
    },
    # REGIME_TREND_ON: {} — global weights already trend_on-tuned.
    # REGIME_UNKNOWN: {} — fall back to global weights when classifier can't decide.
}


def direction_for_composite_regime_gated(
    composite_score: float,
    *,
    threshold: float = 0.15,
    regime: str | None = None,
    fear_greed_score: float | None = None,
    fear_greed_bearish_threshold: float = -0.2,
    bullish_threshold: float | None = None,
    bearish_threshold: float | None = None,
) -> str:
    """`direction_for_composite` with a bear-side regime gate.

    Bearish calls in this corpus hit ~33% (anti-predictive vs 50% base
    rate; Section 14/15 finding). The default direction logic emits
    bearish whenever composite < −threshold. This gate keeps bullish
    and neutral logic untouched, but DOWNGRADES bearish candidates to
    neutral unless BOTH:
      - regime is `chop` (not `trend_on`)
      - `fear_greed_score` is fear or extreme_fear (post v1.4 inversion
        that maps to score ≤ −0.2)

    When regime is None or `unknown`, no gate is applied — fall back to
    standard behavior so we don't silently change behavior on records
    that pre-date the regime classifier.
    """
    direction = direction_for_composite(
        composite_score,
        threshold=threshold,
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
    )
    if direction != "bearish":
        return direction
    # Bearish candidate — apply the gate
    if regime is None or regime == REGIME_UNKNOWN:
        return direction
    if regime == REGIME_CHOP:
        if fear_greed_score is not None and fear_greed_score <= fear_greed_bearish_threshold:
            return direction
    return "neutral"


def apply_regime_to_factor_scores(
    factor_scores: list[dict[str, Any]],
    weights: dict[str, float],
    regime: str,
) -> list[dict[str, Any]]:
    """Return factor_scores with regime-conditional sign flips applied.

    For each factor in `REGIME_SIGN_FLIPS[regime]`, negate its score AND
    recompute `weighted_score = score × weight` (since `weighted_score`
    is pre-computed at emission time in `pipeline.add_factor`, not by
    `compute_composite`). The rationale string is appended with the flip
    annotation so downstream renderers / debugging can trace the change.

    Returns a NEW list — does NOT mutate `factor_scores`. Records with
    `data_available=False` are left untouched (no point flipping a zero
    score).

    See `REGIME_SIGN_FLIPS` for the selection criteria and source data.
    """
    flips = REGIME_SIGN_FLIPS.get(regime, {})
    if not flips:
        return factor_scores
    out: list[dict[str, Any]] = []
    for f in factor_scores:
        name = f.get("factor")
        if name in flips and f.get("data_available", True) and f.get("score") is not None:
            flipped = dict(f)
            try:
                new_score = -float(f["score"]) * flips[name] / abs(flips[name])
            except (TypeError, ValueError):
                out.append(f)
                continue
            # The sign multiplier is ±1; equivalent to: new_score = -score
            # when flip=-1. Kept arithmetic explicit so future +1 entries
            # (no-op safeguards) work without special-casing.
            new_score = float(f["score"]) * flips[name]
            flipped["score"] = round(new_score, 6)
            weight = float(weights.get(name, 0.0))
            flipped["weighted_score"] = round(new_score * weight, 6)
            rationale = f.get("rationale") or ""
            flipped["rationale"] = (
                f"{rationale} [regime={regime}: sign flipped per Phase-4 IC]"
            )
            out.append(flipped)
        else:
            out.append(f)
    return out


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


# Per-horizon composite weight overrides (per-horizon emission).
#
# Only ret_20d is overridden, with the IC-signed sparse vector fit at
# min_abs_ic=0.04 and validated +9.9pp median bullish test-hit under strict
# rolling-OOS walk-forward vs the v1.5 global vector (see
# `backtest/results/per_horizon_ic_sweep_findings.md`). ret_5d / ret_60d are
# intentionally ABSENT, so they fall back to the global DEFAULT_FACTOR_WEIGHTS
# — the primary (60d-anchored) composite is therefore unchanged, and per-horizon
# emission is purely additive.
#
# These are SIGNED weights (sign = IC sign), consumed by
# `compute_composite_signed`'s abs-normalized form — the same math the
# walk-forward gate (`backtest.rebuild_records_with_weights`) used, so an
# emitted composite_20d matches the gated recipe.
PER_HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "ret_20d": {
        "market_vix_regime": -0.0794,
        "options_iv_skew": -0.0639,
        "market_fear_greed_regime": 0.0614,
        "peer_relative_valuation": 0.0577,
        "momentum_rsi": -0.0426,
    },
}

# Horizons for which a composite is emitted alongside the primary.
PER_HORIZON_KEYS: tuple[str, ...] = ("ret_5d", "ret_20d", "ret_60d")


def compute_composite_signed(
    factor_scores: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Composite from SIGNED, abs-normalized weights.

    Mirrors `backtest.rebuild_records_with_weights` exactly: each available
    factor contributes ``score * (w / Σ|w|)``, divided by the active weight
    fraction, clipped to [-1, 1]. With all-positive weights this is identical
    to `compute_composite`; with signed weights it models IC-sign inversions
    (a negative weight flips that factor's contribution).

    Returns ``{composite_score, coverage, n_factors, weight_source?}``.
    """
    abs_total = sum(abs(v) for v in weights.values()) or 1.0
    composite_raw = 0.0
    active_weight = 0.0
    n_factors = 0
    for f in factor_scores:
        if not f.get("data_available"):
            continue
        name = f.get("factor")
        w_signed = float(weights.get(name, 0.0))
        if w_signed == 0.0:
            continue
        score = f.get("score")
        if score is None:
            continue
        composite_raw += float(score) * (w_signed / abs_total)
        active_weight += abs(w_signed) / abs_total
        n_factors += 1
    if active_weight > 0:
        composite_raw = composite_raw / active_weight
    composite = max(-1.0, min(1.0, composite_raw))
    return {
        "composite_score": round(composite, 4),
        "coverage": round(active_weight, 4),
        "n_factors": n_factors,
    }


def compute_per_horizon_composites(
    factor_scores: list[dict[str, Any]],
    global_weights: dict[str, float],
    *,
    per_horizon_weights: dict[str, dict[str, float]] | None = None,
    horizons: tuple[str, ...] = PER_HORIZON_KEYS,
    global_composite: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Emit one composite per horizon.

    A horizon present in ``per_horizon_weights`` uses its override vector and is
    computed via `compute_composite_signed` (matching the walk-forward gate).
    A horizon NOT overridden falls back to the global vector — and when
    ``global_composite`` is supplied it reuses that value verbatim, so the
    fallback horizons stay byte-identical to the primary composite_score
    (avoiding a sub-rounding drift between `compute_composite`, which reads each
    row's stored ``weighted_score``, and the recompute-from-``score`` path).

    Result: ``{horizon: {composite_score, coverage?, n_factors?, weight_source}}``.
    """
    phw = per_horizon_weights if per_horizon_weights is not None else PER_HORIZON_WEIGHTS
    out: dict[str, dict[str, Any]] = {}
    for h in horizons:
        override = phw.get(h)
        if override:
            result = compute_composite_signed(factor_scores, override)
            result["weight_source"] = "per_horizon"
        elif global_composite is not None:
            result = {"composite_score": round(global_composite, 4), "weight_source": "global"}
        else:
            result = compute_composite_signed(factor_scores, global_weights)
            result["weight_source"] = "global"
        out[h] = result
    return out


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
