from __future__ import annotations

import math

import pytest

from tradingagents.analysis_only.scoring import (
    CROSS_SECTIONAL_FACTORS,
    DEFAULT_FACTOR_WEIGHTS,
    REGIME_CHOP,
    REGIME_TREND_ON,
    REGIME_UNKNOWN,
    UNIVERSAL_FACTOR_NAMES,
    apply_isotonic_calibration,
    brier_score,
    bucket_for_score,
    compute_composite,
    compute_ticker_fear_greed,
    confidence_for,
    direction_for_composite,
    factor_correlation_matrix,
    fit_isotonic_calibration,
    is_cross_sectional_factor,
    load_isotonic_calibration,
    normalize_cross_section,
    regime_for_market_context,
    reliability_diagram,
    resolve_factor_weights,
    save_isotonic_calibration,
    score_fear_greed_regime,
    score_iv_rank,
    score_iv_skew,
    score_iv_term_structure,
    compute_news_sentiment,
    score_news_sentiment,
    score_sales_multiple_vs_growth,
    score_ticker_fear_greed_regime,
)


# ---------- resolve_factor_weights ----------


def test_default_weights_renormalize_to_one():
    weights = resolve_factor_weights()
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)


def test_overrides_renormalize():
    weights = resolve_factor_weights({"options_net_flow": 1.0})
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)
    # The boosted factor should dominate.
    assert weights["options_net_flow"] > weights["trend_price_vs_sma20"]


def test_negative_overrides_ignored():
    weights = resolve_factor_weights({"momentum_rsi": -5.0})
    default_rsi = DEFAULT_FACTOR_WEIGHTS["momentum_rsi"]
    total = sum(DEFAULT_FACTOR_WEIGHTS.values())
    assert math.isclose(weights["momentum_rsi"], default_rsi / total, abs_tol=1e-5)


def test_unknown_keys_ignored():
    weights = resolve_factor_weights({"not_a_real_factor": 5.0})
    assert "not_a_real_factor" not in weights
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)


# ---------- UNIVERSAL_FACTOR_NAMES + cohort-aware resolve_factor_weights ----------


def test_universal_factor_names_are_all_in_default_weights():
    # Every universal factor must be a recognized weight key — otherwise
    # cohort-aware re-scoring would silently zero-out the wrong factor set.
    missing = UNIVERSAL_FACTOR_NAMES - set(DEFAULT_FACTOR_WEIGHTS.keys())
    assert missing == set(), (
        f"UNIVERSAL_FACTOR_NAMES contains factors not in "
        f"DEFAULT_FACTOR_WEIGHTS: {missing}"
    )


def test_universal_factor_names_nonempty():
    # Sanity: the universal set should not be empty (else cohort=non_tech
    # would zero out all weights and produce divide-by-zero composites).
    assert len(UNIVERSAL_FACTOR_NAMES) >= 1


def test_resolve_factor_weights_cohort_default_no_change():
    base = resolve_factor_weights()
    assert resolve_factor_weights(cohort=None) == base
    assert resolve_factor_weights(cohort="tech") == base


def test_resolve_factor_weights_cohort_non_tech_zeros_non_universal():
    weights = resolve_factor_weights(cohort="non_tech")
    # Every non-universal factor must be zero.
    for name in DEFAULT_FACTOR_WEIGHTS:
        if name not in UNIVERSAL_FACTOR_NAMES:
            assert weights[name] == 0.0, (
                f"non-universal factor {name} should be zeroed under "
                f"cohort=non_tech, got {weights[name]}"
            )
    # Every universal factor that had a positive default weight should
    # have a positive weight after renormalization.
    for name in UNIVERSAL_FACTOR_NAMES:
        if DEFAULT_FACTOR_WEIGHTS[name] > 0:
            assert weights[name] > 0


def test_resolve_factor_weights_cohort_non_tech_renormalizes_to_one():
    weights = resolve_factor_weights(cohort="non_tech")
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)


def test_resolve_factor_weights_cohort_non_tech_respects_overrides():
    # Overrides apply first, then the non_tech zeroing.
    weights = resolve_factor_weights(
        {"momentum_rsi": 0.5},  # universal — should survive at higher weight
        cohort="non_tech",
    )
    # momentum_rsi (universal) gets the override-bumped weight and survives.
    assert weights["momentum_rsi"] > 0
    # market_spy_trend (non-universal) is zeroed even if defaulted >0.
    assert weights["market_spy_trend"] == 0.0
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)


def test_resolve_factor_weights_unknown_cohort_value_no_change():
    # An unrecognized cohort value falls back to tech-equivalent (no zeroing).
    base = resolve_factor_weights()
    weird = resolve_factor_weights(cohort="some_unknown_label")
    assert weird == base


# ---------- bucket_for_score ----------


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, "bullish"),
        (0.01, "bullish"),
        (0.0, "neutral"),
        (-0.01, "bearish"),
        (-1.0, "bearish"),
        (None, "neutral"),
    ],
)
def test_bucket_for_score(score, expected):
    assert bucket_for_score(score) == expected


# ---------- score_fear_greed_regime ----------


@pytest.mark.parametrize(
    "score,rating,expected_score,expected_available",
    [
        # v1.4 (Section 24): inverted from contrarian to momentum-following
        # for the tech-focused trading universe. Phase 2 cohort IC showed
        # -0.36 on 24-tech core; the previous contrarian mapping produced
        # the wrong directional signal for the universe we trade.
        (10.0, "extreme fear", -0.4, True),
        (35.0, "fear", -0.2, True),
        (50.0, "neutral", 0.0, True),
        (65.0, "greed", 0.2, True),
        (85.0, "extreme greed", 0.4, True),
        (None, None, 0.0, False),
    ],
)
def test_score_fear_greed_regime(
    score,
    rating,
    expected_score,
    expected_available,
):
    actual_score, rationale, available = score_fear_greed_regime(score, rating)
    assert actual_score == expected_score
    assert available is expected_available
    assert rationale


# ---------- score_sales_multiple_vs_growth ----------


def test_score_sales_multiple_vs_growth_rewards_high_growth_reasonable_multiple():
    score, rationale, available, raw = score_sales_multiple_vs_growth(
        price_to_sales=12.0,
        revenue_growth=0.5,
        peer_ev_to_revenue_median=16.0,
    )
    assert score > 0
    assert available is True
    assert raw is not None
    assert rationale


def test_score_sales_multiple_vs_growth_penalizes_low_growth_high_multiple():
    score, rationale, available, raw = score_sales_multiple_vs_growth(
        price_to_sales=15.0,
        revenue_growth=0.02,
    )
    assert score < 0
    assert available is True
    assert raw is not None
    assert rationale


def test_score_sales_multiple_vs_growth_handles_missing_data():
    score, rationale, available, raw = score_sales_multiple_vs_growth(
        price_to_sales=None,
        revenue_growth=0.2,
    )
    assert score == 0.0
    assert available is False
    assert raw is None
    assert rationale


# ---------- direction_for_composite ----------


@pytest.mark.parametrize(
    "composite,expected",
    [
        # Boundaries reflect the v1.1 default threshold = 0.15 (Section 13).
        (0.5, "bullish"),
        (0.15, "bullish"),     # exactly at threshold → bullish
        (0.149, "neutral"),    # just below threshold
        (0.0, "neutral"),
        (-0.149, "neutral"),
        (-0.15, "bearish"),    # exactly at -threshold → bearish
        (-0.99, "bearish"),
    ],
)
def test_direction_for_composite(composite, expected):
    assert direction_for_composite(composite) == expected


def test_direction_threshold_customizable():
    assert direction_for_composite(0.15, threshold=0.1) == "bullish"
    # Force the old default to confirm callers can opt back if needed.
    assert direction_for_composite(0.19, threshold=0.2) == "neutral"


# ---------- direction_for_composite asymmetric kwargs (Section 15) ----------


def test_direction_asymmetric_bullish_loose_bearish_tight():
    # bullish at 0.10, bearish at 0.30 → composite 0.12 is bullish,
    # composite -0.25 is neutral (under the tight bearish gate).
    assert direction_for_composite(
        0.12, bullish_threshold=0.10, bearish_threshold=0.30
    ) == "bullish"
    assert direction_for_composite(
        -0.25, bullish_threshold=0.10, bearish_threshold=0.30
    ) == "neutral"
    assert direction_for_composite(
        -0.30, bullish_threshold=0.10, bearish_threshold=0.30
    ) == "bearish"


def test_direction_asymmetric_single_side_override_falls_back_to_scalar():
    # Only bullish overridden → bearish side still uses the scalar 0.15.
    assert direction_for_composite(0.12, bullish_threshold=0.10) == "bullish"
    assert direction_for_composite(-0.14, bullish_threshold=0.10) == "neutral"
    assert direction_for_composite(-0.15, bullish_threshold=0.10) == "bearish"
    # Only bearish overridden → bullish side still uses the scalar 0.15.
    assert direction_for_composite(0.14, bearish_threshold=0.40) == "neutral"
    assert direction_for_composite(0.15, bearish_threshold=0.40) == "bullish"
    assert direction_for_composite(-0.39, bearish_threshold=0.40) == "neutral"


def test_direction_asymmetric_scalar_threshold_unchanged_when_no_overrides():
    # Regression guard: passing neither override = pure scalar behavior.
    assert direction_for_composite(0.15) == "bullish"
    assert direction_for_composite(-0.15) == "bearish"
    assert direction_for_composite(0.10) == "neutral"


# ---------- confidence_for ----------


def test_confidence_floor_at_zero_signal():
    assert confidence_for(composite_score=0.0, coverage=1.0) == 0.5


def test_confidence_caps_at_0_95():
    assert confidence_for(composite_score=1.0, coverage=1.0) == 0.95


def test_confidence_scales_with_coverage():
    high = confidence_for(composite_score=0.6, coverage=1.0)
    low = confidence_for(composite_score=0.6, coverage=0.5)
    assert high > low > 0.5


def test_confidence_uses_absolute_value():
    pos = confidence_for(composite_score=0.5, coverage=1.0)
    neg = confidence_for(composite_score=-0.5, coverage=1.0)
    assert pos == neg


# ---------- compute_composite ----------


def _factor(name, pillar, weight, score, *, available=True):
    return {
        "factor": name,
        "pillar": pillar,
        "score": score,
        "weight": weight,
        "weighted_score": weight * score,
        "data_available": available,
    }


def test_composite_clipped_to_unit_interval():
    rows = [_factor("a", "x", 1.0, 5.0)]
    out = compute_composite(rows)
    assert out["composite_score"] == 1.0

    rows = [_factor("a", "x", 1.0, -5.0)]
    out = compute_composite(rows)
    assert out["composite_score"] == -1.0


def test_composite_coverage_drops_unavailable_weight():
    rows = [
        _factor("a", "x", 0.5, 1.0, available=True),
        _factor("b", "x", 0.5, 1.0, available=False),
    ]
    out = compute_composite(rows, weights={"a": 0.5, "b": 0.5})
    assert out["coverage"] == 0.5
    assert out["active_weight"] == 0.5
    assert out["total_weight"] == 1.0


def test_composite_pillar_breakdown():
    rows = [
        _factor("a", "technical", 0.5, 1.0),
        _factor("b", "technical", 0.5, -1.0),
        _factor("c", "fundamental", 1.0, 1.0),
    ]
    out = compute_composite(rows)
    assert math.isclose(out["pillar_scores"]["technical"], 0.0)
    assert math.isclose(out["pillar_scores"]["fundamental"], 1.0)


def test_composite_handles_empty_rows():
    out = compute_composite([])
    assert out["composite_score"] == 0.0
    assert out["pillar_scores"] == {}


def test_composite_handles_all_unavailable():
    rows = [_factor("a", "x", 1.0, 1.0, available=False)]
    out = compute_composite(rows)
    # When no row is data_available, the function must not crash and must
    # report zero signal — the row's weighted_score cannot leak into the
    # composite. Falls back to total_weight to avoid division by zero.
    assert out["coverage"] in (0.0, 1.0)
    assert out["composite_score"] == 0.0


def test_composite_ignores_weighted_score_on_unavailable_rows():
    """Regression guard: data_available=False rows must NEVER contribute
    to the composite, even if their weighted_score field is non-zero
    (which could happen if a future change to the pipeline writes a
    score before the availability check, or if a caller hand-builds rows)."""
    rows = [
        _factor("kept", "x", 0.5, 1.0, available=True),
        # Hand-build a row that lies about availability but still has
        # weighted_score set (the fixture's helper sets it unconditionally).
        _factor("ghost", "x", 0.5, 1.0, available=False),
    ]
    out = compute_composite(rows, weights={"kept": 0.5, "ghost": 0.5})
    # Only "kept" should contribute → composite = score = 1.0.
    assert out["composite_score"] == 1.0
    assert out["coverage"] == 0.5


# ---------- score_iv_term_structure ----------


def test_iv_term_unavailable_when_slope_missing():
    score, reason, available = score_iv_term_structure(None)
    assert available is False
    assert score == 0.0


def test_iv_term_steep_contango_negative_in_v1_5():
    # v1.5 (Section 27): contango now scores negative (complacency signal).
    score, _, _ = score_iv_term_structure(0.08)
    assert score == -0.5


def test_iv_term_mild_contango_quarter_negative_in_v1_5():
    score, _, _ = score_iv_term_structure(0.03)
    assert score == -0.25


def test_iv_term_flat_zero():
    score, _, _ = score_iv_term_structure(0.0)
    assert score == 0.0


def test_iv_term_mild_backwardation_positive_in_v1_5():
    # v1.5: backwardation now scores positive (mean-reversion bullish).
    score, _, _ = score_iv_term_structure(-0.03)
    assert score == 0.5


def test_iv_term_deep_backwardation_strongly_positive_in_v1_5():
    score, _, _ = score_iv_term_structure(-0.08)
    assert score == 1.0


# ---------- score_iv_skew ----------


def test_iv_skew_unavailable_when_missing():
    score, _, available = score_iv_skew(None)
    assert available is False
    assert score == 0.0


def test_iv_skew_normal_range_zero():
    for s in (0.0, 0.05, 0.10, -0.02):
        score, _, _ = score_iv_skew(s)
        assert score == 0.0, f"skew={s} should score 0, got {score}"


def test_iv_skew_extreme_positive_bearish():
    score, _, _ = score_iv_skew(0.20)
    assert score == -0.5


def test_iv_skew_strong_call_skew_mildly_bearish():
    score, _, _ = score_iv_skew(-0.10)
    assert score == -0.25


# ---------- score_iv_rank ----------


def test_iv_rank_unavailable_when_missing():
    score, _, available = score_iv_rank(None)
    assert available is False
    assert score == 0.0


def test_iv_rank_high_bearish():
    score, _, _ = score_iv_rank(0.85)
    assert score == -0.5


def test_iv_rank_mid_neutral():
    score, _, _ = score_iv_rank(0.55)
    assert score == 0.0


def test_iv_rank_below_average_mild_bullish():
    score, _, _ = score_iv_rank(0.30)
    assert score == 0.25


def test_iv_rank_low_bullish():
    score, _, _ = score_iv_rank(0.10)
    assert score == 0.5


# ---------- factor weights ----------


def test_iv_skew_stays_at_zero_in_v1_5():
    # options_iv_skew has never crossed the noise floor. Stays at 0.
    assert DEFAULT_FACTOR_WEIGHTS["options_iv_skew"] == 0.0


def test_iv_term_structure_promoted_in_v1_5():
    # v1.5 (Section 27): post-regen IC analysis shows options_iv_term_structure
    # is cohort-universal at 60d (core IC -0.109, canary -0.105; cohorts agree).
    # Sign inverted in score_iv_term_structure (contango → negative score),
    # weight promoted 0.00 → 0.04 to match the IC magnitude.
    assert DEFAULT_FACTOR_WEIGHTS["options_iv_term_structure"] == 0.04


def test_iv_rank_reduced_in_v1_8():
    # v1.8 (2026-06-06): reduced 0.04 → 0.02 after multi-regime IC
    # degradation. The Phase 1 +0.107 signal didn't generalize forward;
    # combined corpus core IC = −0.017 (sign-cons 54% — random). Cutting
    # weight rather than zeroing pending further evidence.
    assert DEFAULT_FACTOR_WEIGHTS["options_iv_rank"] == 0.02


def test_fear_greed_restored_in_v1_4():
    # v1.4 (Section 24) restored market_fear_greed_regime to 0.05 after the
    # Phase 2 cohort analysis confirmed the inversion is tech-specific. The
    # sign of score_fear_greed_regime was flipped (fear → bearish) at the
    # same time, so the factor now contributes with momentum-following
    # semantics for the tech / AI / semi universe.
    assert DEFAULT_FACTOR_WEIGHTS["market_fear_greed_regime"] == 0.05


# ---------- cross-sectional normalization ----------


def test_normalize_cross_section_rank_maps_extremes_to_endpoints():
    raw = {"A": 0.0, "B": 1.0, "C": 0.5, "D": 0.75, "E": 0.25}
    out = normalize_cross_section(raw, method="rank")
    assert out["A"] == -1.0
    assert out["B"] == 1.0
    assert out["E"] < out["C"] < out["D"]


def test_normalize_cross_section_handles_ties_with_average_rank():
    raw = {"A": 1.0, "B": 1.0, "C": 2.0}
    out = normalize_cross_section(raw, method="rank")
    assert out["A"] == out["B"]
    assert out["C"] > out["A"]


def test_normalize_cross_section_passes_through_none():
    raw = {"A": 0.5, "B": None, "C": 1.0, "D": 0.0}
    out = normalize_cross_section(raw, method="rank")
    assert out["B"] is None
    assert out["C"] > out["A"] > out["D"]


def test_normalize_cross_section_skips_when_below_min_universe():
    raw = {"A": 1.0, "B": 2.0}
    out = normalize_cross_section(raw, method="rank", min_universe=3)
    assert all(v is None for v in out.values())


def test_normalize_cross_section_zscore_clips_and_rescales():
    raw = {f"S{i}": float(i) for i in range(11)}
    out = normalize_cross_section(raw, method="zscore")
    vals = [v for v in out.values() if v is not None]
    assert min(vals) >= -1.0
    assert max(vals) <= 1.0


def test_normalize_cross_section_zscore_falls_back_to_rank_on_zero_std():
    raw = {"A": 5.0, "B": 5.0, "C": 5.0, "D": 5.0}
    out = normalize_cross_section(raw, method="zscore")
    assert all(v is not None for v in out.values())
    assert all(v == out["A"] for v in out.values())


def test_is_cross_sectional_factor_membership():
    assert is_cross_sectional_factor("momentum_return_20d")
    assert not is_cross_sectional_factor("momentum_rsi")
    assert not is_cross_sectional_factor("market_fear_greed_regime")


def test_cross_sectional_set_excludes_self_normalizing_factors():
    for self_norm in (
        "momentum_rsi",
        "market_fear_greed_regime",
        "market_vix_regime",
        "options_iv_rank",
        "options_iv_skew",
        "options_iv_term_structure",
    ):
        assert self_norm not in CROSS_SECTIONAL_FACTORS


# ---------- factor correlation matrix ----------


def test_factor_correlation_matrix_perfect_correlation():
    rows = [{"a": float(i), "b": float(2 * i)} for i in range(40)]
    m = factor_correlation_matrix(rows, min_paired=10)
    assert m["a"]["a"] == 1.0
    assert m["a"]["b"] == 1.0
    assert m["b"]["a"] == 1.0


def test_factor_correlation_matrix_anticorrelation():
    rows = [{"a": float(i), "b": float(-i)} for i in range(40)]
    m = factor_correlation_matrix(rows, min_paired=10)
    assert m["a"]["b"] == -1.0


def test_factor_correlation_matrix_below_min_paired_returns_none():
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    m = factor_correlation_matrix(rows, min_paired=5)
    assert m["a"]["b"] is None


def test_factor_correlation_matrix_handles_missing_pairs():
    rows = [
        {"a": 1.0, "b": 2.0},
        {"a": 2.0},
        {"a": 3.0, "b": 6.0},
        {"a": 4.0, "b": 8.0},
    ] * 10
    m = factor_correlation_matrix(rows, min_paired=10)
    assert m["a"]["b"] is not None
    assert m["a"]["b"] > 0.9


# ---------- Phase 4 regime labels ----------


def test_regime_trend_on_when_spy_up_and_vix_calm():
    mc = {"spy_above_50dma": True, "vix_level": 15.0}
    assert regime_for_market_context(mc) == REGIME_TREND_ON


def test_regime_chop_when_spy_down():
    mc = {"spy_above_50dma": False, "vix_level": 15.0}
    assert regime_for_market_context(mc) == REGIME_CHOP


def test_regime_chop_when_vix_elevated():
    mc = {"spy_above_50dma": True, "vix_level": 25.0}
    assert regime_for_market_context(mc) == REGIME_CHOP


def test_regime_unknown_when_market_context_empty():
    assert regime_for_market_context({}) == REGIME_UNKNOWN
    assert regime_for_market_context(None) == REGIME_UNKNOWN


def test_regime_unknown_when_fields_missing():
    assert regime_for_market_context({"spy_above_50dma": True}) == REGIME_UNKNOWN
    assert regime_for_market_context({"vix_level": 15.0}) == REGIME_UNKNOWN


def test_regime_unknown_on_unparseable_vix():
    mc = {"spy_above_50dma": True, "vix_level": "n/a"}
    assert regime_for_market_context(mc) == REGIME_UNKNOWN


def test_regime_threshold_is_strict_less_than():
    mc_below = {"spy_above_50dma": True, "vix_level": 19.99}
    mc_at = {"spy_above_50dma": True, "vix_level": 20.0}
    assert regime_for_market_context(mc_below) == REGIME_TREND_ON
    assert regime_for_market_context(mc_at) == REGIME_CHOP


# ---------- apply_regime_to_factor_scores (v1.7 candidate) ----------


def _flip_test_factors():
    return [
        {
            "factor": "market_fear_greed_regime", "pillar": "context",
            "score": 0.4, "weight": 0.05, "weighted_score": 0.02,
            "data_available": True, "rationale": "Greed (momentum-bullish).",
        },
        {
            "factor": "fund_profit_margins", "pillar": "fundamental",
            "score": 0.5, "weight": 0.08, "weighted_score": 0.04,
            "data_available": True, "rationale": "Strong margins.",
        },
        {
            "factor": "options_iv_rank", "pillar": "options",
            "score": 0.5, "weight": 0.04, "weighted_score": 0.02,
            "data_available": True, "rationale": "Low IV rank.",
        },
        {
            "factor": "trend_price_vs_sma20", "pillar": "technical",
            "score": 1.0, "weight": 0.08, "weighted_score": 0.08,
            "data_available": True, "rationale": "Above SMA20.",
        },
    ]


def _flip_test_weights():
    return {
        "market_fear_greed_regime": 0.05,
        "fund_profit_margins": 0.08,
        "options_iv_rank": 0.04,
        "trend_price_vs_sma20": 0.08,
    }


def test_apply_regime_trend_on_is_passthrough():
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = _flip_test_factors()
    out = apply_regime_to_factor_scores(fs, _flip_test_weights(), REGIME_TREND_ON)
    # trend_on has no flips → output equals input (identity, not even a copy
    # since no flips means no work needed).
    assert out == fs


def test_apply_regime_unknown_is_passthrough():
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = _flip_test_factors()
    out = apply_regime_to_factor_scores(fs, _flip_test_weights(), REGIME_UNKNOWN)
    assert out == fs


def test_apply_regime_chop_flips_three_factors_and_keeps_others():
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = _flip_test_factors()
    weights = _flip_test_weights()
    out = apply_regime_to_factor_scores(fs, weights, REGIME_CHOP)
    by = {f["factor"]: f for f in out}
    # Three configured factors should flip sign and recompute weighted_score.
    assert by["market_fear_greed_regime"]["score"] == -0.4
    assert by["market_fear_greed_regime"]["weighted_score"] == pytest.approx(-0.02)
    assert by["fund_profit_margins"]["score"] == -0.5
    assert by["fund_profit_margins"]["weighted_score"] == pytest.approx(-0.04)
    assert by["options_iv_rank"]["score"] == -0.5
    assert by["options_iv_rank"]["weighted_score"] == pytest.approx(-0.02)
    # Untouched factor stays identical.
    assert by["trend_price_vs_sma20"]["score"] == 1.0
    assert by["trend_price_vs_sma20"]["weighted_score"] == 0.08


def test_apply_regime_chop_does_not_mutate_input():
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = _flip_test_factors()
    original = [dict(f) for f in fs]
    apply_regime_to_factor_scores(fs, _flip_test_weights(), REGIME_CHOP)
    assert fs == original  # input untouched


def test_apply_regime_chop_skips_unavailable_factors():
    """A factor with data_available=False should not be flipped — keeps
    score=0 and a 'no signal' rationale."""
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = [
        {
            "factor": "market_fear_greed_regime", "pillar": "context",
            "score": 0.0, "weight": 0.05, "weighted_score": 0.0,
            "data_available": False, "rationale": "F&G unavailable.",
        },
    ]
    out = apply_regime_to_factor_scores(fs, _flip_test_weights(), REGIME_CHOP)
    assert out == fs


def test_apply_regime_chop_rationale_annotated():
    from tradingagents.analysis_only.scoring import apply_regime_to_factor_scores
    fs = _flip_test_factors()
    out = apply_regime_to_factor_scores(fs, _flip_test_weights(), REGIME_CHOP)
    fg = next(f for f in out if f["factor"] == "market_fear_greed_regime")
    assert "regime=chop" in fg["rationale"]
    assert "sign flipped" in fg["rationale"]


def test_regime_sign_flips_constant_shape():
    """Sanity: every flip target must be in DEFAULT_FACTOR_WEIGHTS (else
    the lookup `weights.get(name, 0.0)` would silently return 0)."""
    from tradingagents.analysis_only.scoring import REGIME_SIGN_FLIPS
    for regime, flips in REGIME_SIGN_FLIPS.items():
        for name, mult in flips.items():
            assert name in DEFAULT_FACTOR_WEIGHTS, (
                f"{regime}.{name} not in DEFAULT_FACTOR_WEIGHTS"
            )
            assert mult in (-1, 1), f"{regime}.{name} sign must be ±1"


# ---------- direction_for_composite_regime_gated (v1.7 bear-side gate) ----------


def test_regime_gate_bullish_call_is_passthrough():
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    # Bullish candidate always passes through regardless of regime/F&G.
    assert direction_for_composite_regime_gated(
        0.6, regime=REGIME_TREND_ON, fear_greed_score=0.4,
    ) == "bullish"
    assert direction_for_composite_regime_gated(
        0.6, regime=REGIME_CHOP, fear_greed_score=-0.4,
    ) == "bullish"


def test_regime_gate_neutral_call_is_passthrough():
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    assert direction_for_composite_regime_gated(
        0.05, regime=REGIME_TREND_ON, fear_greed_score=None,
    ) == "neutral"


def test_regime_gate_bearish_blocked_in_trend_on():
    """Section 15: bearish in trend_on is anti-predictive. Gate blocks it."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    out = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_TREND_ON, fear_greed_score=-0.4,
    )
    assert out == "neutral"


def test_regime_gate_bearish_blocked_in_chop_when_fear_greed_neutral():
    """Chop + neutral F&G is not enough — need fear signal too."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    out = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_CHOP, fear_greed_score=0.0,
    )
    assert out == "neutral"


def test_regime_gate_bearish_blocked_in_chop_when_greed():
    """Chop + greed = ambiguous risk-off signal, gate keeps blocking."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    out = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_CHOP, fear_greed_score=+0.2,
    )
    assert out == "neutral"


def test_regime_gate_bearish_passes_in_chop_with_fear():
    """The intended pass-through case: chop AND F&G at fear or worse."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    # F&G at exactly -0.2 (fear bucket) clears the threshold.
    out = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_CHOP, fear_greed_score=-0.2,
    )
    assert out == "bearish"
    # Extreme fear (-0.4) also passes.
    out2 = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_CHOP, fear_greed_score=-0.4,
    )
    assert out2 == "bearish"


def test_regime_gate_unknown_regime_falls_back_to_standard():
    """When the classifier returns 'unknown' (missing inputs), don't
    silently gate — fall back to standard direction logic."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    out = direction_for_composite_regime_gated(
        -0.4, regime=REGIME_UNKNOWN, fear_greed_score=None,
    )
    assert out == "bearish"


def test_regime_gate_none_regime_falls_back_to_standard():
    """When the caller doesn't pass regime at all, behavior must match
    `direction_for_composite` exactly (no surprise default-gate)."""
    from tradingagents.analysis_only.scoring import direction_for_composite_regime_gated
    out = direction_for_composite_regime_gated(-0.4)
    assert out == "bearish"


# ---------- direction-conditional isotonic calibration (Section 29) ----------


def test_fit_calibration_by_direction_emits_top_level_and_by_direction():
    from tradingagents.analysis_only.scoring import (
        fit_isotonic_calibration_by_direction,
    )
    n = 60
    composites = [(i - n / 2) / (n / 2) for i in range(n)]  # span [-1, 1]
    hits = [1 if c > 0 else 0 for c in composites]
    directions = ["bullish" if c > 0.15 else "bearish" if c < -0.15 else "neutral" for c in composites]
    cal = fit_isotonic_calibration_by_direction(
        composites, hits, directions, min_obs_per_direction=5,
    )
    assert cal["method"] == "isotonic_pav_by_direction"
    assert "fit" in cal           # top-level all-directions still present
    assert "by_direction" in cal
    assert set(cal["by_direction"].keys()) == {"bullish", "bearish", "neutral"}


def test_fit_calibration_by_direction_undersampled_falls_back():
    from tradingagents.analysis_only.scoring import (
        fit_isotonic_calibration_by_direction,
    )
    composites = [0.6, 0.4, -0.3, 0.0]
    hits = [1, 1, 0, 0]
    directions = ["bullish", "bullish", "bearish", "neutral"]
    cal = fit_isotonic_calibration_by_direction(
        composites, hits, directions, min_obs_per_direction=30,
    )
    # Each per-direction fit will be fallback=True (n<30).
    for d in ("bullish", "bearish", "neutral"):
        assert cal["by_direction"][d].get("fallback") is True


def test_apply_calibration_directional_uses_direction_curve_when_available():
    from tradingagents.analysis_only.scoring import (
        apply_isotonic_calibration_directional,
    )
    cal = {
        "fit": [
            {"x_lower": -1.0, "x_upper": 1.0, "hit_rate": 0.30, "n_obs": 100},
        ],
        "by_direction": {
            "bullish": {
                "fit": [{"x_lower": 0.2, "x_upper": 1.0, "hit_rate": 0.78, "n_obs": 60}],
                "fallback": False,
            },
            "bearish": {
                # Anti-predictive: low confidence regardless of composite.
                "fit": [{"x_lower": -1.0, "x_upper": -0.2, "hit_rate": 0.18, "n_obs": 25}],
                "fallback": False,
            },
        },
    }
    # Bullish composite uses bullish curve → 0.78.
    assert apply_isotonic_calibration_directional(0.5, "bullish", calibration=cal) == 0.78
    # Bearish composite uses bearish curve → 0.18.
    assert apply_isotonic_calibration_directional(-0.5, "bearish", calibration=cal) == 0.18


def test_apply_calibration_directional_falls_back_when_direction_missing():
    from tradingagents.analysis_only.scoring import (
        apply_isotonic_calibration_directional,
    )
    cal = {
        "fit": [
            {"x_lower": -1.0, "x_upper": 1.0, "hit_rate": 0.30, "n_obs": 100},
        ],
        "by_direction": {
            # No 'neutral' key → fall back to top-level fit.
        },
    }
    out = apply_isotonic_calibration_directional(0.0, "neutral", calibration=cal)
    assert out == 0.30


def test_apply_calibration_directional_falls_back_on_fallback_subfit():
    """Even when 'bullish' key exists, if it's fallback=True (undersampled)
    we should NOT use it — fall back to the top-level curve."""
    from tradingagents.analysis_only.scoring import (
        apply_isotonic_calibration_directional,
    )
    cal = {
        "fit": [
            {"x_lower": -1.0, "x_upper": 1.0, "hit_rate": 0.30, "n_obs": 100},
        ],
        "by_direction": {
            "bullish": {"fit": [], "fallback": True, "fallback_reason": "n<30"},
        },
    }
    out = apply_isotonic_calibration_directional(0.5, "bullish", calibration=cal)
    assert out == 0.30


def test_confidence_for_uses_direction_curve_when_provided():
    """confidence_for(direction=...) routes through the directional lookup
    when the calibration has a by_direction block."""
    cal = {
        "fit": [{"x_lower": -1.0, "x_upper": 1.0, "hit_rate": 0.50, "n_obs": 100}],
        "by_direction": {
            "bullish": {
                "fit": [{"x_lower": 0.2, "x_upper": 1.0, "hit_rate": 0.85, "n_obs": 60}],
                "fallback": False,
            },
            "bearish": {
                "fit": [{"x_lower": -1.0, "x_upper": -0.2, "hit_rate": 0.15, "n_obs": 30}],
                "fallback": False,
            },
        },
    }
    # Same composite +0.6 — bullish curve says 0.85, top-level says 0.50.
    bull = confidence_for(0.6, coverage=1.0, calibration=cal, direction="bullish")
    top  = confidence_for(0.6, coverage=1.0, calibration=cal)  # no direction
    assert bull == 0.85
    # Top-level falls back through the directional path too when direction
    # isn't provided. It picks the all-directions curve → 0.50, then mixes
    # with coverage 1.0 → 0.50. Bound by floor/cap.
    assert top == 0.50
    # Sanity: bear gets the bear-curve hit rate.
    bear = confidence_for(-0.6, coverage=1.0, calibration=cal, direction="bearish")
    assert bear == 0.50  # 0.15 floor-clamped to 0.5


# ---------- Phase 5 isotonic calibration ----------


def test_fit_isotonic_calibration_monotone_on_clean_signal():
    composites = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0] * 20
    hits = [0, 0, 0, 0, 1, 1, 1] * 20
    cal = fit_isotonic_calibration(composites, hits, min_obs=10)
    assert not cal["fallback"]
    assert cal["n_observations"] == len(composites)
    fitted = [seg["hit_rate"] for seg in cal["fit"]]
    assert all(b >= a for a, b in zip(fitted, fitted[1:]))


def test_fit_isotonic_calibration_fallback_when_below_min_obs():
    cal = fit_isotonic_calibration([0.1, 0.2], [0, 1], min_obs=30)
    assert cal["fallback"]
    assert cal["fit"] == []


def test_apply_isotonic_calibration_uses_fit():
    cal = fit_isotonic_calibration(
        [-0.8, -0.4, 0.0, 0.4, 0.8] * 20,
        [0, 0, 0, 1, 1] * 20,
        min_obs=10,
    )
    low = apply_isotonic_calibration(-0.8, calibration=cal)
    high = apply_isotonic_calibration(0.8, calibration=cal)
    assert low is not None and high is not None
    assert low <= high


def test_apply_isotonic_calibration_returns_none_on_fallback():
    cal = {"fit": [], "fallback": True}
    assert apply_isotonic_calibration(0.5, calibration=cal) is None


def test_apply_isotonic_calibration_clips_out_of_range_input():
    cal = fit_isotonic_calibration(
        [-0.5, 0.0, 0.5] * 20, [0, 0, 1] * 20, min_obs=10
    )
    extreme_high = apply_isotonic_calibration(5.0, calibration=cal)
    extreme_low = apply_isotonic_calibration(-5.0, calibration=cal)
    assert extreme_high is not None
    assert extreme_low is not None
    assert extreme_low <= extreme_high


def test_brier_score_perfect_forecast_is_zero():
    assert brier_score([0.0, 1.0, 1.0, 0.0], [0, 1, 1, 0]) == 0.0


def test_brier_score_random_guesses_quarter():
    b = brier_score([0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0])
    assert b == 0.25


def test_brier_score_none_on_mismatch():
    assert brier_score([], []) is None
    assert brier_score([0.5], [0, 1]) is None


def test_reliability_diagram_groups_by_bin():
    probs = [0.05, 0.55, 0.55, 0.95, 0.95]
    hits = [0, 1, 0, 1, 1]
    bins = reliability_diagram(probs, hits, n_bins=10)
    assert bins[0]["n"] == 1
    assert bins[5]["n"] == 2
    assert bins[9]["n"] == 2
    assert bins[5]["observed_hit_rate"] == 0.5


def test_save_and_load_isotonic_calibration_roundtrip(tmp_path):
    cal = fit_isotonic_calibration(
        [-0.5, 0.0, 0.5] * 20, [0, 0, 1] * 20, min_obs=10
    )
    p = tmp_path / "cal.json"
    save_isotonic_calibration(cal, p)
    loaded = load_isotonic_calibration(p)
    assert loaded == cal


def test_load_isotonic_calibration_missing_returns_none(tmp_path):
    assert load_isotonic_calibration(tmp_path / "does_not_exist.json") is None


def test_confidence_for_uses_calibration_when_supplied():
    cal = fit_isotonic_calibration(
        [-0.5, 0.0, 0.5] * 20, [0, 0, 1] * 20, min_obs=10
    )
    heuristic = confidence_for(0.4, 1.0)
    calibrated = confidence_for(0.4, 1.0, calibration=cal)
    # Must still be in [floor, cap].
    assert 0.5 <= calibrated <= 0.95
    # And different from heuristic (we'd be surprised if they coincide exactly).
    assert heuristic != calibrated or cal["fallback"]


# ---------- compute_ticker_fear_greed (v1.7) ----------


def test_ticker_fear_greed_extreme_fear_when_all_inputs_maxed_fear():
    out = compute_ticker_fear_greed(
        iv_rank=0.95,                  # top quintile  -> 10
        iv_skew=0.20,                  # extreme put-skew -> 10
        drawdown_from_52w_high=-0.40,  # 40% off high -> 10
        rsi_14=22.0,                   # oversold -> 10
        net_flow_notional=-3_000_000,  # heavy put flow -> 10
    )
    assert out["status"] == "ok"
    assert out["available_components"] == 5
    assert out["rating"] == "extreme_fear"
    assert out["score"] <= 25.0
    # All 5 sub-scores should be present.
    assert set(out["components"].keys()) == {
        "iv_rank",
        "iv_skew",
        "drawdown_from_52w_high",
        "rsi_14",
        "net_flow_notional",
    }


def test_ticker_fear_greed_extreme_greed_when_all_inputs_maxed_greed():
    out = compute_ticker_fear_greed(
        iv_rank=0.10,                  # bottom quintile -> 90
        iv_skew=-0.10,                 # strong call-skew -> 90
        drawdown_from_52w_high=-0.01,  # near 52w high -> 90
        rsi_14=82.0,                   # overbought -> 90
        net_flow_notional=3_500_000,   # heavy call flow -> 90
    )
    assert out["status"] == "ok"
    assert out["available_components"] == 5
    assert out["rating"] == "extreme_greed"
    assert out["score"] >= 75.0


def test_ticker_fear_greed_neutral_when_inputs_mid_range():
    out = compute_ticker_fear_greed(
        iv_rank=0.50,                  # mid -> 50
        iv_skew=0.04,                  # ~normal positive -> 50
        drawdown_from_52w_high=-0.10,  # 10% drawdown -> 50
        rsi_14=55.0,                   # constructive -> 60
        net_flow_notional=0.0,         # mixed -> 50
    )
    assert out["status"] == "ok"
    assert out["available_components"] == 5
    # Score in [0, 100] and (with these inputs) sits in the neutral
    # band around 50-55.
    assert 45.0 <= out["score"] <= 65.0
    # The exact rating bucket depends on whether the mean lands in
    # neutral (45..55) or greed (>55); both are acceptable for "mid".
    assert out["rating"] in {"neutral", "greed"}


def test_ticker_fear_greed_missing_components_skipped():
    # 3 inputs available, 2 missing -> still emits but with only 3 components.
    out = compute_ticker_fear_greed(
        iv_rank=0.90,                  # -> 10
        iv_skew=None,                  # skipped
        drawdown_from_52w_high=-0.20,  # -> 30
        rsi_14=None,                   # skipped
        net_flow_notional=-1_500_000,  # -> 25
    )
    assert out["status"] == "ok"
    assert out["available_components"] == 3
    assert set(out["components"].keys()) == {
        "iv_rank",
        "drawdown_from_52w_high",
        "net_flow_notional",
    }
    # Mean of 10, 30, 25 ≈ 21.67 → extreme_fear.
    assert out["rating"] == "extreme_fear"


def test_ticker_fear_greed_requires_min_components():
    # Only 2 inputs available -> unavailable.
    out = compute_ticker_fear_greed(
        iv_rank=0.50,
        iv_skew=0.04,
        drawdown_from_52w_high=None,
        rsi_14=None,
        net_flow_notional=None,
    )
    assert out["status"] == "unavailable"
    assert out["available_components"] == 2
    # The components dict still captures what we did get, for debugging.
    assert set(out["components"].keys()) == {"iv_rank", "iv_skew"}
    # Score / rating must NOT be present when unavailable.
    assert "score" not in out
    assert "rating" not in out


def test_ticker_fear_greed_all_missing_unavailable():
    out = compute_ticker_fear_greed()
    assert out["status"] == "unavailable"
    assert out["available_components"] == 0
    assert out["components"] == {}


def test_ticker_fear_greed_score_bounded_to_unit_range():
    # Sweep across a few realistic combinations and confirm the score
    # never escapes [0, 100].
    cases = [
        dict(iv_rank=0.0, iv_skew=-0.10, drawdown_from_52w_high=0.0,
             rsi_14=99.0, net_flow_notional=10_000_000),
        dict(iv_rank=1.0, iv_skew=0.50, drawdown_from_52w_high=-0.99,
             rsi_14=0.0, net_flow_notional=-10_000_000),
        dict(iv_rank=0.6, iv_skew=0.10, drawdown_from_52w_high=-0.12,
             rsi_14=40.0, net_flow_notional=500_000),
    ]
    for kwargs in cases:
        out = compute_ticker_fear_greed(**kwargs)
        assert out["status"] == "ok"
        assert 0.0 <= out["score"] <= 100.0


def test_ticker_fear_greed_custom_min_components():
    # Caller can require all 5 for the strictest acceptance gate.
    out = compute_ticker_fear_greed(
        iv_rank=0.5,
        iv_skew=0.0,
        drawdown_from_52w_high=-0.05,
        rsi_14=50.0,
        net_flow_notional=None,
        min_components=5,
    )
    assert out["status"] == "unavailable"
    assert out["available_components"] == 4


# ---------- score_ticker_fear_greed_regime (v1.7) ----------


@pytest.mark.parametrize(
    "score,expected_score,expected_available",
    [
        # v1.8 (2026-06-06): sign INVERTED to classical contrarian after
        # multi-regime IC analysis (core IC -0.095, 73% sign-cons).
        # Fear → bullish (mean reversion), greed → bearish (extension).
        (10.0, 0.4, True),    # extreme fear → bullish
        (35.0, 0.2, True),    # fear → mild bullish
        (50.0, 0.0, True),    # neutral
        (65.0, -0.2, True),   # greed → mild bearish
        (85.0, -0.4, True),   # extreme greed → bearish
    ],
)
def test_score_ticker_fear_greed_regime_buckets_v1_8(
    score, expected_score, expected_available
):
    actual_score, rationale, available = score_ticker_fear_greed_regime(score)
    assert actual_score == expected_score
    assert available is expected_available
    assert rationale


def test_score_ticker_fear_greed_regime_extreme_greed_is_bearish_v1_8():
    """v1.8: extreme greed (score≥75) maps to −0.4 (extension-risk bearish)."""
    s, rationale, avail = score_ticker_fear_greed_regime(90.0)
    assert s == -0.4
    assert avail is True
    assert "v1.8 inverted" in rationale


def test_score_ticker_fear_greed_regime_extreme_fear_is_bullish_v1_8():
    """v1.8: extreme fear (score≤25) maps to +0.4 (mean-reversion bullish)."""
    s, rationale, avail = score_ticker_fear_greed_regime(5.0)
    assert s == 0.4
    assert avail is True
    assert "v1.8 inverted" in rationale


def test_score_ticker_fear_greed_regime_unavailable():
    s, rationale, avail = score_ticker_fear_greed_regime(None)
    assert s == 0.0
    assert avail is False
    assert rationale


# ---------- v1.8 weight commit guards ----------


def test_ticker_fear_greed_regime_promoted_in_v1_8():
    """v1.8 (2026-06-06): promoted from 0.00 to 0.02 after multi-regime
    cohort IC analysis (combined corpus, 10,241 records) showed stable
    negative IC (-0.095, 73% sign-cons) across both bull and chop regimes.
    Sign inverted in `score_ticker_fear_greed_regime` at the same time.
    Conservative weight relative to |IC|."""
    assert DEFAULT_FACTOR_WEIGHTS["ticker_fear_greed_regime"] == 0.02


def _news_item(
    title: str = "",
    description: str = "",
    published_utc: str = "2025-08-20T12:00:00Z",
    insights: list[dict] | None = None,
) -> dict:
    item: dict = {
        "title": title,
        "description": description,
        "published_utc": published_utc,
    }
    if insights is not None:
        item["insights"] = insights
    return item


def test_news_sentiment_weight_present_in_defaults_at_zero():
    # Section 28 discipline: ship at weight=0 pending IC validation.
    assert DEFAULT_FACTOR_WEIGHTS.get("news_sentiment") == 0.0


def test_compute_news_sentiment_positive_via_insights():
    items = [
        _news_item(insights=[{"sentiment": "positive"}], published_utc="2025-08-19T10:00:00Z"),
        _news_item(insights=[{"sentiment": "positive"}], published_utc="2025-08-20T10:00:00Z"),
        _news_item(insights=[{"sentiment": "negative"}], published_utc="2025-08-21T10:00:00Z"),
        _news_item(insights=[{"sentiment": "positive"}], published_utc="2025-08-21T11:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 4
    assert out["n_positive"] == 3
    assert out["n_negative"] == 1
    assert out["n_with_insights"] == 4
    assert out["n_keyword_fallback"] == 0
    assert out["net_sentiment"] == pytest.approx(0.5)
    assert out["window_end"] == "2025-08-22"


def test_compute_news_sentiment_negative_via_insights():
    items = [
        _news_item(insights=[{"sentiment": "negative"}], published_utc="2025-08-20T10:00:00Z"),
        _news_item(insights=[{"sentiment": "negative"}], published_utc="2025-08-20T11:00:00Z"),
        _news_item(insights=[{"sentiment": "negative"}], published_utc="2025-08-21T10:00:00Z"),
        _news_item(insights=[{"sentiment": "positive"}], published_utc="2025-08-21T11:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 4
    assert out["net_sentiment"] == pytest.approx(-0.5)


def test_compute_news_sentiment_neutral_when_balanced():
    items = [
        _news_item(insights=[{"sentiment": "positive"}], published_utc="2025-08-20T10:00:00Z"),
        _news_item(insights=[{"sentiment": "negative"}], published_utc="2025-08-21T10:00:00Z"),
        _news_item(insights=[{"sentiment": "neutral"}], published_utc="2025-08-21T11:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 3
    assert out["n_neutral"] == 1
    assert out["n_positive"] == 1
    assert out["n_negative"] == 1
    assert out["net_sentiment"] == pytest.approx(0.0)


def test_compute_news_sentiment_keyword_fallback_when_insights_missing():
    items = [
        _news_item(title="Acme beats earnings, raises guidance",
                   published_utc="2025-08-20T10:00:00Z"),
        _news_item(title="Acme misses badly, downgrade incoming",
                   published_utc="2025-08-21T10:00:00Z"),
        _news_item(title="Acme inks new partnership and growth deal",
                   published_utc="2025-08-21T11:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 3
    assert out["n_with_insights"] == 0
    assert out["n_keyword_fallback"] == 3
    # 2 positive + 1 negative → net = +1/3
    assert out["net_sentiment"] == pytest.approx(0.3333, abs=1e-3)
    assert out["n_positive"] == 2
    assert out["n_negative"] == 1


def test_compute_news_sentiment_keyword_neutral_no_signal_words():
    items = [
        _news_item(title="Acme files report with regulators",
                   published_utc="2025-08-20T10:00:00Z"),
        _news_item(title="Acme attends industry conference",
                   published_utc="2025-08-21T10:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 2
    assert out["n_neutral"] == 2
    assert out["net_sentiment"] == pytest.approx(0.0)


def test_compute_news_sentiment_filters_outside_window():
    items = [
        _news_item(insights=[{"sentiment": "positive"}],
                   published_utc="2025-07-01T10:00:00Z"),  # too old
        _news_item(insights=[{"sentiment": "negative"}],
                   published_utc="2025-08-21T10:00:00Z"),
        _news_item(insights=[{"sentiment": "positive"}],
                   published_utc="2025-09-01T10:00:00Z"),  # in the future
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    # Only the 2025-08-21 article falls inside (2025-08-08, 2025-08-22].
    assert out["n_articles"] == 1
    assert out["n_negative"] == 1
    assert out["net_sentiment"] == pytest.approx(-1.0)


def test_compute_news_sentiment_empty_returns_no_articles():
    out = compute_news_sentiment([], as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 0
    assert out["net_sentiment"] is None
    assert out["window_end"] == "2025-08-22"


def test_compute_news_sentiment_drops_items_without_timestamps():
    items = [
        {"title": "no timestamp", "insights": [{"sentiment": "positive"}]},
        _news_item(insights=[{"sentiment": "positive"}],
                   published_utc="2025-08-20T10:00:00Z"),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 1


def test_compute_news_sentiment_bad_as_of_returns_no_articles():
    out = compute_news_sentiment(
        [_news_item(insights=[{"sentiment": "positive"}])],
        as_of_date="not-a-date",
    )
    assert out["n_articles"] == 0
    assert out["net_sentiment"] is None


def test_compute_news_sentiment_insights_mixed_signs():
    # Two insights on the same article — net negative within the article.
    items = [
        _news_item(
            insights=[
                {"sentiment": "positive"},
                {"sentiment": "negative"},
                {"sentiment": "negative"},
            ],
            published_utc="2025-08-20T10:00:00Z",
        ),
    ]
    out = compute_news_sentiment(items, as_of_date="2025-08-22", lookback_days=14)
    assert out["n_articles"] == 1
    assert out["n_negative"] == 1
    assert out["n_with_insights"] == 1


def test_score_news_sentiment_insufficient_articles_not_available():
    # Default min_articles=3.
    score, rationale, available = score_news_sentiment(net_score=0.5, n_articles=2)
    assert score == 0.0
    assert available is False
    assert rationale


def test_score_news_sentiment_strong_positive():
    score, rationale, available = score_news_sentiment(net_score=0.6, n_articles=10)
    assert score == 0.7
    assert available is True
    assert "positive" in rationale.lower()


def test_score_news_sentiment_mild_positive():
    score, _r, available = score_news_sentiment(net_score=0.25, n_articles=5)
    assert score == 0.4
    assert available is True


def test_score_news_sentiment_strong_negative():
    score, _r, available = score_news_sentiment(net_score=-0.7, n_articles=12)
    assert score == -0.7
    assert available is True


def test_score_news_sentiment_mild_negative():
    score, _r, available = score_news_sentiment(net_score=-0.3, n_articles=4)
    assert score == -0.4
    assert available is True


def test_score_news_sentiment_neutral_band_is_data_available():
    score, _r, available = score_news_sentiment(net_score=0.1, n_articles=6)
    assert score == 0.0
    # Still data_available=True — mixed sentiment is a real signal of "no signal".
    assert available is True


def test_score_news_sentiment_min_articles_override():
    score, _r, available = score_news_sentiment(
        net_score=0.6, n_articles=3, min_articles=5,
    )
    assert score == 0.0
    assert available is False


def test_score_news_sentiment_none_net_not_available():
    score, _r, available = score_news_sentiment(net_score=None, n_articles=10)
    assert score == 0.0
    assert available is False
