from __future__ import annotations

import math

import pytest

from tradingagents.analysis_only.scoring import (
    CROSS_SECTIONAL_FACTORS,
    DEFAULT_FACTOR_WEIGHTS,
    REGIME_CHOP,
    REGIME_TREND_ON,
    REGIME_UNKNOWN,
    apply_isotonic_calibration,
    brier_score,
    bucket_for_score,
    compute_composite,
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
    score_sales_multiple_vs_growth,
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
        (10.0, "extreme fear", 0.4, True),
        (35.0, "fear", 0.2, True),
        (50.0, "neutral", 0.0, True),
        (65.0, "greed", -0.2, True),
        (85.0, "extreme greed", -0.4, True),
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


def test_iv_term_steep_contango_positive():
    score, _, _ = score_iv_term_structure(0.08)
    assert score == 0.5


def test_iv_term_mild_contango_quarter_positive():
    score, _, _ = score_iv_term_structure(0.03)
    assert score == 0.25


def test_iv_term_flat_zero():
    score, _, _ = score_iv_term_structure(0.0)
    assert score == 0.0


def test_iv_term_mild_backwardation_negative():
    score, _, _ = score_iv_term_structure(-0.03)
    assert score == -0.5


def test_iv_term_deep_backwardation_strongly_negative():
    score, _, _ = score_iv_term_structure(-0.08)
    assert score == -1.0


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


def test_iv_factors_have_zero_weight_initially():
    for f in (
        "options_iv_term_structure",
        "options_iv_skew",
        "options_iv_rank",
    ):
        assert DEFAULT_FACTOR_WEIGHTS[f] == 0.0


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
