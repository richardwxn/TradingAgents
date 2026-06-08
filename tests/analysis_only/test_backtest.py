from __future__ import annotations

import pytest

from tradingagents.analysis_only.backtest import (
    BacktestRecord,
    FactorRecord,
    bucket_by_composite,
    bucket_by_confidence,
    bucket_by_direction,
    compute_return_metrics,
    compute_return_metrics_by_direction,
    explode_records_to_factors,
    fit_weights_from_records,
    ic_signed_weights,
    is_hit,
    pearson_correlation,
    rebuild_records_with_weights,
    recommend_asymmetric_thresholds,
    recommend_direction_threshold,
    render_asymmetric_sweep_markdown,
    render_factor_by_ticker_markdown,
    render_factor_summary_markdown,
    render_summary_markdown,
    render_threshold_sweep_markdown,
    spearman_correlation,
    split_train_score,
    summarize_all,
    summarize_bucket,
    summarize_factor,
    summarize_factors,
    summarize_factors_by_ticker,
    sweep_direction_threshold,
    sweep_direction_threshold_asymmetric,
    regime_walk_forward_backtest,
    regime_walk_forward_timeline_to_dict,
    walk_forward_backtest,
    walk_forward_refit_dates,
    walk_forward_timeline_to_dict,
)


def _r(
    symbol="X",
    as_of="2026-01-01",
    direction="bullish",
    confidence=0.7,
    composite=0.4,
    rets=None,
):
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction=direction,
        confidence=confidence,
        composite_score=composite,
        forward_returns=rets or {},
    )


# ---------- is_hit ----------


@pytest.mark.parametrize(
    "direction,ret,expected",
    [
        ("bullish", 0.05, True),
        ("bullish", -0.01, False),
        ("bullish", None, None),
        ("bearish", -0.05, True),
        ("bearish", 0.01, False),
        ("neutral", 0.01, True),
        ("neutral", 0.05, False),
        ("neutral", -0.01, True),
        ("nonsense", 0.05, None),
    ],
)
def test_is_hit(direction, ret, expected):
    assert is_hit(direction, ret) is expected


def test_is_hit_custom_neutral_band():
    assert is_hit("neutral", 0.03, neutral_band=0.05) is True
    assert is_hit("neutral", 0.06, neutral_band=0.05) is False


# ---------- bucketing ----------


def test_bucket_by_direction_separates_all_buckets():
    recs = [
        _r(direction="bullish"),
        _r(direction="bearish"),
        _r(direction="bearish"),
        _r(direction="neutral"),
    ]
    out = bucket_by_direction(recs)
    assert len(out["bullish"]) == 1
    assert len(out["bearish"]) == 2
    assert len(out["neutral"]) == 1


def test_bucket_by_confidence_handles_boundary_inclusive_lower():
    recs = [
        _r(confidence=0.5),
        _r(confidence=0.59),
        _r(confidence=0.7),
        _r(confidence=0.95),
        _r(confidence=None),
    ]
    out = bucket_by_confidence(recs)
    keys = list(out.keys())
    first_bucket = next(k for k in keys if k.startswith("[0.50"))
    assert len(out[first_bucket]) == 2
    assert len(out["unknown"]) == 1
    # bucket [0.70, 0.80) catches the 0.7 record
    second_bucket = next(k for k in keys if k.startswith("[0.70"))
    assert len(out[second_bucket]) == 1


def test_bucket_by_composite_includes_extremes():
    recs = [
        _r(composite=-1.0),
        _r(composite=-0.3),
        _r(composite=0.0),
        _r(composite=0.5),
        _r(composite=1.0),
        _r(composite=None),
    ]
    out = bucket_by_composite(recs)
    # Sum of all bucketed records (excluding 'unknown') should equal 5.
    total_bucketed = sum(len(v) for k, v in out.items() if k != "unknown")
    assert total_bucketed == 5
    assert len(out["unknown"]) == 1


# ---------- summarize_bucket ----------


def test_summarize_bucket_no_returns():
    out = summarize_bucket([_r(rets={})])
    assert out["count"] == 1
    assert out["count_with_return"] == 0
    assert out["mean_forward_return"] is None
    assert out["hit_rate"] is None


def test_summarize_bucket_basic_stats():
    recs = [
        _r(direction="bullish", rets={"ret_20d": 0.10}),
        _r(direction="bullish", rets={"ret_20d": -0.05}),
        _r(direction="bullish", rets={"ret_20d": 0.04}),
        _r(direction="bullish", rets={"ret_20d": 0.06}),
    ]
    out = summarize_bucket(recs, return_field="ret_20d")
    assert out["count_with_return"] == 4
    assert out["mean_forward_return"] == pytest.approx(0.0375, abs=1e-4)
    # 3 of 4 are positive while we're bullish.
    assert out["hit_rate"] == pytest.approx(0.75)


def test_summarize_bucket_skips_missing_returns():
    recs = [
        _r(direction="bullish", rets={"ret_20d": 0.10}),
        _r(direction="bullish", rets={}),
    ]
    out = summarize_bucket(recs, return_field="ret_20d")
    assert out["count"] == 2
    assert out["count_with_return"] == 1
    assert out["hit_rate"] == 1.0


def test_summarize_bucket_neutral_band_applies():
    recs = [
        _r(direction="neutral", rets={"ret_20d": 0.01}),
        _r(direction="neutral", rets={"ret_20d": -0.005}),
        _r(direction="neutral", rets={"ret_20d": 0.05}),
    ]
    out = summarize_bucket(recs, return_field="ret_20d", neutral_band=0.02)
    # First two are inside ±2%, third is outside.
    assert out["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ---------- summarize_all + render ----------


def test_summarize_all_has_all_horizons_and_buckets():
    recs = [
        _r(direction="bullish", confidence=0.8, composite=0.6,
           rets={"ret_5d": 0.02, "ret_20d": 0.05}),
        _r(direction="bearish", confidence=0.65, composite=-0.4,
           rets={"ret_5d": -0.01, "ret_20d": -0.03}),
        _r(direction="neutral", confidence=0.55, composite=0.0,
           rets={"ret_5d": 0.0, "ret_20d": 0.01}),
    ]
    summary = summarize_all(recs, return_fields=["ret_5d", "ret_20d"])
    assert summary["total_records"] == 3
    assert set(summary["by_horizon"].keys()) == {"ret_5d", "ret_20d"}
    for horizon_block in summary["by_horizon"].values():
        assert horizon_block["overall"]["count"] == 3
        # All three directions should be represented in by_direction.
        assert set(horizon_block["by_direction"].keys()) >= {
            "bullish", "bearish", "neutral"
        }
        # Confidence + composite buckets should each have at least one
        # non-empty entry (they're populated only when records land in
        # the bucket).
        assert horizon_block["by_confidence"]
        assert horizon_block["by_composite"]


def test_render_summary_markdown_handles_empty_summary():
    md = render_summary_markdown({"total_records": 0, "by_horizon": {}})
    assert "Total records: **0**" in md
    assert md.endswith("\n")


def test_render_summary_markdown_writes_buckets():
    recs = [
        _r(direction="bullish", confidence=0.8, composite=0.6,
           rets={"ret_5d": 0.02, "ret_20d": 0.05}),
        _r(direction="bearish", confidence=0.65, composite=-0.4,
           rets={"ret_5d": -0.01, "ret_20d": -0.03}),
    ]
    summary = summarize_all(recs, return_fields=["ret_20d"])
    md = render_summary_markdown(summary)
    assert "Horizon: `ret_20d`" in md
    assert "By direction" in md
    assert "By confidence bucket" in md
    assert "By composite score bucket" in md


# ---------- per-factor IC ----------


def _fr(
    factor="trend",
    pillar="technical",
    score=0.5,
    weighted_score=0.05,
    bucket="bullish",
    data_available=True,
    rets=None,
    adj_rets=None,
    as_of="2026-01-01",
    symbol="X",
):
    return FactorRecord(
        symbol=symbol,
        as_of_date=as_of,
        factor=factor,
        pillar=pillar,
        score=score,
        weighted_score=weighted_score,
        bucket=bucket,
        data_available=data_available,
        forward_returns=rets or {},
        benchmark_adjusted_returns=adj_rets or {},
    )


def test_pearson_correlation_perfect_positive_and_negative():
    assert pearson_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pearson_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_pearson_correlation_zero_variance_returns_none():
    assert pearson_correlation([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_pearson_correlation_too_few_points():
    assert pearson_correlation([1, 2], [3, 4]) is None


def test_spearman_correlation_handles_monotonic_nonlinear():
    # Pearson would be < 1.0 for a quadratic; Spearman should be 1.0.
    assert spearman_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)


def test_spearman_correlation_handles_ties():
    # Tied scores should still produce a finite Spearman.
    ic = spearman_correlation([0.2, 0.2, 0.7, -0.7], [0.01, 0.03, 0.05, -0.02])
    assert ic is not None
    assert -1.0 <= ic <= 1.0


def test_summarize_factor_basic_signal():
    recs = [
        _fr(score=1.0, rets={"ret_20d": 0.10}),
        _fr(score=0.7, rets={"ret_20d": 0.06}),
        _fr(score=0.2, rets={"ret_20d": 0.02}),
        _fr(score=-0.2, rets={"ret_20d": -0.01}),
        _fr(score=-0.7, rets={"ret_20d": -0.05}),
        _fr(score=-1.0, rets={"ret_20d": -0.08}),
    ]
    out = summarize_factor(recs, return_field="ret_20d")
    assert out["n_paired"] == 6
    # Perfectly monotonic → Spearman IC = 1.0.
    assert out["spearman_ic"] == pytest.approx(1.0)
    # Long-short = mean(bull) - mean(bear) = 0.06 - (-0.0467) ≈ 0.107
    assert out["long_short_spread"] > 0.10
    assert out["hit_rate_when_bullish"] == 1.0
    assert out["hit_rate_when_bearish"] == 1.0


def test_summarize_factor_skips_missing_score_and_return():
    recs = [
        _fr(score=None, rets={"ret_20d": 0.10}),
        _fr(score=0.5, rets={}),
        _fr(score=0.5, data_available=False, rets={"ret_20d": 0.05}),
        _fr(score=0.5, rets={"ret_20d": 0.05}),
        _fr(score=-0.5, rets={"ret_20d": -0.05}),
        _fr(score=0.5, rets={"ret_20d": 0.05}),
    ]
    out = summarize_factor(recs, return_field="ret_20d")
    assert out["n_paired"] == 3
    assert out["n_missing_score"] == 2  # None + data_available=False
    assert out["n_missing_return"] == 1


def test_summarize_factor_long_short_none_when_only_one_side():
    recs = [
        _fr(score=0.8, rets={"ret_20d": 0.05}),
        _fr(score=0.5, rets={"ret_20d": 0.03}),
        _fr(score=0.3, rets={"ret_20d": 0.02}),
    ]
    out = summarize_factor(recs, return_field="ret_20d")
    assert out["long_short_spread"] is None
    assert out["mean_ret_when_bearish"] is None


def test_summarize_factor_uses_benchmark_adjusted_when_requested():
    recs = [
        _fr(
            score=0.8,
            rets={"ret_20d": 0.10},
            adj_rets={"ret_20d": 0.02},
        ),
        _fr(
            score=-0.8,
            rets={"ret_20d": 0.05},  # raw is positive but bearish on factor
            adj_rets={"ret_20d": -0.03},
        ),
    ]
    out_raw = summarize_factor(recs, return_field="ret_20d", use_benchmark_adjusted=False)
    out_adj = summarize_factor(recs, return_field="ret_20d", use_benchmark_adjusted=True)
    # Raw long-short: 0.10 - 0.05 = 0.05; adj long-short: 0.02 - (-0.03) = 0.05.
    # Same number here, but verify they came from different sources.
    assert out_raw["mean_ret_when_bullish"] == pytest.approx(0.10)
    assert out_adj["mean_ret_when_bullish"] == pytest.approx(0.02)
    assert out_adj["mean_ret_when_bearish"] == pytest.approx(-0.03)


def test_explode_records_to_factors_groups_by_factor_name():
    rec = BacktestRecord(
        symbol="X",
        as_of_date="2026-01-01",
        direction="bullish",
        confidence=0.7,
        composite_score=0.4,
        forward_returns={"ret_20d": 0.05},
        factor_scores=[
            {
                "factor": "trend_sma20",
                "pillar": "technical",
                "score": 0.7,
                "weighted_score": 0.056,
                "bucket": "bullish",
                "data_available": True,
            },
            {
                "factor": "fund_growth",
                "pillar": "fundamental",
                "score": -0.3,
                "weighted_score": -0.024,
                "bucket": "bearish",
                "data_available": True,
            },
        ],
    )
    out = explode_records_to_factors([rec])
    assert set(out.keys()) == {"trend_sma20", "fund_growth"}
    trend = out["trend_sma20"][0]
    assert trend.score == 0.7
    assert trend.pillar == "technical"
    assert trend.forward_returns == {"ret_20d": 0.05}


def test_explode_records_to_factors_skips_records_without_scores():
    recs = [
        BacktestRecord(
            symbol="X", as_of_date="2026-01-01", direction="neutral",
            confidence=0.5, composite_score=0.0, factor_scores=None,
        ),
        BacktestRecord(
            symbol="Y", as_of_date="2026-01-08", direction="bullish",
            confidence=0.7, composite_score=0.3, factor_scores=[
                {"factor": "f1", "pillar": "p1", "score": 0.5,
                 "weighted_score": 0.04, "bucket": "bullish",
                 "data_available": True},
            ],
        ),
    ]
    out = explode_records_to_factors(recs)
    assert list(out.keys()) == ["f1"]


def test_summarize_factors_emits_per_horizon_stats():
    recs_by_factor = {
        "f1": [
            _fr(score=0.7, rets={"ret_5d": 0.02, "ret_20d": 0.05}),
            _fr(score=-0.7, rets={"ret_5d": -0.01, "ret_20d": -0.03}),
            _fr(score=0.7, rets={"ret_5d": 0.03, "ret_20d": 0.06}),
        ]
    }
    out = summarize_factors(
        recs_by_factor, return_fields=["ret_5d", "ret_20d"]
    )
    assert set(out["f1"]["by_horizon"].keys()) == {"ret_5d", "ret_20d"}
    assert out["f1"]["total_observations"] == 3
    # Spearman should be positive for both horizons (score & return co-move).
    assert out["f1"]["by_horizon"]["ret_20d"]["spearman_ic"] > 0


# ---------- per-ticker stratification ----------


def test_summarize_factors_by_ticker_separates_tickers():
    # Build a factor that's strong on AAA (positive IC) and weak on BBB.
    recs = []
    for i, score in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0, 0.7, -0.7, 0.2, -0.2, 0.0]):
        recs.append(_fr(
            symbol="AAA",
            score=score,
            as_of=f"2026-01-{i + 1:02d}",
            rets={"ret_20d": score * 0.10},  # perfectly correlated
        ))
    # BBB: scores random vs returns → near-zero IC.
    bbb_scores = [0.5, -0.5, 0.3, -0.3, 0.7, -0.7, 0.1, -0.1, 0.2, -0.2]
    bbb_rets = [0.02, 0.01, -0.03, 0.04, -0.01, 0.02, 0.05, -0.02, 0.01, 0.0]
    for i, (s, r) in enumerate(zip(bbb_scores, bbb_rets)):
        recs.append(_fr(
            symbol="BBB",
            score=s,
            as_of=f"2026-02-{i + 1:02d}",
            rets={"ret_20d": r},
        ))
    out = summarize_factors_by_ticker(
        {"f1": recs}, return_fields=["ret_20d"], min_obs_per_ticker=5
    )
    block = out["f1"]["by_horizon"]["ret_20d"]
    assert block["n_tickers_evaluated"] == 2
    assert "AAA" in block["per_ticker_ic"]
    assert "BBB" in block["per_ticker_ic"]
    # AAA should have IC ≈ +1.0; BBB much lower magnitude.
    assert block["per_ticker_ic"]["AAA"] == pytest.approx(1.0, abs=1e-6)
    assert abs(block["per_ticker_ic"]["BBB"]) < abs(block["per_ticker_ic"]["AAA"])


def test_summarize_factors_by_ticker_respects_min_obs_threshold():
    # Only 3 records per ticker → below default 8 threshold → no tickers evaluated.
    recs = [
        _fr(symbol="A", score=0.5, rets={"ret_20d": 0.05}),
        _fr(symbol="A", score=-0.5, rets={"ret_20d": -0.05}),
        _fr(symbol="A", score=0.0, rets={"ret_20d": 0.0}),
    ]
    out = summarize_factors_by_ticker(
        {"f1": recs}, return_fields=["ret_20d"], min_obs_per_ticker=8
    )
    block = out["f1"]["by_horizon"]["ret_20d"]
    assert block["n_tickers_evaluated"] == 0
    assert block["median_ic_across_tickers"] is None
    assert block["consistency_pct"] is None


def test_summarize_factors_by_ticker_consistency_unanimous():
    # Three tickers, all with strong positive IC → consistency = 1.0.
    def make(symbol):
        return [
            _fr(symbol=symbol, score=s, as_of=f"2026-01-{i + 1:02d}",
                rets={"ret_20d": s * 0.1})
            for i, s in enumerate([-1, -0.5, 0, 0.5, 1.0, 0.7, -0.7, 0.3])
        ]
    recs = make("A") + make("B") + make("C")
    out = summarize_factors_by_ticker(
        {"f1": recs}, return_fields=["ret_20d"], min_obs_per_ticker=5
    )
    block = out["f1"]["by_horizon"]["ret_20d"]
    assert block["n_tickers_evaluated"] == 3
    assert block["consistency_pct"] == 1.0
    assert block["median_ic_across_tickers"] > 0


# ---------- counterfactual rebuild ----------


def _full_rec(symbol="X", as_of="2026-01-01", factor_scores=None, rets=None):
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction="neutral",
        confidence=0.5,
        composite_score=0.0,
        forward_returns=rets or {},
        factor_scores=factor_scores or [],
    )


def test_rebuild_records_with_weights_flips_direction_via_negative_weight():
    rec = _full_rec(
        factor_scores=[
            {"factor": "f_positive", "pillar": "p", "score": 1.0,
             "weight": 0.5, "weighted_score": 0.5, "data_available": True,
             "bucket": "bullish", "rationale": ""},
            {"factor": "f_contrarian", "pillar": "p", "score": 1.0,
             "weight": 0.5, "weighted_score": 0.5, "data_available": True,
             "bucket": "bullish", "rationale": ""},
        ],
        rets={"ret_20d": 0.05},
    )
    # Original (both positive weights): composite +1.0 → bullish.
    rebuilt_pos = rebuild_records_with_weights(
        [rec], weights={"f_positive": 1.0, "f_contrarian": 1.0}
    )
    assert rebuilt_pos[0].direction == "bullish"
    assert rebuilt_pos[0].composite_score == pytest.approx(1.0)
    # Flip contrarian to negative → composite 0 → neutral.
    rebuilt_flipped = rebuild_records_with_weights(
        [rec], weights={"f_positive": 1.0, "f_contrarian": -1.0}
    )
    assert rebuilt_flipped[0].direction == "neutral"
    assert rebuilt_flipped[0].composite_score == pytest.approx(0.0, abs=1e-6)


def test_rebuild_records_with_weights_drops_factors_not_in_weights():
    rec = _full_rec(
        factor_scores=[
            {"factor": "kept", "pillar": "p", "score": 0.8,
             "weight": 0.5, "weighted_score": 0.4, "data_available": True,
             "bucket": "bullish", "rationale": ""},
            {"factor": "dropped", "pillar": "p", "score": -0.8,
             "weight": 0.5, "weighted_score": -0.4, "data_available": True,
             "bucket": "bearish", "rationale": ""},
        ],
    )
    rebuilt = rebuild_records_with_weights([rec], weights={"kept": 1.0})
    # `dropped` is excluded → composite = score of `kept` = 0.8.
    assert rebuilt[0].composite_score == pytest.approx(0.8, abs=1e-4)
    assert rebuilt[0].direction == "bullish"


def test_rebuild_records_with_weights_skips_unavailable_factors():
    rec = _full_rec(
        factor_scores=[
            {"factor": "live", "pillar": "p", "score": 0.6,
             "weight": 0.5, "weighted_score": 0.3, "data_available": True,
             "bucket": "bullish", "rationale": ""},
            {"factor": "missing", "pillar": "p", "score": 1.0,
             "weight": 0.5, "weighted_score": 0.5, "data_available": False,
             "bucket": "bullish", "rationale": ""},
        ],
    )
    rebuilt = rebuild_records_with_weights(
        [rec], weights={"live": 1.0, "missing": 1.0}
    )
    # `missing` is excluded → composite from `live` alone = 0.6.
    assert rebuilt[0].composite_score == pytest.approx(0.6, abs=1e-4)


def test_ic_signed_weights_keeps_only_strong_signals():
    factor_summary = {
        "strong_pos": {
            "by_horizon": {
                "ret_20d": {"spearman_ic": 0.30, "n_paired": 100}
            }
        },
        "strong_neg": {
            "by_horizon": {
                "ret_20d": {"spearman_ic": -0.20, "n_paired": 100}
            }
        },
        "weak": {
            "by_horizon": {
                "ret_20d": {"spearman_ic": 0.02, "n_paired": 100}
            }
        },
        "small_n": {
            "by_horizon": {
                "ret_20d": {"spearman_ic": 0.40, "n_paired": 5}
            }
        },
        "no_ic": {
            "by_horizon": {
                "ret_20d": {"spearman_ic": None, "n_paired": 100}
            }
        },
    }
    weights = ic_signed_weights(factor_summary, horizon="ret_20d", min_abs_ic=0.05, min_n=50)
    assert set(weights.keys()) == {"strong_pos", "strong_neg"}
    assert weights["strong_pos"] == 0.30
    assert weights["strong_neg"] == -0.20


# ---------- direction-threshold sweep ----------


def _rec_with_composite(symbol, as_of, composite, rets=None):
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction="neutral",
        confidence=0.5,
        composite_score=composite,
        forward_returns=rets or {},
    )


def test_sweep_direction_threshold_uses_as_emitted_composites_when_no_weights():
    """If weights=None the sweep reclassifies direction directly from
    composite_score rather than rebuilding from factor_scores. Verify
    the bucket counts match what `direction_for_composite` would say."""
    recs = [
        _rec_with_composite("A", "2026-01-01", 0.30),   # bull @ 0.10/0.15/0.20/0.25
        _rec_with_composite("B", "2026-01-08", 0.18),   # bull @ 0.10/0.15, neutral @ 0.20+
        _rec_with_composite("C", "2026-01-15", 0.12),   # bull @ 0.10, neutral @ 0.15+
        _rec_with_composite("D", "2026-01-22", 0.05),   # neutral everywhere
        _rec_with_composite("E", "2026-01-29", -0.18),  # bear @ 0.10/0.15
        _rec_with_composite("F", "2026-02-05", -0.30),  # bear everywhere
    ]
    out = sweep_direction_threshold(
        recs, weights=None, thresholds=[0.10, 0.15, 0.20],
        return_fields=["ret_20d"],
    )
    by_thr = out["by_threshold"]
    assert by_thr["0.100"]["counts"] == {"bullish": 3, "bearish": 2, "neutral": 1}
    assert by_thr["0.150"]["counts"] == {"bullish": 2, "bearish": 2, "neutral": 2}
    assert by_thr["0.200"]["counts"] == {"bullish": 1, "bearish": 1, "neutral": 4}


def test_sweep_direction_threshold_under_weights_rebuilds():
    """If weights is provided, sweep rebuilds the composite from
    factor_scores first, then thresholds. Verify directions reflect
    the rebuilt composite, not the saved one."""
    rec = BacktestRecord(
        symbol="X",
        as_of_date="2026-01-01",
        direction="neutral",
        confidence=0.5,
        composite_score=0.99,  # ← stale saved composite, should be ignored
        forward_returns={"ret_20d": 0.05},
        factor_scores=[
            {"factor": "f1", "pillar": "p", "score": -0.4,
             "weight": 0.0, "weighted_score": 0.0, "data_available": True,
             "bucket": "bearish", "rationale": ""},
        ],
    )
    out = sweep_direction_threshold(
        [rec], weights={"f1": 1.0}, thresholds=[0.30, 0.50],
        return_fields=["ret_20d"],
    )
    # Rebuilt composite = -0.4 (the single factor under unit weight).
    # At thr=0.30 → bearish; at thr=0.50 → neutral.
    assert out["by_threshold"]["0.300"]["counts"]["bearish"] == 1
    assert out["by_threshold"]["0.500"]["counts"]["neutral"] == 1


def test_sweep_direction_threshold_direction_conditional_stats():
    """Per-threshold, per-horizon stats must respect the direction
    bucketing induced by that threshold."""
    recs = [
        _rec_with_composite("A", "2026-01-01", 0.50, rets={"ret_20d": 0.10}),
        _rec_with_composite("B", "2026-01-08", 0.30, rets={"ret_20d": 0.05}),
        _rec_with_composite("C", "2026-01-15", 0.10, rets={"ret_20d": -0.03}),
        _rec_with_composite("D", "2026-01-22", -0.40, rets={"ret_20d": -0.08}),
    ]
    out = sweep_direction_threshold(
        recs, weights=None, thresholds=[0.15], return_fields=["ret_20d"]
    )
    block = out["by_threshold"]["0.150"]["by_horizon"]["ret_20d"]["by_direction"]
    # A, B → bullish (both positive returns → 100% bullish hit)
    assert block["bullish"]["count_with_return"] == 2
    assert block["bullish"]["hit_rate"] == 1.0
    # C → neutral (composite=0.10 < 0.15)
    assert block["neutral"]["count_with_return"] == 1
    # D → bearish, hit
    assert block["bearish"]["count_with_return"] == 1
    assert block["bearish"]["hit_rate"] == 1.0


def test_recommend_direction_threshold_picks_highest_hit_above_min_n():
    sweep = {
        "by_threshold": {
            "0.100": {
                "counts": {"bullish": 100, "bearish": 20, "neutral": 30},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 0.55, "count_with_return": 100},
                        "bearish": {"hit_rate": 0.30, "count_with_return": 20},
                        "neutral": {"hit_rate": 0.10, "count_with_return": 30},
                    }}
                },
            },
            "0.150": {
                "counts": {"bullish": 80, "bearish": 15, "neutral": 55},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 0.65, "count_with_return": 80},
                        "bearish": {"hit_rate": 0.40, "count_with_return": 15},
                        "neutral": {"hit_rate": 0.12, "count_with_return": 55},
                    }}
                },
            },
            "0.200": {
                "counts": {"bullish": 50, "bearish": 8, "neutral": 92},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 0.72, "count_with_return": 50},
                        "bearish": {"hit_rate": 0.50, "count_with_return": 8},
                        "neutral": {"hit_rate": 0.10, "count_with_return": 92},
                    }}
                },
            },
            "0.300": {
                "counts": {"bullish": 10, "bearish": 3, "neutral": 137},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 0.90, "count_with_return": 10},
                        "bearish": {"hit_rate": 0.67, "count_with_return": 3},
                        "neutral": {"hit_rate": 0.05, "count_with_return": 137},
                    }}
                },
            },
        }
    }
    rec = recommend_direction_threshold(
        sweep, horizon="ret_20d", min_n_bullish=30, min_n_bearish=5
    )
    # 0.300 has best bullish hit (0.90) but only n=10 < 30 → drops out.
    # 0.200 has next best (0.72) at n=50 ≥ 30 → wins.
    assert rec["bullish_pick"]["threshold"] == 0.200
    assert rec["bullish_pick"]["hit_rate"] == 0.72
    # Bearish: 0.300 has n=3 < 5 → drops. 0.200 has hit 0.50 at n=8 ≥ 5 → wins.
    assert rec["bearish_pick"]["threshold"] == 0.200


def test_recommend_returns_none_when_no_threshold_meets_minimum():
    sweep = {
        "by_threshold": {
            "0.500": {
                "counts": {"bullish": 5, "bearish": 1, "neutral": 200},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 1.0, "count_with_return": 5},
                        "bearish": {"hit_rate": 1.0, "count_with_return": 1},
                        "neutral": {"hit_rate": 0.0, "count_with_return": 200},
                    }}
                },
            },
        }
    }
    rec = recommend_direction_threshold(
        sweep, horizon="ret_20d", min_n_bullish=30, min_n_bearish=5
    )
    assert rec["bullish_pick"] is None
    assert rec["bearish_pick"] is None


def test_render_threshold_sweep_markdown_includes_all_thresholds():
    sweep = {
        "n_records": 4,
        "weights_supplied": False,
        "thresholds": [0.10, 0.20],
        "neutral_band": 0.02,
        "use_benchmark_adjusted": False,
        "by_threshold": {
            "0.100": {
                "counts": {"bullish": 3, "bearish": 1, "neutral": 0},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 0.67, "count_with_return": 3,
                                    "mean_forward_return": 0.05},
                        "bearish": {"hit_rate": 1.0, "count_with_return": 1,
                                    "mean_forward_return": -0.03},
                        "neutral": {"hit_rate": None, "count_with_return": 0,
                                    "mean_forward_return": None},
                    }}
                },
            },
            "0.200": {
                "counts": {"bullish": 1, "bearish": 1, "neutral": 2},
                "by_horizon": {
                    "ret_20d": {"by_direction": {
                        "bullish": {"hit_rate": 1.0, "count_with_return": 1,
                                    "mean_forward_return": 0.10},
                        "bearish": {"hit_rate": 1.0, "count_with_return": 1,
                                    "mean_forward_return": -0.05},
                        "neutral": {"hit_rate": 0.5, "count_with_return": 2,
                                    "mean_forward_return": 0.01},
                    }}
                },
            },
        },
    }
    md = render_threshold_sweep_markdown(sweep)
    assert "0.100" in md and "0.200" in md
    assert "Horizon `ret_20d`" in md
    assert "Bull hit" in md and "Bear hit" in md


# ---------- asymmetric direction-threshold sweep (Section 15) ----------


def test_reclassify_direction_asymmetric_kwargs():
    recs = [
        _rec_with_composite("A", "2026-01-01", 0.20),   # bull if bt<=0.20
        _rec_with_composite("B", "2026-01-08", 0.08),   # bull only if bt<=0.08
        _rec_with_composite("C", "2026-01-15", -0.18),  # bear if br<=0.18
        _rec_with_composite("D", "2026-01-22", -0.35),  # bear if br<=0.35
    ]
    out = _reclassify_direction_via_sweep(
        recs, bullish_threshold=0.15, bearish_threshold=0.30
    )
    # bt=0.15, br=0.30 → A bull, B neutral, C neutral (|-0.18|<0.30), D bear.
    assert [r.direction for r in out] == [
        "bullish", "neutral", "neutral", "bearish"
    ]


def test_reclassify_direction_requires_at_least_one_threshold():
    from tradingagents.analysis_only.backtest import _reclassify_direction
    with pytest.raises(ValueError):
        _reclassify_direction([_rec_with_composite("X", "2026-01-01", 0.0)])


def test_rebuild_records_with_weights_honors_asymmetric_thresholds():
    """When asymmetric thresholds are supplied, rebuild uses them instead
    of the symmetric direction_threshold."""
    rec = BacktestRecord(
        symbol="X",
        as_of_date="2026-01-01",
        direction="neutral",
        confidence=0.5,
        composite_score=0.0,
        forward_returns={"ret_20d": 0.0},
        factor_scores=[
            {"factor": "f1", "pillar": "p", "score": -0.25,
             "weight": 0.0, "weighted_score": 0.0, "data_available": True,
             "bucket": "bearish", "rationale": ""},
        ],
    )
    # symmetric @ 0.15 → composite -0.25 is bearish.
    out_sym = rebuild_records_with_weights(
        [rec], weights={"f1": 1.0}, direction_threshold=0.15
    )
    assert out_sym[0].direction == "bearish"
    # asymmetric: bullish=0.15, bearish=0.30 → -0.25 falls inside neutral.
    out_asym = rebuild_records_with_weights(
        [rec], weights={"f1": 1.0},
        bullish_threshold=0.15, bearish_threshold=0.30,
    )
    assert out_asym[0].direction == "neutral"


def test_sweep_asymmetric_produces_one_cell_per_pair():
    recs = [
        _rec_with_composite("A", "2026-01-01", 0.20, rets={"ret_20d": 0.05}),
        _rec_with_composite("B", "2026-01-08", -0.20, rets={"ret_20d": -0.05}),
        _rec_with_composite("C", "2026-01-15", 0.0, rets={"ret_20d": 0.01}),
    ]
    sweep = sweep_direction_threshold_asymmetric(
        recs, weights=None,
        bullish_thresholds=[0.10, 0.25],
        bearish_thresholds=[0.10, 0.25],
        return_fields=["ret_20d"],
    )
    assert set(sweep["by_cell"].keys()) == {
        "0.100|0.100", "0.100|0.250", "0.250|0.100", "0.250|0.250",
    }
    # bt=0.10 / br=0.10 → A bull, B bear, C neutral.
    c1 = sweep["by_cell"]["0.100|0.100"]["counts"]
    assert c1 == {"bullish": 1, "bearish": 1, "neutral": 1}
    # bt=0.25 / br=0.25 → all three are neutral.
    c4 = sweep["by_cell"]["0.250|0.250"]["counts"]
    assert c4 == {"bullish": 0, "bearish": 0, "neutral": 3}


def test_sweep_asymmetric_bullish_counts_depend_only_on_bullish_threshold():
    """Cells sharing the same bullish_threshold must report identical
    bullish counts and bullish hit-rates regardless of bearish_threshold."""
    recs = [
        _rec_with_composite("A", "2026-01-01", 0.20, rets={"ret_20d": 0.10}),
        _rec_with_composite("B", "2026-01-08", 0.10, rets={"ret_20d": -0.05}),
        _rec_with_composite("C", "2026-01-15", -0.30, rets={"ret_20d": -0.10}),
    ]
    sweep = sweep_direction_threshold_asymmetric(
        recs, weights=None,
        bullish_thresholds=[0.15],
        bearish_thresholds=[0.10, 0.25, 0.40],
        return_fields=["ret_20d"],
    )
    bullish_counts = {
        k: v["counts"]["bullish"] for k, v in sweep["by_cell"].items()
    }
    bullish_hits = {
        k: v["by_horizon"]["ret_20d"]["by_direction"]["bullish"]["hit_rate"]
        for k, v in sweep["by_cell"].items()
    }
    assert len(set(bullish_counts.values())) == 1
    assert len(set(bullish_hits.values())) == 1


def test_recommend_asymmetric_returns_none_when_no_bearish_clears_floor():
    """The headline expected outcome on the current corpus: every cell
    has bearish hit-rate < 0.50 → bearish_pick is None."""
    sweep = {
        "by_cell": {
            "0.150|0.100": {
                "bullish_threshold": 0.15, "bearish_threshold": 0.10,
                "counts": {"bullish": 50, "bearish": 20, "neutral": 100},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.70, "count_with_return": 50},
                    "bearish": {"hit_rate": 0.30, "count_with_return": 20},
                    "neutral": {"hit_rate": 0.15, "count_with_return": 100},
                }}},
            },
            "0.150|0.300": {
                "bullish_threshold": 0.15, "bearish_threshold": 0.30,
                "counts": {"bullish": 50, "bearish": 8, "neutral": 112},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.70, "count_with_return": 50},
                    "bearish": {"hit_rate": 0.25, "count_with_return": 8},
                    "neutral": {"hit_rate": 0.20, "count_with_return": 112},
                }}},
            },
        }
    }
    rec = recommend_asymmetric_thresholds(
        sweep, horizon="ret_20d",
        min_n_bullish=30, min_n_bearish=5,
        bearish_precision_floor=0.50,
    )
    assert rec["bullish_pick"] is not None
    assert rec["bullish_pick"]["threshold"] == 0.15
    assert rec["bearish_pick"] is None
    assert rec["bearish_precision_floor"] == 0.50


def test_recommend_asymmetric_picks_qualifying_bearish_when_one_exists():
    sweep = {
        "by_cell": {
            "0.150|0.100": {
                "bullish_threshold": 0.15, "bearish_threshold": 0.10,
                "counts": {"bullish": 50, "bearish": 20, "neutral": 100},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.70, "count_with_return": 50},
                    "bearish": {"hit_rate": 0.35, "count_with_return": 20},
                    "neutral": {"hit_rate": 0.15, "count_with_return": 100},
                }}},
            },
            "0.150|0.400": {
                "bullish_threshold": 0.15, "bearish_threshold": 0.40,
                "counts": {"bullish": 50, "bearish": 6, "neutral": 114},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.70, "count_with_return": 50},
                    "bearish": {"hit_rate": 0.67, "count_with_return": 6},
                    "neutral": {"hit_rate": 0.20, "count_with_return": 114},
                }}},
            },
        }
    }
    rec = recommend_asymmetric_thresholds(
        sweep, horizon="ret_20d",
        min_n_bullish=30, min_n_bearish=5,
        bearish_precision_floor=0.50,
    )
    assert rec["bearish_pick"] is not None
    assert rec["bearish_pick"]["threshold"] == 0.40
    assert rec["bearish_pick"]["hit_rate"] == 0.67


def test_render_asymmetric_sweep_markdown_contains_grid_axes():
    sweep = {
        "n_records": 3,
        "weights_supplied": True,
        "neutral_band": 0.02,
        "by_cell": {
            "0.100|0.100": {
                "bullish_threshold": 0.10, "bearish_threshold": 0.10,
                "counts": {"bullish": 2, "bearish": 1, "neutral": 0},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.50, "count_with_return": 2},
                    "bearish": {"hit_rate": 1.00, "count_with_return": 1},
                    "neutral": {"hit_rate": None, "count_with_return": 0},
                }}},
            },
            "0.100|0.300": {
                "bullish_threshold": 0.10, "bearish_threshold": 0.30,
                "counts": {"bullish": 2, "bearish": 0, "neutral": 1},
                "by_horizon": {"ret_20d": {"by_direction": {
                    "bullish": {"hit_rate": 0.50, "count_with_return": 2},
                    "bearish": {"hit_rate": None, "count_with_return": 0},
                    "neutral": {"hit_rate": 0.0, "count_with_return": 1},
                }}},
            },
        },
    }
    md = render_asymmetric_sweep_markdown(sweep)
    assert "bear=0.100" in md and "bear=0.300" in md
    assert "**0.100**" in md
    assert "Horizon `ret_20d`" in md


def _reclassify_direction_via_sweep(recs, *, bullish_threshold, bearish_threshold):
    """Tiny shim so tests can exercise _reclassify_direction via its
    asymmetric kwargs without importing the private name directly."""
    sweep = sweep_direction_threshold_asymmetric(
        recs, weights=None,
        bullish_thresholds=[bullish_threshold],
        bearish_thresholds=[bearish_threshold],
        return_fields=["ret_20d"],
    )
    key = f"{bullish_threshold:.3f}|{bearish_threshold:.3f}"
    counts = sweep["by_cell"][key]["counts"]
    # Reconstruct individual directions in record order from the underlying
    # private helper, since the sweep only returns aggregates.
    from tradingagents.analysis_only.backtest import _reclassify_direction
    return _reclassify_direction(
        recs,
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
    )


# ---------- filter_by_date_range (used for train/test split) ----------


def test_filter_by_date_range_inclusive_bounds():
    from backtest import filter_by_date_range

    recs = [
        _full_rec(as_of="2026-01-01"),
        _full_rec(as_of="2026-02-15"),
        _full_rec(as_of="2026-03-31"),
        _full_rec(as_of="2026-05-22"),
    ]
    only_lower = filter_by_date_range(recs, date_from="2026-02-15")
    assert [r.as_of_date for r in only_lower] == [
        "2026-02-15", "2026-03-31", "2026-05-22"
    ]
    only_upper = filter_by_date_range(recs, date_to="2026-03-31")
    assert [r.as_of_date for r in only_upper] == [
        "2026-01-01", "2026-02-15", "2026-03-31"
    ]
    both = filter_by_date_range(recs, date_from="2026-02-15", date_to="2026-03-31")
    assert [r.as_of_date for r in both] == ["2026-02-15", "2026-03-31"]
    no_filter = filter_by_date_range(recs)
    assert no_filter == recs


# ---------- equivalence: rebuild_records_with_weights ↔ scoring.compute_composite ----------


def test_rebuild_matches_compute_composite_for_pure_positive_weights():
    """The counterfactual rebuild must produce the same composite as the
    pipeline's `compute_composite` given identical inputs. Without this,
    every `--weights-override` lift number is a fiction."""
    from tradingagents.analysis_only.scoring import (
        compute_composite,
        resolve_factor_weights,
    )

    # Hand-built factor scores spanning all three buckets and missing data.
    factor_scores = [
        {"factor": "f_a", "pillar": "p1", "score": 0.8, "weight": 0.0,
         "weighted_score": 0.0, "data_available": True, "bucket": "bullish"},
        {"factor": "f_b", "pillar": "p1", "score": -0.5, "weight": 0.0,
         "weighted_score": 0.0, "data_available": True, "bucket": "bearish"},
        {"factor": "f_c", "pillar": "p2", "score": 0.3, "weight": 0.0,
         "weighted_score": 0.0, "data_available": True, "bucket": "bullish"},
        {"factor": "f_missing", "pillar": "p2", "score": 1.0, "weight": 0.0,
         "weighted_score": 0.0, "data_available": False, "bucket": "bullish"},
    ]
    # Raw v1-style weights (not normalized).
    raw_weights = {"f_a": 0.10, "f_b": 0.20, "f_c": 0.30, "f_missing": 0.05}

    # Path A: rebuild_records_with_weights.
    rec = BacktestRecord(
        symbol="X", as_of_date="2026-01-01", direction="neutral",
        confidence=0.5, composite_score=0.0,
        factor_scores=factor_scores,
    )
    rebuilt = rebuild_records_with_weights([rec], weights=raw_weights)
    rebuilt_composite = rebuilt[0].composite_score

    # Path B: emulate the pipeline. The pipeline normalizes weights via
    # resolve_factor_weights, then per row computes `weighted_score =
    # score * weight`, then calls compute_composite which divides by
    # active weight. To emulate, we need the same normalized weights.
    abs_total = sum(abs(v) for v in raw_weights.values())
    normalized = {k: v / abs_total for k, v in raw_weights.items()}
    rows_for_pipeline = []
    for f in factor_scores:
        w = normalized.get(f["factor"], 0.0)
        score = f["score"]
        rows_for_pipeline.append({
            **f,
            "weight": w,
            "weighted_score": score * w,
        })
    pipeline_result = compute_composite(
        rows_for_pipeline, weights=normalized
    )
    pipeline_composite = pipeline_result["composite_score"]

    # They must agree to within rounding (both round to 4 decimals).
    assert rebuilt_composite == pytest.approx(pipeline_composite, abs=1e-4), (
        f"Equivalence broken: rebuild={rebuilt_composite}, "
        f"compute_composite={pipeline_composite}"
    )


def test_rebuild_matches_compute_composite_on_real_v1_weights():
    """End-to-end equivalence check using the committed v1 weight vector
    against a realistic 22-factor row set. Guards against silent
    divergence after future changes to either module."""
    import json
    from pathlib import Path

    from tradingagents.analysis_only.scoring import compute_composite

    weights_path = Path(__file__).resolve().parents[2] / "configs" / "proposed_weights_v1.json"
    raw = json.loads(weights_path.read_text())
    v1_weights = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    # Mock a realistic row set: every v1 factor with a varied score.
    rng_scores = [
        0.5, -0.3, 0.7, 0.0, -0.8, 0.2, -0.5, 1.0, -1.0, 0.4,
        -0.7, 0.6, -0.2, 0.3, -0.4, 0.8, -0.6, 0.1, -0.9, 0.5,
        -0.1, 0.9, 0.0,
    ]
    factor_scores = [
        {
            "factor": factor,
            "pillar": "x",
            "score": rng_scores[i % len(rng_scores)],
            "weight": 0.0,
            "weighted_score": 0.0,
            "data_available": i % 7 != 0,  # mark every 7th row missing
            "bucket": "x",
        }
        for i, factor in enumerate(v1_weights.keys())
    ]

    # Path A: rebuild.
    rec = BacktestRecord(
        symbol="X", as_of_date="2026-01-01", direction="neutral",
        confidence=0.5, composite_score=0.0,
        factor_scores=factor_scores,
    )
    rebuilt = rebuild_records_with_weights([rec], weights=v1_weights)
    rebuilt_composite = rebuilt[0].composite_score
    rebuilt_direction = rebuilt[0].direction

    # Path B: emulate pipeline → compute_composite.
    abs_total = sum(abs(v) for v in v1_weights.values()) or 1.0
    normalized = {k: v / abs_total for k, v in v1_weights.items()}
    rows = []
    for f in factor_scores:
        w = normalized.get(f["factor"], 0.0)
        rows.append({**f, "weight": w, "weighted_score": f["score"] * w})
    pipeline_result = compute_composite(rows, weights=normalized)
    pipeline_composite = pipeline_result["composite_score"]

    assert rebuilt_composite == pytest.approx(pipeline_composite, abs=1e-4), (
        f"Equivalence broken on real v1 weights: rebuild={rebuilt_composite}, "
        f"compute_composite={pipeline_composite}, direction={rebuilt_direction}"
    )


def test_render_factor_by_ticker_markdown_sorts_by_abs_ic():
    summary = {
        "weak": {
            "pillar": "x",
            "by_horizon": {
                "ret_20d": {
                    "n_tickers_evaluated": 5,
                    "median_ic_across_tickers": 0.02,
                    "mean_ic_across_tickers": 0.01,
                    "consistency_pct": 0.6,
                    "per_ticker_ic": {},
                }
            },
        },
        "strong": {
            "pillar": "y",
            "by_horizon": {
                "ret_20d": {
                    "n_tickers_evaluated": 8,
                    "median_ic_across_tickers": -0.4,
                    "mean_ic_across_tickers": -0.35,
                    "consistency_pct": 0.875,
                    "per_ticker_ic": {},
                }
            },
        },
    }
    md = render_factor_by_ticker_markdown(summary, return_fields=["ret_20d"])
    assert md.index("strong") < md.index("weak")
    assert "Consistency" in md


def test_render_factor_summary_markdown_sorts_by_abs_ic():
    summary = {
        "weak_factor": {
            "pillar": "x",
            "by_horizon": {
                "ret_20d": {
                    "n_paired": 10, "spearman_ic": 0.05, "pearson_ic": 0.04,
                    "n_bullish_score": 5, "n_bearish_score": 5,
                    "mean_ret_when_bullish": 0.01, "mean_ret_when_bearish": 0.0,
                    "long_short_spread": 0.01,
                    "hit_rate_when_bullish": 0.5, "hit_rate_when_bearish": 0.5,
                },
            },
        },
        "strong_factor": {
            "pillar": "y",
            "by_horizon": {
                "ret_20d": {
                    "n_paired": 10, "spearman_ic": -0.40, "pearson_ic": -0.35,
                    "n_bullish_score": 5, "n_bearish_score": 5,
                    "mean_ret_when_bullish": -0.02, "mean_ret_when_bearish": 0.05,
                    "long_short_spread": -0.07,
                    "hit_rate_when_bullish": 0.2, "hit_rate_when_bearish": 0.1,
                },
            },
        },
    }
    md = render_factor_summary_markdown(summary, return_fields=["ret_20d"])
    # Strong factor row should appear before weak factor row.
    assert md.index("strong_factor") < md.index("weak_factor")


# ---------- walk-forward backtest (Phase 3) ----------


def _wf_record(symbol: str, as_of: str, factor_score: float, ret_20d: float):
    """Build a record with one factor; useful for walk-forward boundary tests."""
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction="neutral",
        confidence=0.5,
        composite_score=0.0,
        forward_returns={"ret_20d": ret_20d},
        factor_scores=[
            {
                "factor": "signal",
                "pillar": "technical",
                "score": factor_score,
                "weight": 1.0,
                "weighted_score": factor_score,
                "data_available": True,
                "bucket": "neutral",
            }
        ],
    )


def _weekly_dates(start: str, n_weeks: int) -> list[str]:
    from datetime import datetime, timedelta
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    return [(d0 + timedelta(days=7 * i)).isoformat() for i in range(n_weeks)]


def test_walk_forward_refit_dates_picks_correct_anchors():
    dates = _weekly_dates("2024-01-05", 60)
    recs = [_wf_record("A", d, 0.5, 0.01) for d in dates]
    anchors = walk_forward_refit_dates(
        recs, refit_freq_weeks=4, first_refit_after_weeks=26
    )
    assert anchors[0] == dates[26]
    if len(anchors) >= 2:
        from datetime import datetime
        a0 = datetime.strptime(anchors[0], "%Y-%m-%d").date()
        a1 = datetime.strptime(anchors[1], "%Y-%m-%d").date()
        assert (a1 - a0).days >= 28


def test_walk_forward_refit_dates_empty_when_corpus_too_short():
    dates = _weekly_dates("2024-01-05", 10)
    recs = [_wf_record("A", d, 0.5, 0.01) for d in dates]
    anchors = walk_forward_refit_dates(
        recs, refit_freq_weeks=4, first_refit_after_weeks=26
    )
    assert anchors == []


def test_split_train_score_respects_gap_boundary():
    dates = _weekly_dates("2024-01-05", 80)
    recs = [_wf_record("A", d, 0.5, 0.01) for d in dates]
    anchor = dates[40]
    train, score = split_train_score(
        recs,
        anchor_date=anchor,
        train_window_weeks=26,
        score_window_weeks=4,
        gap_weeks=4,
    )
    train_dates = sorted(r.as_of_date for r in train)
    score_dates = sorted(r.as_of_date for r in score)
    from datetime import datetime, timedelta
    a = datetime.strptime(anchor, "%Y-%m-%d").date()
    gap_boundary = (a - timedelta(weeks=4)).isoformat()
    assert all(d < gap_boundary for d in train_dates)
    assert all(d >= anchor for d in score_dates)
    assert all(d < (a + timedelta(weeks=4)).isoformat() for d in score_dates)
    # No overlap between train and score.
    assert set(train_dates).isdisjoint(set(score_dates))


def test_split_train_score_no_overlap_for_a_grid_of_anchors():
    """Sweep many anchors; train ∩ score must always be empty (regression
    test for the no-lookahead invariant)."""
    dates = _weekly_dates("2023-07-14", 150)
    recs = [_wf_record("A", d, 0.5, 0.01) for d in dates]
    anchors = walk_forward_refit_dates(
        recs, refit_freq_weeks=4, first_refit_after_weeks=26
    )
    assert anchors, "need anchors for this test"
    for anchor in anchors:
        train, score = split_train_score(
            recs,
            anchor_date=anchor,
            train_window_weeks=52,
            score_window_weeks=4,
            gap_weeks=4,
        )
        train_set = {r.as_of_date for r in train}
        score_set = {r.as_of_date for r in score}
        assert train_set.isdisjoint(score_set), (
            f"overlap at anchor={anchor}: {train_set & score_set}"
        )


def test_fit_weights_from_records_returns_signed_weight_above_min_ic():
    dates = _weekly_dates("2024-01-05", 40)
    recs = []
    for i in range(80):
        date_idx = i % len(dates)
        symbol = f"S{i % 5}"
        score = (i % 5) * 0.2 - 0.4
        ret = (i % 5) * 0.02 - 0.04
        recs.append(_wf_record(symbol, dates[date_idx], score, ret))
    weights = fit_weights_from_records(
        recs, horizon="ret_20d", min_abs_ic=0.05, min_n=10
    )
    assert weights.get("signal", 0.0) > 0


def test_walk_forward_backtest_only_scores_records_outside_gap():
    """Every record in the rebuilt output must come from inside a scoring
    window, never from training-only weeks or the gap region."""
    dates = _weekly_dates("2023-07-14", 130)
    recs = []
    for i, d in enumerate(dates):
        s = ((i % 7) - 3) / 3.0
        recs.append(_wf_record(f"S{i % 5}", d, s, s * 0.05))
    rebuilt, timeline = walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
        horizon="ret_20d",
        min_abs_ic=0.0,
        min_n=10,
    )
    if not timeline:
        return
    earliest_anchor = timeline[0].anchor_date
    assert all(r.as_of_date >= earliest_anchor for r in rebuilt)
    # Each rebuilt record must fall inside SOME step's scoring window.
    for r in rebuilt:
        in_window = any(
            step.anchor_date <= r.as_of_date < step.score_end_exclusive
            for step in timeline
        )
        assert in_window, (
            f"rebuilt record {r.as_of_date} outside any scoring window"
        )


def test_walk_forward_backtest_train_records_never_overlap_score_window():
    """Build a labelled signal at each date and ensure the timeline's
    train window for step N ends strictly before step N's anchor minus gap."""
    dates = _weekly_dates("2023-07-14", 130)
    recs = [_wf_record(f"S{i % 4}", d, 0.3, 0.02) for i, d in enumerate(dates)]
    _, timeline = walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
    )
    for step in timeline:
        assert step.train_end_exclusive <= step.anchor_date
        assert step.train_start <= step.train_end_exclusive
        # Gap of >= 4 weeks (28 days).
        from datetime import datetime
        gap_days = (
            datetime.strptime(step.anchor_date, "%Y-%m-%d")
            - datetime.strptime(step.train_end_exclusive, "%Y-%m-%d")
        ).days
        assert gap_days >= 28


# ---------- regime walk-forward (Phase 4) ----------


def _wf_record_with_regime(
    symbol: str, as_of: str, factor_score: float, ret_20d: float, regime: str,
):
    rec = _wf_record(symbol, as_of, factor_score, ret_20d)
    if regime == "trend_on":
        rec.market_context = {"spy_above_50dma": True, "vix_level": 15.0}
    elif regime == "chop":
        rec.market_context = {"spy_above_50dma": False, "vix_level": 25.0}
    else:
        rec.market_context = {}
    return rec


def test_regime_walk_forward_falls_back_to_global_when_below_threshold():
    dates = _weekly_dates("2023-07-14", 130)
    recs = []
    for i, d in enumerate(dates):
        regime = "trend_on" if i % 4 == 0 else "chop"
        recs.append(_wf_record_with_regime(
            f"S{i % 5}", d, 0.3, 0.02, regime,
        ))
    _, timeline = regime_walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=10,
        min_abs_ic=0.0,
        min_samples_per_regime=1000,
    )
    for step in timeline:
        assert step.regimes_used == []
        assert "trend_on" in step.regimes_fellback_to_global or "chop" in step.regimes_fellback_to_global


def test_regime_walk_forward_ships_regime_when_enough_samples():
    dates = _weekly_dates("2023-07-14", 130)
    recs = []
    for i, d in enumerate(dates):
        for sym_idx in range(12):
            score = ((sym_idx + i) % 5 - 2) * 0.4
            ret = score * 0.05
            recs.append(_wf_record_with_regime(
                f"S{sym_idx}", d, score, ret, "chop",
            ))
    _, timeline = regime_walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=10,
        min_abs_ic=0.0,
        min_samples_per_regime=100,
        require_regime_ic_ge_global=False,
    )
    assert timeline
    any_chop_used = any("chop" in step.regimes_used for step in timeline)
    assert any_chop_used


def test_regime_walk_forward_falls_back_when_ic_lift_too_small():
    dates = _weekly_dates("2023-07-14", 130)
    recs = []
    for i, d in enumerate(dates):
        for sym_idx in range(12):
            score = ((sym_idx + i) % 5 - 2) * 0.4
            ret = score * 0.05
            recs.append(_wf_record_with_regime(
                f"S{sym_idx}", d, score, ret, "chop",
            ))
    _, timeline = regime_walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=10,
        min_abs_ic=0.0,
        min_samples_per_regime=100,
        min_regime_ic_lift=0.5,
    )
    assert timeline
    assert any("chop" in step.regimes_fellback_to_global for step in timeline)
    assert any(
        "regime_ic_lift" in step.regime_skip_reasons.get("chop", "")
        for step in timeline
    )


def test_regime_walk_forward_only_ships_eligible_regimes():
    dates = _weekly_dates("2023-07-14", 130)
    recs = []
    for i, d in enumerate(dates):
        for sym_idx in range(12):
            regime = "trend_on" if sym_idx % 2 == 0 else "chop"
            score = ((sym_idx + i) % 5 - 2) * 0.4
            ret = score * 0.05
            recs.append(_wf_record_with_regime(
                f"S{sym_idx}", d, score, ret, regime,
            ))
    _, timeline = regime_walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=52,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=10,
        min_abs_ic=0.0,
        min_samples_per_regime=100,
        require_regime_ic_ge_global=False,
        eligible_regimes=["chop"],
    )
    assert timeline
    assert any("trend_on" in step.regimes_fellback_to_global for step in timeline)
    assert all("trend_on" not in step.regimes_used for step in timeline)
    assert any(
        step.regime_skip_reasons.get("trend_on") == "regime_not_eligible"
        for step in timeline
    )


def test_regime_walk_forward_timeline_serializes():
    dates = _weekly_dates("2023-07-14", 60)
    recs = [
        _wf_record_with_regime(f"S{i%5}", d, 0.3, 0.01, "trend_on")
        for i, d in enumerate(dates)
    ]
    _, timeline = regime_walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=26,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=5,
        min_abs_ic=0.0,
    )
    payload = regime_walk_forward_timeline_to_dict(timeline)
    if payload:
        first = payload[0]
        for k in (
            "anchor_date", "train_start", "train_end_exclusive",
            "score_end_exclusive", "n_train_total", "n_train_by_regime",
            "n_score_total", "n_score_by_regime", "global_weights",
            "global_ic", "regime_weights", "regime_ics",
            "regime_ic_lifts", "regime_skip_reasons", "regimes_used",
            "regimes_fellback_to_global",
        ):
            assert k in first
        import json as _json
        _json.dumps(payload)


def test_walk_forward_timeline_serializes_to_dict():
    dates = _weekly_dates("2023-07-14", 80)
    recs = [_wf_record(f"S{i % 4}", d, 0.4, 0.01) for i, d in enumerate(dates)]
    _, timeline = walk_forward_backtest(
        recs,
        refit_freq_weeks=4,
        train_window_weeks=26,
        gap_weeks=4,
        first_refit_after_weeks=26,
        min_n=10,
        min_abs_ic=0.0,
    )
    payload = walk_forward_timeline_to_dict(timeline)
    assert isinstance(payload, list)
    if payload:
        first = payload[0]
        for key in (
            "anchor_date",
            "train_start",
            "train_end_exclusive",
            "score_end_exclusive",
            "n_train",
            "n_score",
            "weights",
        ):
            assert key in first
        # Weights JSON-serializable.
        import json as _json
        _json.dumps(payload)


# ---------- Unit X4: compute_return_metrics ----------


def _rm_record(symbol, as_of, direction, ret, horizon="ret_60d"):
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction=direction,
        confidence=0.7,
        composite_score=0.3 if direction == "bullish" else (-0.3 if direction == "bearish" else 0.0),
        forward_returns={horizon: ret},
    )


def test_compute_return_metrics_empty_records():
    out = compute_return_metrics([], horizon="ret_60d")
    assert out["n_records"] == 0
    assert out["n_with_return"] == 0
    assert out["mean_return"] is None
    assert out["sharpe"] is None
    assert out["sortino"] is None
    assert out["profit_factor"] is None


def test_compute_return_metrics_basic_sharpe_math():
    # Six bullish records with known mean/stdev so we can hand-check.
    # rets = [0.10, 0.05, -0.02, 0.08, 0.03, 0.04]
    # mean = 0.04666..., stdev (sample) ~ 0.04274
    # raw Sharpe ~ 1.0918, annualized at 60d: * sqrt(252/60) ~ 2.049
    # final Sharpe ~ 2.237
    rets = [0.10, 0.05, -0.02, 0.08, 0.03, 0.04]
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", r)
        for i, r in enumerate(rets)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    assert out["n_with_return"] == 6
    assert out["mean_return"] == pytest.approx(0.046667, abs=1e-4)
    assert out["sharpe"] is not None
    # Sharpe should be positive and annualized to roughly 2.2.
    assert 1.5 < out["sharpe"] < 3.0
    # Annualization factor for 60-day horizon.
    assert out["annualization_factor"] == pytest.approx(2.0494, abs=1e-3)
    # Hit-rate for bullish records on these returns: 5 / 6 are > 0.
    assert out["hit_rate"] == pytest.approx(5 / 6, abs=1e-4)


def test_compute_return_metrics_profit_factor_and_dd():
    rets = [0.10, -0.05, 0.08, -0.02, 0.04]
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", r)
        for i, r in enumerate(rets)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    # profit factor = (0.10+0.08+0.04) / (0.05+0.02) = 0.22 / 0.07 ~ 3.143
    assert out["profit_factor"] == pytest.approx(3.1428, abs=1e-3)
    # Cumulative P&L (sum): 0.10, 0.05, 0.13, 0.11, 0.15.
    # Drawdown after 0.10 peak: -0.05 (cum 0.05 vs peak 0.10).
    # Drawdown after 0.13 peak: -0.02 (cum 0.11 vs peak 0.13).
    # Max DD is the larger magnitude: -0.05.
    assert out["max_drawdown"] == pytest.approx(-0.05, abs=1e-6)
    assert out["n_positive"] == 3
    assert out["n_negative"] == 2
    assert out["n_zero"] == 0


def test_compute_return_metrics_all_zero_returns():
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "neutral", 0.0)
        for i in range(5)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    assert out["n_with_return"] == 5
    assert out["mean_return"] == 0.0
    # Zero variance → Sharpe / Sortino undefined.
    assert out["sharpe"] is None
    assert out["sortino"] is None
    # No negative returns → profit factor undefined (avoid div by zero).
    assert out["profit_factor"] is None
    assert out["max_drawdown"] == 0.0


def test_compute_return_metrics_all_positive_no_profit_factor():
    # Profit factor needs at least one negative return.
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", 0.02 + i * 0.01)
        for i in range(5)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    assert out["profit_factor"] is None
    assert out["n_negative"] == 0
    assert out["sortino"] is None  # No downside returns.


def test_compute_return_metrics_missing_returns_skipped():
    # Some records have ret_60d, some don't (have ret_20d only).
    recs = [
        BacktestRecord(
            symbol="A", as_of_date="2025-01-01", direction="bullish",
            confidence=0.7, composite_score=0.3,
            forward_returns={"ret_60d": 0.05},
        ),
        BacktestRecord(
            symbol="B", as_of_date="2025-01-02", direction="bullish",
            confidence=0.7, composite_score=0.3,
            forward_returns={"ret_20d": 0.02},  # No ret_60d.
        ),
        BacktestRecord(
            symbol="C", as_of_date="2025-01-03", direction="bullish",
            confidence=0.7, composite_score=0.3,
            forward_returns={"ret_60d": 0.08},
        ),
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    assert out["n_records"] == 3
    assert out["n_with_return"] == 2
    # Only the two with ret_60d returns contribute to the mean.
    assert out["mean_return"] == pytest.approx(0.065, abs=1e-4)


def test_compute_return_metrics_direction_filter():
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.10),
        _rm_record("B", "2025-01-02", "bearish", -0.05),
        _rm_record("C", "2025-01-03", "bullish", 0.04),
        _rm_record("D", "2025-01-04", "bearish", 0.03),
    ]
    bull = compute_return_metrics(recs, horizon="ret_60d", direction_filter="bullish")
    assert bull["n_records"] == 2
    assert bull["n_with_return"] == 2
    assert bull["mean_return"] == pytest.approx(0.07, abs=1e-4)
    # Direction filter pins hit-rate to that direction's rule (both > 0 = bullish hits).
    assert bull["hit_rate"] == 1.0
    bear = compute_return_metrics(recs, horizon="ret_60d", direction_filter="bearish")
    assert bear["n_records"] == 2
    # bearish hit = ret < 0. One of the two bearish records has -0.05 (hit),
    # the other has +0.03 (miss). hit_rate = 0.5.
    assert bear["hit_rate"] == pytest.approx(0.5, abs=1e-4)


def test_compute_return_metrics_winsorized_mean_clips_outliers():
    # Mix of normal and one extreme return; winsorized mean should be
    # smaller than raw mean. We use 10 samples and p=0.1 so the top
    # outlier sits exactly in the trimmed bucket.
    rets = [0.02, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06, 1.00]
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", r)
        for i, r in enumerate(rets)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d", winsorize_p=0.1)
    raw_mean = sum(rets) / len(rets)
    assert out["mean_return"] == pytest.approx(raw_mean, abs=1e-4)
    # Winsorized mean clips top/bottom 10% → outlier should pull down.
    assert out["winsorized_mean"] < raw_mean


def test_compute_return_metrics_horizon_parse_unannualized_fallback():
    # An unrecognized horizon string returns unannualized Sharpe.
    rets = [0.10, 0.05, -0.02, 0.08, 0.03, 0.04]
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", r, horizon="custom_key")
        for i, r in enumerate(rets)
    ]
    out = compute_return_metrics(recs, horizon="custom_key")
    assert out["n_with_return"] == 6
    assert out["horizon_days"] is None
    assert out["annualization_factor"] is None
    # Sharpe is still computed (unannualized).
    assert out["sharpe"] is not None
    # Unannualized Sharpe ≈ 1.092
    assert 0.5 < out["sharpe"] < 2.0


def test_compute_return_metrics_max_drawdown_monotonic_up():
    # A monotonic up-sequence has zero drawdown.
    rets = [0.01, 0.02, 0.03, 0.04, 0.05]
    recs = [
        _rm_record("S", f"2025-01-{i+1:02d}", "bullish", r)
        for i, r in enumerate(rets)
    ]
    out = compute_return_metrics(recs, horizon="ret_60d")
    assert out["max_drawdown"] == 0.0


def test_compute_return_metrics_by_direction_returns_all_three_keys():
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.05),
        _rm_record("B", "2025-01-02", "bearish", -0.04),
        _rm_record("C", "2025-01-03", "neutral", 0.01),
    ]
    out = compute_return_metrics_by_direction(recs, horizon="ret_60d")
    assert set(out.keys()) == {"bullish", "bearish", "neutral"}
    assert out["bullish"]["n_with_return"] == 1
    assert out["bearish"]["n_with_return"] == 1
    assert out["neutral"]["n_with_return"] == 1
    assert out["bullish"]["mean_return"] == pytest.approx(0.05)


def test_compute_return_metrics_by_direction_empty_bucket_present():
    # Only bullish records — bearish/neutral buckets still appear in output.
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.05),
        _rm_record("B", "2025-01-02", "bullish", 0.06),
    ]
    out = compute_return_metrics_by_direction(recs, horizon="ret_60d")
    assert out["bearish"]["n_records"] == 0
    assert out["bearish"]["n_with_return"] == 0
    assert out["bearish"]["mean_return"] is None
    assert out["neutral"]["mean_return"] is None
    assert out["bullish"]["n_with_return"] == 2


def test_summarize_all_includes_return_metrics_block():
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.05),
        _rm_record("B", "2025-01-02", "bullish", 0.03),
        _rm_record("C", "2025-01-03", "bullish", -0.02),
        _rm_record("D", "2025-01-04", "bearish", -0.04),
    ]
    summary = summarize_all(recs, return_fields=["ret_60d"])
    assert "return_metrics" in summary
    assert "ret_60d" in summary["return_metrics"]
    block = summary["return_metrics"]["ret_60d"]
    assert "overall" in block and "by_direction" in block
    assert block["overall"]["n_with_return"] == 4
    assert set(block["by_direction"].keys()) == {"bullish", "bearish", "neutral"}


def test_compute_return_metrics_bearish_pnl_flips_sign():
    # All bearish records with positive forward returns (anti-predictive
    # case). Strategy P&L should be NEGATIVE since shorting a stock that
    # rose loses money. Sharpe should be negative or near zero.
    recs = [
        _rm_record("A", "2025-01-01", "bearish", 0.05),
        _rm_record("B", "2025-01-02", "bearish", 0.08),
        _rm_record("C", "2025-01-03", "bearish", 0.03),
        _rm_record("D", "2025-01-04", "bearish", -0.02),  # The only winning short.
    ]
    out = compute_return_metrics(recs, horizon="ret_60d", direction_filter="bearish")
    assert out["n_with_return"] == 4
    # Mean strategy P&L: (-0.05 -0.08 -0.03 + 0.02) / 4 = -0.035
    assert out["mean_return"] == pytest.approx(-0.035, abs=1e-4)
    assert out["sharpe"] is not None and out["sharpe"] < 0
    # Hit rate uses raw returns: bearish hit when raw < 0. Only D hits.
    assert out["hit_rate"] == pytest.approx(0.25)


def test_compute_return_metrics_bullish_pnl_unchanged():
    # Bullish records: P&L = raw return, so the sign-flip is a no-op.
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.05),
        _rm_record("B", "2025-01-02", "bullish", 0.08),
        _rm_record("C", "2025-01-03", "bullish", -0.02),
    ]
    out_filter = compute_return_metrics(
        recs, horizon="ret_60d", direction_filter="bullish"
    )
    # Mean P&L = mean raw = (0.05 + 0.08 - 0.02) / 3 = 0.0367.
    assert out_filter["mean_return"] == pytest.approx(0.0367, abs=1e-4)
    assert out_filter["sharpe"] is not None and out_filter["sharpe"] > 0


def test_render_summary_markdown_emits_risk_adjusted_section():
    recs = [
        _rm_record("A", "2025-01-01", "bullish", 0.05),
        _rm_record("B", "2025-01-02", "bullish", 0.03),
        _rm_record("C", "2025-01-03", "bearish", -0.04),
    ]
    summary = summarize_all(recs, return_fields=["ret_60d"])
    md = render_summary_markdown(summary)
    assert "Risk-adjusted metrics" in md
    assert "Sharpe (ann.)" in md
    assert "Profit factor" in md
