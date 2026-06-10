"""Tests for shadow ML dataset construction and leakage guards."""

from __future__ import annotations

import pytest

from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.ml_dataset import (
    assert_feature_names_safe,
    build_ml_rows,
    factor_alpha_probabilities,
    split_rows_for_window,
)
from tradingagents.analysis_only.ml_models import spearman_ic
from tradingagents.analysis_only.walk_forward import WalkForwardWindow


FEATURES = ["factor_composite", "factor_confidence", "momentum_rsi"]


def _record(
    symbol: str,
    as_of_date: str,
    *,
    ret_60d: float,
    adj_60d: float | None = None,
    composite: float = 0.1,
) -> BacktestRecord:
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of_date,
        direction="bullish" if composite > 0 else "bearish",
        confidence=0.65,
        composite_score=composite,
        forward_returns={"ret_60d": ret_60d},
        benchmark_adjusted_returns={"ret_60d": adj_60d},
        factor_scores=[
            {
                "factor": "momentum_rsi",
                "score": composite,
                "weighted_score": composite,
                "data_available": True,
            }
        ],
    )


def test_feature_allowlist_rejects_banned_names():
    with pytest.raises(ValueError):
        assert_feature_names_safe(["factor_composite", "ret_60d"])
    with pytest.raises(ValueError):
        assert_feature_names_safe(["future_return"])
    with pytest.raises(ValueError):
        assert_feature_names_safe(["source_path"])


def test_feature_allowlist_does_not_overreach_on_legit_trailing_features():
    # `momentum_return_20d` is a backward-looking 20d return; `valuation_forward_vs_trailing_pe`
    # uses analyst forward EPS estimates (PIT). Neither is a forward LABEL.
    # The leakage regex must NOT reject them.
    assert_feature_names_safe([
        "momentum_return_20d",
        "valuation_forward_vs_trailing_pe",
    ])


def test_feature_allowlist_rejects_obvious_leak_patterns():
    for bad in (
        "ret_20d",
        "alpha_ret_60d",
        "forward_return_60d",
        "future_alpha_60d",
        "realized_return_60d",
        "realized_alpha_60d",
        "benchmark_adjusted_ret_60d",
        "non_pit_marker",
        "live_snapshot",
    ):
        with pytest.raises(ValueError):
            assert_feature_names_safe([bad])


def test_build_rows_drops_injected_forward_return_features():
    rec = _record("NVDA", "2026-01-02", ret_60d=0.25, adj_60d=0.10)
    rows = build_ml_rows(
        [rec],
        feature_names=FEATURES,
        horizons=("ret_60d",),
        extra_candidate_features={
            ("NVDA", "2026-01-02"): {
                "ret_60d": 0.25,
                "future_return": 0.25,
                "benchmark_adjusted_ret_60d": 0.10,
                "momentum_rsi": 0.3,
            }
        },
    )
    assert rows[0].features == {
        "factor_composite": 0.1,
        "factor_confidence": 0.65,
        "momentum_rsi": 0.3,
    }
    assert "ret_60d" not in rows[0].features
    assert "future_return" not in rows[0].features


def test_labels_are_explicit_headline_alpha_and_regression():
    rec = _record("NVDA", "2026-01-02", ret_60d=0.03, adj_60d=-0.02)
    row = build_ml_rows([rec], feature_names=FEATURES, horizons=("ret_60d",))[0]
    assert row.headline_labels["ret_60d"] == 1
    assert row.alpha_labels["ret_60d"] == 0
    assert row.regression_targets["ret_60d"] == -0.02


def test_date_based_split_keeps_same_date_tickers_together_and_embargoes_train():
    records = [
        _record("NVDA", "2024-06-01", ret_60d=0.01),
        _record("AMD", "2024-06-01", ret_60d=0.02),
        _record("NVDA", "2024-06-20", ret_60d=0.03),  # purged by 20d embargo
        _record("NVDA", "2024-07-01", ret_60d=0.04),
        _record("AMD", "2024-07-01", ret_60d=0.05),
    ]
    rows = build_ml_rows(records, feature_names=FEATURES, horizons=("ret_60d",))
    window = WalkForwardWindow(
        train_start="2024-06-01",
        train_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2024-07-31",
    )
    train, test = split_rows_for_window(rows, window, embargo_days=20)
    assert {(r.symbol, r.as_of_date) for r in train} == {
        ("NVDA", "2024-06-01"),
        ("AMD", "2024-06-01"),
    }
    assert {(r.symbol, r.as_of_date) for r in test} == {
        ("NVDA", "2024-07-01"),
        ("AMD", "2024-07-01"),
    }


def test_build_rows_tags_cohort_from_lookup():
    records = [
        _record("NVDA", "2026-01-02", ret_60d=0.10),
        _record("JPM", "2026-01-02", ret_60d=0.05),
        _record("XYZ", "2026-01-02", ret_60d=0.02),
    ]
    rows = build_ml_rows(
        records,
        feature_names=FEATURES,
        horizons=("ret_60d",),
        cohort_lookup={"NVDA": "core", "JPM": "canary"},
    )
    cohorts = {row.symbol: row.cohort for row in rows}
    assert cohorts == {"NVDA": "core", "JPM": "canary", "XYZ": None}


def test_factor_alpha_probabilities_maps_direction_to_probability():
    # bullish + confidence 0.65 → P(alpha=1) = 0.65
    bull = _record("NVDA", "2026-01-02", ret_60d=0.10, composite=0.5)
    # bearish + confidence 0.65 → P(alpha=1) = 0.35
    bear = _record("NVDA", "2026-01-09", ret_60d=-0.05, composite=-0.5)
    rows = build_ml_rows(
        [bull, bear], feature_names=FEATURES, horizons=("ret_60d",),
    )
    probs = factor_alpha_probabilities(rows)
    assert probs[0] == pytest.approx(0.65)
    assert probs[1] == pytest.approx(0.35)


def test_factor_alpha_probabilities_neutral_and_missing_collapse_to_half():
    rec = _record("NVDA", "2026-01-02", ret_60d=0.10, composite=0.5)
    # Force neutral via direct constructor.
    from tradingagents.analysis_only.ml_dataset import MLFeatureRow
    neutral = MLFeatureRow(
        symbol="NVDA",
        as_of_date="2026-01-02",
        features={"factor_confidence": 0.9},
        direction="neutral",
    )
    missing_conf = MLFeatureRow(
        symbol="NVDA",
        as_of_date="2026-01-02",
        features={},
        direction="bullish",
    )
    probs = factor_alpha_probabilities([neutral, missing_conf])
    assert probs == [0.5, 0.5]


def test_adversarial_leak_sentinel_would_be_unrealistic_but_production_is_not():
    records = [
        _record("A", "2024-01-01", ret_60d=0.30, adj_60d=0.20, composite=-0.2),
        _record("B", "2024-01-08", ret_60d=-0.10, adj_60d=-0.20, composite=0.3),
        _record("C", "2024-01-15", ret_60d=0.20, adj_60d=0.10, composite=-0.1),
        _record("D", "2024-01-22", ret_60d=-0.20, adj_60d=-0.30, composite=0.2),
    ]
    leaked_scores = [r.forward_returns["ret_60d"] for r in records]
    labels = [r.benchmark_adjusted_returns["ret_60d"] for r in records]
    assert spearman_ic(leaked_scores, labels) == pytest.approx(1.0)

    rows = build_ml_rows(
        records,
        feature_names=FEATURES,
        horizons=("ret_60d",),
        extra_candidate_features={
            (r.symbol, r.as_of_date): {"ret_60d": r.forward_returns["ret_60d"]}
            for r in records
        },
    )
    production_scores = [row.features["factor_composite"] for row in rows]
    assert abs(spearman_ic(production_scores, labels)) < 1.0
