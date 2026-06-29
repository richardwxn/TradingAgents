"""Tests for shadow ML model metrics and constrained defaults."""

from __future__ import annotations

import pytest

from tradingagents.analysis_only.ml_dataset import MLFeatureRow
from tradingagents.analysis_only.ml_models import (
    brier_score,
    fit_isotonic_calibrator,
    fit_model,
    precision_at_top_k_by_week,
    spearman_ic,
    validate_hist_gbdt_config,
)


def test_hist_gbdt_defaults_are_constrained():
    params = validate_hist_gbdt_config({})
    assert params["max_depth"] == 4
    assert params["min_samples_leaf"] == 50
    assert params["learning_rate"] == 0.05
    assert params["l2_regularization"] == 1.0
    assert params["early_stopping_rounds"] == 50


def test_hist_gbdt_rejects_loose_overfit_prone_settings():
    with pytest.raises(ValueError):
        validate_hist_gbdt_config({"max_depth": 6})
    with pytest.raises(ValueError):
        validate_hist_gbdt_config({"min_samples_leaf": 10})
    with pytest.raises(ValueError):
        validate_hist_gbdt_config({"learning_rate": 0.10})


def test_precision_at_top_k_by_week():
    rows = [
        MLFeatureRow("A", "2024-01-02", {}),
        MLFeatureRow("B", "2024-01-03", {}),
        MLFeatureRow("C", "2024-01-09", {}),
        MLFeatureRow("D", "2024-01-10", {}),
    ]
    scores = [0.9, 0.1, 0.8, 0.7]
    labels = [1, 0, 0, 1]
    # Week 1 top-1 hits, week 2 top-1 misses => mean precision 0.5.
    assert precision_at_top_k_by_week(rows, scores, labels, top_k=1) == pytest.approx(0.5)


def test_basic_metrics():
    assert brier_score([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    assert spearman_ic([1, 2, 3], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_elastic_logit_adapter_if_sklearn_available():
    pytest.importorskip("sklearn")
    X = [[-1.0], [-0.5], [0.5], [1.0]]
    y = [0, 0, 1, 1]
    model = fit_model("elastic_logit", X, y, config={"C": 1.0, "l1_ratio": 0.1})
    scores = model.predict_scores(X)
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_elastic_logit_predict_scores_monotonic_with_signal():
    """Classification probabilities should rise with the informative feature."""
    pytest.importorskip("sklearn")
    # Deterministic, linearly separable single-feature dataset: label flips at 0.
    X = [[float(i)] for i in range(-10, 10)]
    y = [0 if i < 0 else 1 for i in range(-10, 10)]
    model = fit_model("elastic_logit", X, y, config={"C": 1.0, "l1_ratio": 0.1})
    scores = model.predict_scores(X)
    # Correct shape and probability range.
    assert len(scores) == len(X)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Probability of the positive class must be non-decreasing in the feature
    # (ascending X), and strictly higher at the top than the bottom.
    assert all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1))
    assert scores[-1] > scores[0]


def test_ridge_return_predict_scores_monotonic_and_shaped():
    """Regression predictions should track the target and have correct shape."""
    pytest.importorskip("sklearn")
    X = [[float(i)] for i in range(-10, 10)]
    y = [float(i) for i in range(-10, 10)]
    model = fit_model("ridge_return", X, y)
    preds = model.predict_scores(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, float) for p in preds)
    # Predictions increase monotonically with the (ascending) feature.
    assert all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1))
    assert preds[-1] > preds[0]


def test_isotonic_calibrator_is_monotonic_and_bounded():
    """Calibrated probabilities are non-decreasing in the raw score and in [0, 1]."""
    pytest.importorskip("sklearn")
    # Raw scores ascending in [0, 1]; low scores are negatives, high are positives.
    n = 40
    raw = [i / (n - 1) for i in range(n)]
    labels = [0] * (n // 2) + [1] * (n // 2)
    calibrator = fit_isotonic_calibrator(raw, labels)
    assert calibrator is not None
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    calibrated = calibrator.predict(grid)
    assert len(calibrated) == len(grid)
    assert all(0.0 <= p <= 1.0 for p in calibrated)
    assert all(calibrated[i] <= calibrated[i + 1] + 1e-9 for i in range(len(calibrated) - 1))


def test_isotonic_calibrator_degenerate_inputs_return_none_without_crashing():
    """Single-class or tiny-n inputs are handled gracefully (None, no exception)."""
    pytest.importorskip("sklearn")
    # Single class across enough samples -> cannot calibrate -> None.
    single_class = fit_isotonic_calibrator([0.1 * i for i in range(40)], [1] * 40)
    assert single_class is None
    # Too few samples (< 20) even with both classes -> None.
    tiny = fit_isotonic_calibrator([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 1, 1, 1])
    assert tiny is None
