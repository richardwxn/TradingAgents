"""Tests for shadow ML model metrics and constrained defaults."""

from __future__ import annotations

import pytest

from tradingagents.analysis_only.ml_dataset import MLFeatureRow
from tradingagents.analysis_only.ml_models import (
    brier_score,
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
