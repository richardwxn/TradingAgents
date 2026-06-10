"""Model adapters and metrics for shadow ML evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from tradingagents.analysis_only.ml_dataset import MLFeatureRow


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for ML model training. "
            "Install project dependencies first."
        ) from exc


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float | None:
    pairs = [(float(p), int(y)) for p, y in zip(probs, labels)]
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman_ic(scores: Sequence[float], returns: Sequence[float | None]) -> float | None:
    pairs = [(float(s), float(r)) for s, r in zip(scores, returns) if r is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def precision_at_top_k_by_week(
    rows: Sequence[MLFeatureRow],
    scores: Sequence[float],
    labels: Sequence[int | None],
    *,
    top_k: int = 10,
) -> float | None:
    """Mean precision of top-k ranked rows per ISO week."""

    groups: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for row, score, label in zip(rows, scores, labels):
        if label is None:
            continue
        dt = datetime.strptime(row.as_of_date, "%Y-%m-%d")
        iso = dt.isocalendar()
        groups[(iso.year, iso.week)].append((float(score), int(label)))
    weekly: list[float] = []
    for values in groups.values():
        ranked = sorted(values, key=lambda item: item[0], reverse=True)[:top_k]
        if ranked:
            weekly.append(sum(label for _, label in ranked) / len(ranked))
    if not weekly:
        return None
    return sum(weekly) / len(weekly)


def validate_hist_gbdt_config(config: dict[str, Any]) -> dict[str, Any]:
    """Clamp/validate constrained tree defaults from the plan."""

    out = {
        "max_depth": int(config.get("max_depth", 4)),
        "min_samples_leaf": int(config.get("min_samples_leaf", 50)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "l2_regularization": float(config.get("l2_regularization", 1.0)),
        "early_stopping_rounds": int(config.get("early_stopping_rounds", 50)),
        "validation_fraction": float(config.get("validation_fraction", 0.15)),
    }
    if out["max_depth"] > 4:
        raise ValueError("hist_gbdt.max_depth must be <= 4")
    if out["min_samples_leaf"] < 50:
        raise ValueError("hist_gbdt.min_samples_leaf must be >= 50")
    if out["learning_rate"] > 0.05:
        raise ValueError("hist_gbdt.learning_rate must be <= 0.05")
    if out["l2_regularization"] <= 0:
        raise ValueError("hist_gbdt.l2_regularization must be > 0")
    if out["early_stopping_rounds"] < 1:
        raise ValueError("hist_gbdt.early_stopping_rounds must be >= 1")
    return out


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    task: str

    def predict_scores(self, X: Sequence[Sequence[float]]) -> list[float]:
        if self.task == "classification":
            probs = self.estimator.predict_proba(X)
            return [float(row[1]) for row in probs]
        preds = self.estimator.predict(X)
        return [float(v) for v in preds]


def fit_model(
    name: str,
    X: Sequence[Sequence[float]],
    y: Sequence[int | float],
    *,
    config: dict[str, Any] | None = None,
) -> TrainedModel:
    """Fit one supported shadow model."""

    _require_sklearn()
    config = config or {}
    if name == "elastic_logit":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="saga",
                C=float(config.get("C", 0.5)),
                l1_ratio=float(config.get("l1_ratio", 0.25)),
                max_iter=int(config.get("max_iter", 2000)),
            ),
        )
        estimator.fit(X, y)
        return TrainedModel(name=name, estimator=estimator, task="classification")
    if name == "ridge_return":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(StandardScaler(), Ridge(alpha=float(config.get("alpha", 1.0))))
        estimator.fit(X, y)
        return TrainedModel(name=name, estimator=estimator, task="regression")
    if name == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = validate_hist_gbdt_config(config)
        estimator = HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            learning_rate=params["learning_rate"],
            l2_regularization=params["l2_regularization"],
            early_stopping=True,
            n_iter_no_change=params["early_stopping_rounds"],
            validation_fraction=params["validation_fraction"],
        )
        estimator.fit(X, y)
        return TrainedModel(name=name, estimator=estimator, task="classification")
    raise ValueError(f"unsupported ML model: {name}")


@dataclass
class IsotonicCalibrator:
    estimator: Any

    def predict(self, scores: Sequence[float]) -> list[float]:
        return [float(v) for v in self.estimator.predict(list(scores))]


def fit_isotonic_calibrator(scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator | None:
    """Fit per-model calibration on the training window only."""

    _require_sklearn()
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if len(pairs) < 20 or len({y for _, y in pairs}) < 2:
        return None
    from sklearn.isotonic import IsotonicRegression

    xs, ys = zip(*pairs)
    estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    estimator.fit(xs, ys)
    return IsotonicCalibrator(estimator=estimator)
