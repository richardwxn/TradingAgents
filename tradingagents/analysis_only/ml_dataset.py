"""Dataset helpers for shadow ML evaluation.

This module intentionally starts from the existing `BacktestRecord` shape.
It enforces a strict feature allow-list, explicit labels, and date-based
walk-forward splits with a horizon embargo so ML benchmarks cannot pass via
future-return leakage or cross-ticker row-order leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.walk_forward import WalkForwardWindow


HORIZONS: tuple[str, ...] = ("ret_5d", "ret_20d", "ret_60d")
# Names known to indicate forward-return labels or non-PIT metadata. The
# patterns aim at the leakage shapes that finance ML labels actually take,
# WITHOUT collateral damage to legitimate trailing-window features like
# `momentum_return_20d` (a backward-looking 20d return) or
# `valuation_forward_vs_trailing_pe` (a snapshot of analyst forward EPS).
BANNED_FEATURE_RE = re.compile(
    r"("
    r"^ret_\d+d$|_ret_\d+d$|"           # label keys: ret_60d, foo_ret_60d
    r"forward_return|future_\w+|"        # forward_return_*, future_alpha_*
    r"realized_return|realized_alpha|"
    r"benchmark_adjusted|"
    r"source_path|^path$|^file$|"
    r"post_label|non_pit|live_snapshot"
    r")",
    re.IGNORECASE,
)
BASE_FEATURE_NAMES: tuple[str, ...] = ("factor_composite", "factor_confidence")


@dataclass(frozen=True)
class MLFeatureRow:
    """One point-in-time tabular row for ML shadow evaluation."""

    symbol: str
    as_of_date: str
    features: dict[str, float]
    headline_labels: dict[str, int | None] = field(default_factory=dict)
    alpha_labels: dict[str, int | None] = field(default_factory=dict)
    regression_targets: dict[str, float | None] = field(default_factory=dict)
    forward_returns: dict[str, float | None] = field(default_factory=dict)
    benchmark_adjusted_returns: dict[str, float | None] = field(default_factory=dict)
    # Direction is the factor model's predicted direction (bullish/bearish/
    # neutral). Combined with `factor_confidence`, the baseline can emit a
    # calibrated probability of alpha-hit (P(adj_ret > 0)) that is Brier-
    # comparable with ML model outputs in [0,1].
    direction: str | None = None
    # Cohort tag (e.g. "core" / "canary" / "other"). Set by the eval driver
    # so leaderboards can split universal lift from canary-driven memorization.
    cohort: str | None = None


def assert_feature_names_safe(feature_names: Iterable[str]) -> None:
    """Raise if any feature name looks like a label, path, or non-PIT field."""

    banned = sorted({name for name in feature_names if BANNED_FEATURE_RE.search(name)})
    if banned:
        raise ValueError(f"banned ML feature name(s): {', '.join(banned)}")


def sanitize_features(
    candidates: dict[str, Any],
    *,
    allowlist: Sequence[str],
    strict: bool = False,
) -> dict[str, float]:
    """Return numeric candidates that are explicitly allow-listed and safe.

    Unknown keys are ignored. Banned keys are always dropped; with `strict`
    they raise, which is useful for CI and config validation.
    """

    assert_feature_names_safe(allowlist)
    allowed = set(allowlist)
    out: dict[str, float] = {}
    banned_seen: list[str] = []
    for key, value in candidates.items():
        if BANNED_FEATURE_RE.search(key):
            banned_seen.append(key)
            continue
        if key not in allowed:
            continue
        try:
            if value is None:
                continue
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    if strict and banned_seen:
        raise ValueError(f"candidate feature payload contained banned key(s): {banned_seen}")
    return out


def extract_candidate_features(record: BacktestRecord) -> dict[str, Any]:
    """Extract all plausible PIT features before allow-list filtering."""

    features: dict[str, Any] = {
        "factor_composite": record.composite_score,
        "factor_confidence": record.confidence,
    }
    for factor in record.factor_scores or []:
        name = factor.get("factor")
        if not isinstance(name, str) or not name:
            continue
        features[name] = factor.get("score")
    return features


def labels_for_record(record: BacktestRecord, horizon: str) -> tuple[int | None, int | None, float | None]:
    """Return `(headline_hit, alpha_hit, benchmark_adjusted_return)`."""

    ret = record.forward_returns.get(horizon)
    adj = record.benchmark_adjusted_returns.get(horizon)
    headline = None if ret is None else int(float(ret) > 0.0)
    alpha = None if adj is None else int(float(adj) > 0.0)
    regression = None if adj is None else float(adj)
    return headline, alpha, regression


def build_ml_rows(
    records: Iterable[BacktestRecord],
    *,
    feature_names: Sequence[str],
    horizons: Sequence[str] = HORIZONS,
    extra_candidate_features: dict[tuple[str, str], dict[str, Any]] | None = None,
    strict: bool = False,
    cohort_lookup: dict[str, str] | None = None,
) -> list[MLFeatureRow]:
    """Convert `BacktestRecord`s to strict ML rows.

    `extra_candidate_features` is keyed by `(symbol, as_of_date)` and exists
    primarily for adversarial leakage tests. `cohort_lookup` maps an upper-
    cased symbol to its cohort label (e.g. "core" / "canary").
    """

    assert_feature_names_safe(feature_names)
    cohort_lookup = cohort_lookup or {}
    rows: list[MLFeatureRow] = []
    for record in records:
        candidates = extract_candidate_features(record)
        if extra_candidate_features:
            candidates.update(
                extra_candidate_features.get((record.symbol, record.as_of_date), {})
            )
        features = sanitize_features(candidates, allowlist=feature_names, strict=strict)
        headline_labels: dict[str, int | None] = {}
        alpha_labels: dict[str, int | None] = {}
        regression_targets: dict[str, float | None] = {}
        for horizon in horizons:
            headline, alpha, regression = labels_for_record(record, horizon)
            headline_labels[horizon] = headline
            alpha_labels[horizon] = alpha
            regression_targets[horizon] = regression
        rows.append(
            MLFeatureRow(
                symbol=record.symbol,
                as_of_date=record.as_of_date,
                features=features,
                headline_labels=headline_labels,
                alpha_labels=alpha_labels,
                regression_targets=regression_targets,
                forward_returns=dict(record.forward_returns),
                benchmark_adjusted_returns=dict(record.benchmark_adjusted_returns),
                direction=record.direction,
                cohort=cohort_lookup.get((record.symbol or "").upper()),
            )
        )
    return rows


def factor_alpha_probabilities(rows: Sequence[MLFeatureRow]) -> list[float]:
    """Convert (factor_confidence, direction) → P(alpha_hit = 1).

    The factor model's `confidence` is the probability that the predicted
    direction is correct. Translate to a probability of the alpha label
    (`adj_ret > 0`) so Brier scores are comparable with ML probabilities:

      - bullish  → P(alpha=1) = confidence
      - bearish  → P(alpha=1) = 1 - confidence
      - neutral  → P(alpha=1) = 0.5
      - missing  → 0.5
    """

    probs: list[float] = []
    for row in rows:
        conf = row.features.get("factor_confidence")
        if conf is None:
            probs.append(0.5)
            continue
        direction = (row.direction or "").lower()
        if direction == "bullish":
            probs.append(min(1.0, max(0.0, float(conf))))
        elif direction == "bearish":
            probs.append(min(1.0, max(0.0, 1.0 - float(conf))))
        else:
            probs.append(0.5)
    return probs


def rows_to_matrix(
    rows: Sequence[MLFeatureRow],
    *,
    feature_names: Sequence[str],
) -> list[list[float]]:
    """Dense feature matrix with missing allow-listed values filled as 0."""

    assert_feature_names_safe(feature_names)
    return [[float(row.features.get(name, 0.0)) for name in feature_names] for row in rows]


def labels_for_rows(
    rows: Sequence[MLFeatureRow],
    *,
    horizon: str,
    label_kind: str,
) -> list[int | float | None]:
    if label_kind == "headline":
        return [row.headline_labels.get(horizon) for row in rows]
    if label_kind == "alpha":
        return [row.alpha_labels.get(horizon) for row in rows]
    if label_kind == "regression":
        return [row.regression_targets.get(horizon) for row in rows]
    raise ValueError("label_kind must be headline, alpha, or regression")


def complete_cases(
    rows: Sequence[MLFeatureRow],
    labels: Sequence[int | float | None],
) -> tuple[list[MLFeatureRow], list[int | float]]:
    out_rows: list[MLFeatureRow] = []
    out_labels: list[int | float] = []
    for row, label in zip(rows, labels):
        if label is None:
            continue
        out_rows.append(row)
        out_labels.append(label)
    return out_rows, out_labels


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def split_rows_for_window(
    rows: Sequence[MLFeatureRow],
    window: WalkForwardWindow,
    *,
    embargo_days: int,
) -> tuple[list[MLFeatureRow], list[MLFeatureRow]]:
    """Date-based train/test split with horizon embargo.

    Same-date rows always move together because the predicate is based only
    on `as_of_date`, not row order or ticker.
    """

    test_start = _parse_date(window.test_start)
    train_cutoff = test_start - timedelta(days=max(0, int(embargo_days)))
    train: list[MLFeatureRow] = []
    test: list[MLFeatureRow] = []
    for row in rows:
        row_date = _parse_date(row.as_of_date)
        if _parse_date(window.train_start) <= row_date <= _parse_date(window.train_end):
            if row_date <= train_cutoff:
                train.append(row)
        if _parse_date(window.test_start) <= row_date <= _parse_date(window.test_end):
            test.append(row)
    return train, test
