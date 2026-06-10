"""Live-ish shadow ML predictions for paper-trading logs.

These helpers train simple ML models on historical report rows with realized
labels, then score the latest report rows. Outputs are diagnostic only and are
intended for `RecommendationRecord.ml_shadow`; they do not affect production
actions, sizing, or trade tickets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.ml_dataset import (
    build_ml_rows,
    complete_cases,
    labels_for_rows,
    rows_to_matrix,
)
from tradingagents.analysis_only.ml_models import (
    fit_isotonic_calibrator,
    fit_model,
    sigmoid,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _record_from_report(path: Path) -> BacktestRecord | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    symbol = (payload.get("symbol") or "").upper()
    as_of = payload.get("as_of_date")
    if not symbol or not as_of:
        return None
    model_scoring = (payload.get("key_features") or {}).get("model_scoring") or {}
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction=(payload.get("direction") or "neutral").lower(),
        confidence=payload.get("confidence"),
        composite_score=model_scoring.get("composite_score"),
        factor_scores=model_scoring.get("factor_scores") or [],
        source_path=str(path),
    )


def _latest_records_from_signals(signals: dict[str, Any]) -> list[BacktestRecord]:
    records: list[BacktestRecord] = []
    for sig in signals.values():
        if sig is None:
            continue
        source = getattr(sig, "source_path", None)
        if not source:
            continue
        rec = _record_from_report(Path(source))
        if rec is not None:
            records.append(rec)
    return records


def _train_records_with_returns(
    report_paths: Iterable[Path],
    *,
    horizons_days: list[int],
    benchmark: str,
):
    # Import the CLI helper lazily to keep normal imports light.
    from backtest import load_records

    return load_records(
        [str(p) for p in report_paths],
        horizons=horizons_days,
        capture_factor_scores=True,
        capture_market_context=False,
        benchmark_symbol=benchmark,
    )


def compute_shadow_predictions(
    *,
    report_paths: Iterable[Path],
    signals: dict[str, Any],
    as_of_date: str,
    config_path: str | Path = "configs/ml_models.yaml",
    max_train_rows: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Return `{symbol: ml_shadow_payload}` for latest signal rows.

    The payload is intentionally compact and model-centric:
      `{model_name: {horizon: probability_or_score, ...}, ...}`.
    """

    cfg = _load_config(config_path)
    horizons_days = [int(h) for h in cfg.get("horizons", [5, 20, 60])]
    horizons = [f"ret_{h}d" for h in horizons_days]
    benchmark = str(cfg.get("benchmark", "SPY"))
    feature_names = list((cfg.get("features") or {}).get("allowlist") or [])
    model_cfg = cfg.get("models") or {}
    report_paths = list(report_paths)
    if not report_paths or not feature_names or not model_cfg:
        return {}

    train_records = _train_records_with_returns(
        report_paths,
        horizons_days=horizons_days,
        benchmark=benchmark,
    )
    # Drop future/current rows relative to the daily report date. Rows with
    # unavailable labels are also removed later by complete_cases.
    train_records = [r for r in train_records if r.as_of_date < as_of_date]
    if max_train_rows is not None and max_train_rows > 0:
        train_records = sorted(train_records, key=lambda r: r.as_of_date)[-max_train_rows:]
    train_rows = build_ml_rows(
        train_records,
        feature_names=feature_names,
        horizons=horizons,
    )
    current_records = _latest_records_from_signals(signals)
    current_rows = build_ml_rows(
        current_records,
        feature_names=feature_names,
        horizons=horizons,
    )
    if not train_rows or not current_rows:
        return {}

    by_symbol: dict[str, dict[str, Any]] = {
        row.symbol: {
            "config_path": str(config_path),
            "trained_through": max((r.as_of_date for r in train_records), default=None),
            "models": {},
        }
        for row in current_rows
    }
    X_current = rows_to_matrix(current_rows, feature_names=feature_names)
    for horizon in horizons:
        for model_name, params in model_cfg.items():
            label_kind = "regression" if model_name == "ridge_return" else "alpha"
            labels = labels_for_rows(train_rows, horizon=horizon, label_kind=label_kind)
            fit_rows, fit_labels = complete_cases(train_rows, labels)
            if len(fit_rows) < 50 or len(set(fit_labels)) < 2:
                continue
            try:
                model = fit_model(
                    model_name,
                    rows_to_matrix(fit_rows, feature_names=feature_names),
                    fit_labels,
                    config=params or {},
                )
            except Exception:
                continue
            raw_scores = model.predict_scores(X_current)
            if model.task == "regression":
                probs = [sigmoid(score * 10.0) for score in raw_scores]
            else:
                probs = raw_scores

            alpha_train = labels_for_rows(fit_rows, horizon=horizon, label_kind="alpha")
            train_raw = model.predict_scores(rows_to_matrix(fit_rows, feature_names=feature_names))
            train_probs = [sigmoid(s * 10.0) for s in train_raw] if model.task == "regression" else train_raw
            cal_pairs = [(float(p), int(y)) for p, y in zip(train_probs, alpha_train) if y is not None]
            cal = None
            if cal_pairs:
                cal_probs, cal_labels = zip(*cal_pairs)
                cal = fit_isotonic_calibrator(list(cal_probs), list(cal_labels))
            calibrated = cal.predict(probs) if cal else probs
            for row, raw, score in zip(current_rows, raw_scores, calibrated):
                model_block = by_symbol[row.symbol]["models"].setdefault(model_name, {})
                model_block[horizon] = {
                    "score": round(float(score), 6),
                    "raw_score": round(float(raw), 6),
                    "target": "alpha_hit_probability",
                    "calibrated": bool(cal is not None),
                }
    return by_symbol
