#!/usr/bin/env python
"""Evaluate shadow ML models with date-based walk-forward splits.

This script is artifact-only except for optional yfinance benchmark return
attachment inherited from `backtest.load_records`. Production scoring is not
changed; outputs are a leaderboard for offline review.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.ml_dataset import (  # noqa: E402
    HORIZONS,
    build_ml_rows,
    complete_cases,
    factor_alpha_probabilities,
    labels_for_rows,
    rows_to_matrix,
    split_rows_for_window,
)
from tradingagents.analysis_only.ml_models import (  # noqa: E402
    brier_score,
    fit_isotonic_calibrator,
    fit_model,
    precision_at_top_k_by_week,
    sigmoid,
    spearman_ic,
)
from tradingagents.analysis_only.walk_forward import generate_windows  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/ml_models.yaml")
    p.add_argument(
        "--reports-glob", action="append",
        help=(
            "Glob for report JSONs. May be passed multiple times to evaluate "
            "across multiple corpora (e.g. v1.8 + extended 2020-2023). "
            "Falls back to the `reports_glob` key in the config."
        ),
    )
    p.add_argument("--output-dir", default="backtest/results/ml_shadow")
    p.add_argument("--no-prices", action="store_true",
                   help="Assume reports/records already carry forward returns; do not fetch.")
    p.add_argument("--universe-path", default="configs/universe.yaml",
                   help=(
                       "Universe YAML with `core:` and `canary:` lists. Used "
                       "to tag rows for per-cohort leaderboards."
                   ))
    p.add_argument("--leak-sanity", action="store_true",
                   help=(
                       "Run an adversarial leak-injection check before "
                       "evaluation. Forward returns are injected into the "
                       "candidate feature payload; the sanitizer must strip "
                       "them and the model's OOS IC must stay below "
                       "leak_sanity_ic_ceiling, otherwise the run aborts."
                   ))
    return p.parse_args()


def _load_universe_cohorts(path: str) -> dict[str, str]:
    """Map upper-cased ticker → 'core' or 'canary' from universe.yaml.

    Tickers absent from both lists are left out so they default to 'other'.
    """

    try:
        with Path(path).open() as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    lookup: dict[str, str] = {}
    for ticker in payload.get("core") or []:
        lookup[str(ticker).upper()] = "core"
    for ticker in payload.get("canary") or []:
        lookup[str(ticker).upper()] = "canary"
    return lookup


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open() as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _horizon_days(horizon: str) -> int:
    return int(horizon.removeprefix("ret_").removesuffix("d"))


def _mean(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _factor_scores(rows) -> list[float]:
    # Calibrated P(alpha_hit) so the factor baseline can share the
    # probability space with ML models (Brier, threshold 0.5).
    return factor_alpha_probabilities(rows)


def _spy_trend_scores(rows) -> list[float]:
    # The plan's trivial SPY baseline is represented by the existing
    # market_spy_trend factor score. If unavailable, this collapses to 0
    # rather than leaking current/future market data from outside the row.
    return [float(row.features.get("market_spy_trend", 0.0)) for row in rows]


def _classification_metrics(rows, scores, *, horizon: str, top_k: int) -> dict[str, Any]:
    headline = [row.headline_labels.get(horizon) for row in rows]
    alpha = [row.alpha_labels.get(horizon) for row in rows]
    ret_adj = [row.benchmark_adjusted_returns.get(horizon) for row in rows]
    pred = [1 if s > 0.5 else 0 for s in scores]
    pred_factor_style = [1 if s > 0.0 else 0 for s in scores]

    def hit_rate(labels, predictions):
        pairs = [(y, p) for y, p in zip(labels, predictions) if y is not None]
        if not pairs:
            return None
        return sum(1 for y, p in pairs if int(y) == int(p)) / len(pairs)

    # Scores from composites/trend are centered around 0; probabilities are
    # centered around 0.5. Pick the sensible threshold based on score range.
    predictions = pred if all(0.0 <= s <= 1.0 for s in scores) else pred_factor_style
    complete_alpha = [int(y) for y in alpha if y is not None]
    complete_alpha_scores = [float(s) for s, y in zip(scores, alpha) if y is not None]
    return {
        "headline_hit_rate": hit_rate(headline, predictions),
        "alpha_hit_rate": hit_rate(alpha, predictions),
        "brier_alpha": brier_score(complete_alpha_scores, complete_alpha)
        if complete_alpha and all(0.0 <= s <= 1.0 for s in complete_alpha_scores)
        else None,
        "ic_benchmark_adjusted": spearman_ic(scores, ret_adj),
        "precision_at_top_k": precision_at_top_k_by_week(
            rows, scores, alpha, top_k=top_k,
        ),
        "n": sum(1 for y in alpha if y is not None),
    }


def _evaluate_baseline(name: str, rows, *, horizon: str, top_k: int) -> dict[str, Any]:
    if name == "always_bullish":
        scores = [1.0] * len(rows)
    elif name == "spy_50dma":
        scores = _spy_trend_scores(rows)
    elif name == "factor_v1_8":
        scores = _factor_scores(rows)
    else:
        raise ValueError(name)
    out = _classification_metrics(rows, scores, horizon=horizon, top_k=top_k)
    out["model"] = name
    return out


def _filter_rows_for_cohort(rows: list, cohort: str) -> list:
    if cohort == "all":
        return rows
    return [r for r in rows if (r.cohort or "other") == cohort]


def _compute_gates_check(
    leaderboard: list[dict[str, Any]], *, gates: dict[str, Any], horizons: list[str],
    model_names: list[str], cohorts: list[str],
) -> list[dict[str, Any]]:
    """Compare each (model, horizon, cohort) against factor_v1_8 baseline.

    Returns one row per ML model per horizon per cohort with PASS/FAIL plus
    the contributing lift numbers.
    """

    min_alpha_lift = float(gates.get("min_alpha_hit_lift", 0.01))
    min_topk_lift = float(gates.get("min_top_k_precision_lift", 0.01))
    max_regression = float(gates.get("max_horizon_regression", 0.01))

    by_key = {(r["horizon"], r["model"], r.get("cohort", "all")): r for r in leaderboard}
    out: list[dict[str, Any]] = []
    for cohort in cohorts:
        for model in model_names:
            for horizon in horizons:
                ml = by_key.get((horizon, model, cohort))
                base = by_key.get((horizon, "factor_v1_8", cohort))
                if ml is None or base is None:
                    continue
                alpha_lift = _diff(ml.get("alpha_hit_rate"), base.get("alpha_hit_rate"))
                topk_lift = _diff(ml.get("precision_at_top_k"), base.get("precision_at_top_k"))
                regressions = []
                for other_h in horizons:
                    other_ml = by_key.get((other_h, model, cohort))
                    other_base = by_key.get((other_h, "factor_v1_8", cohort))
                    if other_ml is None or other_base is None:
                        continue
                    delta = _diff(other_ml.get("alpha_hit_rate"), other_base.get("alpha_hit_rate"))
                    if delta is not None:
                        regressions.append((other_h, delta))
                worst_regression = min((d for _, d in regressions), default=None)
                gate_pass = (
                    alpha_lift is not None
                    and topk_lift is not None
                    and (alpha_lift >= min_alpha_lift or topk_lift >= min_topk_lift)
                    and (worst_regression is None or worst_regression >= -max_regression)
                )
                out.append({
                    "cohort": cohort,
                    "model": model,
                    "horizon": horizon,
                    "alpha_hit_lift": alpha_lift,
                    "precision_at_top_k_lift": topk_lift,
                    "worst_horizon_alpha_delta": worst_regression,
                    "gate_pass": bool(gate_pass),
                })
    return out


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _run_leak_sanity(
    *,
    records,
    feature_names: list[str],
    horizons_names: list[str],
    windows,
    embargo_by_h: dict[str, Any],
    cohort_lookup: dict[str, str],
    ic_ceiling: float,
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Inject forward returns into candidate features and verify the sanitizer drops them.

    Returns a dict with `passed` (bool), `injected_test_ic` (worst OOS IC on a
    sample window), and `inspected_feature_names` (any banned key that
    survived sanitization, which would indicate a bug in the leakage filter).
    """

    inject_payload: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        ret = record.forward_returns.get("ret_60d")
        if ret is None:
            continue
        # Use names guaranteed to trip BANNED_FEATURE_RE; the sanitizer must
        # drop them even though we're injecting them through the test hook.
        inject_payload[(record.symbol, record.as_of_date)] = {
            "ret_60d": float(ret),
            "future_alpha_60d": float(ret),
            "realized_return_60d": float(ret),
        }
    leak_rows = build_ml_rows(
        records,
        feature_names=feature_names,
        horizons=horizons_names,
        extra_candidate_features=inject_payload,
        cohort_lookup=cohort_lookup,
    )
    survived = sorted({
        key for row in leak_rows for key in row.features
        if any(banned in key for banned in ("ret_", "future_", "realized_"))
    })
    if survived:
        return {
            "passed": False,
            "reason": "banned feature(s) survived sanitization",
            "leaked_feature_names": survived,
            "injected_test_ic": None,
            "ic_ceiling": ic_ceiling,
        }
    if not windows:
        return {
            "passed": True,
            "reason": "no walk-forward windows to test, sanitizer-only check passed",
            "leaked_feature_names": [],
            "injected_test_ic": None,
            "ic_ceiling": ic_ceiling,
        }
    horizon = "ret_60d"
    window = windows[-1]
    embargo = int(embargo_by_h.get(horizon, _horizon_days(horizon)))
    train_rows, test_rows = split_rows_for_window(leak_rows, window, embargo_days=embargo)
    if len(train_rows) < 50 or len(test_rows) < 10:
        return {
            "passed": True,
            "reason": "insufficient rows in last window for OOS IC check",
            "leaked_feature_names": [],
            "injected_test_ic": None,
            "ic_ceiling": ic_ceiling,
        }
    raw_labels = labels_for_rows(train_rows, horizon=horizon, label_kind="alpha")
    fit_rows, fit_labels = complete_cases(train_rows, raw_labels)
    if len(set(fit_labels)) < 2:
        return {
            "passed": True,
            "reason": "single-class train labels, OOS IC inconclusive",
            "leaked_feature_names": [],
            "injected_test_ic": None,
            "ic_ceiling": ic_ceiling,
        }
    X_train = rows_to_matrix(fit_rows, feature_names=feature_names)
    model = fit_model(
        "elastic_logit", X_train, fit_labels,
        config=model_cfg.get("elastic_logit") or {},
    )
    X_test = rows_to_matrix(test_rows, feature_names=feature_names)
    test_scores = model.predict_scores(X_test)
    test_returns = [row.benchmark_adjusted_returns.get(horizon) for row in test_rows]
    ic = spearman_ic(test_scores, test_returns)
    return {
        "passed": ic is None or abs(ic) < ic_ceiling,
        "reason": (
            f"OOS IC |{ic:.4f}| {'<' if ic is None or abs(ic) < ic_ceiling else '>='} "
            f"ceiling {ic_ceiling:.2f}"
            if ic is not None
            else "OOS IC unavailable"
        ),
        "leaked_feature_names": [],
        "injected_test_ic": ic,
        "ic_ceiling": ic_ceiling,
    }


def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)
    reports_globs = args.reports_glob or [cfg.get("reports_glob")]
    reports_globs = [g for g in reports_globs if g]
    if not reports_globs:
        raise SystemExit("No --reports-glob provided and no reports_glob in config.")
    seen: set[str] = set()
    paths: list[str] = []
    for pattern in reports_globs:
        for path in sorted(glob.glob(str(pattern))):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise SystemExit(f"No reports matched: {reports_globs}")

    horizons_days = [int(h) for h in cfg.get("horizons", [5, 20, 60])]
    horizon_names = [f"ret_{h}d" for h in horizons_days]
    benchmark = cfg.get("benchmark", "SPY")
    records = load_records(
        paths,
        horizons=horizons_days,
        capture_factor_scores=True,
        capture_market_context=True,
        benchmark_symbol=None if args.no_prices else benchmark,
    )
    feature_names = list((cfg.get("features") or {}).get("allowlist") or [])
    cohort_lookup = _load_universe_cohorts(args.universe_path)
    rows = build_ml_rows(
        records,
        feature_names=feature_names,
        horizons=horizon_names,
        cohort_lookup=cohort_lookup,
    )
    if not rows:
        raise SystemExit("No ML rows built.")

    dates = sorted({row.as_of_date for row in rows})
    wf = cfg.get("walk_forward") or {}
    windows = generate_windows(
        corpus_min_date=dates[0],
        corpus_max_date=dates[-1],
        train_months=int(wf.get("train_months", 18)),
        test_months=int(wf.get("test_months", 3)),
        step_months=int(wf.get("step_months", 1)),
    )
    gates_cfg = cfg.get("gates") or {}
    top_k = int(gates_cfg.get("top_k_per_week", 10))
    leak_ic_ceiling = float(gates_cfg.get("leak_sanity_ic_ceiling", 0.95))
    embargo_by_h = wf.get("embargo_days_by_horizon") or {}
    model_cfg = cfg.get("models") or {}
    model_names = list(model_cfg.keys())
    cohorts_present = sorted({(r.cohort or "other") for r in rows})
    cohort_eval = ["all"] + [c for c in ("core", "canary") if c in cohorts_present]

    leak_report: dict[str, Any] = {"skipped": True}
    if args.leak_sanity:
        leak_report = _run_leak_sanity(
            records=records,
            feature_names=feature_names,
            horizons_names=horizon_names,
            windows=windows,
            embargo_by_h=embargo_by_h,
            cohort_lookup=cohort_lookup,
            ic_ceiling=leak_ic_ceiling,
            model_cfg=model_cfg,
        )
        if not leak_report.get("passed", False):
            raise SystemExit(
                f"Leak sanity check FAILED: {leak_report.get('reason')}\n"
                f"  leaked_feature_names: {leak_report.get('leaked_feature_names')}\n"
                f"  injected_test_ic: {leak_report.get('injected_test_ic')}\n"
                "Aborting before publishing leaderboard."
            )

    per_window: list[dict[str, Any]] = []

    def _record(entry: dict[str, Any]) -> None:
        per_window.append(entry)

    for horizon in horizon_names:
        embargo = int(embargo_by_h.get(horizon, _horizon_days(horizon)))
        for window in windows:
            train_rows, test_rows = split_rows_for_window(rows, window, embargo_days=embargo)
            if len(train_rows) < 50 or len(test_rows) < 10:
                continue

            # Train once on the full train set; evaluate on each cohort
            # filter so the same model is scored on core vs canary subsets.
            trained: dict[str, tuple[Any, bool]] = {}
            for model_name in model_names:
                label_kind = "regression" if model_name == "ridge_return" else "alpha"
                raw_labels = labels_for_rows(train_rows, horizon=horizon, label_kind=label_kind)
                fit_rows, fit_labels = complete_cases(train_rows, raw_labels)
                if len(fit_rows) < 50 or len(set(fit_labels)) < 2:
                    continue
                X_train = rows_to_matrix(fit_rows, feature_names=feature_names)
                model = fit_model(
                    model_name, X_train, fit_labels,
                    config=model_cfg.get(model_name) or {},
                )
                train_scores = model.predict_scores(X_train)
                if model.task == "regression":
                    train_probs = [sigmoid(s * 10.0) for s in train_scores]
                else:
                    train_probs = train_scores
                # Per-model calibration on (train probs → train alpha labels).
                # Aligned pairwise to avoid index drift if any label is None.
                alpha_train = labels_for_rows(fit_rows, horizon=horizon, label_kind="alpha")
                cal_pairs = [(float(p), int(y)) for p, y in zip(train_probs, alpha_train) if y is not None]
                if cal_pairs:
                    cal_probs, cal_labels = zip(*cal_pairs)
                    cal = fit_isotonic_calibrator(list(cal_probs), list(cal_labels))
                else:
                    cal = None
                trained[model_name] = (model, cal)

            for cohort in cohort_eval:
                cohort_test_rows = _filter_rows_for_cohort(test_rows, cohort)
                if len(cohort_test_rows) < 10:
                    continue
                for baseline in ("factor_v1_8", "always_bullish", "spy_50dma"):
                    metrics = _evaluate_baseline(
                        baseline, cohort_test_rows, horizon=horizon, top_k=top_k,
                    )
                    _record({
                        "horizon": horizon,
                        "cohort": cohort,
                        "window": window.as_dict(),
                        **metrics,
                    })
                for model_name, (model, cal) in trained.items():
                    X_test = rows_to_matrix(cohort_test_rows, feature_names=feature_names)
                    test_scores = model.predict_scores(X_test)
                    if model.task == "regression":
                        test_probs = [sigmoid(s * 10.0) for s in test_scores]
                    else:
                        test_probs = test_scores
                    eval_scores = cal.predict(test_probs) if cal else test_probs
                    metrics = _classification_metrics(
                        cohort_test_rows, eval_scores, horizon=horizon, top_k=top_k,
                    )
                    _record({
                        "horizon": horizon,
                        "cohort": cohort,
                        "window": window.as_dict(),
                        "model": model_name,
                        "calibrated": cal is not None,
                        **metrics,
                    })

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in per_window:
        key = (item["horizon"], item["model"], item.get("cohort", "all"))
        bucket = grouped.setdefault(
            key, {"horizon": key[0], "model": key[1], "cohort": key[2], "windows": 0},
        )
        bucket["windows"] += 1
        for metric in (
            "headline_hit_rate",
            "alpha_hit_rate",
            "brier_alpha",
            "ic_benchmark_adjusted",
            "precision_at_top_k",
        ):
            bucket.setdefault(f"_{metric}", []).append(item.get(metric))
    leaderboard: list[dict[str, Any]] = []
    for bucket in grouped.values():
        row = {
            "horizon": bucket["horizon"],
            "model": bucket["model"],
            "cohort": bucket["cohort"],
            "windows": bucket["windows"],
        }
        for key, values in list(bucket.items()):
            if key.startswith("_"):
                row[key[1:]] = _mean(values)
        leaderboard.append(row)
    cohort_order = {c: i for i, c in enumerate(cohort_eval)}
    leaderboard = sorted(
        leaderboard,
        key=lambda r: (cohort_order.get(r["cohort"], 99), r["horizon"], r["model"]),
    )

    # Top-line leak guard: if ANY ML model's mean OOS IC exceeds the ceiling
    # the eval is suspect even without --leak-sanity. Flag it loudly.
    suspect_ml_rows = [
        r for r in leaderboard
        if r["model"] in set(model_names)
        and r.get("ic_benchmark_adjusted") is not None
        and abs(float(r["ic_benchmark_adjusted"])) >= leak_ic_ceiling
    ]
    if suspect_ml_rows and not leak_report.get("skipped"):
        leak_report["passed"] = False
        leak_report["reason"] = (
            "post-eval OOS IC exceeds ceiling — possible leakage in features"
        )
    leak_report["suspect_ml_rows"] = [
        {"model": r["model"], "horizon": r["horizon"], "cohort": r["cohort"],
         "ic_benchmark_adjusted": r.get("ic_benchmark_adjusted")}
        for r in suspect_ml_rows
    ]

    gates_check = _compute_gates_check(
        leaderboard,
        gates=gates_cfg,
        horizons=horizon_names,
        model_names=[n for n in model_names],
        cohorts=cohort_eval,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps({
        "config": args.config,
        "n_records": len(records),
        "n_rows": len(rows),
        "n_windows": len(windows),
        "cohorts_present": cohorts_present,
        "leaderboard": leaderboard,
        "gates_check": gates_check,
        "leak_sanity": leak_report,
        "per_window": per_window,
    }, indent=2))

    fmt = lambda v: "n/a" if v is None else f"{float(v):.4f}"
    lines = ["# ML shadow leaderboard", ""]
    if not leak_report.get("skipped"):
        verdict = "PASS" if leak_report.get("passed") else "FAIL"
        lines += [f"**Leak sanity:** {verdict} — {leak_report.get('reason')}", ""]
    if gates_check:
        any_pass = any(g["gate_pass"] for g in gates_check)
        lines += [
            f"**Gates verdict:** {'PASS (≥1 model)' if any_pass else 'FAIL (no ML model beats factor_v1_8)'}",
            "",
            "| Cohort | Model | Horizon | α-hit lift | top-k lift | worst α Δ | gate |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for g in gates_check:
            lines.append(
                f"| {g['cohort']} | {g['model']} | {g['horizon']} | "
                f"{fmt(g['alpha_hit_lift'])} | {fmt(g['precision_at_top_k_lift'])} | "
                f"{fmt(g['worst_horizon_alpha_delta'])} | "
                f"{'PASS' if g['gate_pass'] else '—'} |"
            )
        lines.append("")
    for cohort in cohort_eval:
        rows_for_cohort = [r for r in leaderboard if r["cohort"] == cohort]
        if not rows_for_cohort:
            continue
        lines += [
            f"## Cohort: {cohort}",
            "",
            "| Horizon | Model | Windows | Alpha hit | Top-k precision | IC adj | Brier |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows_for_cohort:
            lines.append(
                f"| {row['horizon']} | {row['model']} | {row['windows']} | "
                f"{fmt(row.get('alpha_hit_rate'))} | {fmt(row.get('precision_at_top_k'))} | "
                f"{fmt(row.get('ic_benchmark_adjusted'))} | {fmt(row.get('brier_alpha'))} |"
            )
        lines.append("")
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_dir / 'leaderboard.json'}")
    print(f"Wrote: {out_dir / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
