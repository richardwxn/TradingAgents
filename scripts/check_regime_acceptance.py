"""Gate regime-conditioned walk-forward results against the global baseline.

Reads the global `summary_walk_forward.json` and the regime-conditioned
`summary_walk_forward_by_regime.json` emitted by `backtest.py`, then enforces
the 60d-first acceptance criteria from the accuracy plan. This script never
writes `configs/regime_weights.json`; a passing result is only a gate signal
that a separate, explicit production write is allowed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_GLOBAL_SUMMARY = "backtest/results/regime_walk_forward_60d/summary_walk_forward.json"
DEFAULT_REGIME_SUMMARY = (
    "backtest/results/regime_walk_forward_60d/"
    "summary_walk_forward_by_regime.json"
)
EPSILON = 1e-12


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _bullish_stats(summary: dict[str, Any], horizon: str) -> dict[str, Any]:
    try:
        stats = summary["by_horizon"][horizon]["by_direction"]["bullish"]
    except KeyError as exc:
        raise KeyError(
            f"missing summary path by_horizon.{horizon}.by_direction.bullish"
        ) from exc
    if not isinstance(stats, dict):
        raise ValueError(f"bullish stats for {horizon} must be an object")
    return stats


def _float_metric(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key)
    if value is None:
        raise ValueError(f"missing numeric metric {key}")
    return float(value)


def _int_metric(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    if value is None:
        raise ValueError(f"missing integer metric {key}")
    return int(value)


def evaluate_regime_acceptance(
    *,
    global_summary: dict[str, Any],
    regime_summary: dict[str, Any],
    primary_horizon: str = "ret_60d",
    secondary_horizons: tuple[str, ...] = ("ret_20d", "ret_5d"),
    min_primary_hit_lift: float = 0.01,
    max_primary_median_regression: float = 0.01,
    max_secondary_hit_regression: float = 0.01,
    min_refit_steps: int = 20,
    min_primary_bullish_count: int = 500,
) -> tuple[bool, list[str], list[str]]:
    """Return (passed, messages, failures) for the regime acceptance gate."""
    messages: list[str] = []
    failures: list[str] = []

    global_primary = _bullish_stats(global_summary, primary_horizon)
    regime_primary = _bullish_stats(regime_summary, primary_horizon)
    global_hit = _float_metric(global_primary, "hit_rate")
    regime_hit = _float_metric(regime_primary, "hit_rate")
    hit_lift = regime_hit - global_hit
    if hit_lift + EPSILON < min_primary_hit_lift:
        failures.append(
            f"{primary_horizon} bullish hit-rate lift {hit_lift:+.2%} "
            f"< required {min_primary_hit_lift:+.2%}"
        )
    messages.append(
        f"{primary_horizon} bullish hit-rate: regime {regime_hit:.2%}, "
        f"global {global_hit:.2%}, lift {hit_lift:+.2%}"
    )

    global_median = _float_metric(global_primary, "median_forward_return")
    regime_median = _float_metric(regime_primary, "median_forward_return")
    median_delta = regime_median - global_median
    if median_delta + EPSILON < -max_primary_median_regression:
        failures.append(
            f"{primary_horizon} bullish median return delta {median_delta:+.2%} "
            f"< allowed {-max_primary_median_regression:+.2%}"
        )
    messages.append(
        f"{primary_horizon} bullish median return: regime {regime_median:.2%}, "
        f"global {global_median:.2%}, delta {median_delta:+.2%}"
    )

    for horizon in secondary_horizons:
        global_stats = _bullish_stats(global_summary, horizon)
        regime_stats = _bullish_stats(regime_summary, horizon)
        secondary_delta = (
            _float_metric(regime_stats, "hit_rate")
            - _float_metric(global_stats, "hit_rate")
        )
        if secondary_delta + EPSILON < -max_secondary_hit_regression:
            failures.append(
                f"{horizon} bullish hit-rate delta {secondary_delta:+.2%} "
                f"< allowed {-max_secondary_hit_regression:+.2%}"
            )
        messages.append(
            f"{horizon} bullish hit-rate delta: {secondary_delta:+.2%}"
        )

    config = regime_summary.get("regime_walk_forward_config") or {}
    n_steps = int(config.get("n_refit_steps") or 0)
    if n_steps < min_refit_steps:
        failures.append(f"refit steps {n_steps} < required {min_refit_steps}")
    messages.append(f"refit steps: {n_steps}")

    count = _int_metric(regime_primary, "count_with_return")
    if count < min_primary_bullish_count:
        failures.append(
            f"{primary_horizon} bullish records with returns {count} "
            f"< required {min_primary_bullish_count}"
        )
    messages.append(f"{primary_horizon} bullish records with returns: {count}")

    return not failures, messages, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-summary-json", default=DEFAULT_GLOBAL_SUMMARY)
    parser.add_argument("--regime-summary-json", default=DEFAULT_REGIME_SUMMARY)
    parser.add_argument("--primary-horizon", default="ret_60d")
    parser.add_argument(
        "--secondary-horizons", nargs="+", default=["ret_20d", "ret_5d"]
    )
    parser.add_argument("--min-primary-hit-lift", type=float, default=0.01)
    parser.add_argument(
        "--max-primary-median-regression", type=float, default=0.01
    )
    parser.add_argument(
        "--max-secondary-hit-regression", type=float, default=0.01
    )
    parser.add_argument("--min-refit-steps", type=int, default=20)
    parser.add_argument("--min-primary-bullish-count", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global_summary = _load_json(args.global_summary_json)
    regime_summary = _load_json(args.regime_summary_json)
    passed, messages, failures = evaluate_regime_acceptance(
        global_summary=global_summary,
        regime_summary=regime_summary,
        primary_horizon=args.primary_horizon,
        secondary_horizons=tuple(args.secondary_horizons),
        min_primary_hit_lift=args.min_primary_hit_lift,
        max_primary_median_regression=args.max_primary_median_regression,
        max_secondary_hit_regression=args.max_secondary_hit_regression,
        min_refit_steps=args.min_refit_steps,
        min_primary_bullish_count=args.min_primary_bullish_count,
    )
    for message in messages:
        print(message)
    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print()
    print("PASS: regime-conditioned weights clear the 60d acceptance gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
