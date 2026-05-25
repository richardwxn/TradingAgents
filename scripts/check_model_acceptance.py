"""Validate backtest and simulator artifacts against model-quality gates.

This script is intentionally artifact-only: it never fetches prices or
regenerates reports. Run the yfinance-backed backtests first, then use this as
the final gate for a proposed scoring / weighting / sizing change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = "backtest/results/test/summary_override.json"
DEFAULT_POLICY = "backtest/results/simulator_test/sim_confidence_weighted.json"
DEFAULT_BENCHMARK = "backtest/results/simulator_test/sim_SPY_baseline.json"

# Phase 3 gate: the honest accuracy headline is the walk-forward summary,
# not the in-sample one. When `--walk-forward-summary-json` is passed
# (or this default file exists), the acceptance check sources hit-rate
# and count from the walk-forward summary instead of `--summary-json`.
DEFAULT_WALK_FORWARD_SUMMARY = "backtest/results/summary_walk_forward.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _direction_stats(
    summary: dict[str, Any],
    *,
    horizon: str,
    direction: str,
) -> dict[str, Any]:
    try:
        stats = summary["by_horizon"][horizon]["by_direction"][direction]
    except KeyError as exc:
        raise KeyError(
            f"missing summary path by_horizon.{horizon}.by_direction.{direction}"
        ) from exc
    if not isinstance(stats, dict):
        raise ValueError(f"summary stats for {horizon}/{direction} must be an object")
    return stats


def _metric(payload: dict[str, Any], name: str) -> float:
    try:
        return float(payload["metrics"][name])
    except KeyError as exc:
        raise KeyError(f"missing simulator metric metrics.{name}") from exc


def _check_min(
    failures: list[str],
    label: str,
    actual: float,
    minimum: float,
) -> None:
    if actual < minimum:
        failures.append(f"{label}: {actual:.4f} < required {minimum:.4f}")


def _check_no_regression(
    failures: list[str],
    label: str,
    actual: float,
    baseline: float,
    tolerance: float,
) -> None:
    floor = baseline - tolerance
    if actual < floor:
        failures.append(
            f"{label}: {actual:.4f} regressed below baseline {baseline:.4f} "
            f"(tolerance {tolerance:.4f})"
        )


def evaluate_acceptance(
    *,
    summary: dict[str, Any],
    policy: dict[str, Any],
    benchmark: dict[str, Any],
    baseline_summary: dict[str, Any] | None = None,
    baseline_policy: dict[str, Any] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    horizon: str = "ret_20d",
    direction: str = "bullish",
    min_hit_rate: float = 0.60,
    min_count: int = 30,
    min_excess_cagr: float = 0.0,
    min_information_ratio: float = 0.0,
    max_drawdown_floor: float = -0.25,
    hit_rate_tolerance: float = 0.0,
    cagr_tolerance: float = 0.0,
    sharpe_tolerance: float = 0.0,
) -> tuple[bool, list[str], list[str]]:
    """Return (passed, messages, failures) for artifact quality checks.

    If `walk_forward_summary` is supplied, the headline hit-rate / count
    used for the gate come from THAT summary, not from `summary`. The
    in-sample `summary` is still printed for context but does not gate.
    """
    messages: list[str] = []
    failures: list[str] = []

    if walk_forward_summary is not None:
        stats = _direction_stats(
            walk_forward_summary, horizon=horizon, direction=direction
        )
        in_sample_stats = _direction_stats(
            summary, horizon=horizon, direction=direction
        )
        messages.append(
            "gate source: WALK-FORWARD summary "
            f"(in-sample {direction} {horizon} hit-rate: "
            f"{float(in_sample_stats.get('hit_rate') or 0):.2%})"
        )
    else:
        stats = _direction_stats(summary, horizon=horizon, direction=direction)
        messages.append("gate source: in-sample summary (no walk-forward provided)")
    hit_rate = float(stats.get("hit_rate") or 0.0)
    count = int(stats.get("count_with_return") or 0)
    policy_cagr = _metric(policy, "cagr")
    policy_sharpe = _metric(policy, "sharpe_annualized")
    policy_max_dd = _metric(policy, "max_drawdown")
    benchmark_cagr = _metric(benchmark, "cagr")
    excess_cagr = policy_cagr - benchmark_cagr

    # Approximate information ratio using the simulator's Sharpe spread. The
    # simulator markdown computes a fuller IR, but per-policy JSONs intentionally
    # keep only core metrics; this remains useful as a cheap acceptance gate.
    benchmark_sharpe = _metric(benchmark, "sharpe_annualized")
    information_ratio_proxy = policy_sharpe - benchmark_sharpe

    _check_min(failures, f"{direction} {horizon} hit-rate", hit_rate, min_hit_rate)
    _check_min(failures, f"{direction} {horizon} count", float(count), float(min_count))
    _check_min(failures, "policy excess CAGR vs benchmark", excess_cagr, min_excess_cagr)
    _check_min(
        failures,
        "policy information-ratio proxy vs benchmark",
        information_ratio_proxy,
        min_information_ratio,
    )
    _check_min(failures, "policy max drawdown", policy_max_dd, max_drawdown_floor)

    messages.extend(
        [
            f"{direction} {horizon} hit-rate: {hit_rate:.2%} (n={count})",
            f"policy CAGR: {policy_cagr:.2%}",
            f"benchmark CAGR: {benchmark_cagr:.2%}",
            f"excess CAGR: {excess_cagr:.2%}",
            f"Sharpe spread proxy: {information_ratio_proxy:+.2f}",
            f"max drawdown: {policy_max_dd:.2%}",
        ]
    )

    if baseline_summary is not None:
        baseline_stats = _direction_stats(
            baseline_summary, horizon=horizon, direction=direction
        )
        baseline_hit = float(baseline_stats.get("hit_rate") or 0.0)
        _check_no_regression(
            failures,
            f"{direction} {horizon} hit-rate",
            hit_rate,
            baseline_hit,
            hit_rate_tolerance,
        )
        messages.append(f"baseline {direction} {horizon} hit-rate: {baseline_hit:.2%}")

    if baseline_policy is not None:
        baseline_cagr = _metric(baseline_policy, "cagr")
        baseline_sharpe = _metric(baseline_policy, "sharpe_annualized")
        _check_no_regression(
            failures,
            "policy CAGR",
            policy_cagr,
            baseline_cagr,
            cagr_tolerance,
        )
        _check_no_regression(
            failures,
            "policy Sharpe",
            policy_sharpe,
            baseline_sharpe,
            sharpe_tolerance,
        )
        messages.append(f"baseline policy CAGR: {baseline_cagr:.2%}")
        messages.append(f"baseline policy Sharpe: {baseline_sharpe:+.2f}")

    return not failures, messages, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--policy-json", default=DEFAULT_POLICY)
    parser.add_argument("--benchmark-json", default=DEFAULT_BENCHMARK)
    parser.add_argument("--baseline-summary-json")
    parser.add_argument("--baseline-policy-json")
    parser.add_argument(
        "--walk-forward-summary-json",
        default=None,
        help=(
            "If set, use this walk-forward summary as the gate source "
            "instead of --summary-json. Pass --walk-forward-auto to "
            "auto-detect the default path."
        ),
    )
    parser.add_argument(
        "--walk-forward-auto",
        action="store_true",
        help=(
            "If true, use DEFAULT_WALK_FORWARD_SUMMARY when it exists "
            "(no need to repeat the path). Honors "
            "--walk-forward-summary-json over the default."
        ),
    )
    parser.add_argument("--horizon", default="ret_20d")
    parser.add_argument("--direction", default="bullish")
    parser.add_argument("--min-hit-rate", type=float, default=0.60)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--min-excess-cagr", type=float, default=0.0)
    parser.add_argument("--min-information-ratio", type=float, default=0.0)
    parser.add_argument("--max-drawdown-floor", type=float, default=-0.25)
    parser.add_argument("--hit-rate-tolerance", type=float, default=0.0)
    parser.add_argument("--cagr-tolerance", type=float, default=0.0)
    parser.add_argument("--sharpe-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wf_path = args.walk_forward_summary_json
    if wf_path is None and args.walk_forward_auto:
        if Path(DEFAULT_WALK_FORWARD_SUMMARY).is_file():
            wf_path = DEFAULT_WALK_FORWARD_SUMMARY
    try:
        passed, messages, failures = evaluate_acceptance(
            summary=_load_json(args.summary_json),
            policy=_load_json(args.policy_json),
            benchmark=_load_json(args.benchmark_json),
            baseline_summary=(
                _load_json(args.baseline_summary_json)
                if args.baseline_summary_json
                else None
            ),
            baseline_policy=(
                _load_json(args.baseline_policy_json)
                if args.baseline_policy_json
                else None
            ),
            walk_forward_summary=(
                _load_json(wf_path) if wf_path else None
            ),
            horizon=args.horizon,
            direction=args.direction,
            min_hit_rate=args.min_hit_rate,
            min_count=args.min_count,
            min_excess_cagr=args.min_excess_cagr,
            min_information_ratio=args.min_information_ratio,
            max_drawdown_floor=args.max_drawdown_floor,
            hit_rate_tolerance=args.hit_rate_tolerance,
            cagr_tolerance=args.cagr_tolerance,
            sharpe_tolerance=args.sharpe_tolerance,
        )
    except Exception as exc:
        print(f"model acceptance: ERROR: {exc}", file=sys.stderr)
        return 2

    print("model acceptance:")
    for line in messages:
        print(f"  - {line}")
    if failures:
        print("\nfailed gates:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
