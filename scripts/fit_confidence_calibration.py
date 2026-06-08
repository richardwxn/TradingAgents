"""Phase 5: fit isotonic calibration of composite -> realized hit rate.

Loads every report under `--reports-glob`, attaches forward returns,
fits a monotone non-decreasing PAV isotonic regression of composite_score
-> realized hit-rate at the configured `--horizon`, and writes the
calibration JSON to `--output` (default: `configs/confidence_calibration.json`).

Also prints the Brier score before vs. after calibration and a reliability
diagram so the user can verify the Phase 5 gate:
  - Brier score improves vs the heuristic baseline.
  - Reliability within +/-5pp of the diagonal across deciles.

The pipeline auto-loads this file when `confidence_calibration_path`
is supplied to `AnalysisOnlyMVP`.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    walk_forward_backtest,
)
from tradingagents.analysis_only.scoring import (  # noqa: E402
    apply_isotonic_calibration,
    brier_score,
    confidence_for,
    fit_isotonic_calibration,
    fit_isotonic_calibration_by_direction,
    reliability_diagram,
    save_isotonic_calibration,
)


def _hit_from_direction(direction: str, ret: float | None) -> int | None:
    if ret is None:
        return None
    if direction == "bullish":
        return 1 if ret > 0 else 0
    if direction == "bearish":
        return 1 if ret < 0 else 0
    if direction == "neutral":
        return 1 if abs(ret) < 0.02 else 0
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    parser.add_argument(
        "--output", default="configs/confidence_calibration.json",
    )
    parser.add_argument("--horizon", default="ret_20d")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=True,
        help=(
            "Fit on the walk-forward OUT-OF-SAMPLE record set instead of "
            "the full corpus (Phase 5 spec). Enabled by default — pass "
            "--no-walk-forward to opt out."
        ),
    )
    parser.add_argument(
        "--no-walk-forward", dest="walk_forward",
        action="store_false",
    )
    parser.add_argument("--wf-refit-freq-weeks", type=int, default=4)
    parser.add_argument("--wf-train-window-weeks", type=int, default=52)
    parser.add_argument("--wf-gap-weeks", type=int, default=4)
    parser.add_argument("--wf-first-refit-after-weeks", type=int, default=26)
    parser.add_argument("--wf-min-n", type=int, default=50)
    parser.add_argument("--wf-min-abs-ic", type=float, default=0.05)
    parser.add_argument(
        "--min-obs", type=int, default=30,
        help="Minimum total observations required to ship a calibration.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2
    print(f"Loading {len(paths)} reports + forward returns...")
    records = load_records(
        paths,
        horizons=args.horizons,
        capture_factor_scores=True,
        capture_market_context=False,
        benchmark_symbol=None,
    )
    if args.walk_forward:
        records, _ = walk_forward_backtest(
            records,
            refit_freq_weeks=args.wf_refit_freq_weeks,
            train_window_weeks=args.wf_train_window_weeks,
            gap_weeks=args.wf_gap_weeks,
            first_refit_after_weeks=args.wf_first_refit_after_weeks,
            horizon=args.horizon,
            min_abs_ic=args.wf_min_abs_ic,
            min_n=args.wf_min_n,
        )
        print(f"Walk-forward OOS records: {len(records)}")

    composites: list[float] = []
    hits: list[int] = []
    coverages: list[float] = []
    directions: list[str] = []
    for r in records:
        if r.composite_score is None:
            continue
        ret = r.forward_returns.get(args.horizon)
        hit = _hit_from_direction(r.direction, ret)
        if hit is None:
            continue
        composites.append(float(r.composite_score))
        hits.append(hit)
        directions.append(str(r.direction or "neutral").lower())
        # Coverage approximation: assume 1.0 unless we can derive it
        # from factor_scores (which we do here for honesty).
        if r.factor_scores:
            active_w = sum(
                float(f.get("weight") or 0.0) for f in r.factor_scores
                if f.get("data_available")
            )
            total_w = sum(
                float(f.get("weight") or 0.0) for f in r.factor_scores
            )
            coverages.append(active_w / total_w if total_w > 0 else 1.0)
        else:
            coverages.append(1.0)

    print(f"Calibration-eligible observations: {len(composites)}")
    if len(composites) < args.min_obs:
        print(
            f"[fail] need >= {args.min_obs} observations to fit "
            f"calibration; got {len(composites)}",
            file=sys.stderr,
        )
        return 1

    # Section 29: produce a direction-conditional calibration. Output
    # retains the top-level `fit` (all-directions curve) as a fallback
    # AND adds `by_direction` with separate curves per direction. The
    # bearish curve is expected to be nearly flat (Section 14/15
    # finding); seeing it explicitly in the fit lets the daily layer
    # downweight bearish exposure properly.
    cal = fit_isotonic_calibration_by_direction(
        composites, hits, directions, min_obs_per_direction=args.min_obs,
    )
    save_isotonic_calibration(cal, args.output)
    by_dir = cal.get("by_direction") or {}
    summary = ", ".join(
        f"{d}: {len(by_dir.get(d, {}).get('fit') or [])} segs"
        for d in ("bullish", "bearish", "neutral")
    )
    print(
        f"Wrote {args.output} "
        f"(all-direction segments={len(cal.get('fit') or [])}; "
        f"by-direction: {summary})"
    )

    # Acceptance gate (per Phase 5 spec) is on the raw calibration map
    # composite -> realized hit rate. Baseline is the simplest naive forecast:
    # everyone gets the base-rate.
    base_rate = sum(hits) / len(hits)
    baseline_probs = [base_rate] * len(hits)
    calibrated_probs = []
    for c in composites:
        prob = apply_isotonic_calibration(c, calibration=cal)
        calibrated_probs.append(prob if prob is not None else base_rate)
    # Also compute heuristic-confidence-as-probability for a more
    # informative diagnostic vs the deployed status quo.
    heuristic_probs = [
        confidence_for(c, cov) for c, cov in zip(composites, coverages)
    ]

    brier_base = brier_score(baseline_probs, hits)
    brier_heur = brier_score(heuristic_probs, hits)
    brier_cal = brier_score(calibrated_probs, hits)
    rel = reliability_diagram(calibrated_probs, hits, n_bins=10)

    print()
    print(f"Base rate (corpus hit rate):    {base_rate:.4f}")
    print(f"Brier score (base-rate naive):  {brier_base}")
    print(f"Brier score (heuristic conf):   {brier_heur}")
    print(f"Brier score (isotonic):         {brier_cal}")
    print(f"  delta vs base-rate:           "
          f"{(brier_base or 0) - (brier_cal or 0):+.4f} (positive = better)")
    print(f"  delta vs heuristic:           "
          f"{(brier_heur or 0) - (brier_cal or 0):+.4f} (positive = better)")
    print()
    print("Reliability (isotonic calibration vs realized):")
    print("  bucket [lo, hi)  n   mean_pred  observed  gap")
    max_gap_pp = 0.0
    for b in rel:
        if b["n"] == 0:
            continue
        gap = (b["observed_hit_rate"] - b["mean_predicted"]) * 100
        if b["n"] >= 10:
            max_gap_pp = max(max_gap_pp, abs(gap))
        warn = "  WARN" if abs(gap) > 5 else ""
        print(
            f"  [{b['lower']:.2f}, {b['upper']:.2f})  "
            f"{b['n']:>3}  {b['mean_predicted']:>8.4f}  "
            f"{b['observed_hit_rate']:>8.4f}  {gap:+6.1f}pp{warn}"
        )
    print()
    print(f"Max |gap| across reliability buckets with n>=10: "
          f"{max_gap_pp:.1f}pp (gate: <= 5pp)")
    gate_brier = (brier_cal or 1.0) < (brier_heur or 0.0)
    gate_reliability = max_gap_pp <= 5.0
    print(f"Phase 5 gate — Brier improves vs heuristic: "
          f"{'PASS' if gate_brier else 'FAIL'}")
    print(f"Phase 5 gate — reliability +/-5pp:           "
          f"{'PASS' if gate_reliability else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
