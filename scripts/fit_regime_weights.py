"""Phase 4: fit per-regime factor weights from the walk-forward corpus.

Reads every report in `--reports-glob`, partitions records by regime
(trend_on vs chop, computed from each record's `market_context`), fits
IC-signed weights per regime, and writes a candidate map of:

    {
      "trend_on": {"factor_a": w_a, ...},
      "chop":     {"factor_a": w_a, ...}
    }

By default this writes under `backtest/results/regime_candidates/` so an
experiment cannot silently become live production config. To write
`configs/regime_weights.json`, pass both `--output configs/regime_weights.json`
and `--allow-production-output` after the walk-forward acceptance gate passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    explode_records_to_factors,
    fit_weights_from_records,
    rebuild_records_with_weights,
    spearman_correlation,
    summarize_factors,
)
from tradingagents.analysis_only.scoring import (  # noqa: E402
    REGIME_CHOP,
    REGIME_TREND_ON,
    REGIME_UNKNOWN,
    regime_for_market_context,
)


def _ic_for(records, horizon: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for r in records:
        if r.composite_score is None:
            continue
        v = r.forward_returns.get(horizon)
        if v is None:
            continue
        pairs.append((float(r.composite_score), float(v)))
    if len(pairs) < 30:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return spearman_correlation(xs, ys)


def build_diagnostics_payload(
    *,
    reports_glob: str,
    horizon: str,
    horizons: list[int],
    min_abs_ic: float,
    min_n: int,
    min_samples: int,
    require_regime_ic_ge_global: bool,
    global_ic: float | None,
    diagnostics: dict[str, dict],
) -> dict:
    return {
        "reports_glob": reports_glob,
        "horizon": horizon,
        "horizons": horizons,
        "min_abs_ic": min_abs_ic,
        "min_n": min_n,
        "min_samples": min_samples,
        "require_regime_ic_ge_global": require_regime_ic_ge_global,
        "global_ic": global_ic,
        "regimes": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-glob", default="reports/analysis_mvp/*.json",
    )
    parser.add_argument(
        "--output",
        default="backtest/results/regime_candidates/regime_weights_candidate.json",
    )
    parser.add_argument(
        "--diagnostics-output",
        default=None,
        help=(
            "Optional diagnostics JSON path. Defaults to "
            "<output-stem>_diagnostics.json."
        ),
    )
    parser.add_argument(
        "--allow-production-output",
        action="store_true",
        help=(
            "Required when --output is configs/regime_weights.json. Use only "
            "after scripts/check_regime_acceptance.py passes."
        ),
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[5, 20, 60],
    )
    parser.add_argument("--horizon", default="ret_20d")
    parser.add_argument("--min-abs-ic", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument(
        "--min-samples", type=int, default=250,
        help="Phase 4 gate: minimum samples per regime to ship a weight vector.",
    )
    parser.add_argument(
        "--require-regime-ic-ge-global",
        action="store_true",
        default=True,
        help=(
            "Drop a regime whose post-rebuild composite IC at --horizon "
            "is below the global IC (Phase 4 gate). Enabled by default."
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if (
        out_path.as_posix() == "configs/regime_weights.json"
        and not args.allow_production_output
    ):
        print(
            "[fail] refusing to write configs/regime_weights.json without "
            "--allow-production-output. Write a candidate artifact first and "
            "run scripts/check_regime_acceptance.py.",
            file=sys.stderr,
        )
        return 2

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2
    print(f"Loading {len(paths)} reports + forward returns...")
    records = load_records(
        paths,
        horizons=args.horizons,
        capture_factor_scores=True,
        capture_market_context=True,
        benchmark_symbol=None,
    )

    by_regime: dict[str, list] = {}
    for r in records:
        regime = regime_for_market_context(r.market_context or {})
        by_regime.setdefault(regime, []).append(r)
    for regime, recs in sorted(by_regime.items()):
        print(f"  regime={regime:<10} n={len(recs)}")

    print()
    print("Fitting global baseline...")
    global_weights = fit_weights_from_records(
        records,
        horizon=args.horizon,
        min_abs_ic=args.min_abs_ic,
        min_n=args.min_n,
    )
    global_rebuilt = rebuild_records_with_weights(
        records, weights=global_weights
    ) if global_weights else []
    global_ic = _ic_for(global_rebuilt, args.horizon)
    print(f"global IC ({args.horizon}): {global_ic}")

    out: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict] = {}
    for regime in (REGIME_TREND_ON, REGIME_CHOP):
        recs = by_regime.get(regime, [])
        diag: dict = {
            "n_records": len(recs),
            "shipped": False,
            "reason": None,
            "ic": None,
            "global_ic": global_ic,
            "ic_lift": None,
            "nonzero_factor_count": 0,
        }
        if len(recs) < args.min_samples:
            diag["reason"] = f"n_records<{args.min_samples}"
            diagnostics[regime] = diag
            continue
        weights = fit_weights_from_records(
            recs,
            horizon=args.horizon,
            min_abs_ic=args.min_abs_ic,
            min_n=args.min_n,
        )
        if not weights:
            diag["reason"] = "no_factors_passed_ic_threshold"
            diagnostics[regime] = diag
            continue
        diag["nonzero_factor_count"] = sum(1 for v in weights.values() if v)
        rebuilt = rebuild_records_with_weights(recs, weights=weights)
        regime_ic = _ic_for(rebuilt, args.horizon)
        diag["ic"] = regime_ic
        if global_ic is not None and regime_ic is not None:
            diag["ic_lift"] = regime_ic - global_ic
        if (
            args.require_regime_ic_ge_global
            and global_ic is not None
            and regime_ic is not None
            and regime_ic < global_ic
        ):
            diag["reason"] = (
                f"regime_ic {regime_ic:.4f} < global_ic {global_ic:.4f}"
            )
            diagnostics[regime] = diag
            continue
        out[regime] = weights
        diag["shipped"] = True
        diagnostics[regime] = diag

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    diag_path = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else out_path.with_name(f"{out_path.stem}_diagnostics.json")
    )
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_payload = build_diagnostics_payload(
        reports_glob=args.reports_glob,
        horizon=args.horizon,
        horizons=args.horizons,
        min_abs_ic=args.min_abs_ic,
        min_n=args.min_n,
        min_samples=args.min_samples,
        require_regime_ic_ge_global=args.require_regime_ic_ge_global,
        global_ic=global_ic,
        diagnostics=diagnostics,
    )
    diag_path.write_text(json.dumps(diag_payload, indent=2, sort_keys=True))
    print()
    print(f"Wrote {len(out)} regime(s) to {out_path}")
    print(f"Wrote diagnostics to {diag_path}")
    print("Diagnostics:")
    for regime, diag in sorted(diagnostics.items()):
        status = "SHIPPED" if diag["shipped"] else f"SKIPPED ({diag['reason']})"
        print(f"  {regime:<10} {status}  n={diag['n_records']}  ic={diag['ic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
