#!/usr/bin/env python
"""Fit per-horizon IC-signed weight vectors from a corpus.

Today the system uses ONE `DEFAULT_FACTOR_WEIGHTS` vector applied to all
forward horizons (5d / 20d / 60d). But factors don't have the same IC
across horizons — `options_iv_term_structure` is strongest at 20-60d,
news/momentum signals at 5d, fundamentals at 60d. A per-horizon weight
vector tunes the composite to each horizon's strongest signals.

This script:
  1. Loads a (multi-)corpus from one or more `--reports-glob` patterns.
  2. Computes `summarize_factors` across [ret_5d, ret_20d, ret_60d].
  3. For each horizon, derives `ic_signed_weights(horizon=h)` with
     configurable `--min-abs-ic` and `--min-n` gates.
  4. Writes a single JSON: `{"ret_5d": {...}, "ret_20d": {...}, "ret_60d": {...}}`.

The output is consumed by `walk_forward_eval.py --weight-source per_horizon_json
--weights-json <path>` (extended in this commit) to gate per-horizon vs the
current single-vector model on the SAME walk-forward windows.

Production scoring is unchanged until the gate passes.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    explode_records_to_factors,
    ic_signed_weights,
    summarize_factors,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reports-glob", action="append", required=True,
        help="Glob for report JSONs. May be passed multiple times.",
    )
    p.add_argument(
        "--horizons", nargs="+", default=["ret_5d", "ret_20d", "ret_60d"],
    )
    p.add_argument(
        "--horizons-int", nargs="+", type=int, default=[5, 20, 60],
        help="Trading-day equivalents of --horizons, used for the price fetch.",
    )
    p.add_argument(
        "--benchmark", default=None,
        help="Optional benchmark symbol for benchmark-adjusted IC. "
        "If omitted, raw forward-return IC is used.",
    )
    p.add_argument("--no-prices", action="store_true",
                   help="Assume corpus records already carry forward returns.")
    p.add_argument(
        "--min-abs-ic", type=float, default=0.05,
        help="Minimum |IC| to keep a factor (defaults match ic_signed_rolling).",
    )
    p.add_argument(
        "--min-n", type=int, default=50,
        help="Minimum paired observations per factor.",
    )
    p.add_argument(
        "--use-benchmark-adjusted", action="store_true",
        help="Compute IC against benchmark-adjusted returns (requires --benchmark).",
    )
    p.add_argument(
        "--output", default="backtest/results/per_horizon_weights.json",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in args.reports_glob:
        for p in sorted(glob.glob(pattern)):
            if p not in seen:
                paths.append(p)
                seen.add(p)
    if not paths:
        raise SystemExit(f"No reports matched: {args.reports_glob}")

    print(f"Loading {len(paths)} reports...")
    t0 = time.time()
    records = load_records(
        paths,
        horizons=args.horizons_int,
        capture_factor_scores=True,
        capture_market_context=False,
        capture_llm_critic=False,
        benchmark_symbol=args.benchmark if not args.no_prices else None,
    )
    print(f"Loaded {len(records)} records in {time.time() - t0:.1f}s.")

    by_factor = explode_records_to_factors(records)
    if not by_factor:
        raise SystemExit("No factor records exploded — corpus missing factor_scores?")
    summary = summarize_factors(
        by_factor,
        return_fields=list(args.horizons),
        use_benchmark_adjusted=args.use_benchmark_adjusted,
    )

    per_horizon: dict[str, dict[str, float]] = {}
    for horizon in args.horizons:
        weights = ic_signed_weights(
            summary, horizon=horizon,
            min_abs_ic=args.min_abs_ic, min_n=args.min_n,
        )
        per_horizon[horizon] = weights
        kept = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        print(
            f"[{horizon}] kept {len(weights)} factors "
            f"(|IC|≥{args.min_abs_ic}, n≥{args.min_n})"
        )
        for name, ic in kept[:8]:
            print(f"    {name:>40s}  IC={ic:+.4f}")
        if len(kept) > 8:
            print(f"    ... ({len(kept) - 8} more)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "horizons": list(args.horizons),
        "min_abs_ic": args.min_abs_ic,
        "min_n": args.min_n,
        "use_benchmark_adjusted": bool(args.use_benchmark_adjusted),
        "n_records": len(records),
        "weights_by_horizon": per_horizon,
    }, indent=2) + "\n")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
