"""Split per-ticker IC table into core vs canary cohorts.

Loads the factor_summary_by_ticker.json from a backtest run and the
configs/universe.yaml definition, then reports for each factor:
- core-only median IC + sign consistency (drives weight tuning)
- canary-only median IC + sign consistency (diagnostic)
- sign agreement between cohorts (True if median IC has same sign)

Use after `python backtest.py --by-ticker` to see whether v1 factor
weights are tech-universe artifacts or generalize cross-sector.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--by-ticker-json",
        default="backtest/results/phase2/factor_summary_by_ticker.json",
        help="Path to factor_summary_by_ticker.json from a backtest run.",
    )
    ap.add_argument(
        "--universe",
        default="configs/universe.yaml",
        help="Universe yaml with `core:` and `canary:` cohorts.",
    )
    ap.add_argument(
        "--horizon",
        default="ret_20d",
        choices=("ret_5d", "ret_20d", "ret_60d"),
    )
    ap.add_argument(
        "--min-tickers",
        type=int,
        default=3,
        help="Minimum tickers in a cohort with non-null IC to report stats.",
    )
    args = ap.parse_args()

    with open(args.universe) as fh:
        univ = yaml.safe_load(fh) or {}
    core = {t.upper() for t in univ.get("core", [])}
    canary = {t.upper() for t in univ.get("canary", [])}

    with open(args.by_ticker_json) as fh:
        data = json.load(fh)

    print(f"# Core vs Canary IC split @ {args.horizon}")
    print(f"Core tickers ({len(core)}):   {' '.join(sorted(core))}")
    print(f"Canary tickers ({len(canary)}): {' '.join(sorted(canary))}")
    print()
    print(
        "| Factor | Core n | Core med IC | Core sign-cons | "
        "Canary n | Canary med IC | Sign agree? |"
    )
    print("|---|---:|---:|---:|---:|---:|:---:|")

    factor_rows = []
    for factor_name, payload in data.items():
        horizon_block = (payload.get("by_horizon") or {}).get(args.horizon) or {}
        per_ticker = horizon_block.get("per_ticker_ic") or {}
        core_ics, canary_ics = [], []
        for tkr, ic in per_ticker.items():
            if ic is None:
                continue
            t_up = tkr.upper()
            if t_up in core:
                core_ics.append(ic)
            elif t_up in canary:
                canary_ics.append(ic)

        def summarize(ics):
            if len(ics) < args.min_tickers:
                return None, None, None
            med = statistics.median(ics)
            cons = sum(1 for x in ics if (x >= 0) == (med >= 0)) / len(ics)
            return len(ics), med, cons

        core_n, core_med, core_cons = summarize(core_ics)
        can_n, can_med, _ = summarize(canary_ics)
        if core_n is None and can_n is None:
            continue

        sign_agree = (
            "—" if core_med is None or can_med is None
            else ("✓" if (core_med >= 0) == (can_med >= 0) else "✗")
        )

        factor_rows.append((
            factor_name,
            core_n, core_med, core_cons,
            can_n, can_med, sign_agree,
        ))

    factor_rows.sort(
        key=lambda r: abs(r[2]) if r[2] is not None else 0, reverse=True,
    )
    for fn, cn, cm, cc, an, am, ag in factor_rows:
        cm_s = f"{cm:+.4f}" if cm is not None else "—"
        cc_s = f"{cc*100:.0f}%" if cc is not None else "—"
        am_s = f"{am:+.4f}" if am is not None else "—"
        print(
            f"| {fn} | "
            f"{cn or '—'} | {cm_s} | {cc_s} | "
            f"{an or '—'} | {am_s} | {ag} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
