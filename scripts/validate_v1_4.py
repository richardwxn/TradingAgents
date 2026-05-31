"""Validate v1.4 weight commit on the Phase 2 corpus via post-hoc rebuild.

Phase 2 reports already on disk were generated under v1.3 (fear_greed
factor at weight 0.00 with contrarian sign). v1.4 changes two things in
production code:

  1. `score_fear_greed_regime` sign is inverted (fear → bearish for tech).
  2. `market_fear_greed_regime` weight goes 0.00 → 0.05.

To validate without regenerating the corpus, we exploit that:
  (existing contrarian score) × (negative weight)
  ≡ (inverted score) × (positive weight)

So we call `rebuild_records_with_weights` with
`market_fear_greed_regime: -0.05` against the v1 weight dict for
everything else, and the resulting composite matches what v1.4
production would compute.

Reports under three slices (TRAIN / TEST / FULL) using the protocol
from Section 13 plus the methodology gap from Section 21 that the user
flagged.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# Allow running from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradingagents.analysis_only.backtest import (
    rebuild_records_with_weights,
    summarize_all,
)
from tradingagents.analysis_only.scoring import DEFAULT_FACTOR_WEIGHTS

from backtest import load_records, filter_by_date_range


TRAIN_TO = "2025-08-29"
TEST_FROM = "2025-09-05"


def make_v1_3_weights() -> dict[str, float]:
    """v1.3: fear_greed at 0.00 (drop-to-zero), everything else as v1 had it."""
    w = dict(DEFAULT_FACTOR_WEIGHTS)
    w["market_fear_greed_regime"] = 0.00
    return w


def make_v1_4_weights() -> dict[str, float]:
    """v1.4 simulated via negative-weight trick.

    Production v1.4 = (sign-flipped score) × (+0.05 weight). The existing
    corpus has the contrarian score, so against that we use -0.05.
    """
    w = dict(DEFAULT_FACTOR_WEIGHTS)
    w["market_fear_greed_regime"] = -0.05
    return w


def main() -> int:
    paths = sorted(glob.glob("reports/analysis_mvp/*.json"))
    print(f"Loading {len(paths)} reports with factor_scores...")
    records = load_records(
        paths,
        horizons=[5, 20, 60],
        capture_factor_scores=True,
        benchmark_symbol="SPY",
    )
    print(f"Loaded {len(records)} records.\n")

    slices = {
        "TRAIN (2023-07-14 → 2025-08-29)":
            filter_by_date_range(records, date_to=TRAIN_TO),
        "TEST (2025-09-05 → 2026-05-22)":
            filter_by_date_range(records, date_from=TEST_FROM),
        "FULL": records,
    }

    return_fields = {5: "ret_5d", 20: "ret_20d", 60: "ret_60d"}.values()

    for slice_name, slice_records in slices.items():
        print(f"\n{'=' * 78}")
        print(f"### {slice_name}  (n={len(slice_records)})")
        print("=" * 78)

        for label, weights in (("v1.3 (fear_greed=0.00)", make_v1_3_weights()),
                                ("v1.4 (fear_greed inverted, weight=0.05)",
                                 make_v1_4_weights())):
            rebuilt = rebuild_records_with_weights(
                slice_records, weights=weights, direction_threshold=0.15,
            )
            summary = summarize_all(
                rebuilt, return_fields=list(return_fields),
            )
            print(f"\n--- {label} ---")
            for horizon in ("ret_5d", "ret_20d", "ret_60d"):
                h = summary["by_horizon"].get(horizon, {})
                by_dir = h.get("by_direction", {})
                bull = by_dir.get("bullish", {})
                bear = by_dir.get("bearish", {})
                neutral = by_dir.get("neutral", {})

                def fmt(b: dict) -> str:
                    n = b.get("count_with_return", 0)
                    hit = b.get("hit_rate")
                    mean = b.get("mean")
                    hit_s = f"{hit*100:5.1f}%" if hit is not None else "    —"
                    mean_s = f"{mean*100:+5.2f}%" if mean is not None else "    —"
                    return f"n={n:>4} hit={hit_s} mean={mean_s}"

                print(
                    f"  {horizon}: bull[{fmt(bull)}]  bear[{fmt(bear)}]  "
                    f"neut[{fmt(neutral)}]"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
