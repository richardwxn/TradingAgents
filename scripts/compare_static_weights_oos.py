#!/usr/bin/env python
"""Head-to-head OOS comparison of two FIXED weight vectors via a clean
chronological holdout.

The rolling walk-forward (`walk_forward_eval.py --weight-source
per_horizon_rolling`) validates a *recipe* (refit-per-window) — it never
tests a single static vector out-of-sample. To decide whether to REPLACE an
already-committed static vector (e.g. `PER_HORIZON_WEIGHTS["ret_20d"]`) with a
freshly-fit one, we need a leakage-free static-vs-static test.

Protocol (per `--split-date`):
  - train = corpus[min .. split]; test = corpus(split .. max], disjoint.
  - Fit a NEW vector on the TRAIN slice only (same recipe as
    `fit_per_horizon_weights.py`: IC-signed weights at --fit-min-abs-ic /
    --fit-min-n). This vector never sees the test slice.
  - Evaluate every candidate vector on the SAME test slice and report the
    bullish test-hit (the model's decision metric — bearish is
    anti-predictive per handoff Section 14/15) plus all-direction test-hit
    and the bullish sample size.

Candidates evaluated on each test slice:
  - `old_committed`     — PER_HORIZON_WEIGHTS[horizon] (the incumbent).
  - `new_train_fit`     — recipe fit on TRAIN only (strict OOS on test).
  - `v1.5_global`       — DEFAULT_FACTOR_WEIGHTS (reference baseline).
  - `new_full_corpus`   — optional, from --new-weights-json. This saw the
    test slice, so its test-hit is IN-SAMPLE — printed for reference only,
    NEVER as the decision basis.

Decision rule: replace the incumbent only if `new_train_fit` beats
`old_committed` on bullish test-hit across the split dates, with adequate
bullish sample size. All scoring goes through the already-unit-tested
`evaluate_window` / `rebuild_records_with_weights` primitives, so this script
adds no new scoring math.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    BacktestRecord,
    explode_records_to_factors,
    ic_signed_weights,
    summarize_factors,
)
from tradingagents.analysis_only.scoring import (  # noqa: E402
    DEFAULT_FACTOR_WEIGHTS,
    PER_HORIZON_WEIGHTS,
)
from tradingagents.analysis_only.walk_forward import (  # noqa: E402
    WalkForwardWindow,
    evaluate_window,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-glob", action="append", required=True)
    p.add_argument("--horizon", default="ret_20d")
    p.add_argument(
        "--split-date", action="append", required=True,
        help="ISO date; train<=split, test>split. May be repeated.",
    )
    p.add_argument("--fit-min-abs-ic", type=float, default=0.04)
    p.add_argument("--fit-min-n", type=int, default=50)
    p.add_argument(
        "--new-weights-json", default=None,
        help="fit_per_horizon_weights.py output; supplies new_full_corpus "
        "(in-sample reference only).",
    )
    return p.parse_args()


def _horizon_to_days(field: str) -> int:
    return int(field.replace("ret_", "").replace("d", ""))


def _fit_recipe(train: list[BacktestRecord], horizon: str,
                min_abs_ic: float, min_n: int) -> dict[str, float]:
    by_factor = explode_records_to_factors(train)
    if not by_factor:
        return {}
    summary = summarize_factors(by_factor, return_fields=[horizon])
    return ic_signed_weights(
        summary, horizon=horizon, min_abs_ic=min_abs_ic, min_n=min_n,
    )


def _eval_static(records: list[BacktestRecord], window: WalkForwardWindow,
                 vector: dict[str, float], horizon: str) -> dict:
    res = evaluate_window(
        records, window,
        weight_fn=(lambda _train, v=vector: dict(v) if v else None),
        horizons=[horizon],
    )
    return res["per_horizon"][horizon]


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

    horizon = args.horizon
    print(f"Loading {len(paths)} reports...")
    records = load_records(
        paths, horizons=[_horizon_to_days(horizon)],
        capture_factor_scores=True, capture_market_context=False,
        capture_llm_critic=False, benchmark_symbol=None,
    )
    dates = sorted(r.as_of_date for r in records)
    corpus_min, corpus_max = dates[0], dates[-1]
    print(f"Loaded {len(records)} records, span {corpus_min} -> {corpus_max}.")

    old_committed = dict(PER_HORIZON_WEIGHTS.get(horizon) or {})
    v15 = dict(DEFAULT_FACTOR_WEIGHTS)
    new_full = None
    if args.new_weights_json:
        payload = json.loads(Path(args.new_weights_json).read_text())
        new_full = dict(
            (payload.get("weights_by_horizon") or {}).get(horizon) or {}
        )

    if not old_committed:
        print(f"[warn] no committed override for {horizon}; "
              f"old_committed will fall back to nothing (skipped).")

    print(f"\n== Head-to-head OOS, horizon={horizon} "
          f"(fit |IC|>={args.fit_min_abs_ic}, n>={args.fit_min_n}) ==")
    overall_new_wins = 0
    overall_evaluable = 0
    for split in args.split_date:
        from datetime import date, timedelta
        y, m, d = (int(x) for x in split.split("-"))
        test_start = (date(y, m, d) + timedelta(days=1)).isoformat()
        window = WalkForwardWindow(
            train_start=corpus_min, train_end=split,
            test_start=test_start, test_end=corpus_max,
        )
        train = [r for r in records if corpus_min <= r.as_of_date <= split]
        new_train_fit = _fit_recipe(
            train, horizon, args.fit_min_abs_ic, args.fit_min_n
        )
        candidates: list[tuple[str, dict[str, float], bool]] = [
            ("old_committed", old_committed, False),
            ("new_train_fit", new_train_fit, False),
            ("v1.5_global", v15, False),
        ]
        if new_full is not None:
            candidates.append(("new_full_corpus*", new_full, True))

        print(f"\n--- split {split}  (train {corpus_min}..{split}, "
              f"test {test_start}..{corpus_max}, n_train={len(train)}) ---")
        print(f"  new_train_fit kept {len(new_train_fit)} factors: "
              + ", ".join(
                  f"{k}={v:+.3f}" for k, v in sorted(
                      new_train_fit.items(), key=lambda kv: -abs(kv[1]))
              ))
        print(f"  {'candidate':18} {'bull_test_hit':>13} {'n_bull':>7} "
              f"{'alldir_test_hit':>16} {'n_test':>7}")
        bull = {}
        for name, vec, leaky in candidates:
            if not vec:
                continue
            block = _eval_static(records, window, vec, horizon)
            bt = block.get("bullish_test_hit")
            nb = block.get("n_bullish_test") or 0
            at = block.get("test_hit")
            nt = block.get("n_test_with_return") or 0
            tag = "  (in-sample)" if leaky else ""
            bt_s = f"{bt*100:6.2f}%" if bt is not None else "   —  "
            at_s = f"{at*100:6.2f}%" if at is not None else "   —  "
            print(f"  {name:18} {bt_s:>13} {nb:>7} {at_s:>16} {nt:>7}{tag}")
            if not leaky:
                bull[name] = (bt, nb)

        oc = bull.get("old_committed", (None, 0))
        nf = bull.get("new_train_fit", (None, 0))
        if oc[0] is not None and nf[0] is not None:
            overall_evaluable += 1
            delta = (nf[0] - oc[0]) * 100
            verdict = "NEW wins" if nf[0] > oc[0] else (
                "tie" if nf[0] == oc[0] else "OLD wins")
            if nf[0] > oc[0]:
                overall_new_wins += 1
            print(f"  => new_train_fit vs old_committed bullish: "
                  f"{delta:+.2f}pp  [{verdict}]")

    print(f"\n== VERDICT: new_train_fit beat old_committed on "
          f"{overall_new_wins}/{overall_evaluable} splits ==")
    print("Decision rule: replace incumbent only on a clear OOS majority "
          "win with adequate bullish n. (new_full_corpus* is in-sample — "
          "reference only.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
