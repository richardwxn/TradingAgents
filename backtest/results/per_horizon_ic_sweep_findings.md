# Per-horizon weight vector findings

Source of record for the `PER_HORIZON_WEIGHTS` overrides in
`tradingagents/analysis_only/scoring.py`. Per-horizon vectors tune the
emitted `per_horizon_composites.<horizon>` to each forward horizon's
strongest IC-signed factors instead of reusing the single global
`DEFAULT_FACTOR_WEIGHTS` vector at every horizon.

## Methodology

1. **Fit** — `scripts/fit_per_horizon_weights.py` over the full corpus
   (`reports/analysis_mvp/*.json`), IC-signed weights per horizon at
   `--min-abs-ic 0.04 --min-n 50`. Output:
   `backtest/results/per_horizon_weights.json`.
2. **Gate** — `scripts/walk_forward_eval.py` over the same corpus,
   comparing three weight sources on identical rolling windows
   (18mo train / 3mo test / 1mo step, 15 windows):
   - `v1.5` — the global `DEFAULT_FACTOR_WEIGHTS` baseline.
   - `per_horizon_rolling` — strict OOS: refit per train window, applied
     to the held-out test slice. This is the honest estimate of whether
     per-horizon adds value.
   - `per_horizon_json` — the static full-corpus fit (in-sample;
     reference only, optimistic).
3. **Commit rule** — promote a horizon's static vector into
   `PER_HORIZON_WEIGHTS` only when `per_horizon_rolling` beats `v1.5` on
   **median bullish test-hit** with a non-pathological overfit gap. The
   model trades bullish-tilted (bearish calls are anti-predictive per
   Section 14/15), so bullish test-hit is the decision metric.

## Corpus

- 3,678 reports, 36 tickers, span 2023-07-14 → 2026-06-26.

## Gate results (bullish test-hit, strict rolling OOS)

| Horizon | v1.5 | per_horizon_rolling | Lift | Overfit gap | Decision |
|---------|------|---------------------|------|-------------|----------|
| ret_5d  | 60.00% | 60.98%            | +0.98pp | +0.26pp  | **reject** (noise) |
| ret_20d | 63.72% | 77.63%            | +13.91pp | -4.17pp | committed (prior) |
| ret_60d | 75.69% | 80.85%            | +5.16pp | +4.17pp  | **commit** |

Note: overall (all-direction) median test-hit *drops* under per-horizon
because the IC-signed vectors flip several factor signs, degrading the
anti-predictive bearish bucket while lifting the bullish bucket the
strategy actually trades. This mirrors the ret_20d commit rationale.

## Static-vs-static head-to-head (leakage-free holdout)

The rolling gate validates a *recipe* (refit per window), not a single
static vector. To decide whether to deploy a fixed full-corpus vector we
also ran `scripts/compare_static_weights_oos.py`: fit a new vector on a
train-only slice, then score every candidate on a disjoint later test
slice neither vector was fit on. The decision metric is OOS bullish
test-hit; `new_full_corpus` is reported but is IN-SAMPLE on the test slice
(it saw all data) so it is reference-only — its number is inflated.

**ret_20d — new full-corpus refit vs the committed incumbent.** Splits
2025-03-31 / 2025-06-30 / 2025-09-30:

| Candidate | split1 | split2 | split3 | OOS? |
|-----------|-------:|-------:|-------:|------|
| old_committed (incumbent) | 85.4% | 75.6% | 79.5% | yes |
| new_train_fit (recipe, OOS) | 65.6% | 62.5% | 67.5% | yes |
| new_full_corpus (in-sample) | 83.7% | 79.3% | 79.8% | NO (ref) |

The incumbent **wins 3/3** by 12–20pp OOS. The new full-corpus fit only
looks competitive in-sample; its honest OOS proxy is well below the
incumbent. **Decision: do NOT refresh ret_20d — keep the curated
5-factor incumbent.** The narrower vector generalizes better; the
larger-corpus refit over-broadens and regresses.

**ret_60d — recipe vs the global fallback (its only prior alternative).**
Splits 2025-01-31 / 2025-02-28 / 2025-03-31:

| Candidate | split1 | split2 | split3 | OOS? |
|-----------|-------:|-------:|-------:|------|
| new_train_fit (recipe, OOS) | 76.3% | 77.1% | 78.9% | yes |
| v1.5_global (fallback) | 71.4% | 74.5% | 74.6% | yes |
| committed full-corpus (in-sample) | 91.5% | 92.8% | 92.9% | NO (ref) |

The recipe **beats global 3/3** by +2.6 to +4.9pp OOS (consistent with
the +5.16pp rolling result). The committed vector's true forward
bullish-hit is ~77% (the ~92% is in-sample inflation), still above
global's ~74%. **Decision: ret_60d commit stands.**

## Committed vectors

- **ret_20d** — original curated 5-factor commit, retained. The
  larger-corpus refit was rejected after losing the leakage-free
  head-to-head above (0/3 OOS splits).
- **ret_60d** — new. 15-factor IC-signed vector; recipe validated OOS vs
  the global fallback (rolling +5.16pp; holdout +2.6/+4.9pp). See
  `backtest/results/per_horizon_weights.json` for the raw fit and the
  `PER_HORIZON_WEIGHTS["ret_60d"]` literal for the committed values.
  Note: in-sample bullish-hit (~92%) overstates the ~77% honest OOS
  level — size expectations off the OOS number.

## Per-horizon confidence calibration (2026-06-29) — NOT deployed

Attempted to (re)fit per-horizon isotonic confidence calibrations for the
committed vectors via
`fit_confidence_calibration.py --recompute-horizon <h> --oos-validate`:

| Horizon | OOS Brier vs heuristic | OOS reliability max-gap (gate ≤5pp) | Deploy? |
|---------|------------------------|-------------------------------------|---------|
| ret_20d | +0.088 (PASS)          | 36.7pp (FAIL)                       | no |
| ret_60d | beats heuristic (PASS) | 16.9pp (FAIL)                       | no |

Both improve Brier but **fail the ±5pp OOS reliability gate** — the
emitted confidence would not match realized hit-rates (large gaps in the
low and high probability buckets). This is the same regime
non-stationarity seen in `filings_recency_signal`: 2025–2026 data shifted
enough that an isotonic map fit across the corpus is locally miscalibrated
out of sample. **Decision: do not deploy refreshed per-horizon
calibration; the deployed `configs/confidence_calibration_20d.json` is left
unchanged.** Revisit after the next corpus regen / when the reliability
gate passes OOS.

## Reproduce the head-to-head

```bash
.venv/bin/python scripts/compare_static_weights_oos.py \
  --reports-glob 'reports/analysis_mvp/*.json' --horizon ret_20d \
  --split-date 2025-03-31 --split-date 2025-06-30 --split-date 2025-09-30 \
  --fit-min-abs-ic 0.04 --fit-min-n 50 \
  --new-weights-json backtest/results/per_horizon_weights.json
```

## Reproduce

```bash
.venv/bin/python scripts/fit_per_horizon_weights.py \
  --reports-glob 'reports/analysis_mvp/*.json' \
  --min-abs-ic 0.04 --min-n 50 \
  --output backtest/results/per_horizon_weights.json

.venv/bin/python scripts/walk_forward_eval.py \
  --reports-glob 'reports/analysis_mvp/*.json' \
  --weight-source v1.5 per_horizon_rolling per_horizon_json \
  --per-horizon-weights-json backtest/results/per_horizon_weights.json \
  --min-abs-ic 0.04 --min-n 50 \
  --output-dir backtest/results/per_horizon_gate
```
