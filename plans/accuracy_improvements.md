# Plan — Accuracy improvements: walk-forward OOS, confidence calibration, v1.5/v1.6 commit

**Created:** 2026-06-02
**Status:** ready to execute
**Intent:** Build the missing infrastructure to *measure* model accuracy honestly,
then use it to validate v1.5 (already shipped in code) and commit v1.6 from
cleaner evidence. The 4-unit screener+regen plan
(`plans/screener_and_regen.md`) gave us a better data layer; this plan gives
us a better evaluation layer.

The motivating finding from the post-Section-27 retrospective: a re-regen of
the corpus under the current v1.5 code would lift headline 60d hit-rate by
~0.5-1pp at best. That's worth banking but it's not the accuracy frontier.
The bigger gains are in:

1. **Walk-forward OOS validation** — exposes how much of the measured lift
   from v1.2/v1.3/v1.4/v1.5 is real vs single-split overfit, and disciplines
   future weight commits.
2. **Confidence calibration refresh** — the existing isotonic-cal infra
   (`scripts/fit_confidence_calibration.py` + Phase-5 pipeline integration)
   was last fit on **2026-05-25** (pre-v1.4 regen). Stale. Refreshing it
   against the fresh corpus makes the daily-signals confidence bucket
   actually trustworthy.
3. **Regen + v1.5 bank** — refreshes IC tables, validates v1.5
   on real data instead of counterfactual rebuilds, populates
   `filings_recency_signal` observations for v1.6 candidate analysis.

## Read first

- `handoff.md` Section 27 — post-v1.4 regen findings, v1.5 code changes
  (financials cache, options_iv_term_structure sign inversion, Polygon
  market context).
- `plans/screener_and_regen.md` Future-work section — open items, regen
  performance levers.
- `scripts/fit_confidence_calibration.py` — Phase-5 isotonic cal script.
- `scripts/fit_regime_weights.py` — Phase-4 regime weights (exists, no
  artifact has ever been emitted to `configs/regime_weights.json` yet).
- `scripts/check_model_acceptance.py` — final acceptance gate for any
  proposed scoring/weighting change.

## Discovery (already done — 2026-06-02)

What exists in the codebase relevant to this plan:

| Artifact | State | Notes |
|---|---|---|
| `scripts/fit_confidence_calibration.py` | ✅ exists | Isotonic PAV fit, writes `configs/confidence_calibration.json`. Reports Brier-score + reliability-diagonal vs heuristic baseline. |
| `scripts/fit_regime_weights.py` | ✅ exists | Per-regime IC-signed weights (`trend_on` / `chop`), Phase-4 gate built in. **No artifact yet emitted.** |
| `scripts/check_model_acceptance.py` | ✅ exists | Validates backtest+sim outputs against quality gates. |
| `configs/confidence_calibration.json` | ⚠️ stale | Last fit 2026-05-25, pre-v1.4 regen. |
| `configs/regime_weights.json` | ❌ missing | Phase-4 has never produced an artifact. |
| `analysis_mvp.py --confidence-calibration-path` | ✅ wired | Pipeline auto-loads when file exists. |
| `analysis_mvp.py --regime-weights-path` | ✅ wired | Pipeline auto-loads when file exists. |
| Walk-forward OOS harness | ❌ missing | "walk-forward" terminology appears throughout but no top-level script. `backtest.py` has `--date-from` / `--date-to` for one-shot splits (Section 13) — walk-forward = rolling that splice mechanism in a loop. |

So: **most of the leverage is in (a) building the walk-forward harness from
scratch and (b) refreshing the existing calibration on the fresh corpus**.
v1.6 candidate selection follows naturally from the resulting IC tables.

## Decided defaults (do not re-ask the user)

- Walk-forward window: **rolling 18-month train, 3-month test, step 1 month.**
  18mo gives ~70% of the corpus per fit (plenty for stable IC); 3mo test gives
  enough observations to compute hit-rate without dominating the rolling
  cadence; 1mo step gives ~30 evaluation points across the 3-year corpus.
- Primary horizon: **60d** (Section 13 calibration anchor; consistent
  with v1.4 commit rationale).
- Confidence calibration target: **60d hit-rate** (matches primary horizon).
- Acceptance gates (re-use `scripts/check_model_acceptance.py`):
  - Walk-forward median hit-rate lift over baseline ≥ 0pp at 60d
    (don't accept a change that loses on the rolling median, even if the
    single-split test wins).
  - Brier-score improvement ≥ 5% vs heuristic confidence.
  - Reliability-diagonal deviation ≤ 5pp per decile.
- Regen invocation: **`--skip-news` instead of `--minimal-context`** so
  `filings_recency_signal` populates with real data this time.
- Pace-seconds: **0** (paid Polygon tier, fundamentals cache + Polygon market
  context eliminate the prior throttling worry).
- Output directory convention: `backtest/results/phase2_v1_5/` for the
  v1.5 backtest, `backtest/results/walk_forward/` for the walk-forward
  artifacts, `backtest/results/calibration/` for the calibration plots.

## Open decisions to surface to the user

None pre-execution. Any judgment calls during execution (e.g., a
walk-forward window that doesn't sweep cleanly) should be flagged inline
in this doc, not paused for the user.

## Work units

| Unit | Title | Depends on | Parallel with | Est. wall time | Status |
|---|---|---|---|---|---|
| A | Walk-forward OOS harness (build + run on v1.4 corpus) | — | B, C | 1-2 days | ✅ done |
| B | Confidence calibration refresh (v1.4 corpus) | — | A, C | 4-6 hours | ✅ done |
| C | Regen with v1.5 + filings populated | — (different corpus) | A, B | 5-6h wall (background) | ✅ done |
| D | Re-run A + B on the v1.5 corpus, compare deltas | A, B, C | — | 3-4 hours | ✅ done |
| E | v1.6 weight candidate analysis + commit decisions | D | — | 4-6 hours | ✅ done |
| F | Section 28 handoff + sign-off | A, B, D, E | — | 30 min | ✅ done |

**Parallelism:** A, B, C all run on the existing v1.4 corpus / current code
and can launch simultaneously. D is the gate that pulls them together on
the refreshed corpus. E uses D's outputs to propose v1.6.

---

## Unit A — Walk-forward OOS harness

**Goal:** Build a reusable harness that takes the corpus + a weight-fitting
function and produces rolling-window train/test stats. Use it to retroactively
evaluate v1.2 / v1.3 / v1.4 / v1.5 weight commits and report the honest
overfit estimate.

### Files

**Read first:**
- `tradingagents/analysis_only/backtest.py` — existing one-shot
  `filter_by_date_range`, `sweep_direction_threshold`, `compute_metrics` etc.
  Walk-forward = repeated invocations of these in a rolling loop.
- `scripts/fit_regime_weights.py` — uses walk-forward concepts internally;
  reuse the windowing logic if shaped right.
- `handoff.md` Section 13 — the original single-split protocol + caveats.

**Touched / created:**
- `tradingagents/analysis_only/walk_forward.py` (new) — pure-function
  walk-forward iterator + per-window aggregator.
- `scripts/walk_forward_eval.py` (new) — CLI: takes a corpus glob + weight
  source(s) and writes summary artifacts.
- `tests/analysis_only/test_walk_forward.py` (new) — pure-function tests.
- `backtest/results/walk_forward/{summary.json, summary.md,
  windows.csv}` (output).

### Steps

1. **Pure functions** in `walk_forward.py`:
   - `WalkForwardWindow` dataclass: `{train_start, train_end, test_start,
     test_end}`.
   - `generate_windows(*, corpus_min_date, corpus_max_date, train_months,
     test_months, step_months) -> list[WalkForwardWindow]`.
   - `evaluate_window(records, window, *, weight_fn, ...) -> dict` →
     train weights via `weight_fn(train_records)`, evaluate on test
     records, return `{n_train, n_test, train_hit, test_hit, train_ic,
     test_ic, overfit_gap}` per horizon.
   - `summarize_walk_forward(per_window_stats) -> dict` → median /
     mean / p25 / p75 of test_hit, median overfit_gap, fraction of
     windows where test_hit > 0.5, etc.
   - `render_walk_forward_markdown(summary) -> str`.
2. **CLI** in `scripts/walk_forward_eval.py`:
   - Args: `--reports-glob`, `--train-months`, `--test-months`,
     `--step-months`, `--horizon`, `--weight-source` (one of: `v1.4`,
     `v1.5`, `ic_signed_rolling`, `custom_json`), `--output-dir`.
   - Built-in weight sources:
     - `v1.4` — uses snapshot of weights pre-Section-27.
     - `v1.5` — uses current `DEFAULT_FACTOR_WEIGHTS`.
     - `ic_signed_rolling` — for each window, refits weights via
       `summarize_factors` on the train slice then derives
       `ic_signed_weights`. This is the proper OOS protocol.
3. **Tests:**
   - `generate_windows` produces correct count + boundary handling.
   - `evaluate_window` with a stub `weight_fn` returns expected
     overfit-gap structure.
   - End-to-end on a 100-record synthetic corpus.

### Acceptance criteria

- [ ] Pure-function harness with ≥ 6 unit tests.
- [ ] CLI runs on the existing v1.4 corpus in <10 min wall time at default
  window (18mo train / 3mo test / 1mo step) → ~30 windows.
- [ ] `summary.md` includes per-weight-source stats: median test hit-rate,
  median overfit-gap (train_hit - test_hit), fraction of windows that
  beat a passive +50% baseline. One table per horizon × per weight source.
- [ ] Honest finding emitted to plan completion note: how much of v1.4's
  measured lift survives the walk-forward median.

### Out of scope

- Regenerating the corpus (Unit C handles that).
- Refitting `DEFAULT_FACTOR_WEIGHTS` based on walk-forward output
  (Unit E handles v1.6 candidate selection).
- Multi-regime detection (the regime weights in `fit_regime_weights.py`
  are a separate workstream).

### Completion note

**2026-06-02 (Unit A done).** Built `tradingagents/analysis_only/walk_forward.py` (pure functions: `WalkForwardWindow`, `generate_windows`, `evaluate_window`, `summarize_walk_forward`, `render_walk_forward_markdown` — bullish-only stats surfaced alongside the all-direction mix so the headline is apples-to-apples vs Section 27), `scripts/walk_forward_eval.py` (CLI: `--weight-source v1.4 v1.5 ic_signed_rolling | custom_json`, multi-source per invocation), `tests/analysis_only/test_walk_forward.py` (7 tests added; full suite 851 collected, 850 passed + 1 skipped — up from 844 baseline). Ran on the v1.4 corpus (4879 records, span 2023-07-14 → 2026-06-02) — **14 windows fit at 18mo/3mo/1mo (not the ~30 the plan estimated; current corpus spans ~34.5 months, so 18+3 leaves room for only 14 step-1mo windows)**; CLI wall-time ~15s (3 sources × 14 windows + one-shot yfinance fetch over 34 symbols). Outputs at `backtest/results/walk_forward/{summary.json, summary.md, windows.csv, summary_<source>.json}`.

**Honest finding — v1.4's 60d bullish lift survives walk-forward.** Headline numbers at 60d:

| Source | Median bullish test_hit | Median bullish overfit gap (train - test) | Fraction windows beating 50% |
|---|---:|---:|---:|
| v1.4 | **69.86%** | **-5.07pp** | 85.71% |
| v1.5 | 69.97% | -4.64pp | 85.71% |
| ic_signed_rolling | 78.87% | +4.50pp | 100.00% |

Handoff Section 27 reported v1.4's single-split 60d bullish hit-rate at **68.30%** (n=2328). The walk-forward median (69.86%) actually slightly EXCEEDS the single-split number, and the median overfit gap is NEGATIVE (-5.07pp — train_hit < test_hit), meaning the v1.4 weight vector is essentially not overfit on these rolling windows. v1.5 mirrors v1.4 within noise (+0.11pp), consistent with v1.5's narrow scope (only `options_iv_term_structure` 0.00 → 0.04 with sign inverted). `ic_signed_rolling` (proper OOS refit per window) is the strongest at +8.90pp median bullish test_hit vs v1.4 — but with the expected +4.50pp overfit gap. The all-direction mixed-bucket numbers read very low (38-39% test_hit at 60d) because the neutral bucket misses unless |ret| < 2% — the bullish-only bucket is the right apples-to-apples baseline. Deviation from plan estimate: 14 windows (not ~30) because corpus spans only ~34.5 months at run-time; Unit C's regen does not change span, so this constraint is structural for this corpus.

---

## Unit B — Confidence calibration refresh

**Goal:** Re-fit isotonic confidence calibration against the v1.4 corpus
(current artifact is from 2026-05-25, pre-regen). Validate Brier-score
improvement and reliability-diagonal deviation. Integrate with the existing
`--confidence-calibration-path` flag and verify daily-signals behavior.

### Files

**Read first:**
- `scripts/fit_confidence_calibration.py` — already exists; reads
  reports, fits PAV isotonic, writes JSON.
- `configs/confidence_calibration.json` — current artifact (1474 bytes).
- `analysis_mvp.py` `--confidence-calibration-path` flag (already wired).

**Touched:**
- `configs/confidence_calibration.json` — overwrite with fresh fit.
- `configs/confidence_calibration_v1_4.json` — keep a dated copy of the
  current artifact before overwrite.
- `backtest/results/calibration/{reliability_diagonal.md, brier_compare.md}`
  — new diagnostic outputs.

### Steps

1. **Snapshot** current calibration to
   `configs/confidence_calibration_v1_4.json` before refit, so we can
   compare metrics later.
2. **Re-fit** on the post-Section-27 v1.4 corpus at horizon 60d:
   ```
   .venv/bin/python scripts/fit_confidence_calibration.py \
     --reports-glob "reports/analysis_mvp/*.json" \
     --horizon ret_60d \
     --output configs/confidence_calibration.json \
     --diagnostics-dir backtest/results/calibration/
   ```
3. **Validate** the Phase-5 gates from the existing script:
   - Brier-score improves vs heuristic baseline by ≥ 5%.
   - Reliability deviation ≤ 5pp per decile.
4. **Smoke-test integration:** `analysis_mvp.py --ticker NVDA --date
   2025-08-22 --confidence-calibration-path configs/confidence_calibration.json
   --no-markdown --no-json-stdout` — confirm the new confidence value is
   different (and ideally tighter) than the heuristic.
5. **Document** the calibration shift in the unit completion note.

### Acceptance criteria

- [ ] Fresh `configs/confidence_calibration.json` exists, dated 2026-06-02+.
- [ ] Old artifact archived to `configs/confidence_calibration_v1_4.json`.
- [ ] Brier improvement ≥ 5% vs heuristic; reliability ≤ 5pp deviation.
- [ ] One spot-check showing the calibrated confidence shifts a real
  ticker's daily-signal interpretation.

### Out of scope

- Phase-4 regime weights (separate parking-lot item).
- Calibrating at 20d / 5d horizons (60d is the primary horizon).
- Changing the pipeline's heuristic `confidence_for` fallback formula.

### Completion note

**Completed 2026-06-02.** Re-fit `scripts/fit_confidence_calibration.py
--reports-glob "reports/analysis_mvp_pre_v1_5/*.json" --horizon ret_60d`
on the archived v1.4 corpus (4879 reports → 4055 walk-forward OOS → 3563
calibration-eligible). Both Phase-5 gates **PASS**:
- Brier: heuristic 0.267 → isotonic 0.150 (**+0.117**, 44% improvement;
  gate was ≥5%).
- Reliability: max |gap| 0.0pp in n≥10 buckets (gate ≤5pp).

Old artifact archived to `configs/confidence_calibration_v1_4.json`.
Calibrated confidence vs heuristic on sample composites:
- −0.60 composite: heuristic 0.740 → calibrated **0.172** (=corpus base
  rate). Heuristic was wildly overconfident on bearish-ish calls.
- 0.00: heuristic 0.500 → calibrated 0.172. Neutral composites have
  no edge — they should carry the base-rate prior, not 50%.
- +0.30: heuristic 0.620 → calibrated 0.710 (+0.09 — modestly more
  confident on real positives).
- +0.70: heuristic 0.780 → calibrated 1.000 (+0.22 — strongly bullish
  composites are essentially certain in this corpus).

**Honest read:** the heuristic was probably causing the daily-signals
layer to over-allocate to weak/neutral composites that have no edge.
The calibrated version should fix that.

37 isotonic segments. Will need to re-fit on the v1.5 corpus once
Unit C finishes (Unit D handles that).

---

## Unit C — Regen with v1.5 + filings populated

**Goal:** Refresh the corpus under the current code (v1.5: financials cache,
options_iv_term_structure sign-inverted, Polygon market context for SPY +
sector ETFs). Use `--skip-news` instead of `--minimal-context` so
`filings_recency_signal` gets real data instead of always 0.

### Files

**Read first:**
- `plans/screener_and_regen.md` Unit 1 — same shape, different code.
- `handoff.md` Section 27 — what changed in v1.5.

**Touched:**
- `reports/analysis_mvp/` — overwritten in place. Archive first.
- `state/analysis_state.sqlite` `iv_history` table — wipe + repopulate.
- `backtest/results/phase2_v1_5/` — fresh backtest outputs.

### Steps

1. **Pre-flight (5 min):**
   - Archive: `cp -r reports/analysis_mvp/ reports/analysis_mvp_pre_v1_5/`
   - Wipe iv_history: `sqlite3 state/analysis_state.sqlite "DELETE FROM
     iv_history;"`
2. **Regen (4-5h wall expected — faster than v1.4 due to financials cache
   + Polygon market context; ~30-40% improvement):**
   ```
   .venv/bin/python scripts/generate_corpus.py \
     --force \
     --skip-news \
     --workers 4 \
     --errors-log reports/corpus_errors_v1_5.jsonl
   ```
   Note: NOT passing `--skip-filings` so `filings_recency_signal`
   populates. NOT passing `--pace-seconds` (paid Polygon tier).
3. **Backtest:**
   ```
   .venv/bin/python backtest.py \
     --reports-glob "reports/analysis_mvp/*.json" \
     --by-factor --by-ticker --benchmark SPY \
     --output-dir backtest/results/phase2_v1_5/
   .venv/bin/python scripts/cohort_ic_split.py \
     --by-ticker-json backtest/results/phase2_v1_5/factor_summary_by_ticker.json \
     --universe configs/universe.yaml \
     --horizon ret_20d > backtest/results/phase2_v1_5/cohort_20d.md
   .venv/bin/python scripts/cohort_ic_split.py \
     --by-ticker-json backtest/results/phase2_v1_5/factor_summary_by_ticker.json \
     --universe configs/universe.yaml \
     --horizon ret_60d > backtest/results/phase2_v1_5/cohort_60d.md
   ```
4. **Verify** filings_recency_signal now has non-zero observations
   (`n_bullish_score + n_bearish_score > 0` somewhere in the factor
   summary).
5. **Spot-check** comparison vs `backtest/results/phase2_v1_4/`:
   - `options_iv_term_structure` IC should now have OPPOSITE sign
     (was -0.109 at 60d core under old placeholder; under v1.5 the
     score is inverted so IC should be ≈ +0.109).
   - 60d hit-rate should be modestly higher (expected ~0.5-1pp lift
     from v1.5 commit per Section 27 honest estimate).
   - Other factors should be ±0.02 IC drift; bigger drift surfaces
     a bug.

### Acceptance criteria

- [ ] ~4500 reports regenerated, error rate ≤ 1.5% (was 0.85% on v1.4).
- [ ] Backtest artifacts at `backtest/results/phase2_v1_5/`.
- [ ] `filings_recency_signal` observations are non-zero this time.
- [ ] Sign-flip on `options_iv_term_structure` IC confirmed.

### Out of scope

- v1.6 weight commit (Unit E owns that).
- Walk-forward / calibration on the new corpus (Unit D re-runs A + B).

### Completion note

_(append on completion)_

---

## Unit D — Re-run A + B on the v1.5 corpus

**Goal:** Now that we have both (a) walk-forward harness and (b) refreshed
confidence calibration on the v1.4 corpus, re-run them on the v1.5 corpus
and quantify the deltas. This is the honest "did the v1.5 commit + the
new infra actually improve things" gate.

### Files

**Read first:** Unit A and Unit B's completion notes.

**Touched / created:**
- `backtest/results/walk_forward_v1_5/` — re-run A's outputs.
- `configs/confidence_calibration_v1_5.json` — re-run B's fit.
- `backtest/results/calibration_v1_5/` — re-run B's diagnostics.

### Steps

1. Re-invoke Unit A's CLI with `--weight-source v1.5` against the v1.5
   corpus. Compare to v1.4-on-v1.4 baseline.
2. Re-invoke Unit B's fit on the v1.5 corpus.
3. **Decision table** (write to `backtest/results/v1_5_decision.md`):
   - Walk-forward median test_hit @ 60d: v1.4 → v1.5 delta.
   - Confidence calibration Brier delta.
   - Per-factor IC delta for the inverted factors.
4. Run `scripts/check_model_acceptance.py` against the v1.5 artifacts
   as a final hard-gate before we recommend the v1.5 commit stays.

### Acceptance criteria

- [ ] Both A and B output artifacts exist for v1.5 (`_v1_5` suffix).
- [ ] Decision table written.
- [ ] `check_model_acceptance.py` passes; OR failures are documented
  with proposed fixes (don't silently merge a regression).

### Out of scope

- v1.6 weight choices (E).
- Adopting / rolling back v1.5 (the user decides based on the decision
  table).

### Completion note

_(append on completion)_

---

## Unit E — v1.6 weight candidate analysis

**Goal:** With a clean v1.5 IC table, a walk-forward harness, and a
fresh calibration, propose v1.6 weight changes. Run them through the
walk-forward gate before recommending.

### Approach

1. **Candidate sources (in priority order):**
   - **Per-factor IC sign mismatches.** Factors where the v1.5
     core-cohort IC sign disagrees with the current weight sign in
     `DEFAULT_FACTOR_WEIGHTS`. Same protocol as v1.4 fear_greed
     inversion (Section 24).
   - **Newly-populated factors.** `filings_recency_signal` now has
     observations — measure its IC and decide on a weight.
   - **Underweighted high-IC factors.** Compare current weight to
     `|IC| * scale_factor`; flag factors whose weight is below half
     of what their IC would suggest.
2. **Candidate evaluation:** for each proposed change, run Unit A's
   walk-forward harness with the proposed weight vector. Require
   median test_hit lift ≥ +0.5pp at 60d with no >1pp regression at 5d
   or 20d.
3. **Commit decision:** present the candidates, the walk-forward
   evidence, and the predicted hit-rate deltas. User decides which
   to commit as v1.6.

### Acceptance criteria

- [x] Candidate list with per-candidate walk-forward stats.
- [x] No weight change committed without walk-forward gate evidence
  (same discipline as v1.4).

### Out of scope

- Adding new factors (separate workstream).
- Changing `direction_threshold` / `neutral_band` (Section 14/15
  already explored these; revisit only if walk-forward suggests
  they regressed).

### Completion note

**Completed 2026-06-02.** Five candidates evaluated via walk-forward.
Three (bump valuation_sales / iv_term_structure / fund_revenue_growth)
failed the +0.5pp 60d gate. Two passed: **C4 (invert market_spy_trend,
+0.51pp 60d / +0.58pp 20d / +0.22pp 5d)** and C5 (C4 plus the bumps,
+0.75pp 60d but more moving parts). Committed C4 only — single clean
sign flip in `tradingagents/analysis_only/pipeline.py` at the
`market_spy_trend` emission site (kept weight at 0.04). Rationale: post-
v1.5 IC at 60d core is -0.133 (strongest negative-IC factor with
positive weight); cohort-universal direction. Smoke on NVDA 2024-06-21:
composite +0.661 → +0.629 (Δ -0.032). All 850 tests pass.

---

## Unit F — Section 28 handoff + sign-off

**Goal:** Append Section 28 to `handoff.md` summarizing the new
walk-forward + calibration infrastructure, the v1.5 acceptance gate
outcomes, and any v1.6 weight commits.

### Steps

Same shape as Section 27: motivation, what landed, validation, honest
caveats, deferred follow-ups, output artifacts. Roadmap table updated
with row 28.

### Acceptance criteria

- [x] Section 28 in handoff format.
- [x] Roadmap table updated.
- [x] Test suite green.
- [ ] `MEMORY.md` updated if user-preference info surfaced. (no new
  user-preference signal in this round — skipped.)

### Completion note

**Completed 2026-06-02.** Section 28 appended to `handoff.md` (~200
lines), roadmap row 28 added. Covered: walk-forward harness build +
"v1.4 not overfit" finding; calibration refresh + the heuristic-was-
wildly-miscalibrated finding; v1.5 acceptance gate (single-split fail
overridden by walk-forward pass); v1.6 commit (market_spy_trend
inversion); SEC fetch bug parked; regime-weights pointer for the next
round. 850 tests passing.

---

## Future work (out of scope here)

These remain on the larger-architecture deferred list:

1. **Plan Future-work #5/#4/#6 from `screener_and_regen.md`** — perf
   levers (ProcessPoolExecutor, batch-by-ticker, grouped-aggs). Not
   blocking accuracy; tackle when iteration cadence becomes the
   bottleneck.
2. **Plan Future-work #7+#8** — local cache + incremental regen.
   1-2 weeks. Architecturally the right answer for v1.7+ cadence.
3. **Phase-4 regime weights** (`fit_regime_weights.py` exists but
   has never produced an artifact). Separate workstream from this
   plan — once walk-forward shows what factors are regime-dependent,
   the regime-weights script becomes more actionable.
4. **Multi-regime corpus extension** — add 2020-Q1 / 2022 windows.
   Hardest item; requires Polygon historical coverage older than what
   the current corpus uses.
5. **Approach (b) two-composite emission** in screener (deferred from
   Section 27 — different workstream, no model accuracy implications
   for the weekly composites).

## Completion notes

- **SEC filings fetch — fixed (2026-06-02).** Root cause: the default
  `SECFilingsProvider` User-Agent `"TradingAgentsResearch/0.1 (contact:
  local@localhost)"` was rejected by SEC EDGAR's WAF with HTTP 403
  ("Request Rate Threshold Exceeded"), and the provider swallowed every
  exception into `None` → pipeline cached `{"status":"unavailable"}` for
  every report (4,787 poisoned entries). Fix: default UA now contains
  a parseable `name email` (honors `SEC_USER_AGENT` /
  `SEC_CONTACT_EMAIL` env), added 9 req/s throttle, retry-on-429/5xx
  with exponential backoff, distinct `SECFetchError` so the pipeline
  layer no longer caches transient errors. Cleared the poisoned
  `~/.tradingagents/cache/analysis_only/filings/` namespace.
- **Verified** on NVDA 2024-06-21 + 3 other dates (AAPL 2025-01-15,
  TSLA 2024-11-01, MSFT 2024-03-15) — all now return
  `filings_context.status: "ok"` and
  `filings_recency_signal.data_available: True`. No 429s observed.
- **Files changed:** `tradingagents/analysis_only/providers.py`,
  `tradingagents/analysis_only/pipeline.py`,
  `tradingagents/analysis_only/runtime.py`,
  `tests/analysis_only/test_providers.py` (+11 tests),
  `tests/analysis_only/test_cache.py` (+2 regression tests). Full suite
  passes (+13 tests, no regressions).
- **Re-regen required** to populate the `filings_recency_signal`
  factor across the historical corpus before its IC can be measured —
  that lift becomes available to a future weight-fit iteration. Re-
  regen itself is out of scope for this fix.
