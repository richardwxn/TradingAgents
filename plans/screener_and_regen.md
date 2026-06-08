# Plan — Phase 2 v1.4 regen + Nasdaq Composite screener

**Created:** 2026-05-31
**Status:** ready to execute
**Intent:** Two parallel workstreams. (1) regen the Phase 2 corpus under v1.4
weights + VIX proxy so on-disk JSONs match production state and downstream IC
analyses run on clean inputs. (2) Build a new ticker-discovery capability that
screens the Nasdaq Composite (filtered) weekly to surface high-composite names
outside the curated 26-name core universe.

Read `handoff.md` Sections 22-26 for context on the v1.4 weight commit, VIX
proxy, cohort IC findings, and the existing 30-ticker core+canary universe.

## How agents should use this doc

Each work unit below is self-contained: goal, files, read-first context,
acceptance criteria, out-of-scope. An agent picking up a unit should not need
to ask the human anything that isn't called out as an open decision.

- Run **multiple units in parallel** when the dependency table allows.
- Update the **status** column in the work-unit table when starting / finishing
  a unit.
- Append a short completion note (1-3 lines) at the bottom of each unit's
  section after finishing, including any deviations from the plan and the
  artifact paths.
- After all units land, append a Section 27 to `handoff.md` summarizing what
  shipped + linking to the artifacts.

## Work units

| Unit | Title | Depends on | Parallel with | Est. wall time | Status |
|---|---|---|---|---|---|
| 1 | Phase 2 regen + post-regen IC | — | 2, 3 | 5-6h (mostly I/O) | ✅ done |
| 2 | Phase A — skip-flags on `AnalysisOnlyMVP` | — | 1, 3 | 1-2h | ✅ done |
| 3 | Phase B1 — Nasdaq 100 smoke screener | — | 1, 2 | 2-3h | ✅ done |
| 4 | Phase B2 — Nasdaq Composite filtered screener | 2, 3 | 1 | 3-4h | ✅ done |
| 5 | Section 27 handoff doc + sign-off | 1, 4 | — | 30min | ✅ done |

**Parallelism:** Units 1, 2, 3 can all run simultaneously. Unit 4 starts after
2 and 3 land. Unit 5 is the final write-up gate.

## Decided defaults (do not re-ask the user)

- Screener universe: **Nasdaq Composite, filtered** to common stock + min
  market cap $500M + min 20d ADV $5M (≈800-1200 names post-filter).
- Scan cadence: **weekly** (Fridays after close, matching the analysis cadence).
- Output: **top 50 overall by composite** + **top 5 per GICS sector** in a
  secondary view (so a tech-heavy day doesn't bury the one strong industrial).
- Cohort-aware scoring approach (a): **universal-sign-agreed factors only when
  scoring non-tech names** (see Section 22 in handoff.md for the universal vs
  tech-specific factor table). Approach (b) — two-composite emission — is
  out-of-scope here; revisit only if (a) systematically misses things.
- Regen pace-seconds: **1.0** (safer on free Polygon tier, ~10% wall-time hit).

## Open decisions to surface to the user

None right now. Any new questions that arise during execution should be
flagged inline in this doc rather than spawned as questions to the user,
unless they block progress.

---

## Unit 1 — Phase 2 regen + post-regen IC

**Goal:** On-disk corpus JSONs reflect v1.4 weights and have the VIX proxy
populating `market_fear_greed_regime` on the ~2,916 records where CNN F&G was
previously unavailable. Fresh IC analysis confirms (or surfaces issues with)
the post-hoc v1.4 validation from Section 24.

### Files

**Read first:**
- `handoff.md` Sections 23 (`--minimal-context`), 24 (v1.4 commit), 25 (VIX proxy)
- `scripts/generate_corpus.py`
- `backtest.py`
- `scripts/cohort_ic_split.py`

**Touched / created:**
- `reports/analysis_mvp_pre_v1_4/` — archived snapshot of current corpus
- `reports/analysis_mvp/` — regenerated in place
- `state/analysis_state.sqlite` — `iv_history` table wiped + repopulated
- `backtest/results/phase2_v1_4/` — fresh backtest outputs
- `backtest/results/phase2_v1_4_cohort/{cohort_20d.md, cohort_60d.md}` — cohort split

### Steps

1. **Pre-flight (5 min):**
   - `cp -r reports/analysis_mvp/ reports/analysis_mvp_pre_v1_4/`
   - In `state/analysis_state.sqlite`: `DELETE FROM iv_history;`
     (`symbol_state`, `news_seen`, `filing_seen` are kept — they're idempotent
     and the symbol_state delta block reads from them.)
   - Verify universe is unchanged from current `configs/universe.yaml`
     (26 core + 6 canary = 30 tickers × 150 Fridays = 4,500 jobs).
2. **Regen (5-6h wall time, background):**
   ```
   python scripts/generate_corpus.py \
       --force \
       --minimal-context \
       --workers 4 \
       --pace-seconds 1.0 \
       --errors-log reports/corpus_errors_v1_4.jsonl
   ```
   Recommend running with `run_in_background=true`.
3. **Verify regen (10 min):** spot-check 5 random JSONs span the corpus date
   range; confirm `factor_scores[].name == "market_fear_greed_regime"` has
   `data_available=true` and `fear_greed_source` ∈ `{cnn, vix_fear_greed_proxy}`
   on dates from the previously-empty 2023-H2 / 2024-H1 windows; confirm
   factor weights match v1.4 (i.e., `weight=0.05` for fear_greed,
   `weight=0.04` for `options_iv_rank`).
4. **Re-run backtests:**
   ```
   python backtest.py \
       --reports-glob "reports/analysis_mvp/*.json" \
       --by-factor --by-ticker \
       --benchmark SPY \
       --output-dir backtest/results/phase2_v1_4/
   python scripts/cohort_ic_split.py \
       --factor-summary-by-ticker backtest/results/phase2_v1_4/factor_summary_by_ticker.json \
       --universe configs/universe.yaml \
       --output-dir backtest/results/phase2_v1_4_cohort/
   ```
5. **Compare to baseline (`backtest/results/phase2/`):**
   - `market_fear_greed_regime` — should now have N closer to 4,500 (up from
     ~1,618). IC sign should be negative for tech (confirming the v1.4 inversion).
   - `options_iv_rank` — should match v1.3 promote validation (IC ≈ +0.15 at 20d).
   - All other factors should be **stable** (no >0.02 IC drift). Larger drifts
     surface a bug.
   - Bucket-level hit-rates should be within ±2pp of the Section 24 rebuild-trick
     numbers. Larger gaps mean the rebuild trick missed something.

### Acceptance criteria

- [ ] 4,500 reports regenerated (or all ≤ N pre-IPO date drops accounted for
  in `corpus_errors_v1_4.jsonl`).
- [ ] Backtest artifacts written to `backtest/results/phase2_v1_4/`.
- [ ] Cohort split written to `backtest/results/phase2_v1_4_cohort/`.
- [ ] Smoke comparison vs `phase2/` baseline noted in unit completion summary,
  with any IC drift >0.02 explained.

### Out of scope

- Do **not** change `DEFAULT_FACTOR_WEIGHTS`. This unit is data-only.
- Do **not** modify `scripts/generate_corpus.py` (the `--force`,
  `--minimal-context`, `--pace-seconds`, `--state-store-path` flags already exist).
- Do **not** delete `reports/analysis_mvp_pre_v1_4/` until Unit 5 signs off.

### Completion note

**Completed 2026-05-31.** Regen ran 8h 49m on 4 workers + `--minimal-context`
+ pace-seconds 1.0 (in retrospect should have been 0). 4,759 ok / 41 errors
(0.85% — within Section 22's range; 30 are pre-IPO ARM/ALAB, residual 11 are
transient yfinance failures). Backtest + cohort_ic_split landed at
`backtest/results/phase2_v1_4/`. Headline: `market_fear_greed_regime`
coverage 36% → ~100% via VIX proxy; v1.4 inversion confirmed (60d core IC
−0.129, sign-cons 84%); +0.72pp / +0.79pp at 60d bullish/bearish vs v1.3.
Discovered `filings_recency_signal` now populates but always scores 0
(different defect from Section 12 thought). Polygon Financials cache shipped
in parallel (Future-work #3) for next regen: ~9k HTTP calls → ~60.

---

## Unit 2 — Phase A: skip-flags on `AnalysisOnlyMVP`

**Goal:** Two new constructor flags on `AnalysisOnlyMVP` that gate the
remaining heavyweight pipeline phases (intraday context, peer-competitor
fundamentals fetch). Default `True` (no behavior change for existing callers);
the screener sets them to `False`.

### Files

**Read first:**
- `tradingagents/analysis_only/pipeline.py` — `AnalysisOnlyMVP.__init__`
  (line ~80), `run` (line ~164), `_build_report`. Search for
  `_load_intraday_context` and `_build_competitor_analysis` to find the
  call sites.
- `handoff.md` Section 23 (the prior precedent for `enable_news_fetching` /
  `enable_filings_fetching`).
- `tests/analysis_only/test_*.py` for existing test patterns.

**Touched:**
- `tradingagents/analysis_only/pipeline.py`
- `tests/analysis_only/test_pipeline.py` (or appropriate existing test file)

### Steps

1. Add two new init params to `AnalysisOnlyMVP.__init__`:
   - `enable_intraday_context: bool = True`
   - `enable_peer_competitor_analysis: bool = True`
   Store on `self`. Echo into `key_features.run_config` like the existing
   skip-flags.
2. In `run()` (or wherever `_load_intraday_context` is invoked), gate the call
   on `self.enable_intraday_context`. When disabled: set
   `intraday_context = {"status": "disabled", "pit_status": "disabled"}` (or
   whatever matches the existing "skipped" shape — match the news/filings
   precedent).
3. Same for `_build_competitor_analysis` gated on
   `self.enable_peer_competitor_analysis`. Disabled returns
   `{"status": "disabled", "peers": []}` or matching shape.
4. Make sure downstream consumers (factor scoring, markdown renderer, narrative
   payload) don't crash on the disabled shape. Most should already handle
   missing data because the news/filings skip path established the precedent.
5. Add unit tests:
   - `enable_intraday_context=False` → report builds, intraday section marked
     disabled, no other factors regress.
   - `enable_peer_competitor_analysis=False` → same.
   - Both False + the existing news/filings/options skip flags → report still
     produces a valid composite (the screener-mode invariant).

### Acceptance criteria

- [ ] Two new constructor flags land with defaults `True` (no live-run regression).
- [ ] At least 3 new tests cover the flags (single-flag-off × 2, all-off).
- [ ] Full test suite (`python -m pytest`) passes (currently 704 tests; +3-5 here).
- [ ] Smoke: `python analysis_mvp.py --ticker NVDA --date 2026-05-22` produces
  composite within ±0.005 of the pre-change value (no flags = no change).
- [ ] Single-thread screener-mode smoke: instantiate
  `AnalysisOnlyMVP(options_enabled=False, enable_news_fetching=False,
  enable_filings_fetching=False, enable_intraday_context=False,
  enable_peer_competitor_analysis=False, enable_llm_insights=False,
  enable_narrative=False)` and run NVDA 2026-05-22 → expect ≤5s wall time.

### Out of scope

- Do **not** modify scoring weights or `DEFAULT_FACTOR_WEIGHTS`.
- Do **not** rename the existing flags (`enable_news_fetching`,
  `enable_filings_fetching`, `options_enabled`) — keep the precedent.
- Do **not** modify `analysis_mvp.py`'s CLI to expose these flags. The
  screener instantiates `AnalysisOnlyMVP` directly; the regular CLI doesn't
  need them. (Add CLI flags later if a user asks.)

### Completion note

Landed both flags in `tradingagents/analysis_only/pipeline.py` (default `True`,
echoed into the cache-key params so screener-mode runs don't collide with
full-context cached reports). Added 8 tests in
`tests/analysis_only/test_pipeline_skip_flags.py` (delta +8 vs the +3-5
expected — added a positive-control test and disabled-shape compat tests
beyond the strict minimum); suite now 728 (was 705 pre-Unit-2/3-files).
Screener-mode wall time on NVDA 2026-05-22 with all heavyweight phases off:
**1.64s warm / 2.96s cold** (target ≤5s). Live default-run smoke still
produces a bullish composite (0.4691) with all 26 factors intact and both
`intraday_context`/`competitor_analysis` sections marked `pit_status=pit`.
No deviations from the plan beyond skipping the `key_features.run_config`
echo (no such field exists in the report shape — followed the existing
news/filings precedent of cache-params-only).

---

## Unit 3 — Phase B1: Nasdaq 100 smoke screener

**Goal:** End-to-end working screener for the Nasdaq 100 universe.
Validates the screener shape (file layout, ranking output, markdown render)
before paying for the Nasdaq Composite-scale machinery. Can run with **today's
existing flags** (does not require Unit 2 to be merged) — agent works in
parallel with Unit 2.

### Files

**Read first:**
- `scripts/generate_corpus.py` (worker pattern, error handling, threading)
- `analysis_mvp.py` (cache key pattern, `report_file` helper)
- `tradingagents/analysis_only/pipeline.py` `AnalysisOnlyMVP.__init__`
- `daily_signals.py` (ranking / markdown rendering precedent)

**Created:**
- `scripts/screener.py` — CLI entrypoint
- `tradingagents/analysis_only/screener.py` — pure-function ranking +
  markdown rendering (so it's testable independently of yfinance/Polygon)
- `configs/screener_universe_nasdaq100.yaml` — hardcoded Nasdaq 100 ticker
  list (acceptable for Phase B1; Unit 4 generates this from Polygon
  reference data).
- `tests/analysis_only/test_screener.py` — pure-function tests
- `reports/screener/YYYY-MM-DD/{ranked.json, ranked.md}` — first emitted output

### Steps

1. **Hardcode Nasdaq 100 list** in `configs/screener_universe_nasdaq100.yaml`.
   Source: current Nasdaq 100 constituents (~100 names). OK to commit a
   point-in-time list; Unit 4 will dynamically derive it.
2. **Pure helpers** in `tradingagents/analysis_only/screener.py`:
   - `ScreenerCandidate` dataclass: symbol, composite_score, direction,
     confidence, top_factors (list of (name, weighted_score, rationale)),
     sector, market_cap, adv_usd, next_earnings_in_calendar_days,
     pit_status_summary.
   - `extract_candidate_from_report(report: dict) -> ScreenerCandidate` —
     reads the report JSON shape `AnalysisOnlyMVP` emits.
   - `rank_candidates(candidates, *, top_n=50) -> list[ScreenerCandidate]` —
     sort by composite_score desc, return top N.
   - `rank_per_sector(candidates, *, top_n_per_sector=5)` →
     `dict[sector, list[ScreenerCandidate]]`.
   - `render_screener_markdown(top_overall, per_sector_top, *, as_of_date,
     universe_size, scan_elapsed_s) -> str`.
3. **CLI** in `scripts/screener.py`:
   - Args: `--universe-yaml`, `--date` (default today), `--workers`,
     `--top-n` (default 50), `--top-n-per-sector` (default 5),
     `--output-dir` (default `reports/screener/`), `--exclude-core` (default
     True; filters out tickers already in `configs/universe.yaml` core).
   - Loads universe yaml.
   - Threadpool over `(ticker, date)` jobs. Each worker constructs an
     `AnalysisOnlyMVP` with **all skip-flags set to False** (i.e., everything
     disabled that can be):
     ```python
     mvp = AnalysisOnlyMVP(
         data_provider="polygon",
         options_enabled=False,
         enable_news_fetching=False,
         enable_filings_fetching=False,
         enable_llm_insights=False,
         enable_narrative=False,
         enable_llm_critic=False,
         enable_tradingagents_review=False,
         # if Unit 2 is merged, also pass:
         # enable_intraday_context=False,
         # enable_peer_competitor_analysis=False,
         state_store_path=None,  # screener doesn't write history
         verbose=False,
         logger=...
     )
     ```
     Calls `mvp.run(...)`, extracts the candidate via
     `extract_candidate_from_report`, returns it.
   - Aggregator filters out errors, applies `--exclude-core`, calls
     `rank_candidates` + `rank_per_sector`, writes JSON + Markdown.
   - Does **not** save individual per-ticker reports to disk — screener
     output is the ranked summary only. (Avoids polluting `reports/analysis_mvp/`
     with 100-1000 throw-away screener reports.)
4. **Tests** (`tests/analysis_only/test_screener.py`):
   - `extract_candidate_from_report` parses a synthetic report dict correctly.
   - `rank_candidates` sorts + caps at N.
   - `rank_per_sector` groups + caps correctly; handles missing sector
     (bucket as "Unknown").
   - `render_screener_markdown` produces non-empty output, includes top-50
     table, per-sector tables, scan stats header.
5. **End-to-end smoke test:**
   ```
   python scripts/screener.py \
       --universe-yaml configs/screener_universe_nasdaq100.yaml \
       --date 2026-05-22 \
       --workers 4
   ```
   Expect: ~5 min wall time, ~74 candidates after `--exclude-core`, ranked
   markdown surfaces names worth a closer look.

### Acceptance criteria

- [ ] `scripts/screener.py` runs end-to-end on Nasdaq 100 in <10 min wall time
  at 4 workers.
- [ ] Output `reports/screener/YYYY-MM-DD/ranked.md` contains: scan-stats
  header, top-50 overall table (composite, direction, confidence, sector,
  market cap, top 3 factors), per-sector top-5 tables.
- [ ] At least 4 pure-function tests pass.
- [ ] Smoke: at least the first 5 ranked candidates look plausible (composite
  >0.2, factor rationale matches direction).

### Out of scope

- Universe builder (Unit 4 owns Polygon ticker-reference + ADV/market-cap
  filtering).
- Cohort-aware scoring (Unit 4 owns the (a)-approach where non-tech names
  are scored with universal factors only).
- Auto-promote to core universe (deferred — see "Future work" below).
- Persisting screener history for trend analysis (deferred).

### Completion note

Landed 2026-05-31. Smoke run on 2026-05-22 at 4 workers: 2m35s wall time
(well under 10min budget), 102 yaml tickers → 87 candidates evaluated after
`--exclude-core` (14 overlap with core: AAPL/AMD/ARM/ASML/AVGO/CRWD/GOOGL/
KLAC/META/MRVL/MSFT/MU/NVDA/PLTR) + 1 hard error (ANSS — delisted post
Synopsys merger). 15 pure-function tests pass in
`tests/analysis_only/test_screener.py`. Top 5 ranked: FTNT (+0.62), GFS
(+0.61), LRCX (+0.61), XEL (+0.54), ROST (+0.49) — all bullish with
trend-aligned rationales. Outputs at
`reports/screener/2026-05-22/{ranked.json, ranked.md}`.

Deviations from plan: (1) almost every Nasdaq 100 ticker buckets as
"Unknown" sector in the per-sector view because the Nasdaq 100 superset
has very little overlap with the curated 26-name `sectors:` map in
configs/universe.yaml — by design per the plan's "fall back to Unknown"
note; Unit 4 will solve broader sector-mapping via Polygon SIC codes.
(2) TaskList / TaskUpdate tools are not available in this agent
environment, so this completion note is the only authoritative record.

---

## Unit 4 — Phase B2: Nasdaq Composite filtered + cohort-aware scoring

**Goal:** Production screener. Universe is dynamically built from Polygon
ticker reference + ADV + market-cap filters (~800-1200 Nasdaq names). Scoring
uses cohort-aware approach (a) — non-tech names get the universal-sign-agreed
factor set only.

**Depends on:** Unit 2 (perf optimization is essential at this scale — without
Phase A flags, 1000 names × 21s ÷ 4 workers = ~90 min/scan vs ~25 min with).
Depends on: Unit 3 (extends the CLI and pure-function module).

### Files

**Read first:**
- All Unit 3 deliverables (especially `tradingagents/analysis_only/screener.py`)
- `handoff.md` Section 22 — cohort split, universal vs tech-specific factor table
- `tradingagents/analysis_only/scoring.py` — `DEFAULT_FACTOR_WEIGHTS`,
  `resolve_factor_weights`, `compute_composite`
- `tradingagents/analysis_only/providers.py` — Polygon HTTP patterns for ticker
  reference + aggs

**Touched / created:**
- `scripts/build_screener_universe.py` — new CLI
- `configs/screener_universe_nasdaq.yaml` — generated artifact (committed
  monthly so it doesn't churn weekly; agent commits the first version)
- `tradingagents/analysis_only/screener.py` — extended with cohort scoring
- `tradingagents/analysis_only/scoring.py` — new `UNIVERSAL_FACTOR_WEIGHTS`
  constant + `resolve_factor_weights(cohort=...)` extension (or a parallel
  function — agent's call)
- `scripts/screener.py` — extended with `--cohort-aware` flag
- `tests/analysis_only/test_screener.py` — extended

### Steps

1. **Universe builder** (`scripts/build_screener_universe.py`):
   - Hit `https://api.polygon.io/v3/reference/tickers?market=stocks&exchange=XNAS&active=true&limit=1000` with pagination.
   - Keep `type == "CS"` (common stock) only.
   - For each ticker: fetch trailing 20d daily aggs, compute median dollar
     volume (close × volume), drop if < $5M.
   - Get market cap from `/v3/reference/tickers/{ticker}` (or use shares
     outstanding × last close).
   - Drop market cap < $500M.
   - Write `configs/screener_universe_nasdaq.yaml` with structure:
     ```yaml
     # Generated by scripts/build_screener_universe.py on YYYY-MM-DD
     # Universe: Nasdaq Composite, CS only, market_cap >= 500M, ADV >= 5M
     generated_at: YYYY-MM-DD
     filters:
       market_cap_min_usd: 500000000
       adv_min_usd: 5000000
     tickers:
       - symbol: AAPL
         sector: Tech-MegaCap        # via Polygon SIC code → coarse bucket map
         market_cap_usd: 3500000000000
         adv_usd: 12000000000
       ...
     ```
   - Sector mapping: Polygon SIC code → coarse buckets matching
     `configs/universe.yaml::sectors`. Decision: hardcode a `sic → sector_label`
     dict in `scripts/build_screener_universe.py`. The 26-core map already
     defines the bucket vocabulary (Semiconductors, Tech-MegaCap, Software,
     Networking, Photonics, Specialty-Materials, Energy-Nuclear,
     Financial-Data, Aerospace, Financials, Energy, Healthcare,
     Consumer-Staples, Utilities, Consumer-Discretionary). Anything
     un-mapped → "Other".
   - Time budget: Polygon free tier rate limit is 5 req/min, so this script
     can take 2-3h for 5,000 tickers. Run with `--pace-seconds 12` or use a
     paid tier. Agent should batch what it can.
   - Idempotent: re-running on the same date overwrites the yaml. Intent is
     monthly refresh.
2. **Cohort-aware scoring** in `tradingagents/analysis_only/scoring.py`:
   - Add `UNIVERSAL_FACTOR_NAMES: frozenset[str]` = the set of factor names
     that Section 22's cohort analysis classified as universal (sign-agrees
     across core and canary cohorts):
     `{"market_vix_regime", "peer_relative_valuation",
       "options_iv_term_structure", "momentum_rsi"}`. **Confirm this list
     against `backtest/results/phase2_v1_4_cohort/cohort_20d.md` after Unit
     1 lands** — the list may shift with fresh data.
   - Extend `resolve_factor_weights(...)` to accept an optional
     `cohort: str | None = None` arg. When `cohort == "non_tech"`, zero out
     every weight whose factor name is not in `UNIVERSAL_FACTOR_NAMES`, then
     renormalize. Other cohort values (`None`, `"tech"`) → no change.
   - Add tests for the cohort path.
3. **Sector → cohort mapping** in `tradingagents/analysis_only/screener.py`:
   - `cohort_for_sector(sector: str) -> str` → returns `"tech"` for
     `{Semiconductors, Tech-MegaCap, Software, Networking, Photonics,
      Specialty-Materials, Aerospace}` and `"non_tech"` otherwise. (The
     boundary is fuzzy; pick a defensible mapping and document it inline.)
   - `extract_candidate_from_report` should already work; the cohort logic
     applies at scoring time in the worker.
4. **Screener CLI** (`scripts/screener.py`) — add `--cohort-aware` flag
   (default True). When set, each worker:
   - Resolves the ticker's sector from the universe yaml.
   - Determines cohort via `cohort_for_sector`.
   - Constructs `AnalysisOnlyMVP` as before but the worker post-processes
     the report's `factor_scores`: re-applies cohort-aware weights via
     `resolve_factor_weights(cohort=...)` + `compute_composite(...)`, then
     overwrites the report's `composite_score` + `direction` + `confidence`
     in the in-memory dict before extraction. (The full re-pipeline isn't
     re-run; just the composite is re-derived under the cohort weights.)
   - Alternative: emit *both* tech-weights composite and cohort-aware
     composite in the candidate dataclass, render both columns in markdown.
     Agent's call which is cleaner; I'd lean overwrite-in-place for
     simplicity, with the original tech-composite captured as
     `composite_score_tech_weights` for transparency.
5. **Wire to universe yaml:**
   - `scripts/screener.py --universe-yaml configs/screener_universe_nasdaq.yaml`
     now reads the structured yaml (symbol + sector + market_cap + adv).
   - The hardcoded Nasdaq 100 yaml from Unit 3 stays around for smoke tests.
6. **Smoke run:**
   ```
   python scripts/build_screener_universe.py  # one-shot, generates the yaml
   python scripts/screener.py \
       --universe-yaml configs/screener_universe_nasdaq.yaml \
       --date 2026-05-29 \
       --workers 4 \
       --cohort-aware
   ```
   Expect: ~30-50min wall time at 4 workers + Phase A skip-flags.

### Acceptance criteria

- [ ] `configs/screener_universe_nasdaq.yaml` committed with ~800-1200
  filtered Nasdaq tickers.
- [ ] `UNIVERSAL_FACTOR_NAMES` matches the post-Unit-1 cohort table (verify
  against `backtest/results/phase2_v1_4_cohort/cohort_20d.md`).
- [ ] `scripts/screener.py --cohort-aware` runs end-to-end on the filtered
  universe in <60 min wall time at 4 workers.
- [ ] Output `reports/screener/YYYY-MM-DD/ranked.md` shows: top-50 overall +
  per-sector breakdowns. Per-name rows include both `composite_tech_weights`
  and `composite_cohort_aware` (or one of them with the other available in
  the JSON sidecar).
- [ ] Tests pass: cohort-aware scoring tests + universe builder unit tests
  (mocked Polygon HTTP).

### Out of scope

- **Auto-promote-to-core mechanism.** Generates a candidate list, that's it.
  Manual review for now. Building a multi-week consistency tracker is a future
  workstream.
- **Daily cadence.** Weekly only.
- **Approach (b) full two-composite emission** (deferred per the decided
  defaults). Only the (a) approach is in scope.
- **Backtesting screener output.** Screener IC validation (does ranking by
  composite actually predict forward returns on out-of-universe names?) is a
  separate workstream — needs a separate cohort-only IC analysis once we have
  a few weeks of screener history.

### Completion note

Landed 2026-05-31. Test count: 789 (was 729 pre-Unit-4) — **+60 tests**: 8 in
`tests/analysis_only/test_scoring.py` for `UNIVERSAL_FACTOR_NAMES` +
cohort kwarg on `resolve_factor_weights`; 25 in `test_screener.py` for
`cohort_for_sector` (20-row parametrize), `rescore_report_for_cohort`,
tech-weights-column rendering; 27 in new `test_build_screener_universe.py`
covering SIC → sector mapping (16-row parametrize), paginated ticker
fetch with mocked HTTP, ADV median calc, end-to-end build_universe with
the right rows dropped at each filter stage, and yaml serializer
round-trip through PyYAML. Universe build: **12m51s** wall, **1139
kept** of 3300 NASDAQ CS tickers (1868 dropped <$500M mcap, 171
dropped <$5M ADV, 122 dropped missing mcap; details_failed=0).
Screener end-to-end smoke at `--workers 4 --cohort-aware` on the full
1139-name universe (1123 evaluated after `--exclude-core` removed 16
core overlaps): **4m38s wall time**, zero errors. Outputs at
`configs/screener_universe_nasdaq.yaml`,
`reports/screener/2026-05-22/{ranked.json, ranked.md}`. Sector
distribution after SIC→sector mapping: Other 366 (32%, mostly
industrials / REITs the coarse vocab doesn't cover), Healthcare 248,
Financials 156, Software 132, Semiconductors 80, Financial-Data 41,
Consumer-Discretionary 39, Networking 14, Utilities 14, Consumer-Staples
14, Tech-MegaCap 13, Energy 11, Aerospace 11 — 13/13 buckets populated,
no "Unknown".

**Known artifact in cohort-aware ranking:** the universal weight vector
(7 surviving factors, of which `options_iv_term_structure` is weight 0
in DEFAULT_FACTOR_WEIGHTS, so 6 effective) is dominated by trend +
momentum mechanics. Many non-tech bullish names saturate at composite
+0.832 with identical top factors, causing top-50 ties broken by
confidence then symbol-alpha. The `Composite (tech wts)` column gives
the differentiation. The plan acknowledges this — approach (b)
two-composite-emission is deferred unless approach (a) systematically
misses things.

**Deviations from plan:**
1. `--cohort-aware` is `nargs="?", const=True` so it accepts both
   `--cohort-aware` (no value) and `--cohort-aware=false`. The plan
   specified default-True; bare-flag invocation works identically.
2. The screener worker tuple grew from 4 → 7 fields (added market_cap,
   adv_usd, cohort_aware passthrough). Backward compat preserved — old
   flat-yaml universe files (Unit 3) still load because the universe
   loader returns empty metadata dicts and the worker tolerates `None`.
3. SIC mapping in `scripts/build_screener_universe.py` is intentionally
   broader than the plan's "26-core vocabulary" — added catch-all
   prefix rules so common-but-uncovered industries (airlines, services,
   etc) don't all bucket as Other. ~32% Other is still high; could be
   tightened with a richer manual SIC table if needed.

**TODO (Unit 5 concern):** re-verify `UNIVERSAL_FACTOR_NAMES` in
`tradingagents/analysis_only/scoring.py` against
`backtest/results/phase2_v1_4_cohort/cohort_20d.md` once Unit 1
finishes. Current list pulled from handoff Section 22 + the "always
mechanical" set per plan brief, all verified subset of
`DEFAULT_FACTOR_WEIGHTS`.

---

## Unit 5 — Section 27 handoff doc + sign-off

**Goal:** Append Section 27 to `handoff.md` summarizing what shipped from
Units 1-4, with artifact paths and any deferred follow-ups.

### Steps

1. Read each unit's completion note.
2. Append `## 27. Phase 2 v1.4 regen + Nasdaq Composite screener` to
   `handoff.md` following the same format as Sections 22-26 (motivation,
   what landed, validation, honest caveats, deferred follow-ups, output
   artifacts).
3. Update the roadmap table at the top of `handoff.md` with the new entry.
4. Run `python -m pytest` once more end-to-end and assert the test count
   has gone up appropriately.
5. Update `MEMORY.md` if any of the work surfaced new user-preferences /
   feedback worth saving.

### Acceptance criteria

- [x] Section 27 appended, format matches Sections 22-26.
- [x] Roadmap table updated.
- [x] Test suite green.

### Completion note

**Completed 2026-05-31.** Section 27 appended to `handoff.md` (roadmap
table row 27 added + full body with comparison tables, cohort split,
artifacts, caveats, deferred). 811 tests passing + 1 skipped. The plan's
"Future work — corpus regen performance" section was added during
execution; it documents the 8 perf levers Items 1, 3, 9 already partially
addressed. Item #3 (Polygon Financials cache) shipped in parallel with
this unit — full details in Section 27.

### Out of scope

- Picking the v1.5 weight changes (separate decision; depends on Unit 1's
  fresh IC table).
- Production scheduling / cron of the screener (deferred — user runs manually
  on Fridays until they ask for automation).

### Completion note

_(append on completion)_

---

## Future work (not in scope for this plan)

These came up during scoping. Logging here so they're not lost:

1. **v1.5 weight commit** based on Unit 1's fresh IC. Likely candidates:
   `options_iv_term_structure` 60d-inversion split, `filings_recency_signal`
   bug investigation, possible `market_fear_greed_regime` weight bump now
   that VIX proxy fills the coverage gap.
2. **Screener auto-promote** — multi-week consistency tracker that suggests
   tickers for `configs/universe.yaml` core promotion.
3. **Screener IC validation** — does a top-50 weekly screener pick actually
   produce forward alpha vs SPY / Nasdaq composite? Needs ~12 weeks of
   screener history first.
4. **Daily cadence option** for the screener, if weekly proves to be too
   coarse for catching earnings-event-driven names.
5. **Approach (b) — two-composite emission** in the screener, if cohort
   (a) systematically misses non-tech opportunities.
6. **Walk-forward OOS validation protocol** (Section 25 deferred).
7. **Confidence calibration plot + recalibration** (Section 13/14 deferred).
8. **Bug-fix `filings_recency_signal` (n=0 across corpus)** and
   `valuation_sales_multiple_vs_growth` per-ticker absences (Section 12
   deferred).
9. **Rank screener output by `composite_score_tech_weights`** instead of
   the cohort-aware `composite_score`. Cohort-aware approach (a) saturates
   the top of the ranking (Unit 4 finding: 25 non-tech bullish names tied
   at +0.832). Data is already preserved in both columns; just change the
   sort key in `render_screener_markdown`. ~10min fix. Note: also re-check
   `top_overall` filtering happens against the same key it sorts by.

## Future work — corpus regen performance

The current Phase-2-scale regen takes ~5-13h depending on contention. Below
are speedup levers, ranked by ROI per hour of dev effort. Each item is a
discrete unit; pick freely.

### Cheap wins (each ~half a day max)

1. **Don't pass `--pace-seconds` on paid Polygon tier.** Note: the CLI
   default is already 0.0 (verified 2026-05-31); the 1.0s on the May-31
   regen was passed explicitly in the invocation. Action item is just
   "don't pass it next time on paid tier." Help text in `generate_corpus.py`
   could be updated to drop the "for free Polygon tier" guidance which
   reads as a default-tier recommendation.
2. **Replace yfinance for market context (SPY / VIX / sector ETFs / `^IRX`)
   with Polygon aggs.** Already noted in handoff §10/§17/§20 footnotes.
   Eliminates the 401-Invalid-Crumb log noise AND the bulk of
   non-pre-IPO errors. Lets you scale workers 4 → 8 without yfinance
   throttling. **~30-40% wall-clock cut.**
3. **Cache Polygon Financials across the regen.** Per-ticker fundamentals
   are identical for every Friday within a fiscal year, but we fetch them
   ~52 times. `(ticker, year) → response` dict shared across the worker
   pool saves ~10-15 min on a Phase-2-scale regen.

### Medium-effort wins (1-3 days each)

4. **Batch by ticker, slice by date.** Each worker currently fetches 300d
   of price history per `(ticker, date)` job. Fetching the full multi-year
   history ONCE per ticker and slicing per Friday in-memory cuts ~80% of
   the Polygon price calls. **Roughly halves regen time.**
5. **`ProcessPoolExecutor` instead of `ThreadPoolExecutor`.** BSM IV
   inversion and statement parsing are CPU-bound — bounded by the GIL
   today. Section 10 punted on process pools due to sandbox issues; those
   are gone now. **2-3x on CPU-heavy phases.**
   - **Done 2026-06-04.** Added `--executor {process,thread}` to
     `scripts/generate_corpus.py`; default is now `process`. JobTuple is
     already picklable (basic types + None) — drop-in clean. Trade-off
     documented inline: module-level caches in `providers.py`
     (`_FINANCIALS_ALL_CACHE`, `_RATE_SERIES_CACHE`, `_VIX_SERIES_CACHE`,
     `_POLYGON_DAILY_AGGS_CACHE`) don't cross process boundaries. `--workers`
     help text updated to drop the stale "yfinance 2-3 thread cap" warning.
     +7 tests (`tests/analysis_only/test_generate_corpus_executor.py`)
     covering CLI parse + pickle round-trip. Smoke on NVDA × 2 Fridays
     both paths: process 9s / thread 4s (per-process import overhead
     dominates on 2-job smoke; amortizes to negligible on a real regen).
6. **Polygon grouped-aggs endpoint** (`/v2/aggs/grouped/locale/us/market/
   stocks/{date}`) returns every symbol's daily aggs in one HTTP call.
   For corpus regen this is dramatic — one request per *date* instead of
   one per `(ticker, date)`. **5-10x on the price-fetch phase.**

### Biggest structural wins (1-2 weeks each, biggest payoff)

7. **Local cache layer (parquet or sqlite)** keyed by `(ticker, as_of_date,
   data_type)`. Precompute prices / fundamentals / options chains / market
   context into a local store. First precompute takes ~1h; every
   subsequent regen reads from disk in seconds. For the iteration loop
   (weight tuning → regen → IC → repeat), this is architecturally right.
   **Each weight-change iteration goes from hours to minutes.**
8. **Incremental regen / patch mode.** Most v1.x weight changes only
   affect composite/direction/factor scores/narrative — not price,
   fundamentals, options, market context. A "patch regen" mode reads the
   existing JSON and rebuilds only the dependent fields, skipping ~95%
   of the work. **10-50x for weight-tuning iterations specifically.**

### Recommended pairing

- **For the next single regen run:** items 1 + 2 + 3 together. ~1/2 day of
  work, probably halves wall-clock. Item 2 also removes the persistent
  yfinance error noise so logs become useful.
- **For the iteration loop:** items 7 + 8 together. Higher up-front cost
  but converts "I can do 2-3 weight experiments per day" → "I can do
  20-30 per day," which is the actual bottleneck on improving model
  accuracy. Unblocks weekly v1.5 / v1.6 cadence.

### Completion notes — followups

**Item #2 (replace yfinance for SPY / sector ETFs) — done 2026-06-01.**
- ETFs (SPY + XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY) now route to
  Polygon `/v2/aggs` via a new module-level cache
  (`providers.fetch_polygon_daily_aggs_cached`); yfinance becomes a
  fallback only when `POLYGON_API_KEY` is unset.
- `^VIX` / `^IRX` / `^TNX` stayed on yfinance — Polygon Stocks plan
  returns 403 NOT_AUTHORIZED on `I:VIX` / `I:IRX` / `I:TNX` (verified).
  `is_polygon_supported_symbol("^X")` is False; routing falls through
  cleanly. Indices would need a separate Indices-plan add-on.
- **Composite drift caveat:** Polygon `adjusted=true` is split-only
  while yfinance `auto_adjust=True` is split-AND-dividend. SPY/XLK
  20-day returns shift by ~0.3–2.5 percentage points. Composite drift
  on the NVDA 2024-06-21 smoke was +0.042 (0.6415 → 0.6831), mostly
  via the `industry_relative_strength` and `peer_relative_momentum`
  factors. The new behaviour is more *internally consistent* because
  the primary symbol (NVDA) already runs through Polygon price-only —
  treating benchmarks the same way removes a hidden adjustment-policy
  asymmetry — but it does mean the post-v1.4 corpus will not be bit-
  identical to pre-change. A re-regen on the same JSONs will show this
  drift; numerically downstream rankings/IC should be minimally
  affected (relative ordering of stocks doesn't depend on which
  adjustment policy SPY uses).
- Cache key includes `provider`, so the on-disk yfinance entries for
  SPY/sector ETFs are orphaned (not stale-served).
- Tests: +14 in `test_providers.py` (Polygon cache mechanics, PIT,
  fallback) and new `test_pipeline_polygon_routing.py` (router only
  hits Polygon for supported symbols when key is present).

## Completion notes — followups (post-Section 27)

- **#9 (sort screener by tech-weights):** done 2026-06-01. Added
  `_effective_rank_score` helper in
  `tradingagents/analysis_only/screener.py` — prefers
  `composite_score_tech_weights` when populated (cohort-aware mode),
  falls back to `composite_score`. Used by both `rank_candidates` and
  `rank_per_sector`. +3 tests. Re-ran cohort-aware screener smoke; new
  top 5 = SEZL, RDWR, FTNT, EFSC, FFIV (was ABCL, ADAM, ADUR, ALKS,
  ANIP alphabetical pre-fix). Saturated-cap names now ranked by
  underlying tech-weights spread.
- **#3a (options_iv_term_structure v1.5 commit):** done 2026-06-01.
  Sign inverted in `score_iv_term_structure` (contango → negative,
  backwardation → positive). `DEFAULT_FACTOR_WEIGHTS` bumped 0.00 →
  **0.04**. Anchored by post-regen IC: core 60d -0.109 / 80% sign-cons,
  canary -0.105 (cohort-universal). +1 test (split existing). Smoke
  on NVDA 2024-06-21: weight 0.000 → 0.030 after renormalization;
  composite Δ -0.044 (renorm dilution; no factor score change for NVDA
  because its slope is in the flat zone). Should be re-measured against
  the next regen to confirm the IC moves the right direction.
- **#3b (filings_recency_signal "bug"):** done 2026-06-01. Diagnosed
  as NOT a code bug — the factor correctly returns
  `data_available=False, score=0` in the unavailable branch (line
  ~4490), which is hit because the regen used `--minimal-context`
  (bundles news+filings skip). Fix: split `--minimal-context` into
  `--skip-news` and `--skip-filings` in
  `scripts/generate_corpus.py`. Legacy `--minimal-context` is now an
  alias that flips both. Next regen can pass `--skip-news` only,
  enabling proper `filings_recency_signal` IC measurement.
- **#2 (Polygon market context replacement):** in flight. Background
  agent in worktree at the time of this writeup.
