# Temp Handoff — Accuracy follow-ups (2026-06-04)

**Status:** working doc for 4 parallel agent workstreams. Each unit below
is self-contained; agents do NOT need to ask the human for context.

## Where the project stands

- **Model version:** v1.7 (handoff Section 28 + this session).
- **Test count:** 940 passing + 1 skipped.
- **Active branch:** `codex/current-change-pr` from main checkout.
- **Last regen:** v1.7 corpus generated 2026-06-03, **2h 43m**, 4,764 ok / 36 errors.
  Location: `reports/analysis_mvp/*.json` (4,880 reports). PIT-correct
  except `analyst_consensus` (live snapshot).
- **Recent v1.7 changes shipped this session:**
  - Regime-conditional sign flips for chop regime (3 factors:
    `market_fear_greed_regime`, `fund_profit_margins`, `options_iv_rank`).
    Walk-forward verified +1.10pp at 60d.
  - Bear-side regime gate — bearish calls only fire in chop + fear.
  - Direction-conditional confidence calibration — separate isotonic
    curves per direction. Bearish curve plateaus at ~0 (anti-predictive).
  - `ticker_fear_greed_regime` factor at weight=0 (pending IC).
  - Polygon grouped-aggs endpoint for screener universe builder.

## Read first (for ALL agents)

- `handoff.md` Section 28 — v1.5-v1.6 work + walk-forward + calibration infra.
- `plans/accuracy_improvements.md` Sections D-F — v1.5/v1.6 commit history.
- `tradingagents/analysis_only/scoring.py` — `DEFAULT_FACTOR_WEIGHTS`,
  `regime_for_market_context`, `apply_regime_to_factor_scores`,
  `direction_for_composite_regime_gated`, `fit_isotonic_calibration_by_direction`,
  `apply_isotonic_calibration_directional`.
- `tradingagents/analysis_only/pipeline.py` — `_build_report` is the
  factor emission + composite computation site (line ~4700).
- `tradingagents/analysis_only/backtest.py` — backtest record / IC
  computation.
- `portfolio/simulator.py` — Section 17 walk-forward portfolio simulator.

## Operational constraints (read carefully)

- Repo Python venv: `.venv/bin/python`. Use it for tests/smoke.
- POLYGON_API_KEY is set; SEC_USER_AGENT defaults work.
- The corpus at `reports/analysis_mvp/` is **production**. Don't touch.
  Each agent has a worktree; stay inside it.
- Don't touch `state/analysis_state.sqlite`.
- Don't run `scripts/generate_corpus.py` against the production corpus dir.
- Test discipline: ship new factors at **weight=0** until IC validates
  on a regenerated corpus (matches v1.5-v1.7 pattern).
- Walk-forward gate (`scripts/walk_forward_eval.py`) for any weight/sign
  change: ≥+0.5pp at 60d bullish, no >1pp regression at 5d/20d.

## Decided defaults

- Primary horizon: **60d**.
- All new factors: **weight=0** pending IC.
- Test target: `.venv/bin/python -m pytest -q` must pass with no
  regressions vs baseline 940 + 1 skipped.

---

## Unit X1 — Polygon news sentiment factor

**Goal:** New per-ticker factor scoring news sentiment via Polygon's
historical news endpoint. Currently `--skip-news` is the regen default
because yfinance news was unreliable; Polygon's news IS on the paid plan
and is PIT-correct (timestamped articles, queryable as_of date).

**Scope:**

1. New `PolygonNewsProvider` in `tradingagents/analysis_only/providers.py`:
   - Hit `https://api.polygon.io/v2/reference/news?ticker={X}&published_utc.lte={as_of}&limit=50`
   - Module-level cache + lock (mirror `_FINANCIALS_ALL_CACHE` pattern).
   - Returns list of dicts with `title`, `description`, `published_utc`,
     `tickers`, `insights` (Polygon includes sentiment classifications
     when available — `insights[].sentiment` is `"positive"|"neutral"|"negative"`).

2. New scoring function in `scoring.py`:
   - `compute_news_sentiment(news_items, as_of_date, *, lookback_days=14)` —
     compute net sentiment score over trailing window. Use Polygon's
     `insights[].sentiment` when present; fall back to simple keyword
     scoring (positive_words / negative_words from a hardcoded list) on
     `title + description` when insights are missing.
   - `score_news_sentiment(net_score, n_articles) -> tuple[float, str, bool]` —
     factor score function. Require ≥3 articles to emit a non-zero score
     (else `data_available=False`).
   - Add `"news_sentiment": 0.00` to `DEFAULT_FACTOR_WEIGHTS`.

3. Pipeline wiring (`tradingagents/analysis_only/pipeline.py`):
   - Add `enable_news_factor` constructor flag (default True). Distinct
     from existing `enable_news_fetching` which gates the LLM-context
     news payload — this one specifically pulls Polygon news for the
     factor.
   - Around the existing news-loading area, call `PolygonNewsProvider`,
     compute `compute_news_sentiment`, emit `add_factor("news_sentiment", ...)`.
   - Persist `key_features.news_sentiment` block with the computed score
     + article count + window.

4. Tests:
   - `tests/analysis_only/test_providers.py` — Polygon news cache dedup,
     PIT filter (`published_utc.lte`), no-key fallback, response parsing.
   - `tests/analysis_only/test_scoring.py` — `compute_news_sentiment`
     positive / negative / neutral / insufficient-articles cases.
     `score_news_sentiment` factor function tests.

5. Smoke verification:
   - `.venv/bin/python analysis_mvp.py --ticker NVDA --date 2025-08-22 --no-markdown --no-json-stdout --output-dir /tmp/news_smoke`
   - Verify `key_features.news_sentiment.status == "ok"` with N≥3 articles
     and a sensible score.
   - At least 2 other ticker/date combos.

**Acceptance:**
- Full test suite passes (`.venv/bin/python -m pytest -q`). +10-15 new tests expected.
- Smoke produces `news_sentiment` factor at weight=0, `data_available=True`
  on >50% of recent dates.
- Composite delta from baseline: **0.000** (weight=0).

**Out of scope:**
- IC measurement / weight commit (separate round after regen).
- LLM-based sentiment (keyword + Polygon insights only).
- Modifying `enable_news_fetching` (that's the existing flag for LLM context).

---

## Unit X4 — Volatility/P&L-based backtest metrics

**Goal:** Backtest currently reports hit-rate only. Add return-based +
risk-adjusted metrics using the existing `portfolio/simulator.py` to
expose what hit-rate hides (some wins are tiny, some losses are huge).

**Scope:**

1. New function in `tradingagents/analysis_only/backtest.py`:
   - `compute_return_metrics(records, *, horizon, direction_filter=None) -> dict` —
     returns `{mean_return, median_return, sharpe, sortino, max_drawdown,
     winsorized_mean, hit_rate, profit_factor}` for the records that have
     a forward_return at `horizon`. Profit factor = sum(positive returns)
     / |sum(negative returns)|.
   - `compute_return_metrics_by_direction(records, *, horizon)` — same
     but bucketed by direction. Returns `{bullish: {...}, bearish: {...},
     neutral: {...}}`.

2. Extend `backtest.py` summary output:
   - In the existing `summary.json` output, add a `return_metrics`
     block at top level alongside `total_records` + `by_horizon`.
   - In `summary.md`, add a new "Risk-adjusted metrics" section per
     horizon with the Sharpe/Sortino/MaxDD/ProfitFactor values.

3. Tests:
   - `tests/analysis_only/test_backtest.py` — `compute_return_metrics`
     on a small synthetic record set; assert known Sharpe/Sortino math.
     Test direction-filter, missing returns, all-zero returns.

4. Smoke:
   - `.venv/bin/python backtest.py --reports-glob "reports/analysis_mvp/*.json" --by-factor --benchmark SPY --output-dir /tmp/x4_smoke`
   - Verify new `return_metrics` block + Sharpe values are sensible.

**Acceptance:**
- Tests pass; +8-12 new tests expected.
- Sharpe at 60d bullish should be in the 1.5-3.0 range (the v1.7 corpus
  was a bull tape, so positive Sharpe expected).
- Sharpe at 60d bearish should be **negative or near-zero** (bearish
  bucket is anti-predictive per Section 14/15).

**Out of scope:**
- Trading-cost modeling beyond what already exists in
  `portfolio/simulator.py`.
- New direction-classification logic.

---

## Unit X5 — Position-level stop-loss simulation

**Goal:** Extend `portfolio/simulator.py` to model intra-week stops.
Current simulator carries weekly positions without stops; Section 16's
1.5×ATR rule should be plugged in. May reveal that certain entry signals
are unprofitable AFTER stops.

**Scope:**

1. `portfolio/simulator.py`:
   - Add `stop_loss_atr_multiple: float = 1.5` to `SimulationConfig`.
   - In the per-week loop, after entry: compute stop level = `entry - (atr_14 * stop_multiple)`.
     If `min_close_during_week <= stop_level`, exit at the stop (not at
     end-of-week close). Adjust realized return accordingly.
   - Need access to **intra-week min close** — the simulator currently
     uses end-of-week close only. Reach to yfinance / Polygon for the
     trading-week's daily lows.
     **PIT note:** load each week's intra-week prices ONCE upfront, not
     per-symbol per-day (use the grouped-aggs approach from Future-work #6
     that already shipped).
   - Track `n_stops_hit` in the per-week stats; expose in `SimulationResult`.

2. `configs/sizing.yaml`:
   - Add `stop_loss_atr_multiple: 1.5` (matching the simulator default
     and Section 16's daily layer setting).

3. Tests:
   - `tests/portfolio/test_simulator.py` — stop fires when intra-week
     min crosses the level; stop does NOT fire when min stays above;
     stop-loss factor=0 disables stops; multiple positions with mixed
     stop/no-stop behaviors.

4. Smoke:
   - `.venv/bin/python portfolio_simulate.py` (existing CLI) — verify
     the new `n_stops_hit` field appears in the policy comparison
     markdown.

**Acceptance:**
- Tests pass; +5-8 new tests expected.
- Smoke run completes; `n_stops_hit > 0` on the v1.7 corpus (bull tape
  had pullbacks that hit ATR stops).
- Sharpe with stops should be ≥ Sharpe without stops on the bullish
  policy (stops cut tail losses; if Sharpe DROPS materially, document
  it as a finding).

**Out of scope:**
- Trailing stops (only fixed-ATR for now).
- Stop on bearish positions (model is long-only).
- Take-profit logic.

---

## Unit X1-data — Multi-regime corpus extension

**Goal:** Add 2020-Q1 (COVID crash) and 2022-H2 (rate-hike bear) to the
corpus. The hardest deferred item but biggest payoff — current corpus is
a single bull-tape regime; multi-regime corpus stress-tests every
regime-sensitive factor.

This is **data work, not code**. Spawns as a background bash command, not
an agent.

**Scope:**

1. **Pre-flight:** check Polygon Stocks plan coverage on:
   - Daily prices going back to 2018-01 — should work on paid tier.
   - Options chains 2020-2022 — Section 19 noted Polygon Options Starter
     plan returns CURRENT chains only. **IV factors WILL be data_available=False
     on these older dates.** Accept this; factor stays as-emitted (score=0
     when unavailable, doesn't pollute IC).
   - SEC EDGAR — works back any historical date.
   - VIX series — `^VIX` on yfinance stays unchanged.

2. **Date range expansion:**
   - Current corpus: 2023-07-14 → 2026-05-22 (150 Fridays).
   - Add: 2020-01-03 → 2023-07-07 (~185 Fridays) as a SECOND regen pass.
   - Write to a SEPARATE dir `reports/analysis_mvp_extended/` first, then
     merge if it looks good.

3. **Regen command:**
   ```
   .venv/bin/python -u scripts/generate_corpus.py \
     --force --skip-news --workers 4 --executor process \
     --start 2020-01-03 --end 2023-07-07 \
     --output-dir reports/analysis_mvp_extended/ \
     --errors-log reports/corpus_errors_extended.jsonl \
     > reports/regen_extended.log 2>&1
   ```
   Expected ~3-4h wall time at process-pool workers.

4. **Post-regen analysis:**
   - Walk-forward on the extended corpus → does v1.6 / v1.7 commit hold?
   - Cohort IC split → does `market_spy_trend` sign hold in 2020 bear?
   - Factor IC stability → which factors are regime-rally-only artifacts?

5. **Decision deliverable:** `backtest/results/multi_regime_findings.md` —
   honest read on which v1.x commits survive multi-regime evidence.

**Acceptance:**
- ~185 new reports in `reports/analysis_mvp_extended/`, error rate <2%.
- Backtest produces clean IC table for the extended period.
- Findings doc identifies any factor whose sign flips across regimes.

**Out of scope:**
- Committing any weight change based on the findings (that's a separate
  weight-commit round after we see the data).
- Merging the two corpora into one (do that after the findings doc
  signs off).

---

## Coordination

- Agents X1 / X4 / X5 run in worktree isolation. The data extension
  runs as a background bash from the main checkout (it only writes to
  `reports/analysis_mvp_extended/`, which no agent touches).
- Each agent: append a 3-5 line completion note to this doc under a
  new `## Completion notes` section at the end. State what landed,
  test count delta, smoke output, and any deviation.
- Test count baseline: **940 + 1 skipped**.
- The merge agent (main session) will cherry-pick worktree changes
  back to main and run the full suite.

## Completion notes

### Unit X1 — Polygon news sentiment factor (2026-06-04)
- Landed `PolygonNewsProvider` (existing class promoted to module-level
  `_POLYGON_NEWS_CACHE` + lock + `reset_polygon_news_cache`), new
  `compute_news_sentiment` + `score_news_sentiment` in `scoring.py`,
  `"news_sentiment": 0.00` in `DEFAULT_FACTOR_WEIGHTS`, new
  `enable_news_factor` pipeline flag, `_load_news_sentiment` helper,
  factor emission in `_build_report` (pillar=`sentiment`, weight=0),
  and `key_features.news_sentiment` block.
- Worktree test count: 705 → 732 passed (+27 new) + 1 skipped, no
  regressions. (Worktree baseline differs from the handoff's 940 by
  unrelated test selection; delta is the load-bearing number.)
- Smoke (3 ticker/date combos, all `pit_status=pit`,
  `data_available=True`, `weighted_score=0.0`, composite delta=0):
  NVDA 2025-08-22 → n=50 net=+0.52 score=+0.7;
  AAPL 2025-07-25 → n=50 net=+0.40 score=+0.4;
  TSLA 2026-03-14 → n=50 net=+0.24 score=+0.4. Polygon `insights`
  populated all three (kw_fallback=0).
- No deviations from scope.

### Unit X5 — Position-level stop-loss simulation (2026-06-04)
- Added `stop_loss_atr_multiple: float = 1.5` to `SimulationConfig`,
  optional `atr_14` to `WeeklyObservation`, and `n_stops_hit` on
  `WeeklyState` + `SimulationResult.metrics`. `run_simulation` now
  accepts `intra_week_min_close` and exits long positions at
  `entry - atr_14 * multiple` when the week's min close pierces it
  (long-only; multiple=0 disables). Markdown comparison gains a
  "Stops hit" column. `portfolio_simulate.py` extracts ATR from
  `key_features.technical.atr_14`, builds the min-close map from
  the daily yfinance pull, and adds `--stop-loss-atr-multiple`.
- Worktree test count: 883 → 891 passed (+8 new) + 1 skipped, no
  regressions. (Worktree baseline differs from handoff's 940 by
  unrelated test selection; delta is the load-bearing number.)
- Smoke (full v1.7 corpus, equal_weight_bullish, 158 of 160 weeks
  had min-close data): with stops → Sharpe +3.44, MaxDD -11.77%,
  n_stops_hit=132, CAGR +112.87%; without stops (mult=0) → Sharpe
  +2.97, MaxDD -15.80%, CAGR +98.79%. Stops add +0.47 Sharpe and
  cut MaxDD ~4pp on the bullish policy. Test slice (2025-11-21 →
  2026-05-22, 32 weeks): stops add +0.68 Sharpe (+5.07 vs +4.39).
- Deviations: `configs/sizing.yaml` already had
  `stop_loss_atr_multiple: 1.5` (carried from `SizingConfig`); no
  edit needed there. Did NOT touch `providers.py`; intra-week mins
  reuse the existing yfinance daily pull instead of grouped-aggs
  (which is not yet committed on this worktree's base).

### Unit X4 — Volatility/P&L-based backtest metrics (2026-06-04)
- Landed `compute_return_metrics` + `compute_return_metrics_by_direction`
  in `tradingagents/analysis_only/backtest.py` (Sharpe/Sortino
  annualized via `sqrt(252/horizon_days)`, max-drawdown on
  cumulative-sum P&L for numerical stability with extreme single-bet
  returns, winsorized mean, profit factor, direction-aware strategy
  P&L sign-flip for bearish records). `summarize_all` now emits a
  top-level `return_metrics` block keyed by horizon and
  `render_summary_markdown` writes a "Risk-adjusted metrics" table
  per horizon in `summary.md`. Files: `tradingagents/analysis_only/
  backtest.py`, `tests/analysis_only/test_backtest.py`.
- Worktree test count: backtest module 69 → 85 (+16 new); full suite
  passes 721 + 1 skipped, no regressions. (Worktree collection
  differs from handoff's 940 baseline; delta is the load-bearing
  number.)
- Smoke (`/tmp/x4_smoke`, full v1.7 corpus, 4905 records, SPY
  benchmark) at 60d: bullish n=2216 Sharpe +0.90 Sortino +1.74,
  bearish n=535 Sharpe -0.84 Sortino -0.64, neutral n=1584
  Sharpe +0.89. Bearish Sharpe is **negative** on the bull tape as
  predicted (anti-predictive bucket). Bullish Sharpe (+0.90) is
  positive but below the plan's 1.5-3.0 expectation — the IID
  single-stock dispersion (stdev ~0.32 on 60d returns) sets a
  realistic ceiling, lower than the portfolio-level Sharpe in
  Section 17 which assumes equal-weight basket aggregation.
- Deviation: max-drawdown switched from compounded `(1+r)` to
  cumulative-sum P&L. Bearish records with raw forward returns >100%
  produce `1 + (-r) < 0`, breaking the compounded formula
  numerically; equal-sized-bet cumulative P&L is well-defined for
  any single-record magnitude and aligns with Section 16's "R units"
  convention. Markdown labels the column "Max DD (R units)" so the
  unit is unambiguous.
