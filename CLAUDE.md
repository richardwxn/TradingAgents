# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a fork of the upstream **TradingAgents** LLM multi-agent framework that adds a large **custom quantitative pipeline** on top. Two systems coexist:

1. **Upstream LLM multi-agent framework** — `tradingagents/agents/`, `tradingagents/graph/`, and `cli/`. A LangGraph workflow of analyst/researcher/trader/risk agents that debate and produce a trade decision. Entry points: the `tradingagents` console script, `python -m cli.main`, or `main.py`.

2. **Custom deterministic quant pipeline** (the focus of most fork work) — `tradingagents/analysis_only/` plus the top-level scripts (`analysis_mvp.py`, `daily_signals.py`, `backtest.py`, `analysis_ui.py`, `trade_tickets.py`, etc.) and the `portfolio/` package. This system scores tickers with **deterministic factor weights** (no LLM required for the core signal) and drives a paper/operator trading workflow.

The two systems share the `tradingagents/llm_clients/` multi-provider client layer and `tradingagents/dataflows/` data providers, but have separate entry points and largely separate concerns. Most day-to-day development happens in system #2.

## Authoritative design doc

`handoff.md` (~175KB) is the design-of-record for the quant pipeline. Code comments reference it as **"handoff Section N"** (e.g. factor weight rationale in `scoring.py` cites "handoff Section 12 / 22"). When changing scoring weights, factor cohorts, thresholds, or calibration, read the relevant handoff section first — the magic numbers are deliberate and backtest-justified.

## Environment & commands

There is **no committed `.venv`**; the Makefile expects one at `.venv/bin/python` (override with `PYTHON=...`). `uv.lock` is present (project uses [uv]); create the env with `uv sync` (or `pip install .` into a venv), Python ≥3.10 (3.13 recommended).

Tests (fast, deterministic, no API keys needed — `tests/conftest.py` injects placeholder keys so nothing hangs on missing credentials):

```bash
make test                 # all: .venv/bin/python -m pytest tests
make test-analysis        # tests/analysis_only
make test-portfolio       # tests/portfolio
.venv/bin/python -m pytest tests/test_scoring.py::test_name   # single test
```

Model-quality checks are **kept manual** (they use historical reports under `reports/analysis_mvp/*.json` and live yfinance prices, so they are slow and non-deterministic). Run them when scoring, weights, thresholds, report generation, or sizing change:

```bash
make model-backtest-override   # backtest reports with configs/proposed_weights_v1.json
make model-backtest-train      # train-window backtest
make model-backtest-test       # test-window (OOS) backtest
make portfolio-sim             # portfolio policy simulation
make tune-model                # weight tuning sweep (configs/tuning.yaml)
make model-acceptance          # scripts/check_model_acceptance.py gate
make test-all                  # fast suite + prints the manual checklist above
```

## Quant pipeline data flow

```
analysis_mvp.py  ──>  reports/analysis_mvp/<TICKER>.json   (AnalysisReport)
   (AnalysisOnlyMVP in tradingagents/analysis_only/pipeline.py)
        │  per-ticker: fetch factors -> scoring -> composite -> direction/confidence
        ▼
daily_signals.py ──>  reports/daily_signals/*.md
        │  composites refresh WEEKLY (analysis_mvp); this is the DAILY layer that
        │  diffs the user's positions ledger (portfolio/positions.json) against the
        │  target portfolio from latest composites + current prices -> actions/limits/stops
        ▼
trade_tickets.py / trade_workflow.py ──> reports/trade_tickets, reports/trade_workflow
        │  broker-gated operator packet. NEVER calls Robinhood / NEVER places orders;
        │  the JSON is the source of truth for external Codex/Robinhood-MCP execution.

backtest.py ──> scores past reports vs forward returns (yfinance is point-in-time, so
                back-dated scoring is leak-safe). IC by factor/ticker, hit-rate buckets.

analysis_ui.py ──> local ThreadingHTTPServer UI tying reports, signals, sizing,
                   execution, ML-gate shadow, and HTML report rendering together.
```

### Key modules

- `tradingagents/analysis_only/pipeline.py` — `AnalysisReport` dataclass + the orchestrator that fetches per-factor evidence from providers and assembles a report.
- `tradingagents/analysis_only/scoring.py` — **pure, I/O-free, deterministic** scoring: factor weights, composite arithmetic, direction thresholds, confidence mapping, regime gating, per-horizon (20d) composites. Unit-tested directly. This is the heart of the signal; changes here move the strategy.
- `tradingagents/analysis_only/providers.py` — Polygon (default), FMP, SEC EDGAR, VIX/Fear-Greed data providers.
- `tradingagents/analysis_only/{forecast,options_iv,bsm,ml_models,ml_shadow,walk_forward,tuning,calibration}` — price-range/scenario forecasting, options IV surface, Black-Scholes, the ML gate (+ shadow report), walk-forward eval, weight tuning, confidence calibration.
- `portfolio/` — `sizing.py`, `execution.py`/`execution_simulator.py`, `risk.py`, `options.py`, `signals.py`, `paper_trading.py`, `simulator.py`, `snapshot.py`. The deterministic position/execution layer downstream of reports.
- `configs/` — weights (`proposed_weights_v1.json`), calibration (`confidence_calibration*.json`), sizing/execution/tuning/universe YAML. The pipeline is config-driven; prefer editing config over hardcoding.

## LLM provider layer

`tradingagents/llm_clients/` is a unified multi-provider client (OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen/DashScope, GLM/Zhipu, MiniMax, OpenRouter, Ollama, Azure) via `create_llm_client` + `factory.py`; `model_catalog.py`/`validators.py` hold the valid model IDs (`VALID_MODELS`), `capabilities.py` per-model feature flags.

Config for the agent framework lives in `tradingagents/default_config.py` (`DEFAULT_CONFIG`). Keys can be overridden by `TRADINGAGENTS_*` env vars (mapping table `_ENV_OVERRIDES` in that file — add a row there to expose a new key, no entry-point edits needed). Data-source selection is via the `data_vendors` / `tool_vendors` config (`yfinance` | `alpha_vantage` | `fmp`). API keys are read from provider-specific env vars (see README and `.env.example`); SEC EDGAR needs only `SEC_USER_AGENT`/`SEC_CONTACT_EMAIL`.

## Conventions

- New scoring/factor logic must stay **pure and deterministic** in `scoring.py` so it can be unit-tested without market data; the pipeline orchestrator owns all I/O and passes evidence in.
- Forward-return / backtest code must respect **point-in-time** correctness — only consider data with `as_of_date <=` the run date (`daily_signals` `--as-of` back-dating relies on this).
- Trade execution stays **broker-gated**: repo code generates tickets/packets but never places live orders.
- Tests live under `tests/` mirroring the source (`tests/analysis_only/`, `tests/portfolio/`, `tests/scripts/`); pytest markers `unit`/`integration`/`smoke` are defined, `--strict-markers` is on.
