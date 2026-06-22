# Data Handover And Corpus Regeneration

This repo can run on a fresh machine with only source code, but some pages and
audits are only useful after restoring or rebuilding generated data. The most
important artifact is the historical analysis corpus under
`reports/analysis_mvp/`.

## Recommended New-Machine Order

1. Install the Python environment.

```bash
uv sync
```

If `uv sync` is not available, create a virtualenv and install the repo:

```bash
uv venv --python 3.13 .venv
uv pip install -e .
```

2. Copy secrets into `.env`.

Required for best corpus quality:

- `POLYGON_API_KEY`
- `SEC_USER_AGENT` or `SEC_CONTACT_EMAIL`

Required only for LLM report sections:

- `OPENAI_API_KEY` or whichever LLM provider key you use

3. Prefer restoring generated data from the old machine or backup.

Fastest path:

```bash
rsync -av old-machine:/path/to/tradingagent/reports/analysis_mvp/ reports/analysis_mvp/
rsync -av old-machine:/path/to/tradingagent/state/ state/
rsync -av old-machine:/path/to/tradingagent/backtest/results/ backtest/results/
rsync -av old-machine:/path/to/tradingagent/reports/paper_trading/ reports/paper_trading/
```

If those folders are restored, you usually do not need to regenerate the full
corpus. Run the validation steps below instead.

## What Each Artifact Is For

- `reports/analysis_mvp/*.json`: historical per-ticker analysis reports. Needed
  for backtests, ML shadow training, model acceptance, and Best Buy ranking.
- `state/analysis_state.sqlite`: local state store, including IV history and
  recent symbol state. Useful but can be rebuilt.
- `backtest/results/`: derived backtest/model audit outputs. Regenerable from
  `reports/analysis_mvp`.
- `reports/paper_trading/recommendations/*.jsonl`: daily recommendation logs.
  The ML Gate page reads these logs.
- `reports/daily_signals/`, `reports/trade_tickets/`, `reports/trade_workflow/`:
  daily/operator outputs. Regenerate as needed for the current trading day.

## Full Historical Corpus Regeneration

Use this when `reports/analysis_mvp/` is missing or incomplete.

The generator is resumable: existing `TICKER_YYYY-MM-DD.json` files are skipped
unless `--force` is passed.

First, inspect the job count:

```bash
.venv/bin/python scripts/generate_corpus.py --dry-run
```

Default range is weekly Fridays from `2023-07-14` through `2026-05-22`, using
`configs/universe.yaml` core + canary tickers.

Recommended full run:

```bash
mkdir -p reports state
MPLCONFIGDIR=state/.mplcache .venv/bin/python scripts/generate_corpus.py \
  --data-provider polygon \
  --workers 4 \
  --executor process \
  --output-dir reports/analysis_mvp \
  --state-store-path state/analysis_state.sqlite \
  --errors-log reports/corpus_errors.jsonl
```

If the provider throttles or errors, retry with fewer workers and pacing:

```bash
MPLCONFIGDIR=state/.mplcache .venv/bin/python scripts/generate_corpus.py \
  --data-provider polygon \
  --workers 2 \
  --executor process \
  --pace-seconds 1.5 \
  --output-dir reports/analysis_mvp \
  --state-store-path state/analysis_state.sqlite \
  --errors-log reports/corpus_errors.jsonl
```

For ML Gate and backtests, LLM sections are not required. The corpus generator
intentionally disables narrative and LLM insight generation, which keeps the
backfill much faster and cheaper.

Use `--force` only after a scoring/schema change when you intentionally want to
rewrite existing historical reports:

```bash
MPLCONFIGDIR=state/.mplcache .venv/bin/python scripts/generate_corpus.py --force
```

## Faster Partial Regeneration

Regenerate only the current core watchlist:

```bash
.venv/bin/python scripts/generate_corpus.py \
  --tickers NVDA AMD AVGO MU TSM ALAB COHR FIG GLW LEU NET RKLB AAPL MSFT GOOGL META ORCL ARM ASML KLAC ANET CRWD MRVL LITE AAOI SPCX INTC SNDK \
  --workers 4 \
  --data-provider polygon
```

Regenerate one ticker:

```bash
.venv/bin/python scripts/generate_corpus.py --tickers NVDA --workers 2
```

Regenerate a shorter date window:

```bash
.venv/bin/python scripts/generate_corpus.py \
  --start 2025-11-21 \
  --end 2026-05-22 \
  --workers 4
```

## Validate The Corpus

Count reports and dates:

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

rows = []
for path in Path("reports/analysis_mvp").glob("*.json"):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue
    rows.append((payload.get("as_of_date"), payload.get("symbol")))

print("reports:", len(rows))
print("dates:", len(set(d for d, _ in rows)))
print("symbols:", len(set(s for _, s in rows)))
print("latest dates:", Counter(d for d, _ in rows).most_common(10))
PY
```

Run fast tests:

```bash
make test-analysis
make test-portfolio
```

Run model/backtest outputs:

```bash
make model-backtest-override
make model-backtest-train
make model-backtest-test
make portfolio-sim
make model-acceptance
```

## Regenerate Paper Trading Logs With ML Shadow Scores

ML Gate reads paper-trading recommendation logs, not raw analysis reports.
After the corpus exists, generate a recommendation log for a date:

```bash
.venv/bin/python daily_signals.py \
  --as-of 2026-06-19 \
  --reports-glob "reports/analysis_mvp/*.json" \
  --positions portfolio/positions.json \
  --sizing-config configs/sizing.yaml \
  --output-dir reports/daily_signals \
  --paper-log-dir reports/paper_trading \
  --ml-shadow-config configs/ml_models.yaml
```

Do not pass `--no-prices`; ML shadow scoring is skipped when prices are
disabled. If ML Gate still shows `missing`, the historical corpus likely does
not have enough older labeled rows before that recommendation date.

## Regenerate Daily Operator Outputs

For a normal morning run:

```bash
make morning
```

For a manual dry run on a specific date:

```bash
scripts/morning_report.sh --force --date 2026-06-19 --no-notify
```

For a faster run that reuses existing reports:

```bash
scripts/morning_report.sh --force --date 2026-06-19 --skip-refresh --no-notify
```

The morning workflow writes:

- `reports/daily_signals/YYYY-MM-DD.{md,json}`
- `reports/trade_tickets/YYYY-MM-DD.{md,json}`
- `reports/trade_workflow/*`
- `reports/morning/YYYY-MM-DD.log`

## Expected Runtime

Runtime depends heavily on API throttling, network quality, universe size, and
date range.

- Restoring from backup: minutes.
- Full deterministic corpus: tens of minutes to several hours.
- Backtests from an existing corpus: usually minutes.
- LLM daily reports: minutes to hours depending on ticker count and whether
  full TradingAgents Review is enabled.

For ML Gate specifically, restore or regenerate `reports/analysis_mvp/` first,
then rerun `daily_signals.py` with `--ml-shadow-config`.
