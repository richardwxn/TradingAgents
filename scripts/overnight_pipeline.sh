#!/usr/bin/env bash
# Overnight pipeline (2026-06-04 → 06-05).
# Sequenced long-running stages. Each stage logs to reports/overnight/<NN>_*.log
# and writes a one-line status to reports/overnight/STATUS.txt. On per-stage
# failure the script continues to the next stage. Final SUMMARY.md is written
# at the end.
#
# Stages:
#   0. Wait for any in-flight generate_corpus.py (X1-data extended regen).
#   1. Archive prod corpus + wipe iv_history + production-corpus regen
#      with v1.7 code baked in (news_sentiment, ticker_fear_greed_regime,
#      regime sign flips, bear gate, calibrated confidence).
#   2. Backtest fresh corpus, IC by factor + by ticker, benchmark SPY.
#   3. Generate weight candidates + walk-forward sweep.
#   4. Stop-loss multiplier sweep on the portfolio simulator.
#   5. End-to-end strategy backtest at current v1.7 weights + 1.5x ATR stops.
#   6. Cohort split on combined corpus when extended data is available.
#   7. Write SUMMARY.md.
#
# Run unattended via:  bash scripts/overnight_pipeline.sh > reports/overnight/orchestrator.log 2>&1

set -u
cd /Users/xwen/TradingAgents

PY=.venv/bin/python
OUT=reports/overnight
mkdir -p "$OUT"
STATUS_FILE="$OUT/STATUS.txt"
: > "$STATUS_FILE"

log_stage() { echo "[$(date '+%F %T')] $1" | tee -a "$STATUS_FILE"; }

# -----------------------------------------------------------------------------
# Stage 0 — wait for in-flight generate_corpus.py (X1-data extended regen).
# -----------------------------------------------------------------------------
log_stage "Stage 0: waiting for any in-flight generate_corpus..."
while pgrep -f "generate_corpus.py" > /dev/null 2>&1; do
    sleep 60
done
log_stage "Stage 0 done. No generate_corpus running."

# -----------------------------------------------------------------------------
# Stage 1 — production-corpus regen under v1.7 code.
# -----------------------------------------------------------------------------
log_stage "Stage 1 start: archive prod corpus + wipe iv_history + regen."
if [ ! -d reports/analysis_mvp_pre_v1_7_final ]; then
    cp -r reports/analysis_mvp reports/analysis_mvp_pre_v1_7_final
    log_stage "  archived → reports/analysis_mvp_pre_v1_7_final/"
else
    log_stage "  archive exists, skipping cp"
fi
sqlite3 state/analysis_state.sqlite "DELETE FROM iv_history;" 2>/dev/null
log_stage "  iv_history table wiped"

"$PY" -u scripts/generate_corpus.py \
    --force --skip-news \
    --workers 4 --executor process \
    --errors-log reports/corpus_errors_v1_7_final.jsonl \
    > "$OUT/01_regen_prod.log" 2>&1
S1_RC=$?
log_stage "Stage 1 done (exit $S1_RC). See $OUT/01_regen_prod.log"

# -----------------------------------------------------------------------------
# Stage 2 — backtest the fresh corpus.
# -----------------------------------------------------------------------------
log_stage "Stage 2 start: backtest + factor IC."
"$PY" -u backtest.py \
    --reports-glob "reports/analysis_mvp/*.json" \
    --by-factor --by-ticker --benchmark SPY \
    --output-dir backtest/results/phase2_v1_7_final/ \
    > "$OUT/02_backtest.log" 2>&1
S2_RC=$?
log_stage "Stage 2 done (exit $S2_RC). See $OUT/02_backtest.log"

# -----------------------------------------------------------------------------
# Stage 3 — walk-forward weight candidate sweep.
# -----------------------------------------------------------------------------
log_stage "Stage 3 start: generate candidates + walk-forward sweep."
mkdir -p /tmp/overnight_candidates /tmp/overnight_wf

"$PY" -c "
import json
from tradingagents.analysis_only.scoring import DEFAULT_FACTOR_WEIGHTS
base = dict(DEFAULT_FACTOR_WEIGHTS)
cs = {
    'baseline_v1_7': base,
    'news_002':       {**base, 'news_sentiment': 0.02},
    'news_004':       {**base, 'news_sentiment': 0.04},
    'news_006':       {**base, 'news_sentiment': 0.06},
    'tfg_002':        {**base, 'ticker_fear_greed_regime': 0.02},
    'tfg_004':        {**base, 'ticker_fear_greed_regime': 0.04},
    'tfg_006':        {**base, 'ticker_fear_greed_regime': 0.06},
    'news_tfg_combo': {**base, 'news_sentiment': 0.03, 'ticker_fear_greed_regime': 0.03},
    'bump_iv_term':   {**base, 'options_iv_term_structure': 0.06},
    'bump_industry':  {**base, 'industry_relative_strength': 0.10},
    'bump_vsg':       {**base, 'valuation_sales_multiple_vs_growth': 0.06},
    'cut_iv_rank':    {**base, 'options_iv_rank': 0.02},
    'all_new_004':    {**base, 'news_sentiment': 0.04, 'ticker_fear_greed_regime': 0.04, 'options_iv_term_structure': 0.06},
}
import os
os.makedirs('/tmp/overnight_candidates', exist_ok=True)
for n, w in cs.items():
    json.dump(w, open(f'/tmp/overnight_candidates/{n}.json', 'w'), indent=2)
print(f'wrote {len(cs)} candidate weight files')
" >> "$OUT/03_walk_forward_sweep.log" 2>&1

for c in /tmp/overnight_candidates/*.json; do
    name=$(basename "$c" .json)
    echo "=== candidate: $name ===" >> "$OUT/03_walk_forward_sweep.log"
    "$PY" -u scripts/walk_forward_eval.py \
        --reports-glob "reports/analysis_mvp/*.json" \
        --weight-source custom_json --weights-json "$c" \
        --output-dir "/tmp/overnight_wf/$name/" \
        >> "$OUT/03_walk_forward_sweep.log" 2>&1 || true
done
S3_RC=$?
log_stage "Stage 3 done. See $OUT/03_walk_forward_sweep.log"

# -----------------------------------------------------------------------------
# Stage 4 — stop-loss multiplier sweep.
# -----------------------------------------------------------------------------
log_stage "Stage 4 start: stop-loss multiplier sweep."
mkdir -p /tmp/overnight_stops
for mult in 1.0 1.25 1.5 1.75 2.0 2.5; do
    echo "=== stop_loss_atr_multiple=$mult ===" >> "$OUT/04_stop_loss_sweep.log"
    "$PY" -u portfolio_simulate.py \
        --stop-loss-atr-multiple "$mult" \
        --output-dir "/tmp/overnight_stops/mult_${mult}/" \
        >> "$OUT/04_stop_loss_sweep.log" 2>&1 || true
done
S4_RC=$?
log_stage "Stage 4 done. See $OUT/04_stop_loss_sweep.log"

# -----------------------------------------------------------------------------
# Stage 5 — end-to-end strategy backtest.
# -----------------------------------------------------------------------------
log_stage "Stage 5 start: end-to-end strategy backtest."
"$PY" -u portfolio_simulate.py \
    --output-dir backtest/results/phase2_v1_7_final/simulator/ \
    > "$OUT/05_strategy_backtest.log" 2>&1
S5_RC=$?
log_stage "Stage 5 done (exit $S5_RC). See $OUT/05_strategy_backtest.log"

# -----------------------------------------------------------------------------
# Stage 6 — cohort split on combined corpus (if extended is ready).
# -----------------------------------------------------------------------------
log_stage "Stage 6 start: cohort split."
EXT_COUNT=$(ls reports/analysis_mvp_extended/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$EXT_COUNT" -gt 100 ]; then
    "$PY" -u scripts/cohort_ic_split.py \
        --by-ticker-json backtest/results/phase2_v1_7_final/factor_summary_by_ticker.json \
        --universe configs/universe.yaml \
        --horizon ret_60d \
        > backtest/results/phase2_v1_7_final/cohort_60d.md 2>&1
    "$PY" -u scripts/cohort_ic_split.py \
        --by-ticker-json backtest/results/phase2_v1_7_final/factor_summary_by_ticker.json \
        --universe configs/universe.yaml \
        --horizon ret_20d \
        > backtest/results/phase2_v1_7_final/cohort_20d.md 2>&1
    log_stage "Stage 6 done. Extended corpus has $EXT_COUNT files."
else
    log_stage "Stage 6 SKIPPED — extended corpus has only $EXT_COUNT files."
fi

# -----------------------------------------------------------------------------
# Stage 7 — write SUMMARY.md.
# -----------------------------------------------------------------------------
log_stage "Stage 7 start: assemble SUMMARY.md."
SUMMARY="$OUT/SUMMARY.md"
{
    echo "# Overnight pipeline summary"
    echo
    echo "Generated: $(date)"
    echo
    echo "## Stage statuses"
    echo
    cat "$STATUS_FILE"
    echo
    echo "## Stage 1: production regen"
    echo
    tail -5 "$OUT/01_regen_prod.log" 2>/dev/null | grep -v "HTTP\|Invalid\|Failed to retrieve" | head
    echo
    echo "## Stage 2: backtest summary headline"
    echo
    head -10 backtest/results/phase2_v1_7_final/summary.md 2>/dev/null | sed -n '1,40p'
    echo
    echo "## Stage 3: walk-forward sweep — bullish 60d test_hit per candidate"
    echo
    echo '| candidate | 60d_bull | 60d_bear |'
    echo '|---|---:|---:|'
    for c in /tmp/overnight_candidates/*.json; do
        name=$(basename "$c" .json)
        sjson="/tmp/overnight_wf/$name/summary_custom_json.json"
        if [ -f "$sjson" ]; then
            "$PY" -c "
import json,sys
d=json.load(open('$sjson'))['summary']['per_horizon']['ret_60d']
print(f'| $name | {d[\"median_bullish_test_hit\"]*100:.2f}% | {d[\"median_bearish_test_hit\"]*100 if d.get(\"median_bearish_test_hit\") is not None else 0:.2f}% |')
"
        fi
    done
    echo
    echo "## Stage 4: stop-loss multiplier sweep"
    echo
    echo '| ATR multiplier | wall outputs |'
    echo '|---|---|'
    for mult in 1.0 1.25 1.5 1.75 2.0 2.5; do
        if [ -d "/tmp/overnight_stops/mult_${mult}" ]; then
            mf=$(find "/tmp/overnight_stops/mult_${mult}" -name '*.json' 2>/dev/null | head -1)
            echo "| $mult | $mf |"
        fi
    done
    echo
    echo "## Stage 5: end-to-end strategy backtest"
    echo
    tail -20 "$OUT/05_strategy_backtest.log" 2>/dev/null
    echo
    echo "## Stage 6: cohort split"
    echo
    if [ -f backtest/results/phase2_v1_7_final/cohort_60d.md ]; then
        head -30 backtest/results/phase2_v1_7_final/cohort_60d.md
    else
        echo "(skipped — see status file)"
    fi
    echo
    echo "---"
    echo "Outputs:"
    echo "- Production corpus: \`reports/analysis_mvp/\` (regenerated)"
    echo "- Pre-regen archive: \`reports/analysis_mvp_pre_v1_7_final/\`"
    echo "- Backtest: \`backtest/results/phase2_v1_7_final/\`"
    echo "- Walk-forward sweep: \`/tmp/overnight_wf/\`"
    echo "- Stop-loss sweep: \`/tmp/overnight_stops/\`"
    echo "- Stage logs: \`$OUT/\`"
} > "$SUMMARY"

log_stage "Pipeline complete. See $SUMMARY"
