.PHONY: test test-analysis test-portfolio test-all model-acceptance model-backtest model-backtest-override model-backtest-train model-backtest-test portfolio-sim morning morning-dry model-recal model-recal-promote

PYTHON ?= .venv/bin/python

# --- Confidence-calibration auto-recal (plan B-8) -------------------------
# Re-fit the Phase-5 isotonic confidence calibration from the report corpus.
# `model-recal` writes to a REVIEW path (CAL_REGEN) and NEVER clobbers the
# live file, so the operator can diff the new curve before promoting it.
# `model-recal-promote` archives the live file then copies the reviewed
# candidate into place — and only does so when both Phase-5 gates PASS.
#
# Horizon note: the live primary file was fit on the default 20d horizon.
# The plan prose (accuracy_improvements.md) names 60d as the primary target;
# override with `make model-recal CAL_HORIZON=ret_60d` if that is intended.
CAL_REPORTS ?= reports/analysis_mvp/*.json
CAL_LIVE    ?= configs/confidence_calibration.json
CAL_REGEN   ?= configs/confidence_calibration.regen.json
CAL_HORIZON ?= ret_20d
CAL_LOG     ?= /tmp/cal_fit.log

test:
	$(PYTHON) -m pytest tests

# Run the full morning report pipeline (trading-day gated). Mirrors what the
# launchd agent (deploy/launchd/com.tradingagent.morning.plist) runs each
# weekday morning.
morning:
	PYTHON=$(PYTHON) scripts/morning_report.sh

# Manual dry run: ignore the trading-day gate and skip notifications. Add
# ARGS="--no-llm --skip-refresh" etc. to go faster.
morning-dry:
	PYTHON=$(PYTHON) scripts/morning_report.sh --force --no-notify $(ARGS)

test-analysis:
	$(PYTHON) -m pytest tests/analysis_only

test-portfolio:
	$(PYTHON) -m pytest tests/portfolio

test-all:
	scripts/test_all.sh

model-acceptance:
	$(PYTHON) scripts/check_model_acceptance.py

model-backtest:
	$(PYTHON) backtest.py \
		--reports-glob "reports/analysis_mvp/*.json" \
		--by-factor \
		--by-ticker \
		--benchmark SPY \
		--output-dir backtest/results

model-backtest-override:
	$(PYTHON) backtest.py \
		--reports-glob "reports/analysis_mvp/*.json" \
		--weights-override configs/proposed_weights_v1.json \
		--by-factor \
		--by-ticker \
		--benchmark SPY \
		--output-dir backtest/results

model-backtest-train:
	$(PYTHON) backtest.py \
		--reports-glob "reports/analysis_mvp/*.json" \
		--weights-override configs/proposed_weights_v1.json \
		--date-from 2025-11-21 \
		--date-to 2026-02-27 \
		--output-dir backtest/results/train

model-backtest-test:
	$(PYTHON) backtest.py \
		--reports-glob "reports/analysis_mvp/*.json" \
		--weights-override configs/proposed_weights_v1.json \
		--date-from 2026-02-28 \
		--date-to 2026-05-22 \
		--output-dir backtest/results/test

portfolio-sim:
	$(PYTHON) portfolio_simulate.py \
		--reports-glob "reports/analysis_mvp/*.json" \
		--policies equal_weight_bullish top_n_bullish confidence_weighted \
		--include-benchmark \
		--output-dir backtest/results/simulator_full

tune-model:
	$(PYTHON) tune_model.py \
		--config configs/tuning.yaml \
		--output-dir backtest/results/tuning

# Re-fit calibration to the REVIEW path (never the live file) and surface the
# Phase-5 gate verdicts. Diff CAL_REGEN against CAL_LIVE, then run
# `make model-recal-promote` once you are satisfied.
model-recal:
	$(PYTHON) scripts/fit_confidence_calibration.py \
		--reports-glob "$(CAL_REPORTS)" \
		--horizon $(CAL_HORIZON) \
		--output $(CAL_REGEN) 2>&1 | tee $(CAL_LOG)
	@echo ""
	@echo "Wrote review candidate -> $(CAL_REGEN) (live file $(CAL_LIVE) untouched)."
	@echo "Diff before promoting:  git --no-pager diff --no-index $(CAL_LIVE) $(CAL_REGEN)"
	@echo "Promote when satisfied: make model-recal-promote"

# Gated promotion: archive the live file to a dated copy, then copy the
# reviewed candidate over the live path — only if BOTH Phase-5 gates PASS
# in the captured log (the script's exit code does NOT gate quality).
model-recal-promote:
	@test -f $(CAL_REGEN) || { echo "No candidate at $(CAL_REGEN); run 'make model-recal' first." >&2; exit 1; }
	@test -f $(CAL_LOG) || { echo "No gate log at $(CAL_LOG); run 'make model-recal' first." >&2; exit 1; }
	@grep -q 'reliability +/-5pp:           PASS' $(CAL_LOG) || { echo "GATE FAIL: reliability not within +/-5pp — not promoting." >&2; exit 2; }
	@grep -q 'Brier improves vs heuristic: PASS' $(CAL_LOG) || { echo "GATE FAIL: Brier does not beat heuristic — not promoting." >&2; exit 2; }
	cp $(CAL_LIVE) configs/confidence_calibration_v1_$$(date +%Y%m%d).json
	cp $(CAL_REGEN) $(CAL_LIVE)
	@echo "Promoted $(CAL_REGEN) -> $(CAL_LIVE) (prior archived to configs/confidence_calibration_v1_$$(date +%Y%m%d).json)."
