.PHONY: test test-analysis test-portfolio test-all model-acceptance model-backtest model-backtest-override model-backtest-train model-backtest-test portfolio-sim

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m pytest tests

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
