#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"

echo "== Fast correctness suite =="
"${PYTHON_BIN}" -m pytest tests

echo
echo "== Optional model-quality checks =="
echo "Run these when scoring, weights, thresholds, report generation, or sizing changes:"
echo
echo "  make model-backtest-override"
echo "  make model-backtest-train"
echo "  make model-backtest-test"
echo "  make portfolio-sim"
echo "  make tune-model"
echo "  make model-acceptance"
echo
echo "The model-quality checks use historical reports and yfinance-backed prices, so they stay manual."
