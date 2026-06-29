"""Characterization ("golden values") tests for the deterministic scoring
weight tables.

These tests intentionally hard-code the *exact* current values of
`DEFAULT_FACTOR_WEIGHTS` and `UNIVERSAL_FACTOR_NAMES`. They are a safety net,
not a correctness assertion: factor weights are backtest-justified magic
numbers (handoff Section 12 / 22 / 24 / 27), and the
`UNIVERSAL_FACTOR_NAMES` membership is the cohort-IC sign-agreed set used by
the screener's `--cohort-aware` non-tech scoring path.

This pins the stale `TODO(unit5)` near `UNIVERSAL_FACTOR_NAMES` in
`tradingagents/analysis_only/scoring.py`: any *accidental* change to a weight
value or to the universal-factor set will fail here. A *deliberate* change
(backed by cohort IC re-verification against
`backtest/results/phase2_v1_4_cohort/cohort_20d.md`) must update the golden
snapshots below in the same commit, which makes the change explicit and
reviewable.

Deterministic and offline: imports the pure scoring module only, no network,
no market data, no API keys.
"""
from __future__ import annotations

from tradingagents.analysis_only.scoring import (
    DEFAULT_FACTOR_WEIGHTS,
    UNIVERSAL_FACTOR_NAMES,
)


# Golden snapshot of every default factor weight. If you change a value here
# you MUST have backtest evidence (these numbers move the live strategy).
_GOLDEN_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "trend_price_vs_sma20": 0.08,
    "trend_sma20_vs_sma50": 0.08,
    "trend_sma50_vs_sma200": 0.08,
    "momentum_rsi": 0.06,
    "momentum_macd_hist": 0.05,
    "momentum_return_20d": 0.05,
    "breakout_60d": 0.05,
    "fund_revenue_growth": 0.08,
    "fund_earnings_growth": 0.08,
    "fund_profit_margins": 0.08,
    "fund_fcf_growth": 0.08,
    "valuation_forward_vs_trailing_pe": 0.04,
    "valuation_sales_multiple_vs_growth": 0.03,
    "industry_relative_strength": 0.08,
    "peer_relative_momentum": 0.05,
    "peer_relative_valuation": 0.06,
    "market_spy_trend": 0.04,
    "market_vix_regime": 0.03,
    "market_fear_greed_regime": 0.05,
    "intraday_momentum_rsi": 0.04,
    "intraday_breakout_signal": 0.00,
    "filings_recency_signal": 0.00,
    "options_net_flow": 0.05,
    "options_iv_term_structure": 0.04,
    "options_iv_skew": 0.00,
    "options_iv_rank": 0.02,
    "news_sentiment": 0.00,
    "ticker_fear_greed_regime": 0.02,
}


# Golden snapshot of the cohort-IC-validated universal factor set (the (1)
# sign-agreed + (2) mechanical-direction members documented above the
# definition in scoring.py).
_GOLDEN_UNIVERSAL_FACTOR_NAMES: frozenset[str] = frozenset({
    "market_vix_regime",
    "peer_relative_valuation",
    "options_iv_term_structure",
    "momentum_rsi",
    "trend_price_vs_sma20",
    "trend_sma20_vs_sma50",
    "valuation_forward_vs_trailing_pe",
})


def test_default_factor_weights_golden():
    # Exact dict equality also catches added/removed keys, not just value
    # drift. A failure here means a weight changed without updating this
    # snapshot — confirm there is backtest evidence before re-baselining.
    assert DEFAULT_FACTOR_WEIGHTS == _GOLDEN_DEFAULT_FACTOR_WEIGHTS


def test_default_factor_weights_key_set_golden():
    # Separate, sharper message for the add/remove-a-factor case.
    assert set(DEFAULT_FACTOR_WEIGHTS.keys()) == set(
        _GOLDEN_DEFAULT_FACTOR_WEIGHTS.keys()
    ), (
        "DEFAULT_FACTOR_WEIGHTS key set changed: "
        f"added={set(DEFAULT_FACTOR_WEIGHTS) - set(_GOLDEN_DEFAULT_FACTOR_WEIGHTS)} "
        f"removed={set(_GOLDEN_DEFAULT_FACTOR_WEIGHTS) - set(DEFAULT_FACTOR_WEIGHTS)}"
    )


def test_universal_factor_names_golden():
    # Pins the cohort-aware (non-tech) scoring factor set. Re-verify against
    # backtest/results/phase2_v1_4_cohort/cohort_20d.md before changing.
    assert UNIVERSAL_FACTOR_NAMES == _GOLDEN_UNIVERSAL_FACTOR_NAMES, (
        "UNIVERSAL_FACTOR_NAMES changed: "
        f"added={UNIVERSAL_FACTOR_NAMES - _GOLDEN_UNIVERSAL_FACTOR_NAMES} "
        f"removed={_GOLDEN_UNIVERSAL_FACTOR_NAMES - UNIVERSAL_FACTOR_NAMES}"
    )
