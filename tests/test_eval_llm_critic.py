from __future__ import annotations

from scripts import eval_llm_critic
from tradingagents.analysis_only.backtest import BacktestRecord


def _record(direction: str, ret_20d: float) -> BacktestRecord:
    return BacktestRecord(
        symbol="NVDA",
        as_of_date="2026-05-31",
        direction=direction,
        confidence=0.7,
        composite_score=0.2,
        forward_returns={"ret_20d": ret_20d},
    )


def test_direction_signed_return_flips_bearish_records():
    assert eval_llm_critic._direction_signed_return(
        _record("bullish", 0.05),
        "ret_20d",
    ) == 0.05
    assert eval_llm_critic._direction_signed_return(
        _record("bearish", -0.05),
        "ret_20d",
    ) == 0.05
    assert eval_llm_critic._direction_signed_return(
        _record("neutral", 0.05),
        "ret_20d",
    ) is None


def test_phase6_gate_uses_expected_sign_and_ignores_veto():
    horizons = {
        "ret_20d": {
            "invalidation_prob_30d": {
                "n": 600,
                "ic_incremental": -0.04,
            },
            "confidence_adjustment": {
                "n": 600,
                "ic_incremental": -0.01,
            },
        },
        "ret_60d": {
            "invalidation_prob_30d": {
                "n": 600,
                "ic_incremental": 0.02,
            },
            "confidence_adjustment": {
                "n": 600,
                "ic_incremental": 0.02,
            },
        },
    }
    passed, candidates, best, notes = eval_llm_critic._phase6_gate_evaluation(
        horizons,
        min_n=500,
        threshold=0.03,
    )
    assert passed is True
    assert best["field"] == "invalidation_prob_30d"
    assert best["expected_signed_ic"] == 0.04
    assert any("legacy veto" in note for note in notes)
    assert len(candidates) == 4


def test_phase6_gate_requires_minimum_sample_count():
    horizons = {
        "ret_20d": {
            "invalidation_prob_30d": {
                "n": 499,
                "ic_incremental": -0.20,
            },
            "confidence_adjustment": {
                "n": 499,
                "ic_incremental": 0.20,
            },
        },
    }
    passed, candidates, best, notes = eval_llm_critic._phase6_gate_evaluation(
        horizons,
        min_n=500,
        threshold=0.03,
    )
    assert passed is False
    assert candidates == []
    assert best is None
    assert "minimum OOS sample count" in notes[0]
