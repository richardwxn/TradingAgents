from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_model_acceptance.py"
SPEC = importlib.util.spec_from_file_location("check_model_acceptance", SCRIPT)
assert SPEC and SPEC.loader
check_model_acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_model_acceptance)


def _summary(hit_rate=0.7, count=40):
    return {
        "by_horizon": {
            "ret_20d": {
                "by_direction": {
                    "bullish": {
                        "hit_rate": hit_rate,
                        "count_with_return": count,
                    }
                }
            }
        }
    }


def _sim(cagr=0.2, sharpe=1.5, max_drawdown=-0.05):
    return {
        "metrics": {
            "cagr": cagr,
            "sharpe_annualized": sharpe,
            "max_drawdown": max_drawdown,
        }
    }


def test_acceptance_passes_with_minimum_gates():
    passed, messages, failures = check_model_acceptance.evaluate_acceptance(
        summary=_summary(hit_rate=0.7, count=40),
        policy=_sim(cagr=0.25, sharpe=1.8, max_drawdown=-0.05),
        benchmark=_sim(cagr=0.1, sharpe=0.5, max_drawdown=-0.08),
    )
    assert passed is True
    assert failures == []
    assert any("hit-rate" in m for m in messages)


def test_acceptance_fails_low_hit_rate():
    passed, _, failures = check_model_acceptance.evaluate_acceptance(
        summary=_summary(hit_rate=0.55, count=40),
        policy=_sim(cagr=0.25, sharpe=1.8, max_drawdown=-0.05),
        benchmark=_sim(cagr=0.1, sharpe=0.5, max_drawdown=-0.08),
    )
    assert passed is False
    assert any("hit-rate" in f for f in failures)


def test_acceptance_fails_policy_regression_against_baseline():
    passed, _, failures = check_model_acceptance.evaluate_acceptance(
        summary=_summary(hit_rate=0.7, count=40),
        policy=_sim(cagr=0.20, sharpe=1.0, max_drawdown=-0.05),
        benchmark=_sim(cagr=0.1, sharpe=0.5, max_drawdown=-0.08),
        baseline_summary=_summary(hit_rate=0.72, count=40),
        baseline_policy=_sim(cagr=0.30, sharpe=1.5, max_drawdown=-0.05),
    )
    assert passed is False
    assert any("policy CAGR" in f for f in failures)
    assert any("policy Sharpe" in f for f in failures)
    assert any("hit-rate" in f for f in failures)
