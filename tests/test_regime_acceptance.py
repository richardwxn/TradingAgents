from __future__ import annotations

from scripts.check_regime_acceptance import evaluate_regime_acceptance


def _summary(
    *,
    hit_5d: float,
    hit_20d: float,
    hit_60d: float,
    median_60d: float,
    count_60d: int = 500,
    n_steps: int = 20,
) -> dict:
    def _stats(hit: float, median: float = 0.05, count: int = 500) -> dict:
        return {
            "count": count,
            "count_with_return": count,
            "mean_forward_return": median,
            "median_forward_return": median,
            "p25_forward_return": -0.01,
            "p75_forward_return": 0.10,
            "hit_rate": hit,
        }

    return {
        "total_records": 1000,
        "by_horizon": {
            "ret_5d": {"by_direction": {"bullish": _stats(hit_5d)}},
            "ret_20d": {"by_direction": {"bullish": _stats(hit_20d)}},
            "ret_60d": {
                "by_direction": {
                    "bullish": _stats(hit_60d, median_60d, count_60d)
                }
            },
        },
        "regime_walk_forward_config": {"n_refit_steps": n_steps},
    }


def test_regime_acceptance_passes_on_exact_boundaries():
    global_summary = _summary(
        hit_5d=0.60,
        hit_20d=0.65,
        hit_60d=0.70,
        median_60d=0.10,
    )
    regime_summary = _summary(
        hit_5d=0.59,  # exact allowed -1pp secondary regression
        hit_20d=0.64,
        hit_60d=0.71,  # exact required +1pp primary lift
        median_60d=0.09,  # exact allowed -1pp median regression
        count_60d=500,
        n_steps=20,
    )
    passed, _messages, failures = evaluate_regime_acceptance(
        global_summary=global_summary,
        regime_summary=regime_summary,
    )
    assert passed is True
    assert failures == []


def test_regime_acceptance_fails_when_primary_hit_rate_regresses():
    global_summary = _summary(
        hit_5d=0.577,
        hit_20d=0.669,
        hit_60d=0.756,
        median_60d=0.139,
    )
    regime_summary = _summary(
        hit_5d=0.620,
        hit_20d=0.697,
        hit_60d=0.701,
        median_60d=0.108,
        count_60d=876,
        n_steps=32,
    )
    passed, _messages, failures = evaluate_regime_acceptance(
        global_summary=global_summary,
        regime_summary=regime_summary,
    )
    assert passed is False
    assert any("ret_60d bullish hit-rate lift" in f for f in failures)
    assert any("ret_60d bullish median return delta" in f for f in failures)


def test_regime_acceptance_fails_on_insufficient_steps_and_count():
    global_summary = _summary(
        hit_5d=0.60,
        hit_20d=0.65,
        hit_60d=0.70,
        median_60d=0.10,
    )
    regime_summary = _summary(
        hit_5d=0.62,
        hit_20d=0.66,
        hit_60d=0.72,
        median_60d=0.10,
        count_60d=499,
        n_steps=19,
    )
    passed, _messages, failures = evaluate_regime_acceptance(
        global_summary=global_summary,
        regime_summary=regime_summary,
    )
    assert passed is False
    assert any("refit steps" in f for f in failures)
    assert any("bullish records with returns" in f for f in failures)
