from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portfolio.simulator import SimulationConfig, run_simulation
from portfolio.sizing import SizingConfig
from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.tuning import (
    CandidateConfig,
    DateSlice,
    TuningBounds,
    TuningConfig,
    TuningGates,
    evaluate_candidate,
    generate_random_candidates,
    normalize_weights,
    refine_candidates,
    run_benchmark,
    score_candidate,
)


def _config(**overrides):
    cfg = TuningConfig(
        seed=42,
        max_random_candidates=5,
        refine_top_n=2,
        refine_variants_per_candidate=2,
        report_glob="reports/analysis_mvp/*.json",
        base_weights_path="configs/proposed_weights_v1.json",
        benchmark="SPY",
        horizons=(5, 20, 60),
        search_slice=DateSlice("search", "2026-01-02", "2026-01-30"),
        holdout_slice=DateSlice("holdout", "2026-02-06", "2026-02-27"),
        bounds=TuningBounds(
            bullish_threshold=(0.10, 0.25),
            bearish_threshold=(0.15, 0.40),
            neutral_bands=(0.01, 0.02),
            weight_multiplier=(0.5, 1.5),
            policies=("equal_weight_bullish", "confidence_weighted"),
            max_per_name=(0.08, 0.20),
            max_long_exposure=(0.40, 0.90),
            top_n=(3, 5),
            stale_signal_decay=(0.25, 1.0),
        ),
        gates=TuningGates(
            min_bullish_20d_count=1,
            min_avg_long_exposure=0.01,
            max_drawdown_floor=-0.50,
            require_excess_cagr_positive=False,
            enable_bearish=False,
        ),
        universe=("AAA",),
        initial_capital=10_000.0,
        cost_per_side_bps=0.0,
        preserve_zero_weights=True,
    )
    return cfg.__class__(**{**cfg.__dict__, **overrides})


def _candidate(**overrides):
    c = CandidateConfig(
        candidate_id="c",
        source="test",
        weights={"signal": 1.0, "zero": 0.0},
        bullish_threshold=0.10,
        bearish_threshold=0.25,
        neutral_band=0.02,
        policy="equal_weight_bullish",
        max_per_name=0.20,
        max_long_exposure=0.20,
        top_n=3,
        stale_signal_decay=1.0,
    )
    return c.__class__(**{**c.__dict__, **overrides})


def _record(as_of: str, score: float, ret: float):
    return BacktestRecord(
        symbol="AAA",
        as_of_date=as_of,
        direction="neutral",
        confidence=0.7,
        composite_score=0.0,
        forward_returns={"ret_20d": ret},
        factor_scores=[
            {
                "factor": "signal",
                "pillar": "technical",
                "score": score,
                "data_available": True,
            },
            {
                "factor": "zero",
                "pillar": "technical",
                "score": 1.0,
                "data_available": True,
            },
        ],
    )


def test_normalize_weights_preserves_proportions():
    out = normalize_weights({"a": 2.0, "b": 1.0})
    assert out["a"] == pytest.approx(2 / 3)
    assert out["b"] == pytest.approx(1 / 3)


def test_generate_random_candidates_is_deterministic_and_preserves_zero_weights():
    cfg = _config()
    base = {"a": 0.8, "zero": 0.0}
    first = generate_random_candidates(config=cfg, base_weights=base)
    second = generate_random_candidates(config=cfg, base_weights=base)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    assert all(c.weights["zero"] == 0.0 for c in first)
    assert all(sum(c.weights.values()) == pytest.approx(1.0) for c in first)


def test_refine_candidates_keeps_zero_weights_zero():
    cfg = _config()
    refined = refine_candidates(
        config=cfg,
        top_candidates=[_candidate()],
        start_index=10,
    )
    assert len(refined) == cfg.refine_variants_per_candidate
    assert all(c.weights["zero"] == 0.0 for c in refined)


def test_score_candidate_rejects_low_sample_count():
    sim = run_simulation(
        weeks=[date(2026, 1, 2), date(2026, 1, 9)],
        observations={},
        prices=pd.DataFrame({"AAA": [100.0, 101.0]}, index=[date(2026, 1, 2), date(2026, 1, 9)]),
        sizing_config=SizingConfig(),
        sim_config=SimulationConfig(initial_capital=10_000.0, cost_per_side_bps=0.0),
    )
    bench = sim
    out = score_candidate(
        candidate=_candidate(),
        slice_name="search",
        bullish_stats={"hit_rate": 1.0, "mean_forward_return": 0.1, "count_with_return": 0},
        simulation=sim,
        benchmark=bench,
        gates=TuningGates(min_bullish_20d_count=1),
    )
    assert out.rejected is True
    assert "bullish_20d_count<1" in out.rejection_reasons


def test_evaluate_candidate_ranks_known_better_candidate_above_weaker_one():
    cfg = _config()
    records = [
        _record("2026-01-02", score=0.8, ret=0.10),
        _record("2026-01-09", score=0.7, ret=0.08),
        _record("2026-01-16", score=0.6, ret=0.06),
    ]
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 121.0], "SPY": [100.0, 101.0, 102.0]},
        index=[date(2026, 1, 2), date(2026, 1, 9), date(2026, 1, 16)],
    )
    bench = run_benchmark(
        weeks=[date(2026, 1, 2), date(2026, 1, 9), date(2026, 1, 16)],
        prices=prices,
        config=cfg,
    )
    good = evaluate_candidate(
        candidate=_candidate(candidate_id="good", bullish_threshold=0.10),
        records=records,
        prices=prices,
        benchmark=bench,
        date_slice=cfg.search_slice,
        config=cfg,
    )
    weak = evaluate_candidate(
        candidate=_candidate(candidate_id="weak", bullish_threshold=0.95),
        records=records,
        prices=prices,
        benchmark=bench,
        date_slice=cfg.search_slice,
        config=cfg,
    )
    assert good.score > weak.score
    assert good.rejected is False
    assert weak.rejected is True
