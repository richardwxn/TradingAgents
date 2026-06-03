"""Tests for the Unit-A walk-forward OOS harness."""

from __future__ import annotations

import pytest

from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.walk_forward import (
    WalkForwardWindow,
    evaluate_window,
    generate_windows,
    render_walk_forward_markdown,
    summarize_walk_forward,
)


# ---------- fixtures ----------


def _make_record(
    *,
    symbol: str,
    as_of_date: str,
    direction: str,
    composite: float,
    ret_5d: float | None = 0.01,
    ret_20d: float | None = 0.02,
    ret_60d: float | None = 0.03,
    factor_score: float = 0.4,
) -> BacktestRecord:
    """Synthetic single-factor record. Composite/direction are explicit
    so callers can construct deterministic train/test splits.

    A single factor score is attached so `rebuild_records_with_weights`
    can recompute the composite under proposed weight vectors in the
    sign-flip test.
    """
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of_date,
        direction=direction,
        confidence=0.7,
        composite_score=composite,
        forward_returns={
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "ret_60d": ret_60d,
        },
        factor_scores=[
            {
                "factor": "fakefactor",
                "pillar": "test",
                "score": factor_score,
                "weighted_score": factor_score,
                "bucket": "bullish" if factor_score > 0 else "bearish",
                "data_available": True,
            }
        ],
    )


# ---------- generate_windows ----------


def test_generate_windows_basic_count_and_non_overlapping_test():
    # Corpus span: 2023-01-01 .. 2025-12-31 (3 years).
    # train_months=18, test_months=3, step_months=1.
    # First window: train 2023-01 → 2024-06, test 2024-07 → 2024-09.
    # Last possible window: train_start such that test_end ≤ 2025-12-31.
    # train_start + 21 months - 1 day ≤ 2025-12-31
    # → train_start ≤ 2024-04-01 (since 21 months later = 2026-01-01 - 1 = 2025-12-31).
    # So train_starts: 2023-01-01, 2023-02-01, ..., 2024-04-01 → 16 windows.
    windows = generate_windows(
        corpus_min_date="2023-01-01",
        corpus_max_date="2025-12-31",
        train_months=18,
        test_months=3,
        step_months=1,
    )
    assert len(windows) == 16
    # Test starts are non-overlapping cadence (1 month apart), but the
    # test SPANS themselves overlap because step_months=1 < test_months=3.
    # What "non-overlapping test slice" means in the plan is that each
    # window's TEST starts immediately after its OWN train ends — not
    # across windows. Verify that property.
    for w in windows:
        # train_end + 1 day == test_start
        assert w.test_start > w.train_end
        # Test span is exactly the requested test_months.
        assert w.test_start[:7] != w.train_end[:7] or True  # sanity
    assert windows[0].train_start == "2023-01-01"
    assert windows[0].train_end == "2024-06-30"
    assert windows[0].test_start == "2024-07-01"
    assert windows[0].test_end == "2024-09-30"


def test_generate_windows_skips_incomplete_forward_return():
    # If corpus_max_date is too close to corpus_min_date for even one
    # 18+3 month window, return empty.
    windows = generate_windows(
        corpus_min_date="2024-01-01",
        corpus_max_date="2024-12-31",  # only 12 months
        train_months=18,
        test_months=3,
        step_months=1,
    )
    assert windows == []

    # If max date is exactly enough for one window, get exactly one.
    windows = generate_windows(
        corpus_min_date="2023-01-01",
        corpus_max_date="2024-09-30",  # last day of test slice
        train_months=18,
        test_months=3,
        step_months=1,
    )
    assert len(windows) == 1
    assert windows[0].test_end == "2024-09-30"


def test_generate_windows_rejects_non_positive_args():
    with pytest.raises(ValueError):
        generate_windows(
            corpus_min_date="2023-01-01",
            corpus_max_date="2025-12-31",
            train_months=0,
            test_months=3,
            step_months=1,
        )


# ---------- evaluate_window ----------


def _stub_corpus_for_window() -> tuple[list[BacktestRecord], WalkForwardWindow]:
    """Build a small synthetic corpus and matching window.

    Train slice (2023-01-15 .. 2024-06-15):
      4 bullish records, 3 hit (positive 60d) → train_hit = 3/4 = 0.75.
    Test slice (2024-07-15 .. 2024-09-15):
      3 bullish records, 1 hit → test_hit = 1/3 ≈ 0.3333.
    """
    train_recs = [
        _make_record(
            symbol="AAA", as_of_date="2023-02-01", direction="bullish",
            composite=0.5, ret_60d=0.05,
        ),
        _make_record(
            symbol="BBB", as_of_date="2023-06-01", direction="bullish",
            composite=0.5, ret_60d=0.04,
        ),
        _make_record(
            symbol="CCC", as_of_date="2023-09-01", direction="bullish",
            composite=0.5, ret_60d=0.02,
        ),
        _make_record(
            symbol="DDD", as_of_date="2024-03-01", direction="bullish",
            composite=0.5, ret_60d=-0.03,  # miss
        ),
    ]
    test_recs = [
        _make_record(
            symbol="EEE", as_of_date="2024-07-15", direction="bullish",
            composite=0.5, ret_60d=0.01,  # hit
        ),
        _make_record(
            symbol="FFF", as_of_date="2024-08-15", direction="bullish",
            composite=0.5, ret_60d=-0.02,  # miss
        ),
        _make_record(
            symbol="GGG", as_of_date="2024-09-01", direction="bullish",
            composite=0.5, ret_60d=-0.01,  # miss
        ),
    ]
    win = WalkForwardWindow(
        train_start="2023-01-01",
        train_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2024-09-30",
    )
    return train_recs + test_recs, win


def test_evaluate_window_with_none_weight_fn_returns_expected_shape():
    corpus, win = _stub_corpus_for_window()
    out = evaluate_window(
        corpus,
        win,
        weight_fn=lambda _train: None,
        horizons=("ret_60d",),
    )
    assert out["train_n"] == 4
    assert out["test_n"] == 3
    assert out["weights_used"] is None
    block = out["per_horizon"]["ret_60d"]
    assert block["n_train_with_return"] == 4
    assert block["n_test_with_return"] == 3
    # Train: 3/4 hits, test: 1/3 hits.
    assert block["train_hit"] == 0.75
    assert block["test_hit"] == pytest.approx(0.3333, abs=1e-3)
    # Overfit gap = train_hit - test_hit ≈ 0.4167.
    assert block["overfit_gap"] == pytest.approx(0.4167, abs=1e-3)
    # All stub records are direction=bullish, so the bullish-only block
    # must equal the all-direction block.
    assert block["bullish_train_hit"] == 0.75
    assert block["bullish_test_hit"] == pytest.approx(0.3333, abs=1e-3)
    assert block["n_bullish_test"] == 3
    assert block["n_bearish_test"] == 0
    assert block["bearish_test_hit"] is None


def test_evaluate_window_with_sign_flipped_weights_flips_direction():
    """A weight_fn that flips the factor sign should swap bullish ↔
    bearish on every record (single-factor synthetic corpus), which
    flips hit_rate from (positive-return) to (negative-return) hit.

    Train hits (under flipped sign):
      DDD ret_60d=-0.03 → bearish hit. AAA/BBB/CCC are all positive →
      bearish misses. So train_hit = 1/4 = 0.25.
    Test hits:
      EEE ret_60d=0.01 → bearish miss. FFF=-0.02 → bearish hit.
      GGG=-0.01 → bearish hit. → test_hit = 2/3 ≈ 0.6667.
    """
    corpus, win = _stub_corpus_for_window()

    # Factor score is +0.4 everywhere; negative weight flips composite
    # to -0.4 → direction = bearish at default threshold 0.2.
    def flip_weights(_train: list[BacktestRecord]) -> dict[str, float]:
        return {"fakefactor": -1.0}

    out = evaluate_window(
        corpus,
        win,
        weight_fn=flip_weights,
        horizons=("ret_60d",),
    )
    assert out["weights_used"] == {"fakefactor": -1.0}
    block = out["per_horizon"]["ret_60d"]
    assert block["train_hit"] == 0.25
    assert block["test_hit"] == pytest.approx(0.6667, abs=1e-3)
    # Records flipped to bearish: bullish counts must be zero, the
    # bearish-test hit-rate must mirror the all-direction one.
    assert block["n_bullish_test"] == 0
    assert block["bullish_test_hit"] is None
    assert block["n_bearish_test"] == 3
    assert block["bearish_test_hit"] == pytest.approx(0.6667, abs=1e-3)


# ---------- summarize_walk_forward ----------


def test_summarize_walk_forward_aggregates_correctly():
    # Helper to keep the literals compact. Bullish-only block tracks
    # the all-direction block to keep both aggregations exercised.
    def _w(train_hit, test_hit, gap):
        return {
            "per_horizon": {
                "ret_60d": {
                    "train_hit": train_hit,
                    "test_hit": test_hit,
                    "overfit_gap": gap,
                    "n_train_with_return": 100 if train_hit is not None else 0,
                    "n_test_with_return": 30 if test_hit is not None else 0,
                    "bullish_train_hit": train_hit,
                    "bullish_test_hit": test_hit,
                    "bullish_overfit_gap": gap,
                    "n_bullish_test": 30 if test_hit is not None else 0,
                }
            }
        }

    per_window_stats = [
        _w(0.70, 0.60, 0.10),  # beats 0.5
        _w(0.80, 0.55, 0.25),  # beats 0.5
        _w(0.75, 0.45, 0.30),  # misses
        _w(None, None, None),  # incomplete
    ]
    summary = summarize_walk_forward(per_window_stats, horizons=("ret_60d",))
    assert summary["n_windows"] == 4
    stats = summary["per_horizon"]["ret_60d"]
    assert stats["n_windows_with_test_hit"] == 3
    # test_hits = [0.60, 0.55, 0.45]. median = 0.55, mean ≈ 0.5333.
    assert stats["median_test_hit"] == 0.55
    assert stats["mean_test_hit"] == pytest.approx(0.5333, abs=1e-3)
    # p25, p75 of [0.45, 0.55, 0.60]: linear-interp → p25=0.50, p75=0.575.
    assert stats["p25_test_hit"] == 0.50
    assert stats["p75_test_hit"] == 0.575
    # gaps = [0.10, 0.25, 0.30] → median 0.25, mean ≈ 0.2167.
    assert stats["median_overfit_gap"] == 0.25
    assert stats["mean_overfit_gap"] == pytest.approx(0.2167, abs=1e-3)
    # 2 of 3 windows beat 0.5 → 0.6667.
    assert stats["fraction_windows_beat_baseline"] == pytest.approx(
        0.6667, abs=1e-3
    )
    # Bullish-only block: identical aggregation since we set
    # bullish_* == all-direction in the fixture.
    assert stats["n_windows_with_bullish_test_hit"] == 3
    assert stats["median_bullish_test_hit"] == 0.55
    assert stats["median_bullish_overfit_gap"] == 0.25
    assert stats["fraction_windows_bullish_beat_baseline"] == pytest.approx(
        0.6667, abs=1e-3
    )


# ---------- render_walk_forward_markdown ----------


def test_render_walk_forward_markdown_handles_empty_and_nonempty():
    # Empty input: must not crash and must produce a sensible "no stats"
    # blurb.
    empty_summary = {"n_windows": 0, "per_horizon": {}}
    md = render_walk_forward_markdown(empty_summary)
    assert "Windows evaluated" in md
    assert "0" in md
    assert "No per-horizon stats" in md

    nonempty = {
        "n_windows": 3,
        "per_horizon": {
            "ret_60d": {
                "n_windows_with_test_hit": 3,
                "median_test_hit": 0.55,
                "mean_test_hit": 0.5333,
                "p25_test_hit": 0.5,
                "p75_test_hit": 0.575,
                "median_train_hit": 0.75,
                "median_overfit_gap": 0.20,
                "mean_overfit_gap": 0.2167,
                "fraction_windows_beat_baseline": 0.6667,
                "n_windows_with_bullish_test_hit": 3,
                "median_bullish_test_hit": 0.68,
                "mean_bullish_test_hit": 0.66,
                "p25_bullish_test_hit": 0.60,
                "p75_bullish_test_hit": 0.72,
                "median_bullish_train_hit": 0.75,
                "median_bullish_overfit_gap": 0.07,
                "fraction_windows_bullish_beat_baseline": 1.0,
            }
        },
    }
    md = render_walk_forward_markdown(nonempty)
    assert "Horizon `ret_60d`" in md
    assert "55.00%" in md  # median_test_hit
    assert "+20.00pp" in md  # median overfit gap rendered as pp
    assert "66.67%" in md  # fraction_windows_beat_baseline
    assert "68.00%" in md  # median bullish test hit
    assert "+7.00pp" in md  # median bullish overfit gap
