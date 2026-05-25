"""Unit tests for portfolio/signals.py (pure functions)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from portfolio.signals import (
    Action,
    PriceContext,
    Position,
    Signal,
    classify_action,
    compute_actions,
    compute_portfolio_value,
    format_daily_report,
    load_latest_signals,
)
from portfolio.sizing import SizingConfig


# ---------- load_latest_signals ----------


def _write_report(
    tmp_path: Path, *, symbol: str, as_of: str, direction: str,
    confidence: float = 0.7, composite: float = 0.5,
) -> Path:
    payload = {
        "symbol": symbol,
        "as_of_date": as_of,
        "direction": direction,
        "confidence": confidence,
        "key_features": {
            "model_scoring": {"composite_score": composite}
        },
    }
    p = tmp_path / f"{symbol}_{as_of}.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_latest_signals_picks_newest_per_ticker(tmp_path):
    _write_report(tmp_path, symbol="NVDA", as_of="2026-05-01", direction="neutral", composite=0.05)
    _write_report(tmp_path, symbol="NVDA", as_of="2026-05-22", direction="bullish", composite=0.48)
    _write_report(tmp_path, symbol="AMD", as_of="2026-05-15", direction="bullish", composite=0.20)
    paths = list(tmp_path.glob("*.json"))
    signals = load_latest_signals(
        paths, universe=["NVDA", "AMD", "MISSING"], as_of=date(2026, 5, 24)
    )
    assert signals["NVDA"] is not None
    assert signals["NVDA"].as_of_date == "2026-05-22"
    assert signals["NVDA"].direction == "bullish"
    assert signals["NVDA"].composite == pytest.approx(0.48)
    assert signals["AMD"] is not None
    assert signals["AMD"].as_of_date == "2026-05-15"
    assert signals["MISSING"] is None


def test_load_latest_signals_respects_as_of_cutoff(tmp_path):
    """A back-dated `as_of` must not pull in future composites."""
    _write_report(tmp_path, symbol="NVDA", as_of="2026-04-10", direction="neutral", composite=0.0)
    _write_report(tmp_path, symbol="NVDA", as_of="2026-05-22", direction="bullish", composite=0.48)
    paths = list(tmp_path.glob("*.json"))
    signals = load_latest_signals(paths, universe=["NVDA"], as_of=date(2026, 4, 15))
    assert signals["NVDA"].as_of_date == "2026-04-10"


def test_load_latest_signals_handles_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_text("{not valid json")
    _write_report(tmp_path, symbol="NVDA", as_of="2026-05-22", direction="bullish", composite=0.48)
    signals = load_latest_signals(list(tmp_path.glob("*.json")), universe=["NVDA"])
    assert signals["NVDA"] is not None


# ---------- classify_action ----------


@pytest.mark.parametrize(
    "current_w, current_shares, target_w, expected",
    [
        (0.0,   0,   0.0,   "SKIP"),   # nothing to do
        (0.0,   0,   0.05,  "BUY"),    # new position
        (0.10,  100, 0.0,   "EXIT"),   # close existing
        (0.05,  50,  0.05,  "HOLD"),   # exactly at target
        (0.05,  50,  0.054, "HOLD"),   # within rebalance tolerance
        (0.05,  50,  0.08,  "ADD"),    # increase
        (0.08,  80,  0.05,  "TRIM"),   # reduce
    ],
)
def test_classify_action(current_w, current_shares, target_w, expected):
    assert classify_action(
        target_weight=target_w,
        current_weight=current_w,
        current_shares=current_shares,
    ) == expected


# ---------- compute_portfolio_value ----------


def test_compute_portfolio_value_with_cash_and_positions():
    positions = {
        "NVDA": Position(shares=100, avg_cost=140.0),
        "AMD":  Position(shares=50,  avg_cost=170.0),
    }
    prices = {
        "NVDA": PriceContext(last_close=150.0, sma20=145.0, atr14=5.0),
        "AMD":  PriceContext(last_close=180.0, sma20=175.0, atr14=4.0),
    }
    total, weights = compute_portfolio_value(cash=10_000.0, positions=positions, prices=prices)
    # 10000 + 100*150 + 50*180 = 10000 + 15000 + 9000 = 34000.
    assert total == pytest.approx(34_000.0)
    assert weights["NVDA"] == pytest.approx(15_000 / 34_000)
    assert weights["AMD"] == pytest.approx(9_000 / 34_000)


def test_compute_portfolio_value_falls_back_to_avg_cost_when_price_missing():
    positions = {"NVDA": Position(shares=100, avg_cost=140.0)}
    prices = {"NVDA": PriceContext(last_close=None, sma20=None, atr14=None)}
    total, _weights = compute_portfolio_value(cash=0.0, positions=positions, prices=prices)
    # No price → falls back to avg_cost so the ledger doesn't silently shrink.
    assert total == pytest.approx(14_000.0)


# ---------- compute_actions (end-to-end pure) ----------


def _ctx(close, sma=None, atr=None):
    return PriceContext(last_close=close, sma20=sma, atr14=atr)


def test_compute_actions_emits_buy_for_new_bullish_signal():
    cfg = SizingConfig(max_per_name=0.10, max_long_exposure=0.50,
                       min_position_weight=0.01)
    signals = {"NVDA": Signal("NVDA", "2026-05-22", "bullish", 0.5, 0.7, "x"),
               "AMD":  None}
    positions = {}
    prices = {"NVDA": _ctx(150.0, 145.0, 5.0), "AMD": _ctx(180.0, None, None)}
    actions, summary = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 24),
    )
    by_sym = {a.symbol: a for a in actions}
    assert by_sym["NVDA"].action == "BUY"
    # target = 10% of 10000 = $1000 → 1000 / 150 = 6 shares.
    assert by_sym["NVDA"].target_shares == 6
    assert by_sym["NVDA"].limit_price == 145.0  # SMA20 < last_close, pullback wins.
    assert by_sym["NVDA"].stop_loss == 145.0 - 1.5 * 5.0  # 137.5
    # AMD has no signal → SKIP with a "no report" note.
    assert by_sym["AMD"].action == "SKIP"
    assert any("No analysis report" in n for n in by_sym["AMD"].notes)


def test_compute_actions_exits_held_name_that_turned_neutral():
    cfg = SizingConfig()
    signals = {"AMD": Signal("AMD", "2026-05-22", "neutral", 0.05, 0.5, "x")}
    positions = {"AMD": Position(shares=50, avg_cost=170.0)}
    prices = {"AMD": _ctx(180.0, 175.0, 4.0)}
    actions, _ = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 24),
    )
    a = actions[0]
    assert a.action == "EXIT"
    assert a.target_shares == 0
    assert a.delta_shares == -50
    # Trim limit = max(last_close, last_close + ATR) = max(180, 184) = 184.
    assert a.limit_price == 184.0


def test_compute_actions_suppresses_bearish_when_disabled():
    cfg = SizingConfig(enable_bearish=False)
    signals = {"AMD": Signal("AMD", "2026-05-22", "bearish", -0.5, 0.7, "x")}
    positions = {}
    prices = {"AMD": _ctx(180.0, 175.0, 4.0)}
    actions, _ = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 24),
    )
    a = actions[0]
    assert a.action == "SKIP"
    assert a.target_weight == 0.0
    assert any("suppressed" in n.lower() for n in a.notes)


def test_compute_actions_flags_stale_composite():
    cfg = SizingConfig(stale_composite_days=7)
    signals = {"NVDA": Signal("NVDA", "2026-05-01", "bullish", 0.5, 0.7, "x")}
    positions = {}
    prices = {"NVDA": _ctx(150.0, 145.0, 5.0)}
    actions, summary = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 22),  # 21 days later
    )
    a = actions[0]
    assert a.signal_age_days == 21
    assert any("Stale composite" in n for n in a.notes)
    assert summary["n_stale_signals"] == 1


def test_compute_actions_flags_held_symbol_outside_universe():
    cfg = SizingConfig()
    signals = {"NVDA": Signal("NVDA", "2026-05-22", "bullish", 0.5, 0.7, "x")}
    positions = {
        "NVDA": Position(shares=10, avg_cost=140.0),
        "OFF_UNIVERSE": Position(shares=5, avg_cost=100.0),
    }
    prices = {
        "NVDA": _ctx(150.0, 145.0, 5.0),
        "OFF_UNIVERSE": _ctx(110.0, None, None),
    }
    actions, _ = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 24),
    )
    off = next(a for a in actions if a.symbol == "OFF_UNIVERSE")
    assert off.action == "REVIEW"
    assert any("outside the configured universe" in n for n in off.notes)


def test_compute_actions_hold_when_target_matches_current():
    cfg = SizingConfig(max_per_name=0.15, max_long_exposure=0.15)
    # Single bullish name with target = max_per_name = 15%.
    # Current portfolio: 1500 cash + position worth 1500 (~10 shares @ $150).
    # So current weight ≈ 50% — won't match target ≈ 15%. Let's contrive a match.
    # Cash 8500, position 100 shares @ 150 → 15000 position, 23500 total → 15000/23500 ≈ 63.8%.
    # Adjust to make math obvious: cash 0, 100 sh @ 1 (1.50 each) is awkward. Use
    # cash 850, 1 share @ 150 → total 1000 → position weight 150/1000=15%.
    cfg = SizingConfig(max_per_name=0.15, max_long_exposure=0.15)
    signals = {"NVDA": Signal("NVDA", "2026-05-22", "bullish", 0.5, 0.7, "x")}
    positions = {"NVDA": Position(shares=1, avg_cost=150.0)}
    prices = {"NVDA": _ctx(150.0, 145.0, 5.0)}
    actions, _ = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=850.0, as_of=date(2026, 5, 24),
    )
    a = actions[0]
    # target=15%, current=15% → HOLD.
    assert a.action == "HOLD"


# ---------- format_daily_report (smoke) ----------


def test_format_daily_report_contains_expected_sections():
    cfg = SizingConfig(universe=("NVDA", "AMD"))
    signals = {
        "NVDA": Signal("NVDA", "2026-05-22", "bullish", 0.5, 0.7, "x"),
        "AMD":  Signal("AMD",  "2026-05-22", "neutral", 0.05, 0.5, "x"),
    }
    positions = {"AMD": Position(shares=10, avg_cost=170.0)}
    prices = {"NVDA": _ctx(150.0, 145.0, 5.0), "AMD": _ctx(180.0, 175.0, 4.0)}
    actions, summary = compute_actions(
        signals=signals, positions=positions, prices=prices,
        config=cfg, cash=10_000.0, as_of=date(2026, 5, 24),
    )
    md = format_daily_report(actions, summary, config=cfg, as_of=date(2026, 5, 24))
    assert "Daily Signals — 2026-05-24" in md
    assert "BUY NVDA" in md
    assert "EXIT AMD" in md
    assert "Portfolio snapshot" in md
    assert "long-or-cash" in md  # bearish-suppressed mode label
