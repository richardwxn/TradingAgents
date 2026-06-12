"""Tests for the 20d-horizon action overlay (derive_horizon_signals +
format_horizon_overlay)."""

from datetime import date

from portfolio.signals import (
    Action,
    Signal,
    derive_horizon_signals,
    format_horizon_overlay,
)


def _sig(symbol, composite, composite_20d, coverage=1.0, direction="bullish"):
    return Signal(
        symbol=symbol,
        as_of_date="2026-05-26",
        direction=direction,
        composite=composite,
        confidence=0.7,
        source_path=f"/tmp/{symbol}.json",
        composite_20d=composite_20d,
        composite_20d_coverage=coverage,
    )


def test_derive_uses_20d_composite_for_direction_and_confidence():
    # Primary bullish (+0.32) but 20d bearish (-0.30): the derived signal must
    # flip to the 20d view.
    signals = {"NVDA": _sig("NVDA", composite=0.32, composite_20d=-0.30)}
    out = derive_horizon_signals(signals)
    d = out["NVDA"]
    assert d is not None
    assert d.composite == -0.30
    assert d.direction == "bearish"          # direction_for_composite(-0.30)
    assert d.confidence is not None and d.confidence > 0.5
    # Original signal is untouched.
    assert signals["NVDA"].composite == 0.32
    assert signals["NVDA"].direction == "bullish"


def test_derive_neutral_when_20d_below_threshold():
    signals = {"AAPL": _sig("AAPL", composite=0.5, composite_20d=0.05)}
    out = derive_horizon_signals(signals)
    assert out["AAPL"].direction == "neutral"  # |0.05| < 0.15 threshold


def test_derive_none_when_no_20d_composite():
    signals = {"MSFT": _sig("MSFT", composite=0.4, composite_20d=None)}
    out = derive_horizon_signals(signals)
    assert out["MSFT"] is None


def test_derive_handles_none_signal():
    out = derive_horizon_signals({"X": None})
    assert out["X"] is None


def test_lower_coverage_lowers_confidence():
    hi = derive_horizon_signals({"A": _sig("A", 0.3, 0.6, coverage=1.0)})["A"]
    lo = derive_horizon_signals({"A": _sig("A", 0.3, 0.6, coverage=0.3)})["A"]
    assert lo.confidence < hi.confidence


def test_calibration_overrides_heuristic_confidence():
    # A bullish 20d calibration that maps high composite -> ~0.9 hit-rate.
    cal = {
        "fit": [
            {"x_lower": -1.0, "x_upper": 0.0, "hit_rate": 0.2},
            {"x_lower": 0.0, "x_upper": 1.0, "hit_rate": 0.9},
        ],
    }
    sig = {"A": _sig("A", composite=0.3, composite_20d=0.5, coverage=1.0)}
    heuristic = derive_horizon_signals(sig)["A"].confidence
    calibrated = derive_horizon_signals(sig, calibration=cal)["A"].confidence
    # Calibrated bullish high-composite confidence should exceed the heuristic
    # and reflect the ~0.9 mapped hit-rate.
    assert calibrated != heuristic
    assert calibrated > 0.7


# ---------- overlay formatter ----------


def _action(symbol, direction, action):
    return Action(
        symbol=symbol, action=action, direction=direction, composite=0.0,
        confidence=0.6, target_weight=0.0, current_weight=0.0, delta_pp=0.0,
        target_shares=0, current_shares=0, delta_shares=0, limit_price=None,
        stop_loss=None, last_close=None, sma20=None, atr14=None,
        signal_age_days=0,
    )


def test_overlay_flags_divergence():
    primary = [_action("NVDA", "bullish", "BUY"), _action("AAPL", "bullish", "HOLD")]
    horizon = [_action("NVDA", "bearish", "TRIM"), _action("AAPL", "bullish", "HOLD")]
    md = format_horizon_overlay(primary, horizon)
    assert "20d horizon overlay" in md
    assert "NVDA" in md and "AAPL" in md
    # NVDA diverges (bearish/TRIM vs bullish/BUY); AAPL agrees.
    assert "⚠ diverges" in md
    assert md.count("⚠ diverges") == 1
    assert "**1** name" in md


def test_overlay_empty_when_no_actions():
    assert format_horizon_overlay([], []) == ""


def test_overlay_skips_review_rows():
    horizon = [_action("ZZZ", None, "REVIEW")]
    primary = [_action("ZZZ", None, "REVIEW")]
    assert format_horizon_overlay(primary, horizon) == ""
