"""Unit tests for portfolio/sizing.py (pure functions)."""

from __future__ import annotations

import pytest

from portfolio.sizing import (
    SizingConfig,
    buy_limit_price,
    compute_target_weights,
    sizing_config_from_dict,
    stop_loss_price,
    trim_limit_price,
)


# ---------- SizingConfig ----------


def test_sizing_config_defaults_pass_validation():
    cfg = SizingConfig()
    assert cfg.policy == "equal_weight_bullish"
    assert cfg.max_per_name == 0.12
    assert cfg.max_long_exposure == 0.80
    assert cfg.enable_bearish is False


def test_sizing_config_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unsupported sizing policy"):
        SizingConfig(policy="bogus_policy")


def test_sizing_config_rejects_bad_per_name():
    with pytest.raises(ValueError, match="max_per_name"):
        SizingConfig(max_per_name=0.0)
    with pytest.raises(ValueError, match="max_per_name"):
        SizingConfig(max_per_name=1.5)


def test_sizing_config_from_dict_ignores_unknown_keys():
    cfg = sizing_config_from_dict({
        "policy": "equal_weight_bullish",
        "max_per_name": 0.10,
        "universe": ["NVDA", "AMD"],
        "enable_tradingagents_review_gate": True,
        "tradingagents_review_apply_to_sizing": False,
        "tradingagents_review_apply_to_tickets": False,
        "tradingagents_review_top_screener_n": 3,
        "unknown_future_field": 42,
        "another_unknown": "value",
    })
    assert cfg.max_per_name == 0.10
    assert cfg.universe == ("NVDA", "AMD")
    assert cfg.enable_tradingagents_review_gate is True
    assert cfg.tradingagents_review_apply_to_sizing is False
    assert cfg.tradingagents_review_apply_to_tickets is False
    assert cfg.tradingagents_review_top_screener_n == 3


def test_sizing_config_from_dict_parses_sector_shock_guard_block():
    cfg = sizing_config_from_dict({
        "sector_shock_guard": {
            "enabled": False,
            "drop_pct": 0.05,
            "new_buy_size_factor": 0.25,
            "existing_position_size_factor": 0.75,
            "etfs": {"Semiconductors": "SMH"},
        }
    })
    assert cfg.sector_shock_guard_enabled is False
    assert cfg.sector_shock_drop_pct == pytest.approx(0.05)
    assert cfg.sector_shock_new_buy_size_factor == pytest.approx(0.25)
    assert cfg.sector_shock_existing_position_size_factor == pytest.approx(0.75)
    assert cfg.sector_shock_etfs["Semiconductors"] == "SMH"


# ---------- equal_weight_bullish ----------


def _sig(direction: str, composite: float | None = None, confidence: float | None = None):
    return {"direction": direction, "composite": composite, "confidence": confidence}


def test_equal_weight_all_neutral_means_all_cash():
    cfg = SizingConfig()
    out = compute_target_weights(
        {"A": _sig("neutral"), "B": _sig("neutral")}, config=cfg
    )
    assert out == {"A": 0.0, "B": 0.0}


def test_equal_weight_single_bullish_caps_at_per_name():
    """1 bullish name → per_name = min(0.80/1, 0.12) = 0.12 (cap binds)."""
    cfg = SizingConfig()  # max_per_name=0.12, max_long=0.80
    out = compute_target_weights({"NVDA": _sig("bullish", 0.5)}, config=cfg)
    assert out["NVDA"] == pytest.approx(0.12)


def test_equal_weight_seven_bullish_per_name_caps_each():
    """7 bullish → per_name = min(0.80/7, 0.12) = 0.114 → 0.114 (per-name binds)."""
    cfg = SizingConfig()
    sigs = {f"T{i}": _sig("bullish", 0.5) for i in range(7)}
    out = compute_target_weights(sigs, config=cfg)
    expected = 0.80 / 7  # 0.1143 < 0.12 → per_name cap doesn't bind, long_exposure does
    for v in out.values():
        assert v == pytest.approx(expected, abs=1e-9)
    assert sum(out.values()) == pytest.approx(0.80, abs=1e-9)


def test_equal_weight_ten_bullish_long_cap_binds():
    """10 bullish → 0.80/10 = 0.08 per name (under 12% per-name cap)."""
    cfg = SizingConfig()
    sigs = {f"T{i}": _sig("bullish", 0.3) for i in range(10)}
    out = compute_target_weights(sigs, config=cfg)
    for v in out.values():
        assert v == pytest.approx(0.08, abs=1e-9)


def test_equal_weight_bearish_suppressed_when_disabled():
    cfg = SizingConfig(enable_bearish=False)
    out = compute_target_weights({
        "A": _sig("bullish", 0.5),
        "B": _sig("bearish", -0.5),
    }, config=cfg)
    # B is dropped (caller is expected to suppress bearish before sizing
    # OR sizing should ignore bearish; we test compute_target_weights only
    # cares about direction == 'bullish'.
    assert out["A"] == pytest.approx(0.12)
    assert out["B"] == 0.0


def test_min_position_weight_prunes_micro_positions():
    """If we'd split across so many names that each is below the floor,
    those positions are pruned to 0."""
    cfg = SizingConfig(min_position_weight=0.05)
    sigs = {f"T{i}": _sig("bullish", 0.3) for i in range(20)}  # 4% each before prune
    out = compute_target_weights(sigs, config=cfg)
    for v in out.values():
        assert v == 0.0  # all under 5% → pruned


# ---------- top_n_bullish ----------


def test_top_n_picks_top_by_composite():
    cfg = SizingConfig(policy="top_n_bullish", top_n=3, max_per_name=0.20)
    sigs = {
        "A": _sig("bullish", 0.10),
        "B": _sig("bullish", 0.50),
        "C": _sig("bullish", 0.30),
        "D": _sig("bullish", 0.40),
        "E": _sig("bullish", 0.20),
    }
    out = compute_target_weights(sigs, config=cfg)
    # Top 3 by composite: B (0.50), D (0.40), C (0.30).
    held = {k for k, v in out.items() if v > 0}
    assert held == {"B", "C", "D"}
    # Each ~26.7%? No — capped at min(0.80/3, 0.20) = 0.20.
    for k in held:
        assert out[k] == pytest.approx(0.20)


# ---------- confidence_weighted ----------


def test_confidence_weighted_proportional_to_composite_times_confidence():
    cfg = SizingConfig(policy="confidence_weighted", max_per_name=0.50)
    sigs = {
        "HIGH": _sig("bullish", 0.8, 0.9),  # raw = 0.72
        "LOW": _sig("bullish", 0.2, 0.5),   # raw = 0.10
    }
    out = compute_target_weights(sigs, config=cfg)
    # Total raw = 0.82 → scale = 0.80 / 0.82 ≈ 0.976.
    # HIGH → 0.72 * 0.976 = 0.702 → capped at 0.50.
    # LOW → 0.10 * 0.976 = 0.098.
    assert out["HIGH"] == pytest.approx(0.50)
    assert out["LOW"] == pytest.approx(0.098, abs=1e-3)


def test_confidence_weighted_falls_back_to_equal_when_all_zero_signal():
    cfg = SizingConfig(policy="confidence_weighted", max_per_name=0.50)
    sigs = {"A": _sig("bullish", 0.0, 0.0), "B": _sig("bullish", 0.0, 0.0)}
    out = compute_target_weights(sigs, config=cfg)
    # All raw = 0 → equal-weight fallback: min(0.80/2, 0.50) = 0.40 each.
    assert out["A"] == pytest.approx(0.40)
    assert out["B"] == pytest.approx(0.40)


# ---------- stale-signal decay ----------


def _sig_with_age(direction: str, composite: float, age_days: int):
    return {
        "direction": direction, "composite": composite,
        "confidence": 0.7, "age_days": age_days,
    }


def test_stale_decay_default_off_no_change():
    """With default stale_signal_decay=1.0, age_days is ignored."""
    cfg = SizingConfig(stale_composite_days=7)  # decay defaults to 1.0
    sigs = {
        "FRESH": _sig_with_age("bullish", 0.5, age_days=2),
        "STALE": _sig_with_age("bullish", 0.5, age_days=90),
    }
    out = compute_target_weights(sigs, config=cfg)
    # Both equal-weighted at min(0.80/2, 0.12) = 0.12.
    assert out["FRESH"] == pytest.approx(0.12)
    assert out["STALE"] == pytest.approx(0.12)


def test_stale_decay_halves_stale_weight_under_equal_weight():
    """Stale signal gets weight * decay; freed allocation becomes cash."""
    cfg = SizingConfig(stale_composite_days=7, stale_signal_decay=0.5)
    sigs = {
        "FRESH": _sig_with_age("bullish", 0.5, age_days=2),
        "STALE": _sig_with_age("bullish", 0.5, age_days=90),
    }
    out = compute_target_weights(sigs, config=cfg)
    assert out["FRESH"] == pytest.approx(0.12)        # untouched
    assert out["STALE"] == pytest.approx(0.06)        # 0.12 * 0.5
    assert sum(out.values()) == pytest.approx(0.18)   # rest stays cash


def test_stale_decay_zero_drops_stale_to_zero():
    cfg = SizingConfig(stale_composite_days=7, stale_signal_decay=0.0)
    sigs = {
        "FRESH": _sig_with_age("bullish", 0.5, age_days=2),
        "STALE": _sig_with_age("bullish", 0.5, age_days=90),
    }
    out = compute_target_weights(sigs, config=cfg)
    assert out["FRESH"] == pytest.approx(0.12)
    assert out["STALE"] == 0.0


def test_stale_decay_at_threshold_does_not_apply():
    """age_days == stale_composite_days is fresh (strict > comparison)."""
    cfg = SizingConfig(stale_composite_days=7, stale_signal_decay=0.5)
    sigs = {"AT_EDGE": _sig_with_age("bullish", 0.5, age_days=7)}
    out = compute_target_weights(sigs, config=cfg)
    assert out["AT_EDGE"] == pytest.approx(0.12)


def test_stale_decay_works_under_confidence_weighted():
    """Decay applies after the policy chooses raw weights."""
    cfg = SizingConfig(
        policy="confidence_weighted",
        max_per_name=0.50,
        stale_composite_days=7,
        stale_signal_decay=0.5,
    )
    sigs = {
        "FRESH": {"direction": "bullish", "composite": 0.4, "confidence": 0.7, "age_days": 2},
        "STALE": {"direction": "bullish", "composite": 0.4, "confidence": 0.7, "age_days": 90},
    }
    out = compute_target_weights(sigs, config=cfg)
    # Without decay both would be 0.40 each (raw 0.28 * scale → 0.40 each, summing to 0.80).
    # With decay only STALE gets halved → STALE = 0.20, FRESH stays 0.40.
    assert out["FRESH"] == pytest.approx(0.40)
    assert out["STALE"] == pytest.approx(0.20)


def test_stale_decay_respects_min_position_weight_prune():
    """If decay drops a weight below min_position_weight, it's pruned to 0."""
    cfg = SizingConfig(
        stale_composite_days=7, stale_signal_decay=0.1,
        min_position_weight=0.05,
    )
    sigs = {
        "FRESH": _sig_with_age("bullish", 0.5, age_days=2),
        "STALE": _sig_with_age("bullish", 0.5, age_days=90),
    }
    out = compute_target_weights(sigs, config=cfg)
    assert out["FRESH"] == pytest.approx(0.12)
    # 0.12 * 0.1 = 0.012 < 0.05 floor → pruned to 0.
    assert out["STALE"] == 0.0


def test_stale_signal_decay_validation():
    with pytest.raises(ValueError, match="stale_signal_decay"):
        SizingConfig(stale_signal_decay=-0.1)
    with pytest.raises(ValueError, match="stale_signal_decay"):
        SizingConfig(stale_signal_decay=1.5)


def test_sizing_config_from_dict_includes_stale_signal_decay():
    cfg = sizing_config_from_dict({"stale_signal_decay": 0.5})
    assert cfg.stale_signal_decay == 0.5


# ---------- price-level heuristics ----------


def test_buy_limit_uses_sma_when_pullback_within_cap():
    # SMA20=95, last=100 → 5% pullback, exactly at default cap → 95.
    assert buy_limit_price(100.0, 95.0, pullback_to="sma20", max_pullback_pct=0.05) == 95.0
    assert buy_limit_price(100.0, 105.0, pullback_to="sma20") == 100.0


def test_buy_limit_caps_at_max_pullback_for_extended_names():
    # last=100, SMA20=70 → 30% pullback. Cap at 5% → floor=95 → limit=95.
    assert buy_limit_price(100.0, 70.0, pullback_to="sma20", max_pullback_pct=0.05) == 95.0
    # Cap at 10% → floor=90 → limit=90.
    assert buy_limit_price(100.0, 70.0, pullback_to="sma20", max_pullback_pct=0.10) == 90.0


def test_buy_limit_falls_back_to_close_when_sma_missing():
    assert buy_limit_price(100.0, None, pullback_to="sma20") == 100.0


def test_buy_limit_returns_none_when_no_close():
    assert buy_limit_price(None, 50.0) is None


def test_trim_limit_waits_for_bounce():
    # last_close=100, ATR=5, atrs=1 → max(100, 105) = 105.
    assert trim_limit_price(100.0, 5.0, atrs=1.0) == 105.0


def test_trim_limit_falls_back_to_close_when_atr_missing():
    assert trim_limit_price(100.0, None) == 100.0


def test_stop_loss_uses_atr_multiple():
    # entry=100, ATR=5, mult=1.5 → 100 - 7.5 = 92.5.
    assert stop_loss_price(100.0, 5.0, atr_multiple=1.5) == 92.5


def test_stop_loss_falls_back_to_percent_when_atr_missing():
    # entry=100, no ATR → 100 * (1 - 0.10) = 90.
    assert stop_loss_price(100.0, None, fallback_pct=0.10) == 90.0


def test_stop_loss_clamps_to_positive():
    # entry=10, ATR=20, mult=1.5 → 10 - 30 = -20, clamped to 0.01.
    assert stop_loss_price(10.0, 20.0, atr_multiple=1.5) == 0.01
