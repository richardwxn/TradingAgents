"""Pure sizing rules for the daily signals layer.

All helpers here are deterministic and I/O-free so the CLI can be unit
tested without yfinance, file IO, or LLMs. The CLI in
`daily_signals.py` is responsible for wiring composites + prices into
these helpers and writing the markdown report.

Sizing policy (default `equal_weight_bullish`):
- Bullish names are equal-weighted, then clipped at `max_per_name`.
- Total long exposure is clipped at `max_long_exposure`.
- Neutral names are 0%. Bearish names are 0% when `enable_bearish=False`
  (the default, per handoff Section 15 finding that bearish calls are
  structurally anti-predictive in the current corpus).
- Positions below `min_position_weight` are pruned to 0% to avoid
  micro-positions that aren't worth the friction.
- Stale-signal decay (opt-in): if a signal carries `age_days` and that
  age exceeds `stale_composite_days`, the post-policy weight is
  multiplied by `stale_signal_decay`. The freed allocation falls to
  cash (no redistribution to fresh names) so reduced conviction
  translates directly into reduced risk.

The simulator (Section 17) will compare this policy against alternatives
and may recommend a different default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_POLICIES = ("equal_weight_bullish", "confidence_weighted", "top_n_bullish")


@dataclass(frozen=True)
class SizingConfig:
    policy: str = "equal_weight_bullish"
    max_per_name: float = 0.12
    max_long_exposure: float = 0.80
    min_position_weight: float = 0.02
    enable_bearish: bool = False
    stop_loss_atr_multiple: float = 1.5
    entry_pullback_to: str = "sma20"
    # Cap on how patient the buy-limit heuristic is allowed to be.
    # If SMA20 would require a pullback deeper than this fraction of
    # last_close (e.g. 5%), use `last_close * (1 - max_entry_pullback_pct)`
    # instead so the limit stays reachable for extended names.
    max_entry_pullback_pct: float = 0.05
    exit_patience_atrs: float = 1.0
    stale_composite_days: int = 7
    stale_signal_decay: float = 1.0
    composite_threshold: float = 0.15
    top_n: int = 5
    universe: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"unsupported sizing policy {self.policy!r}; "
                f"choose from {SUPPORTED_POLICIES}"
            )
        if not 0 < self.max_per_name <= 1:
            raise ValueError("max_per_name must be in (0, 1]")
        if not 0 < self.max_long_exposure <= 1:
            raise ValueError("max_long_exposure must be in (0, 1]")
        if self.min_position_weight < 0:
            raise ValueError("min_position_weight must be >= 0")
        if not 0 <= self.stale_signal_decay <= 1:
            raise ValueError("stale_signal_decay must be in [0, 1]")


def sizing_config_from_dict(data: dict[str, Any]) -> SizingConfig:
    """Build a SizingConfig from a parsed YAML dict, ignoring unknown keys.

    Unknown keys are tolerated so the YAML file can carry comments,
    `policy_*` overrides, or future extension fields without breaking
    older code.
    """
    allowed = {
        "policy", "max_per_name", "max_long_exposure", "min_position_weight",
        "enable_bearish", "stop_loss_atr_multiple", "entry_pullback_to",
        "max_entry_pullback_pct", "exit_patience_atrs",
        "stale_composite_days", "stale_signal_decay",
        "composite_threshold", "top_n",
    }
    kwargs = {k: v for k, v in data.items() if k in allowed}
    universe = data.get("universe") or ()
    if universe:
        kwargs["universe"] = tuple(universe)
    return SizingConfig(**kwargs)


def _eligible_bullish(
    signals: dict[str, dict[str, Any]],
    *,
    config: SizingConfig,
) -> list[str]:
    return [
        sym for sym, sig in signals.items()
        if (sig or {}).get("direction") == "bullish"
    ]


def compute_target_weights(
    signals: dict[str, dict[str, Any]],
    *,
    config: SizingConfig,
) -> dict[str, float]:
    """Map per-ticker signals to per-ticker target weights in [0, 1].

    `signals[sym]` is expected to expose at least:
      - "direction": "bullish" | "bearish" | "neutral" | None
      - "composite": float | None
      - "confidence": float | None

    Returns a weights dict covering every key in `signals` (missing
    entries pruned by the caller if desired).
    """
    weights = {sym: 0.0 for sym in signals}
    bullish = _eligible_bullish(signals, config=config)
    if not bullish:
        return weights

    if config.policy == "equal_weight_bullish":
        per_name = min(config.max_long_exposure / len(bullish), config.max_per_name)
        for sym in bullish:
            weights[sym] = per_name

    elif config.policy == "top_n_bullish":
        ranked = sorted(
            bullish,
            key=lambda s: (signals[s].get("composite") or 0.0),
            reverse=True,
        )[: max(1, config.top_n)]
        per_name = min(config.max_long_exposure / len(ranked), config.max_per_name)
        for sym in ranked:
            weights[sym] = per_name

    elif config.policy == "confidence_weighted":
        raw = {}
        for sym in bullish:
            sig = signals[sym] or {}
            composite = float(sig.get("composite") or 0.0)
            confidence = float(sig.get("confidence") or 0.5)
            raw[sym] = max(0.0, composite) * max(0.0, confidence)
        total_raw = sum(raw.values())
        if total_raw <= 0:
            # All bullish names had composite=0 / confidence=0 — fall back to equal-weight.
            per_name = min(config.max_long_exposure / len(bullish), config.max_per_name)
            for sym in bullish:
                weights[sym] = per_name
        else:
            scale = config.max_long_exposure / total_raw
            for sym, r in raw.items():
                weights[sym] = min(r * scale, config.max_per_name)

    # Stale-signal decay: shrink names whose composite is older than
    # stale_composite_days. Freed weight falls to cash (no redistribution).
    if config.stale_signal_decay < 1.0:
        for sym, sig in signals.items():
            age = (sig or {}).get("age_days")
            if age is not None and age > config.stale_composite_days and weights.get(sym, 0.0) > 0:
                weights[sym] = weights[sym] * config.stale_signal_decay

    # Enforce the aggregate long-exposure cap (per-name cap may have undone
    # the initial scaling under confidence_weighted).
    total_long = sum(weights.values())
    if total_long > config.max_long_exposure and total_long > 0:
        scale = config.max_long_exposure / total_long
        weights = {k: v * scale for k, v in weights.items()}

    # Prune sub-min-weight positions to 0% to avoid micro-positions.
    pruned = {
        k: (v if v >= config.min_position_weight else 0.0)
        for k, v in weights.items()
    }
    return pruned


# ---------- price-level heuristics (NOT backtest-validated, see Section 16) ----------


def buy_limit_price(
    last_close: float | None,
    sma20: float | None,
    *,
    pullback_to: str = "sma20",
    max_pullback_pct: float = 0.05,
) -> float | None:
    """Patient buy entry: target `min(last_close, SMA20)` so we wait for
    a pullback if the name is extended, BUT capped at
    `last_close * (1 - max_pullback_pct)` so the limit stays reachable
    for momentum names where SMA20 is far below.

    Falls back to last_close when SMA20 is unavailable or the
    pullback_to mode isn't `sma20`. Returns None when last_close is
    missing.
    """
    if last_close is None:
        return None
    if pullback_to != "sma20" or sma20 is None or sma20 <= 0:
        return round(last_close, 2)
    floor = last_close * (1.0 - max(0.0, max_pullback_pct))
    target = min(last_close, sma20)
    return round(max(target, floor), 2)


def trim_limit_price(
    last_close: float | None,
    atr14: float | None,
    *,
    atrs: float = 1.0,
) -> float | None:
    """Patient trim/exit: `max(last_close, last_close + atrs * ATR)` so
    we wait for a bounce if the name is weak intraday. Falls back to
    last_close when ATR is unavailable."""
    if last_close is None:
        return None
    if atr14 is not None and atr14 > 0:
        return round(max(last_close, last_close + atrs * atr14), 2)
    return round(last_close, 2)


def stop_loss_price(
    entry_price: float | None,
    atr14: float | None,
    *,
    atr_multiple: float = 1.5,
    fallback_pct: float = 0.10,
) -> float | None:
    """Vol-adjusted stop-loss reminder: `entry - atr_multiple * ATR(14)`.
    Falls back to `entry * (1 - fallback_pct)` when ATR is missing.

    This is a REMINDER for manual review, not an auto-executing order.
    """
    if entry_price is None or entry_price <= 0:
        return None
    if atr14 is not None and atr14 > 0:
        return round(max(0.01, entry_price - atr_multiple * atr14), 2)
    return round(entry_price * (1.0 - fallback_pct), 2)
