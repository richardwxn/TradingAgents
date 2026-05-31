"""Portfolio-level risk awareness for the daily-signals layer.

For a long-only portfolio, sizing IS the primary risk control. Three caps
get applied after `compute_target_weights` runs:

1. **Sector concentration cap** — no sector exceeds `max_sector_exposure`
   of the book. Scales down the highest-weight names in offending sectors.
2. **Portfolio beta budget** — weighted-avg beta of the target portfolio
   stays at or below `max_portfolio_beta`. Scales down high-beta names
   when the budget would be exceeded.
3. **Pairwise correlation cap** — refuses to size up a name whose 60-day
   correlation with the largest held position is above
   `max_pair_correlation`. Caps the new name at the smaller of its
   computed target or `max_per_name * (1 − correlation)` so the overlap
   doesn't double-count exposure.

All helpers are pure: they take resolved numbers (sector labels, betas,
correlation matrix) and produce new weight dicts. Network/yfinance calls
to fetch betas + returns live in the daily-signals CLI, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


# ---------- config + result shapes ----------


@dataclass(frozen=True)
class RiskLimits:
    """Soft caps applied after weight assignment.

    Defaults are intentionally generous so they don't surprise users who
    don't opt into the limits. Set `max_sector_exposure = 1.0` and
    `max_portfolio_beta = float('inf')` to disable the respective cap.
    """

    max_sector_exposure: float = 0.50
    max_portfolio_beta: float = 1.6
    max_pair_correlation: float = 0.85

    def __post_init__(self) -> None:
        if not 0 < self.max_sector_exposure <= 1.0:
            raise ValueError("max_sector_exposure must be in (0, 1]")
        if self.max_portfolio_beta <= 0:
            raise ValueError("max_portfolio_beta must be positive")
        if not -1.0 <= self.max_pair_correlation <= 1.0:
            raise ValueError("max_pair_correlation must be in [-1, 1]")


@dataclass(frozen=True)
class RiskAdjustment:
    """Outcome of applying caps. `adjusted_weights` is the new dict;
    `notes` maps symbol → human-readable reason for any adjustment."""

    adjusted_weights: dict[str, float]
    notes: dict[str, str] = field(default_factory=dict)
    sector_exposure: dict[str, float] = field(default_factory=dict)
    portfolio_beta: float | None = None


# ---------- pure computations ----------


def compute_sector_exposure(
    weights: dict[str, float],
    sector_map: dict[str, str | None],
) -> dict[str, float]:
    """Group weights by sector. Symbols with `None` sector go into "unknown"."""
    out: dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym) or "unknown"
        out[sector] = out.get(sector, 0.0) + float(w)
    return out


def compute_portfolio_beta(
    weights: dict[str, float],
    beta_map: dict[str, float | None],
) -> float | None:
    """Weighted-average beta, ignoring symbols with no beta data.

    Returns `None` when no symbol contributes (no betas available).
    Otherwise normalizes by the weights of symbols that DID contribute,
    so a few missing betas don't artificially deflate the number.
    """
    weighted = 0.0
    active_w = 0.0
    for sym, w in weights.items():
        b = beta_map.get(sym)
        if b is None:
            continue
        weighted += float(b) * float(w)
        active_w += float(w)
    if active_w <= 0:
        return None
    return round(weighted, 4)


def apply_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str | None],
    *,
    max_sector_exposure: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Scale down names so no sector exceeds `max_sector_exposure`.

    Strategy: for each over-cap sector, scale every member proportionally
    so the sector totals exactly the cap. Order-independent: doesn't
    matter which sector goes first because we don't redistribute the
    trimmed weight elsewhere (it becomes uninvested → cash).
    """
    out = dict(weights)
    notes: dict[str, str] = {}
    sector_exposure = compute_sector_exposure(out, sector_map)
    for sector, exposure in sector_exposure.items():
        if exposure <= max_sector_exposure or exposure <= 0:
            continue
        scale = max_sector_exposure / exposure
        members = [s for s, sec in sector_map.items()
                   if (sec or "unknown") == sector and s in out]
        for sym in members:
            old_w = out[sym]
            new_w = old_w * scale
            out[sym] = new_w
            notes[sym] = (
                f"Sector cap ({sector} exposure {exposure*100:.1f}% > "
                f"{max_sector_exposure*100:.0f}% limit) → scaled "
                f"{old_w*100:.1f}% → {new_w*100:.1f}%."
            )
    return out, notes


def apply_beta_budget(
    weights: dict[str, float],
    beta_map: dict[str, float | None],
    *,
    max_portfolio_beta: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Scale down high-beta names so the portfolio beta is within budget.

    Strategy: if current beta > max, compute the scaling factor needed and
    apply it to symbols whose beta is above the portfolio average (i.e.,
    the ones disproportionately driving the overage). Low-beta names get
    left alone so the cap doesn't squash defensive holdings.
    """
    out = dict(weights)
    notes: dict[str, str] = {}
    pf_beta = compute_portfolio_beta(out, beta_map)
    if pf_beta is None or pf_beta <= max_portfolio_beta:
        return out, notes

    # Identify above-avg-beta names (those pulling the budget over).
    avg_beta = pf_beta
    high_beta_syms = [
        sym for sym, b in beta_map.items()
        if b is not None and b > avg_beta and sym in out and out[sym] > 0
    ]
    if not high_beta_syms:
        return out, notes

    # For a long-only portfolio with cash, β_portfolio = Σ w_i × β_i
    # (cash is implicit at β=0, weight = 1 − Σw_i — it doesn't appear in
    # the sum). Scaling high-β names by `scale`:
    #
    #     β_new = scale × β_high_weighted + β_low_weighted ≤ cap
    #     ⇒ scale = (cap − β_low_weighted) / β_high_weighted
    #
    # The freed weight becomes cash (not redistributed), so weight_of_low
    # and weight_of_high don't enter the equation.
    beta_of_high = sum(out[s] * beta_map[s] for s in high_beta_syms)
    beta_of_low = sum(
        out[s] * (beta_map.get(s) or 0.0)
        for s in out if s not in high_beta_syms
    )
    if beta_of_high <= 0:
        return out, notes
    scale = (max_portfolio_beta - beta_of_low) / beta_of_high
    scale = max(0.0, min(1.0, scale))
    for sym in high_beta_syms:
        old_w = out[sym]
        new_w = old_w * scale
        out[sym] = new_w
        notes[sym] = (
            f"Beta budget (portfolio β {pf_beta:.2f} > {max_portfolio_beta:.2f} "
            f"limit, sym β {beta_map[sym]:.2f}) → scaled "
            f"{old_w*100:.1f}% → {new_w*100:.1f}%."
        )
    return out, notes


def apply_correlation_cap(
    weights: dict[str, float],
    correlation_matrix: dict[str, dict[str, float]],
    *,
    max_pair_correlation: float,
    anchor: str | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Cap correlated names against the highest-weight position.

    For each non-anchor symbol whose correlation with the anchor exceeds
    `max_pair_correlation`, cap its weight at `anchor_weight * (1 - rho)`.
    Doesn't redistribute the trimmed weight — it becomes uninvested.

    `anchor` defaults to the symbol with the largest weight. Passing
    `anchor` explicitly is useful when you want to test sensitivity to
    one specific name.
    """
    out = dict(weights)
    notes: dict[str, str] = {}
    if not out:
        return out, notes
    if anchor is None:
        anchor = max(out, key=lambda s: out[s])
    if out.get(anchor, 0.0) <= 0:
        return out, notes
    anchor_w = out[anchor]
    for sym, w in list(out.items()):
        if sym == anchor or w <= 0:
            continue
        rho = (correlation_matrix.get(sym) or {}).get(anchor)
        if rho is None:
            rho = (correlation_matrix.get(anchor) or {}).get(sym)
        if rho is None or rho < max_pair_correlation:
            continue
        # Cap based on how correlated they are; the higher the rho, the
        # smaller the allowable secondary position.
        cap_w = anchor_w * (1.0 - rho)
        if cap_w >= w:
            continue
        old_w = w
        out[sym] = max(0.0, cap_w)
        notes[sym] = (
            f"Correlation cap (ρ={rho:.2f} with anchor {anchor}, "
            f"limit ρ ≥ {max_pair_correlation:.2f}) → scaled "
            f"{old_w*100:.1f}% → {out[sym]*100:.1f}%."
        )
    return out, notes


def apply_all_risk_caps(
    weights: dict[str, float],
    *,
    sector_map: dict[str, str | None],
    beta_map: dict[str, float | None],
    correlation_matrix: dict[str, dict[str, float]],
    limits: RiskLimits,
) -> RiskAdjustment:
    """Apply sector, beta, then correlation caps in that order.

    Order matters: sector cap first (largest blunt cut), then beta budget
    (refines exposure quality), then correlation (refines pairwise overlap).
    The notes dict aggregates reasons across all three passes — a single
    symbol can collect multiple lines if it tripped multiple caps.
    """
    notes: dict[str, str] = {}
    adjusted = dict(weights)

    adjusted, sector_notes = apply_sector_cap(
        adjusted, sector_map,
        max_sector_exposure=limits.max_sector_exposure,
    )
    for s, n in sector_notes.items():
        notes[s] = (notes.get(s, "") + " " + n).strip()

    if math.isfinite(limits.max_portfolio_beta):
        adjusted, beta_notes = apply_beta_budget(
            adjusted, beta_map,
            max_portfolio_beta=limits.max_portfolio_beta,
        )
        for s, n in beta_notes.items():
            notes[s] = (notes.get(s, "") + " " + n).strip()

    adjusted, corr_notes = apply_correlation_cap(
        adjusted, correlation_matrix,
        max_pair_correlation=limits.max_pair_correlation,
    )
    for s, n in corr_notes.items():
        notes[s] = (notes.get(s, "") + " " + n).strip()

    return RiskAdjustment(
        adjusted_weights=adjusted,
        notes=notes,
        sector_exposure=compute_sector_exposure(adjusted, sector_map),
        portfolio_beta=compute_portfolio_beta(adjusted, beta_map),
    )


# ---------- helpers (pure) for market-data prep ----------


def compute_pairwise_correlations(
    returns: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compute the full pairwise correlation matrix from per-symbol return
    series (already aligned by date). Pure-Python, no numpy/scipy.

    Returns a dict-of-dicts indexed by symbol. Symbols with fewer than 20
    observations or zero variance are omitted from one or both directions.
    """
    out: dict[str, dict[str, float]] = {sym: {} for sym in returns}
    symbols = list(returns.keys())
    for i, a in enumerate(symbols):
        ra = returns[a]
        if len(ra) < 20:
            continue
        for b in symbols[i + 1:]:
            rb = returns[b]
            n = min(len(ra), len(rb))
            if n < 20:
                continue
            # Use the last n observations (aligned tail).
            xs = ra[-n:]
            ys = rb[-n:]
            rho = _pearson(xs, ys)
            if rho is None:
                continue
            out.setdefault(a, {})[b] = rho
            out.setdefault(b, {})[a] = rho
    return out


def compute_beta_vs_benchmark(
    asset_returns: list[float],
    benchmark_returns: list[float],
) -> float | None:
    """Single-asset beta: cov(asset, bench) / var(bench).

    Returns None when either series is too short or variance is zero.
    """
    n = min(len(asset_returns), len(benchmark_returns))
    if n < 20:
        return None
    xs = asset_returns[-n:]
    ys = benchmark_returns[-n:]
    mean_y = sum(ys) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n
    if var_y <= 0:
        return None
    mean_x = sum(xs) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    return round(cov / var_y, 4)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sx2 = sum((x - mean_x) ** 2 for x in xs)
    sy2 = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(sx2 * sy2)
    if denom <= 0:
        return None
    return round(cov / denom, 4)
