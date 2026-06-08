from __future__ import annotations

import math

import pytest

from portfolio.risk import (
    RiskAdjustment,
    RiskLimits,
    SectorShock,
    apply_all_risk_caps,
    apply_beta_budget,
    apply_correlation_cap,
    apply_sector_cap,
    apply_sector_shock_guard,
    compute_beta_vs_benchmark,
    compute_pairwise_correlations,
    compute_portfolio_beta,
    compute_sector_exposure,
)


# ---------- RiskLimits validation ----------


def test_risk_limits_defaults_valid():
    RiskLimits()  # should not raise


def test_risk_limits_rejects_zero_sector_cap():
    with pytest.raises(ValueError, match="max_sector_exposure"):
        RiskLimits(max_sector_exposure=0.0)


def test_risk_limits_rejects_bad_correlation():
    with pytest.raises(ValueError, match="max_pair_correlation"):
        RiskLimits(max_pair_correlation=1.5)


# ---------- compute_sector_exposure ----------


def test_sector_exposure_sums_by_sector():
    weights = {"NVDA": 0.10, "AMD": 0.08, "JPM": 0.05}
    sector_map = {"NVDA": "Tech", "AMD": "Tech", "JPM": "Financials"}
    out = compute_sector_exposure(weights, sector_map)
    assert out == {"Tech": pytest.approx(0.18), "Financials": pytest.approx(0.05)}


def test_sector_exposure_unknown_bucket():
    weights = {"NVDA": 0.10, "FOO": 0.05}
    sector_map = {"NVDA": "Tech"}
    out = compute_sector_exposure(weights, sector_map)
    assert out["unknown"] == pytest.approx(0.05)


# ---------- apply_sector_cap ----------


def test_sector_cap_scales_overweight_sector():
    weights = {"NVDA": 0.30, "AMD": 0.30, "AVGO": 0.10, "JPM": 0.10}
    sector_map = {"NVDA": "Tech", "AMD": "Tech", "AVGO": "Tech", "JPM": "Financials"}
    # Tech total = 0.70, cap to 0.50 → scale by 0.50/0.70 ≈ 0.714.
    out, notes = apply_sector_cap(weights, sector_map, max_sector_exposure=0.50)
    tech_sum = out["NVDA"] + out["AMD"] + out["AVGO"]
    assert tech_sum == pytest.approx(0.50, abs=1e-4)
    # Tech members are scaled proportionally.
    assert out["NVDA"] == pytest.approx(0.30 * 0.50 / 0.70, abs=1e-4)
    # Non-Tech is untouched.
    assert out["JPM"] == 0.10
    # Notes populated for tech names only.
    assert {"NVDA", "AMD", "AVGO"} == set(notes.keys())


def test_sector_cap_no_op_when_under_limit():
    weights = {"NVDA": 0.20, "JPM": 0.10}
    sector_map = {"NVDA": "Tech", "JPM": "Financials"}
    out, notes = apply_sector_cap(weights, sector_map, max_sector_exposure=0.50)
    assert out == weights
    assert notes == {}


# ---------- apply_sector_shock_guard ----------


def test_sector_shock_guard_scales_new_buys_and_existing_positions():
    weights = {"NVDA": 0.12, "AMD": 0.10, "JPM": 0.08}
    sector_map = {"NVDA": "Semiconductors", "AMD": "Semiconductors", "JPM": "Financials"}
    shocks = {
        "Semiconductors": SectorShock(
            sector="Semiconductors",
            trigger_symbol="SOXX",
            pct_change=-0.052,
            threshold=0.03,
        )
    }
    out, notes = apply_sector_shock_guard(
        weights,
        sector_map,
        position_shares={"NVDA": 0, "AMD": 10},
        shocks=shocks,
        new_buy_size_factor=0.0,
        existing_position_size_factor=0.5,
    )
    assert out["NVDA"] == 0.0
    assert out["AMD"] == pytest.approx(0.05)
    assert out["JPM"] == pytest.approx(0.08)
    assert "new-buy" in notes["NVDA"]
    assert "existing-position" in notes["AMD"]


def test_sector_shock_guard_no_op_without_matching_sector():
    weights = {"JPM": 0.08}
    out, notes = apply_sector_shock_guard(
        weights,
        {"JPM": "Financials"},
        position_shares={},
        shocks={
            "Semiconductors": SectorShock(
                sector="Semiconductors",
                trigger_symbol="SOXX",
                pct_change=-0.05,
                threshold=0.03,
            )
        },
        new_buy_size_factor=0.0,
        existing_position_size_factor=0.5,
    )
    assert out == weights
    assert notes == {}


# ---------- compute_portfolio_beta ----------


def test_portfolio_beta_includes_cash_drag():
    """Portfolio beta = Σ w_i × β_i with the implicit cash β=0 left out.
    So a 60%-invested portfolio with avg risky β=1.667 has portfolio β=1.0."""
    weights = {"NVDA": 0.40, "JPM": 0.20}
    betas = {"NVDA": 2.0, "JPM": 1.0}
    assert compute_portfolio_beta(weights, betas) == pytest.approx(1.0, abs=1e-4)


def test_portfolio_beta_ignores_missing_betas():
    weights = {"NVDA": 0.40, "JPM": 0.20}
    betas = {"NVDA": 2.0}  # JPM missing → drops out entirely
    # Only NVDA contributes: 0.40 × 2.0 = 0.80
    assert compute_portfolio_beta(weights, betas) == pytest.approx(0.80, abs=1e-4)


def test_portfolio_beta_returns_none_when_no_data():
    assert compute_portfolio_beta({"NVDA": 0.10}, {}) is None


# ---------- apply_beta_budget ----------


def test_beta_budget_scales_high_beta_names():
    # Higher exposure that pushes the cash-drag-inclusive beta over 1.0.
    weights = {"NVDA": 0.40, "AMD": 0.30, "JPM": 0.20}
    betas = {"NVDA": 2.0, "AMD": 1.8, "JPM": 1.0}
    # Portfolio beta = 0.40×2.0 + 0.30×1.8 + 0.20×1.0 = 0.8 + 0.54 + 0.20 = 1.54
    # Push the cap below 1.54 to force adjustment.
    out, notes = apply_beta_budget(weights, betas, max_portfolio_beta=1.20)
    # High-beta names (NVDA, AMD: β > avg) → scaled down. JPM untouched.
    assert out["JPM"] == 0.20
    assert out["NVDA"] < 0.40
    assert "NVDA" in notes or "AMD" in notes
    pf = compute_portfolio_beta(out, betas)
    assert pf <= 1.20 + 1e-3


def test_beta_budget_no_op_when_under_limit():
    weights = {"NVDA": 0.20, "JPM": 0.30}
    betas = {"NVDA": 1.5, "JPM": 1.0}
    out, notes = apply_beta_budget(weights, betas, max_portfolio_beta=1.6)
    assert out == weights
    assert notes == {}


# ---------- apply_correlation_cap ----------


def test_correlation_cap_trims_correlated_secondary():
    weights = {"NVDA": 0.10, "AMD": 0.08}  # NVDA is anchor by largest weight
    corr = {"NVDA": {"AMD": 0.90}, "AMD": {"NVDA": 0.90}}
    out, notes = apply_correlation_cap(weights, corr, max_pair_correlation=0.85)
    # Cap = NVDA_weight * (1 - rho) = 0.10 * 0.10 = 0.01.
    assert out["AMD"] == pytest.approx(0.01, abs=1e-4)
    assert out["NVDA"] == 0.10
    assert "AMD" in notes


def test_correlation_cap_no_op_when_under_limit():
    weights = {"NVDA": 0.10, "AMD": 0.08}
    corr = {"NVDA": {"AMD": 0.60}, "AMD": {"NVDA": 0.60}}
    out, notes = apply_correlation_cap(weights, corr, max_pair_correlation=0.85)
    assert out == weights
    assert notes == {}


def test_correlation_cap_no_op_when_secondary_already_smaller():
    weights = {"NVDA": 0.10, "AMD": 0.005}
    corr = {"NVDA": {"AMD": 0.95}, "AMD": {"NVDA": 0.95}}
    # AMD weight (0.005) is already below the cap 0.10*(1-0.95)=0.005 → no-op.
    out, _ = apply_correlation_cap(weights, corr, max_pair_correlation=0.85)
    assert out["AMD"] == 0.005


def test_correlation_cap_handles_missing_correlations():
    weights = {"NVDA": 0.10, "JPM": 0.05}
    corr = {}  # no data at all
    out, notes = apply_correlation_cap(weights, corr, max_pair_correlation=0.85)
    assert out == weights
    assert notes == {}


# ---------- apply_all_risk_caps ----------


def test_apply_all_returns_full_diagnostic():
    weights = {"NVDA": 0.30, "AMD": 0.30, "AVGO": 0.10, "JPM": 0.10}
    sector_map = {"NVDA": "Tech", "AMD": "Tech", "AVGO": "Tech", "JPM": "Financials"}
    betas = {"NVDA": 2.0, "AMD": 1.8, "AVGO": 1.5, "JPM": 1.0}
    corr = {"NVDA": {"AMD": 0.92, "AVGO": 0.75}}
    adj = apply_all_risk_caps(
        weights,
        sector_map=sector_map,
        beta_map=betas,
        correlation_matrix=corr,
        limits=RiskLimits(
            max_sector_exposure=0.50,
            max_portfolio_beta=1.6,
            max_pair_correlation=0.85,
        ),
    )
    assert isinstance(adj, RiskAdjustment)
    assert adj.sector_exposure["Tech"] <= 0.50 + 1e-4
    assert adj.portfolio_beta is not None and adj.portfolio_beta <= 1.6 + 1e-3
    # NVDA likely picked up notes from sector cap + beta budget (and maybe corr).
    assert "NVDA" in adj.notes


def test_apply_all_no_op_when_within_all_caps():
    weights = {"NVDA": 0.10, "JPM": 0.10}
    sector_map = {"NVDA": "Tech", "JPM": "Financials"}
    betas = {"NVDA": 1.4, "JPM": 1.0}
    corr = {}
    adj = apply_all_risk_caps(
        weights, sector_map=sector_map, beta_map=betas,
        correlation_matrix=corr,
        limits=RiskLimits(
            max_sector_exposure=0.50,
            max_portfolio_beta=1.6,
            max_pair_correlation=0.85,
        ),
    )
    assert adj.adjusted_weights == weights
    assert adj.notes == {}


# ---------- compute_pairwise_correlations ----------


def test_pairwise_correlations_perfect_positive():
    rs = list(range(30))
    out = compute_pairwise_correlations({"A": rs, "B": rs})
    assert out["A"]["B"] == pytest.approx(1.0, abs=1e-4)


def test_pairwise_correlations_perfect_negative():
    rs = [float(x) for x in range(30)]
    neg = [-x for x in rs]
    out = compute_pairwise_correlations({"A": rs, "B": neg})
    assert out["A"]["B"] == pytest.approx(-1.0, abs=1e-4)


def test_pairwise_correlations_short_series_omitted():
    out = compute_pairwise_correlations({"A": [1, 2, 3], "B": [1, 2, 3]})
    # 3 obs is way below the 20-obs threshold.
    assert out.get("A", {}).get("B") is None


# ---------- compute_beta_vs_benchmark ----------


def test_beta_vs_benchmark_unity_when_identical():
    rs = [0.01 * (i - 15) for i in range(30)]  # spans positive + negative
    assert compute_beta_vs_benchmark(rs, rs) == pytest.approx(1.0, abs=1e-4)


def test_beta_vs_benchmark_double_amplitude_gives_two():
    bench = [0.01 * (i - 15) for i in range(30)]
    asset = [r * 2 for r in bench]
    assert compute_beta_vs_benchmark(asset, bench) == pytest.approx(2.0, abs=1e-4)


def test_beta_vs_benchmark_none_when_short_series():
    assert compute_beta_vs_benchmark([0.01, 0.02], [0.01, 0.02]) is None


def test_beta_vs_benchmark_none_when_bench_zero_variance():
    rs = [0.01] * 30
    asset = [float(i) for i in range(30)]
    assert compute_beta_vs_benchmark(asset, rs) is None
