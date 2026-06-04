from __future__ import annotations

from scripts.fit_regime_weights import build_diagnostics_payload


def test_regime_weight_diagnostics_payload_preserves_reasons():
    payload = build_diagnostics_payload(
        reports_glob="reports/analysis_mvp/*.json",
        horizon="ret_60d",
        horizons=[5, 20, 60],
        min_abs_ic=0.05,
        min_n=50,
        min_samples=250,
        require_regime_ic_ge_global=True,
        global_ic=0.23,
        diagnostics={
            "chop": {
                "n_records": 300,
                "shipped": True,
                "reason": None,
                "ic": 0.30,
                "global_ic": 0.23,
                "ic_lift": 0.07,
                "nonzero_factor_count": 12,
            },
            "trend_on": {
                "n_records": 400,
                "shipped": False,
                "reason": "regime_ic 0.2200 < global_ic 0.2300",
                "ic": 0.22,
                "global_ic": 0.23,
                "ic_lift": -0.01,
                "nonzero_factor_count": 9,
            },
        },
    )
    assert payload["global_ic"] == 0.23
    assert payload["regimes"]["chop"]["shipped"] is True
    assert payload["regimes"]["chop"]["nonzero_factor_count"] == 12
    assert payload["regimes"]["trend_on"]["shipped"] is False
    assert "regime_ic" in payload["regimes"]["trend_on"]["reason"]
