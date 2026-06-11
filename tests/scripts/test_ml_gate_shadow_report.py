from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from portfolio.paper_trading import RecommendationRecord, write_recommendations_for_day


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_ml_gate_shadow_report_for_tests",
        repo_root / "scripts" / "ml_gate_shadow_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rec(
    symbol: str,
    *,
    action: str,
    ml_score: float | None,
    target_weight: float = 0.10,
    confidence: float = 0.75,
) -> RecommendationRecord:
    shadow = {}
    if ml_score is not None:
        shadow = {
            "models": {
                "ridge_return": {
                    "ret_60d": {
                        "score": ml_score,
                        "raw_score": ml_score - 0.5,
                        "calibrated": True,
                    }
                }
            }
        }
    return RecommendationRecord(
        as_of_date="2026-06-01",
        symbol=symbol,
        action=action,
        direction="bullish" if action != "TRIM" else "bearish",
        composite=0.2,
        confidence=confidence,
        target_weight=target_weight,
        current_weight=0.0,
        delta_pp=target_weight,
        target_shares=10,
        current_shares=0,
        delta_shares=10,
        limit_price=100.0,
        stop_loss=90.0,
        last_close=100.0,
        sma20=98.0,
        atr14=3.0,
        signal_age_days=0,
        price_source="fixture",
        notes=[],
        ml_shadow=shadow,
    )


def test_shadow_report_summarizes_factor_ml_and_disagreements():
    mod = _load_module()
    recs = {
        "2026-06-01": [
            _rec("ALLOW", action="BUY", ml_score=0.61, target_weight=0.10),
            _rec("BLOCK", action="BUY", ml_score=0.44, target_weight=0.10),
            _rec("NEW", action="SKIP", ml_score=0.72, target_weight=0.0),
            _rec("QUIET", action="SKIP", ml_score=0.33, target_weight=0.0),
        ]
    }
    outcomes = {
        "ALLOW": mod.Outcome(raw_return=0.10, benchmark_return=0.03),
        "BLOCK": mod.Outcome(raw_return=-0.10, benchmark_return=0.02),
        "NEW": mod.Outcome(raw_return=0.20, benchmark_return=0.04),
        "QUIET": mod.Outcome(raw_return=-0.02, benchmark_return=0.01),
    }
    rows = mod.build_shadow_rows(
        recs,
        model="ridge_return",
        horizon="ret_60d",
        outcome_provider=lambda rec: outcomes[rec.symbol],
    )

    summaries, decisions = mod.summarize_all_strategies(
        rows,
        threshold=0.55,
        hybrid_threshold=0.55,
        default_ml_weight=0.05,
        top_k_per_date=10,
        cost_bps=10.0,
    )
    by_strategy = {row["strategy"]: row for row in summaries}

    assert by_strategy["factor_production"]["selected"] == 2
    assert by_strategy["ml_gated_factor"]["selected"] == 1
    assert by_strategy["ml_only"]["selected"] == 2
    assert by_strategy["factor_production"]["alpha_hit_rate"] == pytest.approx(0.5)
    assert by_strategy["ml_gated_factor"]["alpha_hit_rate"] == pytest.approx(1.0)
    assert by_strategy["ml_only"]["alpha_hit_rate"] == pytest.approx(1.0)
    assert by_strategy["factor_production"]["cost_drag"] == pytest.approx(0.0002)
    assert len([d for d in decisions["ml_only"] if d.selected]) == 2

    disagreements = mod.disagreement_rows(rows, threshold=0.55)
    assert [(row["symbol"], row["reason"]) for row in disagreements] == [
        ("BLOCK", "ML blocks production BUY/ADD"),
        ("NEW", "ML would trigger where factor did not"),
    ]


def test_shadow_report_markdown_explains_pending_outcomes():
    mod = _load_module()
    rows = mod.build_shadow_rows(
        {"2026-06-01": [_rec("ALLOW", action="BUY", ml_score=0.61)]},
        model="ridge_return",
        horizon="ret_60d",
    )
    summaries, _ = mod.summarize_all_strategies(
        rows,
        threshold=0.55,
        hybrid_threshold=0.55,
        default_ml_weight=0.05,
        top_k_per_date=10,
        cost_bps=10.0,
    )
    md = mod.render_markdown(
        from_date="2026-06-01",
        to_date="2026-06-01",
        model="ridge_return",
        horizon="ret_60d",
        threshold=0.55,
        hybrid_threshold=0.55,
        benchmark="SPY",
        cost_bps=10.0,
        no_prices=True,
        rows=rows,
        summaries=summaries,
        disagreements=mod.disagreement_rows(rows, threshold=0.55),
    )

    assert "ML Gate shadow report" in md
    assert "Factor production" in md
    assert "ML-gated factor" in md
    assert "pending" in md


def test_shadow_report_cli_writes_no_price_report(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = tmp_path / "paper"
    write_recommendations_for_day(
        [_rec("ALLOW", action="BUY", ml_score=0.61)],
        base_dir=base_dir,
        as_of_date="2026-06-01",
    )
    output = tmp_path / "ml_gate.md"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "ml_gate_shadow_report.py"),
            "--from-date",
            "2026-06-01",
            "--to-date",
            "2026-06-01",
            "--base-dir",
            str(base_dir),
            "--output",
            str(output),
            "--no-prices",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "ML Gate shadow report" in output.read_text()
