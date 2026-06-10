"""Unit tests for portfolio/paper_trading.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio.paper_trading import (
    EXECUTIONS_SCHEMA_VERSION,
    RECOMMENDATIONS_SCHEMA_VERSION,
    ExecutionRecord,
    RecommendationRecord,
    append_execution,
    append_recommendation,
    executions_path,
    join_recommendations_to_executions,
    load_executions,
    load_executions_range,
    load_recommendations,
    load_recommendations_range,
    recommendations_path,
    unattributed_executions,
    write_recommendations_for_day,
)


def _rec(symbol: str, as_of: str, *, action: str = "BUY",
         delta_shares: int = 10, **kw) -> RecommendationRecord:
    base = dict(
        as_of_date=as_of,
        symbol=symbol,
        action=action,
        direction="bullish",
        composite=0.5,
        confidence=0.75,
        target_weight=0.09,
        current_weight=0.0,
        delta_pp=9.0,
        target_shares=delta_shares,
        current_shares=0,
        delta_shares=delta_shares,
        limit_price=145.50,
        stop_loss=130.00,
        last_close=148.00,
        sma20=146.00,
        atr14=4.5,
        signal_age_days=1,
        price_source="yfinance",
        notes=[],
    )
    base.update(kw)
    return RecommendationRecord(**base)


def _exec(symbol: str, trade_date: str, *, side: str = "BUY",
          shares: float = 10, **kw) -> ExecutionRecord:
    base = dict(
        trade_date=trade_date,
        symbol=symbol,
        side=side,
        shares=shares,
        fill_price=146.10,
    )
    base.update(kw)
    return ExecutionRecord(**base)


# ---------- paths ----------


def test_paths_segregate_by_date_and_kind(tmp_path):
    r = recommendations_path(tmp_path, "2026-06-07")
    e = executions_path(tmp_path, "2026-06-07")
    assert r.parent.name == "recommendations"
    assert e.parent.name == "executions"
    assert r.name == "2026-06-07.jsonl"


# ---------- round-trip writers + readers ----------


def test_append_recommendation_creates_dir_and_persists(tmp_path):
    rec = _rec("NVDA", "2026-06-07", ml_shadow={"elastic_logit": {"ret_60d": 0.61}})
    p = append_recommendation(rec, base_dir=tmp_path)
    assert p.exists()
    loaded = load_recommendations(tmp_path, "2026-06-07")
    assert len(loaded) == 1
    assert loaded[0].symbol == "NVDA"
    assert loaded[0].action == "BUY"
    assert loaded[0].ml_shadow["elastic_logit"]["ret_60d"] == 0.61
    assert loaded[0].schema_version == RECOMMENDATIONS_SCHEMA_VERSION


def test_write_recommendations_for_day_is_idempotent(tmp_path):
    # First write.
    write_recommendations_for_day(
        [_rec("NVDA", "2026-06-07"), _rec("AMD", "2026-06-07", action="ADD")],
        base_dir=tmp_path, as_of_date="2026-06-07",
    )
    assert len(load_recommendations(tmp_path, "2026-06-07")) == 2
    # Re-run with smaller set — file must REPLACE, not append.
    write_recommendations_for_day(
        [_rec("NVDA", "2026-06-07")],
        base_dir=tmp_path, as_of_date="2026-06-07",
    )
    after = load_recommendations(tmp_path, "2026-06-07")
    assert len(after) == 1
    assert after[0].symbol == "NVDA"


def test_append_execution_persists_multiple_trades_same_day(tmp_path):
    append_execution(_exec("NVDA", "2026-06-07"), base_dir=tmp_path)
    append_execution(_exec("AMD", "2026-06-07", shares=5), base_dir=tmp_path)
    loaded = load_executions(tmp_path, "2026-06-07")
    assert {e.symbol for e in loaded} == {"NVDA", "AMD"}
    assert all(e.schema_version == EXECUTIONS_SCHEMA_VERSION for e in loaded)


def test_load_recommendations_missing_returns_empty(tmp_path):
    assert load_recommendations(tmp_path, "1999-01-01") == []


def test_load_recommendations_skips_malformed_lines(tmp_path):
    path = recommendations_path(tmp_path, "2026-06-07")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json\n" + _serialize_rec(_rec("NVDA", "2026-06-07")) + "\n")
    loaded = load_recommendations(tmp_path, "2026-06-07")
    assert len(loaded) == 1
    assert loaded[0].symbol == "NVDA"


def _serialize_rec(rec: RecommendationRecord) -> str:
    import json
    from dataclasses import asdict
    return json.dumps(asdict(rec))


# ---------- range loaders ----------


def test_load_recommendations_range_filters_by_date(tmp_path):
    write_recommendations_for_day(
        [_rec("NVDA", "2026-06-01")], base_dir=tmp_path, as_of_date="2026-06-01",
    )
    write_recommendations_for_day(
        [_rec("AMD", "2026-06-05")], base_dir=tmp_path, as_of_date="2026-06-05",
    )
    write_recommendations_for_day(
        [_rec("AVGO", "2026-06-09")], base_dir=tmp_path, as_of_date="2026-06-09",
    )
    out = load_recommendations_range(tmp_path, from_date="2026-06-02", to_date="2026-06-07")
    assert set(out.keys()) == {"2026-06-05"}


def test_load_executions_range_filters_by_date(tmp_path):
    append_execution(_exec("NVDA", "2026-06-01"), base_dir=tmp_path)
    append_execution(_exec("AMD", "2026-06-05"), base_dir=tmp_path)
    out = load_executions_range(tmp_path, from_date="2026-06-02", to_date="2026-06-07")
    assert set(out.keys()) == {"2026-06-05"}


# ---------- join + divergence classification ----------


def test_join_explicit_ref_link_marks_followed(tmp_path):
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07", delta_shares=10)]}
    execs = {"2026-06-07": [_exec(
        "NVDA", "2026-06-07", shares=10, ref_recommendation_date="2026-06-07",
    )]}
    rows = join_recommendations_to_executions(recs, execs)
    assert len(rows) == 1
    assert rows[0]["divergence_type"] == "followed"
    assert rows[0]["executions"][0].symbol == "NVDA"


def test_join_same_day_fallback_link(tmp_path):
    """No ref_recommendation_date but same-day same-symbol trade with
    matching side counts as followed."""
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07", delta_shares=10)]}
    execs = {"2026-06-07": [_exec("NVDA", "2026-06-07", shares=10)]}  # no ref_date
    rows = join_recommendations_to_executions(recs, execs)
    assert rows[0]["divergence_type"] == "followed"


def test_join_partial_fill_classified_as_partial():
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07", delta_shares=100)]}
    execs = {"2026-06-07": [_exec("NVDA", "2026-06-07", shares=30)]}  # 30/100 = 30%
    rows = join_recommendations_to_executions(recs, execs)
    assert rows[0]["divergence_type"] == "partial"


def test_join_no_execution_classified_as_ignored():
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07")]}
    rows = join_recommendations_to_executions(recs, {})
    assert rows[0]["divergence_type"] == "ignored"


def test_join_hold_actions_classified_as_n_a():
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07", action="HOLD", delta_shares=0)]}
    rows = join_recommendations_to_executions(recs, {})
    assert rows[0]["divergence_type"] == "n_a"


def test_join_side_mismatch_does_not_attribute_as_followed():
    """BUY recommended; user SOLD same symbol same day → not a match."""
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07", action="BUY", delta_shares=10)]}
    execs = {"2026-06-07": [_exec("NVDA", "2026-06-07", side="SELL", shares=10)]}
    rows = join_recommendations_to_executions(recs, execs)
    assert rows[0]["divergence_type"] == "ignored"
    assert rows[0]["executions"] == []


# ---------- unattributed executions ----------


def test_unattributed_returns_trades_with_no_matching_recommendation():
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07")]}
    # Discretionary trade on a symbol NOT in recommendations.
    execs = {"2026-06-07": [_exec("TSLA", "2026-06-07", shares=20)]}
    out = unattributed_executions(recs, execs)
    assert len(out) == 1
    assert out[0].symbol == "TSLA"


def test_unattributed_excludes_ref_linked_executions():
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07")]}
    execs = {"2026-06-07": [_exec(
        "NVDA", "2026-06-07", ref_recommendation_date="2026-06-07",
    )]}
    out = unattributed_executions(recs, execs)
    assert out == []


def test_unattributed_excludes_same_day_same_symbol_executions():
    """Even without ref_recommendation_date, same-symbol same-date
    trades on a recommended ticker are attributed."""
    recs = {"2026-06-07": [_rec("NVDA", "2026-06-07")]}
    execs = {"2026-06-07": [_exec("NVDA", "2026-06-07")]}  # no ref_date
    out = unattributed_executions(recs, execs)
    assert out == []
