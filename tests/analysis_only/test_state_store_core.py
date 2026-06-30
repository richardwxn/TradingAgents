from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.analysis_only.state_store import (
    DailyLLMUsage,
    StateStore,
    SymbolState,
)


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(str(tmp_path / "state.sqlite"))


# --- construction / schema ------------------------------------------------


def test_init_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "state.sqlite"
    StateStore(str(nested))
    assert nested.parent.exists()
    assert nested.exists()


def test_init_is_idempotent(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    first = StateStore(str(db))
    first.upsert_symbol_state("AAPL", last_price=1.0, last_signal="long")
    # Re-opening the same DB must not wipe data or fail re-creating tables.
    second = StateStore(str(db))
    assert second.get_symbol_state("AAPL").last_price == pytest.approx(1.0)


# --- symbol state ---------------------------------------------------------


def test_get_symbol_state_default_when_missing(store: StateStore):
    state = store.get_symbol_state("NVDA")
    assert isinstance(state, SymbolState)
    assert state.symbol == "NVDA"
    assert state.last_run_time is None
    assert state.last_price is None
    assert state.last_signal is None
    assert state.last_composite_score is None
    assert state.last_options_unusual_count is None
    assert state.daily_summary_date is None
    assert state.daily_summary_json is None
    assert state.last_as_of_date is None
    assert state.last_confidence is None
    assert state.last_factor_scores is None
    assert state.last_thesis is None


def test_upsert_and_read_back_basic(store: StateStore):
    store.upsert_symbol_state(
        symbol="NVDA",
        last_price=123.45,
        last_signal="long",
        last_composite_score=0.42,
        last_options_unusual_count=7,
    )
    state = store.get_symbol_state("NVDA")
    assert state.last_price == pytest.approx(123.45)
    assert state.last_signal == "long"
    assert state.last_composite_score == pytest.approx(0.42)
    assert state.last_options_unusual_count == 7
    # last_run_time is stamped on every upsert.
    assert state.last_run_time is not None


def test_upsert_persists_json_fields(store: StateStore):
    summary = {"headline": "good quarter", "score": 3}
    factors = [{"name": "value", "z": 1.2}, {"name": "momentum", "z": -0.5}]
    store.upsert_symbol_state(
        symbol="AAPL",
        last_price=10.0,
        last_signal="neutral",
        daily_summary_date="2026-01-09",
        daily_summary_json=summary,
        last_as_of_date="2026-01-09",
        last_confidence=0.66,
        last_factor_scores=factors,
        last_thesis="solid",
    )
    state = store.get_symbol_state("AAPL")
    assert state.daily_summary_date == "2026-01-09"
    assert state.daily_summary_json == summary
    assert state.last_as_of_date == "2026-01-09"
    assert state.last_confidence == pytest.approx(0.66)
    assert state.last_factor_scores == factors
    assert state.last_thesis == "solid"


def test_upsert_updates_core_columns(store: StateStore):
    store.upsert_symbol_state("NVDA", last_price=100.0, last_signal="long")
    store.upsert_symbol_state(
        "NVDA",
        last_price=200.0,
        last_signal="short",
        last_composite_score=-0.3,
        last_options_unusual_count=2,
    )
    state = store.get_symbol_state("NVDA")
    assert state.last_price == pytest.approx(200.0)
    assert state.last_signal == "short"
    assert state.last_composite_score == pytest.approx(-0.3)
    assert state.last_options_unusual_count == 2


def test_upsert_coalesce_preserves_existing_optional_fields(store: StateStore):
    # First write sets the optional/json fields.
    store.upsert_symbol_state(
        symbol="NVDA",
        last_price=100.0,
        last_signal="long",
        daily_summary_date="2026-01-09",
        daily_summary_json={"a": 1},
        last_thesis="thesis-v1",
    )
    # Second write omits them (passes None) -> COALESCE keeps prior values.
    store.upsert_symbol_state(
        symbol="NVDA",
        last_price=110.0,
        last_signal="long",
    )
    state = store.get_symbol_state("NVDA")
    assert state.last_price == pytest.approx(110.0)
    assert state.daily_summary_date == "2026-01-09"
    assert state.daily_summary_json == {"a": 1}
    assert state.last_thesis == "thesis-v1"


def test_symbol_state_isolated_per_symbol(store: StateStore):
    store.upsert_symbol_state("AAPL", last_price=1.0, last_signal="long")
    store.upsert_symbol_state("MSFT", last_price=2.0, last_signal="short")
    assert store.get_symbol_state("AAPL").last_signal == "long"
    assert store.get_symbol_state("MSFT").last_signal == "short"


# --- news tracking --------------------------------------------------------


def test_get_seen_news_ids_empty(store: StateStore):
    assert store.get_seen_news_ids("NVDA") == set()


def test_mark_and_get_news_seen(store: StateStore):
    store.mark_news_seen("NVDA", ["id1", "id2", "id3"])
    assert store.get_seen_news_ids("NVDA") == {"id1", "id2", "id3"}


def test_mark_news_seen_empty_is_noop(store: StateStore):
    store.mark_news_seen("NVDA", [])
    assert store.get_seen_news_ids("NVDA") == set()


def test_mark_news_seen_is_idempotent(store: StateStore):
    store.mark_news_seen("NVDA", ["id1", "id2"])
    # Re-inserting overlapping ids must not raise (INSERT OR IGNORE) or dup.
    store.mark_news_seen("NVDA", ["id2", "id3"])
    assert store.get_seen_news_ids("NVDA") == {"id1", "id2", "id3"}


def test_news_seen_isolated_per_symbol(store: StateStore):
    store.mark_news_seen("NVDA", ["n1"])
    store.mark_news_seen("AMD", ["a1"])
    assert store.get_seen_news_ids("NVDA") == {"n1"}
    assert store.get_seen_news_ids("AMD") == {"a1"}


def test_news_ids_coerced_to_str(store: StateStore):
    store.mark_news_seen("NVDA", [123, 456])  # type: ignore[list-item]
    assert store.get_seen_news_ids("NVDA") == {"123", "456"}


# --- filing tracking ------------------------------------------------------


def test_get_last_seen_filing_none_when_missing(store: StateStore):
    assert store.get_last_seen_filing_accession("NVDA") is None


def test_set_and_get_filing_accession(store: StateStore):
    store.set_last_seen_filing_accession("NVDA", "0000320193-26-000010")
    assert (
        store.get_last_seen_filing_accession("NVDA")
        == "0000320193-26-000010"
    )


def test_set_filing_accession_upserts(store: StateStore):
    store.set_last_seen_filing_accession("NVDA", "acc-1")
    store.set_last_seen_filing_accession("NVDA", "acc-2")
    assert store.get_last_seen_filing_accession("NVDA") == "acc-2"


def test_filing_accession_isolated_per_symbol(store: StateStore):
    store.set_last_seen_filing_accession("NVDA", "nv-1")
    store.set_last_seen_filing_accession("AMD", "amd-1")
    assert store.get_last_seen_filing_accession("NVDA") == "nv-1"
    assert store.get_last_seen_filing_accession("AMD") == "amd-1"


# --- LLM usage caps -------------------------------------------------------


def test_get_llm_usage_default_zero(store: StateStore):
    usage = store.get_llm_usage("2026-01-09")
    assert isinstance(usage, DailyLLMUsage)
    assert usage.usage_date == "2026-01-09"
    assert usage.calls == 0
    assert usage.est_input_tokens == 0
    assert usage.est_output_tokens == 0


def test_add_llm_usage_creates_row(store: StateStore):
    store.add_llm_usage("2026-01-09", calls=2, est_input_tokens=5000, est_output_tokens=1600)
    usage = store.get_llm_usage("2026-01-09")
    assert usage.calls == 2
    assert usage.est_input_tokens == 5000
    assert usage.est_output_tokens == 1600


def test_add_llm_usage_accumulates_same_day(store: StateStore):
    store.add_llm_usage("2026-01-09", calls=2, est_input_tokens=5000, est_output_tokens=1600)
    store.add_llm_usage("2026-01-09", calls=3, est_input_tokens=7500, est_output_tokens=2400)
    usage = store.get_llm_usage("2026-01-09")
    assert usage.calls == 5
    assert usage.est_input_tokens == 12500
    assert usage.est_output_tokens == 4000


def test_add_llm_usage_isolated_per_day(store: StateStore):
    store.add_llm_usage("2026-01-09", calls=2, est_input_tokens=5000, est_output_tokens=1600)
    store.add_llm_usage("2026-01-10", calls=1, est_input_tokens=2500, est_output_tokens=800)
    day1 = store.get_llm_usage("2026-01-09")
    day2 = store.get_llm_usage("2026-01-10")
    assert day1.calls == 2
    assert day2.calls == 1
    assert day2.est_input_tokens == 2500


def test_state_survives_reconnect(tmp_path: Path):
    db = str(tmp_path / "state.sqlite")
    store1 = StateStore(db)
    store1.add_llm_usage("2026-01-09", calls=1, est_input_tokens=10, est_output_tokens=5)
    store1.mark_news_seen("NVDA", ["x"])
    store1.set_last_seen_filing_accession("NVDA", "acc")
    store1.upsert_symbol_state("NVDA", last_price=9.0, last_signal="long")

    store2 = StateStore(db)
    assert store2.get_llm_usage("2026-01-09").calls == 1
    assert store2.get_seen_news_ids("NVDA") == {"x"}
    assert store2.get_last_seen_filing_accession("NVDA") == "acc"
    assert store2.get_symbol_state("NVDA").last_price == pytest.approx(9.0)
