from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.analysis_only.state_store import StateStore


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(str(tmp_path / "state.sqlite"))


def _record(store: StateStore, symbol: str, as_of: str, iv30: float, iv60: float | None = None):
    store.record_iv_snapshot(
        symbol=symbol,
        as_of_date=as_of,
        atm_iv_30d=iv30,
        atm_iv_60d=iv60,
        atm_iv_90d=None,
        skew_25d_30d=None,
        term_slope_30_to_60=None,
    )


def test_record_and_read_back_single_row(store: StateStore):
    _record(store, "NVDA", "2026-01-09", 0.35)
    hist = store.get_iv_history("NVDA", before_date="2026-05-22")
    assert len(hist) == 1
    assert hist[0]["as_of_date"] == "2026-01-09"
    assert hist[0]["atm_iv_30d"] == 0.35


def test_history_query_is_strict_before(store: StateStore):
    # PIT correctness: a run with as_of=X must never see X's own row.
    _record(store, "NVDA", "2026-01-09", 0.35)
    _record(store, "NVDA", "2026-01-16", 0.40)
    hist = store.get_iv_history("NVDA", before_date="2026-01-16")
    assert [r["as_of_date"] for r in hist] == ["2026-01-09"]


def test_history_window_cuts_older_than_n_days(store: StateStore):
    _record(store, "NVDA", "2025-06-01", 0.20)
    _record(store, "NVDA", "2026-01-09", 0.35)
    hist = store.get_iv_history(
        "NVDA",
        before_date="2026-05-22",
        window_days=180,  # 2025-06-01 should drop out
    )
    assert [r["as_of_date"] for r in hist] == ["2026-01-09"]


def test_history_orders_ascending(store: StateStore):
    _record(store, "NVDA", "2026-03-01", 0.40)
    _record(store, "NVDA", "2026-01-09", 0.35)
    _record(store, "NVDA", "2026-02-13", 0.37)
    hist = store.get_iv_history("NVDA", before_date="2026-05-22")
    assert [r["as_of_date"] for r in hist] == [
        "2026-01-09",
        "2026-02-13",
        "2026-03-01",
    ]


def test_history_isolates_per_symbol(store: StateStore):
    _record(store, "NVDA", "2026-01-09", 0.35)
    _record(store, "AMD", "2026-01-09", 0.55)
    nvda_hist = store.get_iv_history("NVDA", before_date="2026-05-22")
    amd_hist = store.get_iv_history("AMD", before_date="2026-05-22")
    assert len(nvda_hist) == 1 and nvda_hist[0]["atm_iv_30d"] == 0.35
    assert len(amd_hist) == 1 and amd_hist[0]["atm_iv_30d"] == 0.55


def test_record_is_upsert_on_same_symbol_date(store: StateStore):
    _record(store, "NVDA", "2026-01-09", 0.35)
    # Rerunning the same date should overwrite, not duplicate.
    _record(store, "NVDA", "2026-01-09", 0.40)
    hist = store.get_iv_history("NVDA", before_date="2026-05-22")
    assert len(hist) == 1
    assert hist[0]["atm_iv_30d"] == 0.40


def test_symbol_match_is_case_insensitive(store: StateStore):
    _record(store, "nvda", "2026-01-09", 0.35)
    hist = store.get_iv_history("NVDA", before_date="2026-05-22")
    assert len(hist) == 1
