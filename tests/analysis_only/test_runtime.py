from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.analysis_only.config import AppConfig, LLMConfig, RuntimeConfig
from tradingagents.analysis_only.runtime import AnalysisRuntime
from tradingagents.analysis_only.state_store import SymbolState


def _make_config(tmp_path: Path, **runtime_overrides) -> AppConfig:
    runtime = RuntimeConfig(**runtime_overrides) if runtime_overrides else RuntimeConfig()
    return AppConfig(
        watchlist=["AAPL", "MSFT"],
        runtime=runtime,
        state_db_path=str(tmp_path / "state.sqlite"),
        output_dir=str(tmp_path / "out"),
    )


@pytest.fixture
def runtime(tmp_path: Path) -> AnalysisRuntime:
    return AnalysisRuntime(_make_config(tmp_path))


def _empty_state(symbol: str = "AAPL", **kwargs) -> SymbolState:
    base = dict(
        symbol=symbol,
        last_run_time=None,
        last_price=None,
        last_signal=None,
        last_composite_score=None,
        last_options_unusual_count=None,
        daily_summary_date=None,
        daily_summary_json=None,
    )
    base.update(kwargs)
    return SymbolState(**base)


# --- construction ---------------------------------------------------------


def test_construction_wires_config_and_state(tmp_path: Path):
    cfg = _make_config(tmp_path)
    rt = AnalysisRuntime(cfg)
    assert rt.config is cfg
    # StateStore created the backing DB file.
    assert Path(cfg.state_db_path).exists()
    # Providers are constructed but no network call has happened.
    assert rt.market_status is not None
    assert rt.news_provider is not None
    assert rt.filings_provider is not None


# --- _extract_news_ids ----------------------------------------------------


def test_extract_news_ids_prefers_id_then_fallbacks(runtime: AnalysisRuntime):
    items = [
        {"id": "ID1", "uuid": "u1", "title": "t1"},
        {"uuid": "U2", "title": "t2"},
        {"article_url": "http://x/3", "title": "t3"},
        {"title": "only-title"},
    ]
    assert runtime._extract_news_ids(items) == ["ID1", "U2", "http://x/3", "only-title"]


def test_extract_news_ids_skips_items_without_identifier(runtime: AnalysisRuntime):
    items = [{"description": "no id here"}, {"id": "keep"}]
    assert runtime._extract_news_ids(items) == ["keep"]


def test_extract_news_ids_coerces_to_str(runtime: AnalysisRuntime):
    assert runtime._extract_news_ids([{"id": 42}]) == ["42"]


# --- _delta_gate_reasons --------------------------------------------------


def test_delta_gate_no_reasons_when_no_prior_state(runtime: AnalysisRuntime):
    prev = _empty_state()
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": 0.9, "options_unusual_count": 10},
        new_direction="long",
    )
    assert reasons == []


def test_delta_gate_composite_shift(runtime: AnalysisRuntime):
    prev = _empty_state(last_composite_score=0.0)
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": 0.5, "options_unusual_count": None},
        new_direction="long",
    )
    assert "composite_shift" in reasons


def test_delta_gate_composite_below_threshold(runtime: AnalysisRuntime):
    prev = _empty_state(last_composite_score=0.0)
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": 0.1, "options_unusual_count": None},
        new_direction="long",
    )
    assert "composite_shift" not in reasons


def test_delta_gate_options_unusual_jump(runtime: AnalysisRuntime):
    prev = _empty_state(last_options_unusual_count=1)
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": None, "options_unusual_count": 3},
        new_direction="long",
    )
    assert "options_unusual_jump" in reasons


def test_delta_gate_options_jump_below_threshold(runtime: AnalysisRuntime):
    prev = _empty_state(last_options_unusual_count=1)
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": None, "options_unusual_count": 2},
        new_direction="long",
    )
    assert "options_unusual_jump" not in reasons


def test_delta_gate_signal_change(runtime: AnalysisRuntime):
    prev = _empty_state(last_signal="short")
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": None, "options_unusual_count": None},
        new_direction="long",
    )
    assert "signal_change" in reasons


def test_delta_gate_no_signal_change_when_same(runtime: AnalysisRuntime):
    prev = _empty_state(last_signal="long")
    reasons = runtime._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": None, "options_unusual_count": None},
        new_direction="long",
    )
    assert "signal_change" not in reasons


def test_delta_gate_signal_change_disabled(tmp_path: Path):
    rt = AnalysisRuntime(_make_config(tmp_path, llm_gate_signal_change_only=False))
    prev = _empty_state(last_signal="short")
    reasons = rt._delta_gate_reasons(
        prev_state=prev,
        new_metrics={"composite_score": None, "options_unusual_count": None},
        new_direction="long",
    )
    assert "signal_change" not in reasons


# --- _extract_report_metrics ----------------------------------------------


def test_extract_report_metrics(runtime: AnalysisRuntime):
    report = SimpleNamespace(
        to_json_dict=lambda: {
            "key_features": {
                "technical": {"close": 101.5},
                "model_scoring": {"composite_score": 0.33},
                "options_flow": {"unusual_count": 4},
            }
        }
    )
    metrics = runtime._extract_report_metrics(report)
    assert metrics["close"] == pytest.approx(101.5)
    assert metrics["composite_score"] == pytest.approx(0.33)
    assert metrics["options_unusual_count"] == 4


def test_extract_report_metrics_missing_sections_defaults(runtime: AnalysisRuntime):
    report = SimpleNamespace(to_json_dict=lambda: {"key_features": {}})
    metrics = runtime._extract_report_metrics(report)
    assert metrics["close"] is None
    assert metrics["composite_score"] is None
    assert metrics["options_unusual_count"] == 0


# --- _apply_llm_quota -----------------------------------------------------


def test_apply_quota_empty_candidates(runtime: AnalysisRuntime):
    assert runtime._apply_llm_quota([], "2026-01-09") == set()


def test_apply_quota_capped_by_per_run(tmp_path: Path):
    rt = AnalysisRuntime(_make_config(tmp_path, llm_max_calls_per_run=2))
    chosen = rt._apply_llm_quota(["A", "B", "C", "D"], "2026-01-09")
    # Per-run cap of 2 selects the first two candidates.
    assert chosen == {"A", "B"}


def test_apply_quota_capped_by_daily_calls(tmp_path: Path):
    rt = AnalysisRuntime(
        _make_config(
            tmp_path,
            llm_max_calls_per_run=10,
            llm_max_calls_per_day=3,
        )
    )
    rt.state.add_llm_usage("2026-01-09", calls=2, est_input_tokens=0, est_output_tokens=0)
    # Only 1 daily call remaining.
    chosen = rt._apply_llm_quota(["A", "B", "C"], "2026-01-09")
    assert chosen == {"A"}


def test_apply_quota_zero_when_daily_exhausted(tmp_path: Path):
    rt = AnalysisRuntime(_make_config(tmp_path, llm_max_calls_per_day=3))
    rt.state.add_llm_usage("2026-01-09", calls=3, est_input_tokens=0, est_output_tokens=0)
    assert rt._apply_llm_quota(["A", "B"], "2026-01-09") == set()


def test_apply_quota_capped_by_input_tokens(tmp_path: Path):
    rt = AnalysisRuntime(
        _make_config(
            tmp_path,
            llm_max_calls_per_run=10,
            llm_max_calls_per_day=10,
            llm_est_input_tokens_per_call=1000,
            llm_max_est_input_tokens_per_day=2500,
            llm_max_est_output_tokens_per_day=10**9,
        )
    )
    # 2500 // 1000 = 2 calls allowed by the input-token budget.
    chosen = rt._apply_llm_quota(["A", "B", "C", "D"], "2026-01-09")
    assert chosen == {"A", "B"}


# --- _select_llm_symbols --------------------------------------------------


def test_select_llm_symbols_off_mode(tmp_path: Path):
    cfg = _make_config(tmp_path)
    cfg.llm = LLMConfig(mode="off")
    rt = AnalysisRuntime(cfg)
    chosen = rt._select_llm_symbols(["A", "B"], {"A": ["x"]}, "2026-01-09")
    assert chosen == set()


def test_select_llm_symbols_always_mode(tmp_path: Path):
    cfg = _make_config(tmp_path)
    cfg.llm = LLMConfig(mode="always")
    cfg.runtime.llm_max_calls_per_run = 10
    rt = AnalysisRuntime(cfg)
    chosen = rt._select_llm_symbols(["A", "B"], {}, "2026-01-09")
    # always mode ignores gate reasons; all candidates eligible (subject to quota).
    assert chosen == {"A", "B"}


def test_select_llm_symbols_selective_only_gated(tmp_path: Path):
    cfg = _make_config(tmp_path)
    cfg.llm = LLMConfig(mode="selective")
    cfg.runtime.llm_max_calls_per_run = 10
    rt = AnalysisRuntime(cfg)
    gate_map = {"A": ["new_headlines"], "B": [], "C": ["price_move", "new_filing"]}
    chosen = rt._select_llm_symbols(["A", "B", "C"], gate_map, "2026-01-09")
    assert chosen == {"A", "C"}
    assert "B" not in chosen


def test_select_llm_symbols_selective_prioritizes_more_reasons(tmp_path: Path):
    cfg = _make_config(tmp_path)
    cfg.llm = LLMConfig(mode="selective")
    cfg.runtime.llm_max_calls_per_run = 1
    rt = AnalysisRuntime(cfg)
    gate_map = {"A": ["one"], "C": ["one", "two", "three"]}
    chosen = rt._select_llm_symbols(["A", "C"], gate_map, "2026-01-09")
    # With a 1-call budget, the symbol with the most gate reasons wins.
    assert chosen == {"C"}


# --- _get_quota_snapshot --------------------------------------------------


def test_quota_snapshot_reflects_usage_and_limits(tmp_path: Path):
    rt = AnalysisRuntime(_make_config(tmp_path, llm_max_calls_per_day=30))
    rt.state.add_llm_usage("2026-01-09", calls=4, est_input_tokens=1000, est_output_tokens=400)
    snap = rt._get_quota_snapshot("2026-01-09")
    assert snap["usage_date"] == "2026-01-09"
    assert snap["calls_used"] == 4
    assert snap["calls_limit"] == 30
    assert snap["est_input_tokens_used"] == 1000
    assert snap["est_output_tokens_used"] == 400


# --- _write_gate_diagnostics ----------------------------------------------


def test_write_gate_diagnostics_writes_json(runtime: AnalysisRuntime, tmp_path: Path):
    out = tmp_path / "diag_out"
    out.mkdir()
    run_time = datetime(2026, 1, 9, 14, 30, 0, tzinfo=timezone.utc)
    diagnostics = [{"symbol": "AAPL", "gate_reasons": ["x"]}]
    path = runtime._write_gate_diagnostics(out, diagnostics, run_time)
    assert path.exists()
    assert path.parent.name == "_runtime"
    assert path.name == "gate_diagnostics_20260109T143000Z.json"
    payload = json.loads(path.read_text())
    assert payload["run_time_utc"] == run_time.isoformat()
    assert payload["llm_mode"] == runtime.config.llm.mode
    assert payload["diagnostics"] == diagnostics
