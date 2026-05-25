from __future__ import annotations

import json

from tradingagents.analysis_only import agent_review
from tradingagents.analysis_only.backtest import BacktestRecord
from tradingagents.analysis_only.reporting.markdown import render_markdown
from tradingagents.analysis_only.tuning import (
    CandidateConfig,
    CandidateEvaluation,
)


def _record(
    symbol: str,
    as_of: str,
    *,
    direction: str,
    composite: float,
    ret_20d: float,
) -> BacktestRecord:
    return BacktestRecord(
        symbol=symbol,
        as_of_date=as_of,
        direction=direction,
        confidence=0.7,
        composite_score=composite,
        forward_returns={"ret_20d": ret_20d},
        factor_scores=[
            {
                "factor": "signal",
                "pillar": "technical",
                "score": composite,
                "data_available": True,
            }
        ],
    )


def _candidate(candidate_id: str = "c1") -> CandidateConfig:
    return CandidateConfig(
        candidate_id=candidate_id,
        source="test",
        weights={"signal": 1.0},
        bullish_threshold=0.10,
        bearish_threshold=0.25,
        neutral_band=0.02,
        policy="equal_weight_bullish",
        max_per_name=0.20,
        max_long_exposure=0.50,
        top_n=3,
        stale_signal_decay=1.0,
    )


def _eval(candidate_id: str = "c1", score: float = 1.0) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=_candidate(candidate_id),
        slice_name="search",
        rejected=False,
        rejection_reasons=(),
        score=score,
        bullish_20d_hit_rate=0.6,
        bullish_20d_mean_return=0.03,
        bullish_20d_count=20,
        portfolio_cagr=0.2,
        benchmark_cagr=0.1,
        excess_cagr=0.1,
        sharpe=1.2,
        benchmark_sharpe=0.8,
        sharpe_spread=0.4,
        max_drawdown=-0.1,
        turnover=1.0,
        avg_long_exposure=0.4,
        avg_max_position=0.12,
    )


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str):
        self._content = content

    def get_llm(self):
        return _FakeLLM(self._content)


def _patch_llm(monkeypatch, payload: dict):
    content = json.dumps(payload)

    def _factory(provider, model, base_url=None, **_kwargs):
        return _FakeClient(content)

    monkeypatch.setattr(
        "tradingagents.llm_clients.factory.create_llm_client",
        _factory,
    )


def _valid_review_payload() -> dict:
    return {
        "factor_hypotheses": [
            {
                "name": "earnings revision breadth",
                "rationale": "Graph researchers repeatedly cited estimate risk.",
                "required_data": ["analyst estimate revisions"],
                "proposed_factor_formula": "net_up_revisions_30d / total_revisions_30d",
                "expected_direction": "bullish",
                "validation_test": "Measure 20d IC out of sample.",
            }
        ],
        "candidate_risk_critiques": [
            {
                "candidate_id": "c1",
                "concern": "Concentration rises when only one name qualifies.",
                "affected_knobs": ["max_per_name", "top_n"],
                "failure_mode": "Single-name drawdown dominates the slice.",
                "severity": "medium",
                "mitigation": "Cap max_per_name lower in holdout validation.",
            }
        ],
        "overfit_explanations": [
            {
                "candidate_id": "c1",
                "evidence": "Search score is materially above holdout score.",
                "suspected_overfit_mechanism": "Threshold tuned to a narrow trend regime.",
                "confidence": 0.7,
                "suggested_gate": "Require positive excess CAGR in each regime.",
            }
        ],
        "feature_recommendations": [
            {
                "feature_or_dataset": "estimate revisions",
                "reason": "It directly tests the researcher concern.",
                "priority": "high",
                "expected_signal_type": "fundamental momentum",
                "validation_plan": "Backtest revision breadth against 20d returns.",
            }
        ],
    }


def test_strip_code_fences_handles_json_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert agent_review._strip_code_fences(raw) == "{\"a\": 1}"


def test_build_review_prompt_contains_required_payload():
    prompt, payload = agent_review.build_review_prompt(
        graph_contexts=[{"symbol": "NVDA", "status": "ok"}],
        candidates=[{"candidate": {"candidate_id": "c1"}}],
    )
    assert "factor_hypotheses" in prompt
    assert payload["graph_contexts"][0]["symbol"] == "NVDA"
    assert payload["tuner_candidates"][0]["candidate"]["candidate_id"] == "c1"


def test_select_representative_contexts_is_deterministic_and_deduped():
    originals = [
        _record("AAA", "2026-01-01", direction="neutral", composite=0.01, ret_20d=0.01),
        _record("BBB", "2026-01-08", direction="bullish", composite=0.2, ret_20d=-0.08),
        _record("CCC", "2026-01-15", direction="bearish", composite=-0.3, ret_20d=0.05),
    ]
    rebuilt = [
        _record("AAA", "2026-01-01", direction="bullish", composite=0.6, ret_20d=0.10),
        _record("BBB", "2026-01-08", direction="bullish", composite=0.2, ret_20d=-0.08),
        _record("CCC", "2026-01-15", direction="bearish", composite=-0.3, ret_20d=0.05),
    ]
    contexts = agent_review.select_representative_contexts(
        originals,
        rebuilt,
        max_contexts=5,
    )
    keys = [(c["symbol"], c["as_of_date"]) for c in contexts]
    assert len(keys) == len(set(keys))
    assert keys[0] == ("AAA", "2026-01-01")
    assert any("largest_direction_change" in c["selection_reasons"] for c in contexts)


def test_generate_structured_review_validates_llm_output(monkeypatch):
    _patch_llm(monkeypatch, _valid_review_payload())
    block = agent_review.generate_structured_review(
        graph_contexts=[{"symbol": "NVDA", "status": "ok"}],
        candidates=[{"candidate": {"candidate_id": "c1"}}],
    )
    assert block["status"] == "ok"
    assert block["analysis"]["factor_hypotheses"][0]["name"] == "earnings revision breadth"


def test_generate_structured_review_reports_schema_errors(monkeypatch):
    _patch_llm(monkeypatch, {"factor_hypotheses": [{"name": "bad"}]})
    block = agent_review.generate_structured_review(
        graph_contexts=[{"symbol": "NVDA", "status": "ok"}],
    )
    assert block["status"] == "llm_schema_validation_error"


def test_run_tuning_agent_review_writes_valid_block_without_real_graph(monkeypatch):
    _patch_llm(monkeypatch, _valid_review_payload())

    def _fake_graph_contexts(contexts, **_kwargs):
        return [
            {
                "symbol": c["symbol"],
                "as_of_date": c["as_of_date"],
                "selection_reasons": c["selection_reasons"],
                "status": "ok",
                "final_trade_decision": "Hold",
            }
            for c in contexts
        ]

    monkeypatch.setattr(agent_review, "run_graph_contexts", _fake_graph_contexts)
    holdout_records = [
        _record("AAA", "2026-01-01", direction="bullish", composite=0.6, ret_20d=0.1),
        _record("BBB", "2026-01-08", direction="bullish", composite=0.4, ret_20d=-0.1),
    ]
    search_eval = _eval("c1")
    block = agent_review.run_tuning_agent_review(
        search_evals=[search_eval],
        holdout_evals={"c1": search_eval},
        holdout_records=holdout_records,
        top_n=1,
        max_contexts=2,
    )
    assert block["status"] == "ok"
    assert block["review_type"] == "tuning"
    assert block["best_candidate_id"] == "c1"


def test_markdown_renderer_handles_tradingagents_review_block():
    payload = {
        "symbol": "NVDA",
        "as_of_date": "2026-05-22",
        "horizon": "swing_1_4_weeks",
        "generated_at_utc": "2026-05-22T00:00:00Z",
        "direction": "bullish",
        "confidence": 0.7,
        "thesis": "Test thesis",
        "key_features": {
            "tradingagents_review": {
                "enabled": True,
                "status": "ok",
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "analysis": _valid_review_payload(),
            }
        },
    }
    md = render_markdown(payload)
    assert "## TradingAgents review" in md
    assert "earnings revision breadth" in md
