"""Phase 6: unit tests for the LLM critic module.

These tests exercise the deterministic plumbing (payload reduction,
prompt hashing, schema validation, code-fence stripping). The actual
LLM call in `run_critic` is monkey-patched: we never hit a real
provider from CI.
"""
from __future__ import annotations

import json

import pytest

from tradingagents.analysis_only import llm_critic


@pytest.fixture
def sample_report() -> dict:
    return {
        "symbol": "NVDA",
        "as_of_date": "2026-05-22",
        "summary": {
            "direction": "bullish",
            "confidence": 0.72,
            "composite_score": 0.41,
            "coverage": 0.87,
        },
        "pillar_scores": {"trend": 0.6, "fundamentals": 0.3},
        "factor_scores": [
            {
                "factor": "trend_price_vs_sma20",
                "pillar": "trend",
                "bucket": "bullish",
                "score": 0.8,
                "weighted_score": 0.064,
                "rationale": "Price above SMA20 by 4%",
                "data_available": True,
            },
            {
                "factor": "broken_factor",
                "weighted_score": None,
                "data_available": False,
            },
        ],
        "technicals": {"close": 950.0, "rsi_14": 62.0, "macd_hist": 1.2},
        "fundamentals": {"trailing_pe": 50.0},
        "options_flow": {"unusual_count": 3, "atm_iv_30d": 0.35},
        "risk_flags": ["Earnings within 7 days"],
        "earnings_calendar": {"next_earnings_date": "2026-05-28"},
        "data_quality": {"pit_warnings": []},
    }


def test_build_payload_only_includes_active_factors(sample_report):
    payload = llm_critic._build_payload(sample_report)
    factors = [f["factor"] for f in payload["top_factors"]]
    assert "broken_factor" not in factors
    assert "trend_price_vs_sma20" in factors


def test_build_payload_includes_required_keys(sample_report):
    payload = llm_critic._build_payload(sample_report)
    for k in (
        "symbol", "as_of_date", "direction", "confidence",
        "composite_score", "pillar_scores", "coverage", "top_factors",
        "snapshot", "options_flow", "risk_flags", "earnings_calendar",
        "data_quality",
    ):
        assert k in payload, f"missing key: {k}"


def test_build_critic_prompt_is_deterministic(sample_report):
    prompt_a, payload_a = llm_critic.build_critic_prompt(sample_report)
    prompt_b, payload_b = llm_critic.build_critic_prompt(sample_report)
    assert prompt_a == prompt_b
    assert payload_a == payload_b
    # Hashing is stable
    assert llm_critic._prompt_hash(prompt_a) == llm_critic._prompt_hash(prompt_b)


def test_strip_code_fences_handles_json_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert llm_critic._strip_code_fences(raw) == "{\"a\": 1}"


def test_strip_code_fences_handles_bare_fence():
    raw = "```\n{\"a\": 1}\n```"
    assert llm_critic._strip_code_fences(raw) == "{\"a\": 1}"


def test_critic_schema_rejects_positive_confidence_adjustment():
    schema = llm_critic._build_critic_schema()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({
            "factor_blindspots": [],
            "invalidation_prob_30d": 0.3,
            "confidence_adjustment": 0.05,
            "veto": False,
            "veto_reason": None,
        })


def test_critic_schema_rejects_too_negative_confidence_adjustment():
    schema = llm_critic._build_critic_schema()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({
            "factor_blindspots": [],
            "invalidation_prob_30d": 0.3,
            "confidence_adjustment": -0.21,
            "veto": False,
            "veto_reason": None,
        })


def test_critic_schema_rejects_invalidation_prob_out_of_range():
    schema = llm_critic._build_critic_schema()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({
            "factor_blindspots": [],
            "invalidation_prob_30d": 1.2,
            "confidence_adjustment": 0.0,
            "veto": False,
            "veto_reason": None,
        })


def test_critic_schema_requires_veto_reason_when_veto_true():
    schema = llm_critic._build_critic_schema()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({
            "factor_blindspots": [],
            "invalidation_prob_30d": 0.4,
            "confidence_adjustment": -0.1,
            "veto": True,
            "veto_reason": None,
        })


def test_critic_schema_accepts_well_formed_output():
    schema = llm_critic._build_critic_schema()
    out = schema.model_validate({
        "factor_blindspots": ["ignored peer de-rating"],
        "invalidation_prob_30d": 0.35,
        "confidence_adjustment": -0.05,
        "veto": False,
        "veto_reason": None,
    })
    assert out.invalidation_prob_30d == 0.35
    assert out.confidence_adjustment == -0.05
    assert out.veto is False


def test_critic_schema_accepts_valid_veto():
    schema = llm_critic._build_critic_schema()
    out = schema.model_validate({
        "factor_blindspots": [],
        "invalidation_prob_30d": 0.7,
        "confidence_adjustment": -0.2,
        "veto": True,
        "veto_reason": "Earnings inside window, composite contradicts snapshot.",
    })
    assert out.veto is True


def test_critic_schema_rejects_too_many_blindspots():
    schema = llm_critic._build_critic_schema()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({
            "factor_blindspots": ["a", "b", "c", "d", "e", "f"],
            "invalidation_prob_30d": 0.3,
            "confidence_adjustment": 0.0,
            "veto": False,
            "veto_reason": None,
        })


# ---------- run_critic (monkeypatched LLM) ----------


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str) -> _FakeResponse:
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, llm: _FakeLLM):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _patch_llm(monkeypatch, response_content: str):
    fake_llm = _FakeLLM(response_content)

    def _factory(provider, model, base_url=None, **_kwargs):
        return _FakeClient(fake_llm)

    monkeypatch.setattr(
        "tradingagents.llm_clients.factory.create_llm_client", _factory
    )


def test_run_critic_returns_validated_output_on_clean_response(
    monkeypatch, sample_report,
):
    payload = {
        "factor_blindspots": ["ignored peer de-rating"],
        "invalidation_prob_30d": 0.35,
        "confidence_adjustment": -0.05,
        "veto": False,
        "veto_reason": None,
    }
    _patch_llm(monkeypatch, json.dumps(payload))
    block = llm_critic.run_critic(
        sample_report, provider="openai", model="gpt-5.4-mini",
    )
    assert block["status"] == "ok"
    assert block["model"] == "gpt-5.4-mini"
    assert block["prompt_version"] == llm_critic.CRITIC_PROMPT_VERSION
    assert len(block["prompt_hash"]) == 16
    assert block["output"]["confidence_adjustment"] == -0.05
    assert block["output"]["veto"] is False


def test_run_critic_handles_code_fenced_response(monkeypatch, sample_report):
    payload = {
        "factor_blindspots": [],
        "invalidation_prob_30d": 0.4,
        "confidence_adjustment": 0.0,
        "veto": False,
        "veto_reason": None,
    }
    raw = "```json\n" + json.dumps(payload) + "\n```"
    _patch_llm(monkeypatch, raw)
    block = llm_critic.run_critic(
        sample_report, provider="openai", model="gpt-5.4-mini",
    )
    assert block["status"] == "ok"


def test_run_critic_reports_json_parse_error(monkeypatch, sample_report):
    _patch_llm(monkeypatch, "not valid json {")
    block = llm_critic.run_critic(
        sample_report, provider="openai", model="gpt-5.4-mini",
    )
    assert block["status"] == "llm_json_parse_error"
    assert "raw_response" in block


def test_run_critic_reports_schema_error_on_invalid_payload(
    monkeypatch, sample_report,
):
    # confidence_adjustment positive -> schema rejects
    bad = {
        "factor_blindspots": [],
        "invalidation_prob_30d": 0.3,
        "confidence_adjustment": 0.5,
        "veto": False,
        "veto_reason": None,
    }
    _patch_llm(monkeypatch, json.dumps(bad))
    block = llm_critic.run_critic(
        sample_report, provider="openai", model="gpt-5.4-mini",
    )
    assert block["status"] == "llm_schema_validation_error"


def test_critic_output_or_none_helper():
    assert llm_critic.critic_output_or_none(None) is None
    assert llm_critic.critic_output_or_none({}) is None
    assert llm_critic.critic_output_or_none({"status": "llm_init_error"}) is None
    out = {"status": "ok", "output": {"veto": False}}
    assert llm_critic.critic_output_or_none(out) == {"veto": False}


# ---------- Phase 7: multi-model disagreement ----------


def _critic_out(*, conf=-0.05, inv=0.4, blindspots=None, veto=False):
    return {
        "confidence_adjustment": conf,
        "invalidation_prob_30d": inv,
        "factor_blindspots": blindspots or [],
        "veto": veto,
        "veto_reason": "x" if veto else None,
    }


def test_compute_llm_disagreement_returns_empty_when_under_two_models():
    out = llm_critic.compute_llm_disagreement([_critic_out()])
    assert out["n_models"] == 1
    assert out["confidence_adjustment_stdev"] is None
    assert out["blindspots_jaccard_mean"] is None


def test_compute_llm_disagreement_perfect_agreement_is_zero():
    same = _critic_out(conf=-0.1, inv=0.5, blindspots=["a", "b"], veto=False)
    out = llm_critic.compute_llm_disagreement([dict(same), dict(same), dict(same)])
    assert out["n_models"] == 3
    assert out["confidence_adjustment_stdev"] == 0.0
    assert out["invalidation_prob_30d_stdev"] == 0.0
    assert out["blindspots_jaccard_mean"] == 0.0
    assert out["veto_agreement_rate"] == 0.0


def test_compute_llm_disagreement_picks_up_blindspot_divergence():
    out = llm_critic.compute_llm_disagreement([
        _critic_out(blindspots=["peer derate"]),
        _critic_out(blindspots=["momentum double-count"]),
    ])
    assert out["blindspots_jaccard_mean"] == 1.0


def test_compute_llm_disagreement_normalizes_blindspot_case():
    out = llm_critic.compute_llm_disagreement([
        _critic_out(blindspots=["Peer Derate"]),
        _critic_out(blindspots=["peer derate"]),
    ])
    assert out["blindspots_jaccard_mean"] == 0.0


def test_compute_llm_disagreement_confidence_stdev_nonzero():
    out = llm_critic.compute_llm_disagreement([
        _critic_out(conf=0.0),
        _critic_out(conf=-0.2),
    ])
    assert out["confidence_adjustment_stdev"] is not None
    assert out["confidence_adjustment_stdev"] > 0


def test_compute_llm_disagreement_veto_agreement_rate():
    out = llm_critic.compute_llm_disagreement([
        _critic_out(veto=True),
        _critic_out(veto=False),
        _critic_out(veto=False),
    ])
    assert out["veto_agreement_rate"] == round(1 / 3, 4)


def test_pairwise_jaccard_distance_basic():
    sets = [{"a", "b"}, {"b", "c"}, {"a", "c"}]
    # All pairs share 1/3 overlap each; distance = 1 - 1/3 = 2/3 per pair.
    d = llm_critic._pairwise_jaccard_distance(sets)
    assert d == round(2 / 3, 4)


def test_pairwise_jaccard_distance_empty_pair_is_zero():
    d = llm_critic._pairwise_jaccard_distance([set(), set()])
    assert d == 0.0
