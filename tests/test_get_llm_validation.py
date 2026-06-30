"""Tests that ``get_llm()`` validates the model name (TODO item #1).

``BaseLLMClient.get_llm()`` runs ``validate_model()`` (via
``warn_if_unknown_model``) before constructing the provider client.
Validation is deliberately *warn-and-continue*, not an error, so
forward-compat / preview model ids keep working (cf. test_anthropic_effort
which exercises future ``claude-*`` ids). These tests lock that contract.
"""

import warnings

import pytest

from tradingagents.llm_clients import anthropic_client as mod
from tradingagents.llm_clients.factory import create_llm_client


@pytest.fixture
def captured_build(monkeypatch):
    """Stub the Anthropic constructor so no network/SDK call happens."""
    captured: dict = {}

    def _fake(**kwargs):
        captured["kwargs"] = kwargs
        return "built-llm"

    monkeypatch.setattr(mod, "NormalizedChatAnthropic", _fake)
    return captured


@pytest.mark.unit
class TestGetLlmValidatesModel:
    def test_known_good_model_builds_without_warning(self, captured_build):
        client = mod.AnthropicClient(model="claude-sonnet-4-6", api_key="x")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llm = client.get_llm()

        assert llm == "built-llm"
        assert captured_build["kwargs"]["model"] == "claude-sonnet-4-6"
        assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    def test_invalid_model_warns_but_still_builds(self, captured_build):
        """Unknown model warns (does not raise) and still returns an LLM."""
        client = mod.AnthropicClient(model="claude-not-a-real-model", api_key="x")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llm = client.get_llm()  # must not raise

        assert llm == "built-llm"
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime) == 1
        assert "claude-not-a-real-model" in str(runtime[0].message)
        assert "anthropic" in str(runtime[0].message)

    def test_factory_path_validates_in_get_llm(self, captured_build):
        """The public create_llm_client(...).get_llm() path warns once."""
        client = create_llm_client("anthropic", "claude-not-a-real-model", api_key="x")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.get_llm()

        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime) == 1
