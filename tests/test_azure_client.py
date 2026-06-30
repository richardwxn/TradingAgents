"""Tests for the Azure OpenAI client (`azure_client.py`).

The Azure deployment client builds a ``NormalizedAzureChatOpenAI`` from the
model name plus the ``AZURE_OPENAI_DEPLOYMENT_NAME`` env var, forwards only an
allowlisted set of kwargs, and accepts any model name (deployments are named
freely on Azure, so ``validate_model`` is always True and never warns).
"""

import warnings

import pytest

from tradingagents.llm_clients import azure_client as mod
from tradingagents.llm_clients.factory import create_llm_client


@pytest.fixture
def captured_build(monkeypatch):
    """Stub the AzureChatOpenAI constructor so no SDK/network call happens."""
    captured: dict = {}

    def _fake(**kwargs):
        captured["kwargs"] = kwargs
        return "built-azure-llm"

    monkeypatch.setattr(mod, "NormalizedAzureChatOpenAI", _fake)
    return captured


@pytest.mark.unit
class TestAzureBuildLlm:
    def test_deployment_defaults_to_model_when_env_unset(self, captured_build, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
        client = mod.AzureOpenAIClient(model="gpt-4o")

        llm = client.get_llm()

        assert llm == "built-azure-llm"
        assert captured_build["kwargs"]["model"] == "gpt-4o"
        # No explicit deployment configured -> falls back to the model name.
        assert captured_build["kwargs"]["azure_deployment"] == "gpt-4o"

    def test_deployment_uses_env_when_set(self, captured_build, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deployment")
        client = mod.AzureOpenAIClient(model="gpt-4o")

        client.get_llm()

        assert captured_build["kwargs"]["azure_deployment"] == "my-deployment"
        assert captured_build["kwargs"]["model"] == "gpt-4o"

    def test_only_allowlisted_kwargs_pass_through(self, captured_build, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
        client = mod.AzureOpenAIClient(
            model="gpt-4o",
            api_key="secret",
            max_retries=7,
            reasoning_effort="high",
            unsupported_kwarg="should-be-dropped",
        )

        client.get_llm()

        kw = captured_build["kwargs"]
        assert kw["api_key"] == "secret"
        assert kw["max_retries"] == 7
        assert kw["reasoning_effort"] == "high"
        assert "unsupported_kwarg" not in kw

    def test_validate_model_accepts_anything(self):
        client = mod.AzureOpenAIClient(model="any-deployment-name")
        assert client.validate_model() is True

    def test_get_llm_does_not_warn_for_arbitrary_model(self, captured_build):
        client = mod.AzureOpenAIClient(model="totally-made-up-deployment")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.get_llm()

        assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    def test_factory_routes_azure_to_azure_client(self):
        client = create_llm_client("azure", "gpt-4o")
        assert isinstance(client, mod.AzureOpenAIClient)
        assert client.get_provider_name() == "azureopenai"
