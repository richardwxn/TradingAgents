"""Tests for OpenAIClient._build_llm provider auth/base-url/class routing.

The multi-provider OpenAI-compatible client resolves a per-provider base
URL + API key, raises a helpful error when a required key is missing, uses
the placeholder ``"ollama"`` key for the local runtime, enables the Responses
API only for native OpenAI, and routes deepseek/minimax to their quirk
subclasses. These branches were previously uncovered.
"""

import pytest

from tradingagents.llm_clients import openai_client as mod


@pytest.fixture
def capture_classes(monkeypatch):
    """Stub all three chat classes; record (class_name, kwargs) per call."""
    calls: dict = {}

    def _make(name):
        def _fake(**kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return f"built-{name}"
        return _fake

    monkeypatch.setattr(mod, "NormalizedChatOpenAI", _make("normalized"))
    monkeypatch.setattr(mod, "DeepSeekChatOpenAI", _make("deepseek"))
    monkeypatch.setattr(mod, "MinimaxChatOpenAI", _make("minimax"))
    return calls


@pytest.mark.unit
class TestProviderAuth:
    def test_missing_api_key_raises_helpful_error(self, capture_classes, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        client = mod.OpenAIClient(model="grok-4.20", provider="xai")

        with pytest.raises(ValueError) as exc:
            client.get_llm()

        msg = str(exc.value)
        assert "xai" in msg
        assert "XAI_API_KEY" in msg

    def test_provider_key_and_default_base_url_are_used(self, capture_classes, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-secret")
        client = mod.OpenAIClient(model="grok-4.20", provider="xai")

        client.get_llm()

        assert capture_classes["kwargs"]["base_url"] == "https://api.x.ai/v1"
        assert capture_classes["kwargs"]["api_key"] == "xai-secret"
        assert capture_classes["name"] == "normalized"

    def test_explicit_base_url_overrides_provider_default(self, capture_classes, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-secret")
        client = mod.OpenAIClient(
            model="grok-4.20", provider="xai", base_url="http://gateway/v1"
        )

        client.get_llm()

        assert capture_classes["kwargs"]["base_url"] == "http://gateway/v1"

    def test_ollama_uses_placeholder_key_without_env(self, capture_classes, monkeypatch):
        # Ollama needs no auth: even with nothing set it must not raise.
        client = mod.OpenAIClient(model="llama3.1", provider="ollama")

        client.get_llm()

        assert capture_classes["kwargs"]["api_key"] == "ollama"
        assert "localhost:11434" in capture_classes["kwargs"]["base_url"]


@pytest.mark.unit
class TestClassRoutingAndResponsesApi:
    def test_native_openai_enables_responses_api_and_no_base_url(self, capture_classes):
        client = mod.OpenAIClient(model="gpt-4.1", provider="openai")

        client.get_llm()

        assert capture_classes["kwargs"]["use_responses_api"] is True
        # openai is not in the provider base-url table and no explicit url given.
        assert "base_url" not in capture_classes["kwargs"]
        assert capture_classes["name"] == "normalized"

    def test_deepseek_routes_to_deepseek_subclass(self, capture_classes, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        client = mod.OpenAIClient(model="deepseek-chat", provider="deepseek")

        client.get_llm()

        assert capture_classes["name"] == "deepseek"
        # Third-party providers do not get the Responses API flag.
        assert "use_responses_api" not in capture_classes["kwargs"]

    def test_minimax_routes_to_minimax_subclass(self, capture_classes, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret")
        client = mod.OpenAIClient(model="MiniMax-M2.7", provider="minimax")

        client.get_llm()

        assert capture_classes["name"] == "minimax"
