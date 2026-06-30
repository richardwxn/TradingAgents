"""Tests for ``create_llm_client`` provider routing (`factory.py`).

The factory lazily imports the right client class per provider, routes every
OpenAI-compatible provider (xai/deepseek/qwen/glm/minimax/ollama/openrouter)
to ``OpenAIClient`` with the provider preserved, and raises ``ValueError`` for
anything unknown. Constructing a client does not build the underlying LLM, so
these tests need no API keys or network.
"""

import pytest

from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.azure_client import AzureOpenAIClient
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.google_client import GoogleClient
from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
class TestFactoryRouting:
    @pytest.mark.parametrize(
        "provider",
        ["openai", "xai", "deepseek", "qwen", "qwen-cn", "glm", "glm-cn",
         "minimax", "minimax-cn", "ollama", "openrouter"],
    )
    def test_openai_compatible_providers_route_to_openai_client(self, provider):
        client = create_llm_client(provider, "some-model")
        assert isinstance(client, OpenAIClient)
        # The lowercased provider is preserved for downstream base-url/auth logic.
        assert client.provider == provider

    def test_provider_is_lowercased(self):
        client = create_llm_client("OpenAI", "gpt-4o")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "openai"

    def test_anthropic_routes_to_anthropic_client(self):
        client = create_llm_client("anthropic", "claude-sonnet-4-6")
        assert isinstance(client, AnthropicClient)

    def test_google_routes_to_google_client(self):
        client = create_llm_client("google", "gemini-2.5-pro")
        assert isinstance(client, GoogleClient)

    def test_azure_routes_to_azure_client(self):
        client = create_llm_client("azure", "gpt-4o")
        assert isinstance(client, AzureOpenAIClient)

    def test_base_url_is_forwarded(self):
        client = create_llm_client("openai", "gpt-4o", base_url="http://proxy/v1")
        assert client.base_url == "http://proxy/v1"

    def test_extra_kwargs_are_forwarded(self):
        client = create_llm_client("anthropic", "claude-sonnet-4-6", api_key="k")
        assert client.kwargs.get("api_key") == "k"

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("not-a-provider", "model")
