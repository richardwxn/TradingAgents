"""Tests for GoogleClient thinking-level mapping and kwarg passthrough.

``GoogleClient._build_llm`` maps the unified ``thinking_level`` kwarg onto the
right provider param per model family:

* Gemini 3 Pro/Flash use ``thinking_level`` directly, except Pro cannot take
  ``"minimal"`` and is bumped to ``"low"``.
* Gemini 2.5 has no ``thinking_level``; it maps to ``thinking_budget``
  (-1 dynamic for "high", 0 otherwise).

``test_google_api_key.py`` covers api_key mapping; these lock the rest.
"""

from unittest.mock import patch

import pytest

from tradingagents.llm_clients.google_client import GoogleClient


def _build_kwargs(model, **client_kwargs):
    with patch(
        "tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI"
    ) as mock_chat:
        GoogleClient(model, **client_kwargs).get_llm()
        return mock_chat.call_args[1]


@pytest.mark.unit
class TestThinkingLevelMapping:
    def test_gemini3_flash_passes_thinking_level_through(self):
        kw = _build_kwargs("gemini-3-flash-preview", thinking_level="minimal")
        assert kw["thinking_level"] == "minimal"
        assert "thinking_budget" not in kw

    def test_gemini3_pro_minimal_is_bumped_to_low(self):
        kw = _build_kwargs("gemini-3.1-pro-preview", thinking_level="minimal")
        assert kw["thinking_level"] == "low"

    def test_gemini3_pro_high_passes_through(self):
        kw = _build_kwargs("gemini-3.1-pro-preview", thinking_level="high")
        assert kw["thinking_level"] == "high"

    def test_gemini25_high_maps_to_dynamic_budget(self):
        kw = _build_kwargs("gemini-2.5-flash", thinking_level="high")
        assert kw["thinking_budget"] == -1
        assert "thinking_level" not in kw

    def test_gemini25_low_maps_to_disabled_budget(self):
        kw = _build_kwargs("gemini-2.5-flash", thinking_level="low")
        assert kw["thinking_budget"] == 0

    def test_no_thinking_level_sets_neither_param(self):
        kw = _build_kwargs("gemini-2.5-flash")
        assert "thinking_level" not in kw
        assert "thinking_budget" not in kw


@pytest.mark.unit
class TestGooglePassthrough:
    def test_base_url_is_forwarded(self):
        kw = _build_kwargs("gemini-2.5-flash", base_url="http://proxy/v1")
        assert kw["base_url"] == "http://proxy/v1"

    def test_no_base_url_omits_key(self):
        kw = _build_kwargs("gemini-2.5-flash")
        assert "base_url" not in kw

    def test_allowlisted_kwargs_pass_through(self):
        kw = _build_kwargs("gemini-2.5-flash", timeout=30, max_retries=4)
        assert kw["timeout"] == 30
        assert kw["max_retries"] == 4
