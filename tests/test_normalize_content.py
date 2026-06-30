"""Tests for ``normalize_content`` (base_client).

Several providers (OpenAI Responses API, Gemini 3) return ``response.content``
as a list of typed blocks like ``[{"type": "reasoning", ...},
{"type": "text", "text": "..."}]``. Downstream agents expect a plain string.
``normalize_content`` extracts and joins the text blocks in place, dropping
reasoning/metadata. Every provider's ``Normalized*.invoke`` relies on it, so it
is exercised directly here.
"""

import pytest

from tradingagents.llm_clients.base_client import normalize_content


class _Resp:
    def __init__(self, content):
        self.content = content


@pytest.mark.unit
class TestNormalizeContent:
    def test_string_content_is_left_unchanged(self):
        resp = _Resp("already a string")
        assert normalize_content(resp).content == "already a string"

    def test_returns_same_object(self):
        resp = _Resp("x")
        assert normalize_content(resp) is resp

    def test_text_blocks_are_joined_with_newline(self):
        resp = _Resp([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])
        assert normalize_content(resp).content == "first\nsecond"

    def test_reasoning_blocks_are_discarded(self):
        resp = _Resp([
            {"type": "reasoning", "summary": "thinking..."},
            {"type": "text", "text": "answer"},
        ])
        assert normalize_content(resp).content == "answer"

    def test_plain_string_items_are_kept(self):
        resp = _Resp(["raw string", {"type": "text", "text": "block"}])
        assert normalize_content(resp).content == "raw string\nblock"

    def test_empty_text_blocks_are_dropped(self):
        resp = _Resp([
            {"type": "text", "text": ""},
            {"type": "text", "text": "kept"},
        ])
        assert normalize_content(resp).content == "kept"

    def test_text_block_missing_text_key_becomes_empty(self):
        resp = _Resp([
            {"type": "text"},
            {"type": "text", "text": "kept"},
        ])
        assert normalize_content(resp).content == "kept"

    def test_empty_list_becomes_empty_string(self):
        resp = _Resp([])
        assert normalize_content(resp).content == ""

    def test_all_non_text_blocks_become_empty_string(self):
        resp = _Resp([{"type": "reasoning"}, {"type": "tool_use"}])
        assert normalize_content(resp).content == ""
