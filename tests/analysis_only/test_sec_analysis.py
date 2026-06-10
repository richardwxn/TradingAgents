"""Tests for SEC filing document fetch, section extraction, and LLM analysis.

Network and LLM calls are fully stubbed.
"""

import json

import pytest

from tradingagents.analysis_only import providers as P
from tradingagents.analysis_only import sec_analysis as SA
from tradingagents.analysis_only.providers import (
    SECFetchError,
    SECFilingsProvider,
    extract_filing_sections,
    strip_html_to_text,
)


# --------------------------------------------------------------------------
# HTML stripping + section extraction
# --------------------------------------------------------------------------

def test_strip_html_removes_tags_and_scripts():
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><script>var x=1;</script><p>Hello&nbsp;world</p>"
        "<div>Second   line</div></body></html>"
    )
    text = strip_html_to_text(html)
    assert "Hello world" in text
    assert "Second line" in text
    assert "var x" not in text
    assert "color:red" not in text


def test_strip_html_plaintext_passthrough():
    assert "plain text body" in strip_html_to_text("plain text body").lower()


def _ten_k_html():
    return (
        "<html><body>"
        "<p>Table of contents Item 1A. Risk Factors .... 12 Item 7. MD&A .... 30</p>"
        "<p>Item 1. Business. We make widgets.</p>"
        "<p>Item 1A. Risk Factors. Our supply chain is concentrated in one region. "
        "Currency fluctuations may hurt margins.</p>"
        "<p>Item 1B. Unresolved Staff Comments. None.</p>"
        "<p>Item 7. Management's Discussion and Analysis. Revenue grew 12% on strong "
        "demand. Operating margin expanded.</p>"
        "<p>Item 7A. Quantitative disclosures.</p>"
        "<p>Item 8. Financial Statements.</p>"
        "</body></html>"
    )


def test_extract_10k_sections():
    sections = extract_filing_sections(_ten_k_html(), "10-K")
    assert set(sections) == {"risk_factors", "mdna"}
    # Uses the LAST "Item 1A" occurrence (real section, not the TOC entry).
    assert "supply chain is concentrated" in sections["risk_factors"]
    assert "Item 1B" not in sections["risk_factors"]  # bounded by end header
    assert "Revenue grew 12%" in sections["mdna"]
    assert "Item 8" not in sections["mdna"]


def test_extract_8k_returns_body():
    html = "<html><body><p>Item 2.02 Results of Operations. We beat estimates.</p></body></html>"
    sections = extract_filing_sections(html, "8-K")
    assert set(sections) == {"body"}
    assert "beat estimates" in sections["body"]


def test_extract_unmapped_form_returns_body():
    sections = extract_filing_sections("<p>some text</p>", "DEF 14A")
    assert "body" in sections


def test_extract_falls_back_to_body_when_no_items():
    # 10-K form but no recognizable item headers -> body fallback.
    sections = extract_filing_sections("<p>no item headers here at all</p>", "10-K")
    assert set(sections) == {"body"}


def test_extract_respects_max_section_chars():
    big = "Item 1A. Risk Factors. " + ("x" * 50000) + " Item 1B. End."
    sections = extract_filing_sections(f"<p>{big}</p>", "10-K", max_section_chars=100)
    assert len(sections["risk_factors"]) <= 100


def test_extract_empty_document():
    assert extract_filing_sections("", "10-K") == {}


# --------------------------------------------------------------------------
# SECFilingsProvider document fetch
# --------------------------------------------------------------------------

def test_filing_document_url_construction():
    sec = SECFilingsProvider(user_agent="Tester test@example.com")
    url = sec.filing_document_url(
        {"cik": "320193", "accession": "0000320193-26-000005", "primary_document": "aapl-10k.htm"}
    )
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-10k.htm"
    )


def test_filing_document_url_none_when_incomplete():
    sec = SECFilingsProvider(user_agent="Tester test@example.com")
    assert sec.filing_document_url({"cik": "320193"}) is None


def test_fetch_filing_document_returns_text(monkeypatch):
    sec = SECFilingsProvider(user_agent="Tester test@example.com")
    monkeypatch.setattr(sec, "_fetch_text", lambda url, host: "<html>doc</html>")
    out = sec.fetch_filing_document(
        {"cik": "1", "accession": "0000000000-26-000001", "primary_document": "x.htm"}
    )
    assert out == "<html>doc</html>"


def test_fetch_filing_document_truncates(monkeypatch):
    sec = SECFilingsProvider(user_agent="Tester test@example.com")
    monkeypatch.setattr(sec, "_fetch_text", lambda url, host: "y" * 500)
    out = sec.fetch_filing_document(
        {"cik": "1", "accession": "a-1", "primary_document": "x.htm"}, max_chars=100
    )
    assert len(out) == 100


def test_fetch_text_retries_then_raises_on_5xx(monkeypatch):
    sec = SECFilingsProvider(user_agent="Tester test@example.com")
    monkeypatch.setattr(sec, "_throttle", lambda: None)
    monkeypatch.setattr(sec, "_sleep_backoff", lambda attempt: None)

    class _Resp:
        status_code = 503

    monkeypatch.setattr(P.requests, "get", lambda *a, **k: _Resp())
    with pytest.raises(SECFetchError):
        sec._fetch_text("https://www.sec.gov/x", host="www.sec.gov")


# --------------------------------------------------------------------------
# LLM analysis layer
# --------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        class _R:
            content = self._content
        return _R()


class _FakeClient:
    def __init__(self, content):
        self._content = content

    def get_llm(self):
        return _FakeLLM(self._content)


def _patch_llm(monkeypatch, content):
    import tradingagents.llm_clients.factory as factory

    monkeypatch.setattr(
        factory, "create_llm_client", lambda **kw: _FakeClient(content)
    )


def test_run_filing_analysis_ok(monkeypatch):
    payload = {
        "summary": "Company grew revenue and flagged supply risk.",
        "key_risks": ["supply chain concentration"],
        "mdna_highlights": ["revenue +12%"],
        "tone": "cautious",
        "notable_changes": [],
    }
    _patch_llm(monkeypatch, json.dumps(payload))
    block = SA.run_filing_analysis(
        {"risk_factors": "risk text", "mdna": "mdna text"},
        {"symbol": "AAPL", "form": "10-K"},
        provider="openai",
        model="gpt-5.4-mini",
    )
    assert block["status"] == "ok"
    assert block["output"]["tone"] == "cautious"
    assert block["output"]["key_risks"] == ["supply chain concentration"]
    assert block["prompt_version"] == SA.SEC_ANALYSIS_PROMPT_VERSION


def test_run_filing_analysis_no_sections():
    block = SA.run_filing_analysis(
        {}, {"symbol": "AAPL"}, provider="openai", model="m"
    )
    assert block["status"] == "no_sections"


def test_run_filing_analysis_strips_code_fences(monkeypatch):
    fenced = "```json\n" + json.dumps({"tone": "positive"}) + "\n```"
    _patch_llm(monkeypatch, fenced)
    block = SA.run_filing_analysis(
        {"body": "x"}, {"symbol": "X"}, provider="openai", model="m"
    )
    assert block["status"] == "ok"
    assert block["output"]["tone"] == "positive"


def test_run_filing_analysis_bad_tone_normalized(monkeypatch):
    _patch_llm(monkeypatch, json.dumps({"tone": "euphoric"}))
    block = SA.run_filing_analysis(
        {"body": "x"}, {"symbol": "X"}, provider="openai", model="m"
    )
    assert block["output"]["tone"] == "neutral"


def test_run_filing_analysis_json_parse_error(monkeypatch):
    _patch_llm(monkeypatch, "not json at all")
    block = SA.run_filing_analysis(
        {"body": "x"}, {"symbol": "X"}, provider="openai", model="m"
    )
    assert block["status"] == "llm_json_parse_error"
    assert "raw_response" in block


def test_run_filing_analysis_init_error(monkeypatch):
    import tradingagents.llm_clients.factory as factory

    def boom(**kw):
        raise RuntimeError("no key")

    monkeypatch.setattr(factory, "create_llm_client", boom)
    block = SA.run_filing_analysis(
        {"body": "x"}, {"symbol": "X"}, provider="openai", model="m"
    )
    assert block["status"] == "llm_init_error"


# --------------------------------------------------------------------------
# analyze_filing orchestration
# --------------------------------------------------------------------------

class _StubSecProvider:
    def __init__(self, latest, document, raise_on=None):
        self._latest = latest
        self._document = document
        self._raise_on = raise_on

    def get_latest_filing(self, symbol, as_of_date=None):
        if self._raise_on == "latest":
            raise SECFetchError("boom")
        return self._latest

    def filing_document_url(self, filing):
        return "https://sec.gov/doc"

    def fetch_filing_document(self, filing):
        if self._raise_on == "document":
            raise SECFetchError("boom")
        return self._document


def test_analyze_filing_end_to_end(monkeypatch):
    _patch_llm(monkeypatch, json.dumps({"summary": "ok", "tone": "neutral"}))
    sec = _StubSecProvider(
        latest={
            "form": "10-K",
            "accession": "a-1",
            "filing_date": "2026-02-10",
            "cik": "1",
        },
        document=_ten_k_html(),
    )
    block = SA.analyze_filing(
        "AAPL", "2026-03-01", provider="openai", model="m", sec_provider=sec
    )
    assert block["status"] == "ok"
    assert block["filing"]["form"] == "10-K"
    assert set(block["sections_found"]) == {"risk_factors", "mdna"}
    assert block["analysis"]["output"]["summary"] == "ok"


def test_analyze_filing_unavailable():
    sec = _StubSecProvider(latest=None, document=None)
    block = SA.analyze_filing(
        "AAPL", "2026-03-01", provider="openai", model="m", sec_provider=sec
    )
    assert block["status"] == "unavailable"


def test_analyze_filing_fetch_error_on_latest():
    sec = _StubSecProvider(latest=None, document=None, raise_on="latest")
    block = SA.analyze_filing(
        "AAPL", "2026-03-01", provider="openai", model="m", sec_provider=sec
    )
    assert block["status"] == "fetch_error"


def test_analyze_filing_fetch_error_on_document():
    sec = _StubSecProvider(
        latest={"form": "10-K", "accession": "a", "filing_date": "2026-01-01", "cik": "1"},
        document=None,
        raise_on="document",
    )
    block = SA.analyze_filing(
        "AAPL", "2026-03-01", provider="openai", model="m", sec_provider=sec
    )
    assert block["status"] == "fetch_error"
    assert block["filing"]["form"] == "10-K"
