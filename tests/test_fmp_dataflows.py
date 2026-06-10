"""Unit tests for the Financial Modeling Prep (FMP) data vendor.

Network is fully stubbed (no FMP_API_KEY required to run these).
"""

import json

import pytest

from tradingagents.dataflows import fmp_common, fmp_fundamentals, fmp_news
from tradingagents.dataflows.fmp_common import (
    FMPError,
    FMPRateLimitError,
    filter_statements_by_date,
    fmp_request,
)


class _StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _StubSession:
    """Returns a queued payload per call, recording requests."""

    def __init__(self, payloads):
        # Accept a single payload or a list (one per sequential call).
        self._payloads = payloads if isinstance(payloads, list) else [payloads]
        self._payloads = list(self._payloads)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        payload = self._payloads.pop(0) if len(self._payloads) > 1 else self._payloads[0]
        if isinstance(payload, _StubResponse):
            return payload
        return _StubResponse(payload)


@pytest.fixture(autouse=True)
def _fmp_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")


# --------------------------------------------------------------------------
# fmp_request / error handling
# --------------------------------------------------------------------------

def test_fmp_request_injects_api_key_and_builds_url():
    session = _StubSession([{"ok": True}])
    out = fmp_request("profile/AAPL", session=session)
    assert out == {"ok": True}
    call = session.calls[0]
    assert call["url"] == "https://financialmodelingprep.com/api/v3/profile/AAPL"
    assert call["params"]["apikey"] == "test-key"


def test_fmp_request_version_segment():
    session = _StubSession([{"ok": True}])
    fmp_request("some/endpoint", version="v4", session=session)
    assert "/api/v4/some/endpoint" in session.calls[0]["url"]


def test_fmp_request_http_429_is_rate_limit():
    session = _StubSession([_StubResponse({}, status_code=429)])
    with pytest.raises(FMPRateLimitError):
        fmp_request("profile/AAPL", session=session)


def test_fmp_request_limit_message_is_rate_limit():
    session = _StubSession([{"Error Message": "Limit Reach. Please upgrade your plan."}])
    with pytest.raises(FMPRateLimitError):
        fmp_request("profile/AAPL", session=session)


def test_fmp_request_other_error_message_raises_fmp_error():
    session = _StubSession([{"Error Message": "Invalid symbol ZZZZ"}])
    with pytest.raises(FMPError):
        fmp_request("profile/ZZZZ", session=session)


def test_fmp_request_missing_key_raises(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        fmp_request("profile/AAPL", session=_StubSession([{}]))


# --------------------------------------------------------------------------
# PIT filtering
# --------------------------------------------------------------------------

def _statements():
    return [
        {"date": "2026-03-31", "fillingDate": "2026-04-20", "revenue": 300},
        {"date": "2025-12-31", "fillingDate": "2026-01-25", "revenue": 280},
        {"date": "2025-09-30", "fillingDate": "2025-10-22", "revenue": 260},
    ]


def test_filter_drops_filings_after_curr_date():
    rows = filter_statements_by_date(_statements(), "2026-02-01")
    dates = [r["date"] for r in rows]
    # Only the rows filed on/before 2026-02-01 survive.
    assert dates == ["2025-12-31", "2025-09-30"]


def test_filter_uses_filling_date_not_period_end():
    # Period ended 2025-12-31 but was not filed until 2026-01-25, so an
    # as-of date between those must NOT see it (look-ahead protection).
    rows = filter_statements_by_date(_statements(), "2026-01-10")
    assert [r["date"] for r in rows] == ["2025-09-30"]


def test_filter_falls_back_to_date_when_no_filling_date():
    rows = [{"date": "2026-05-01", "revenue": 1}, {"date": "2025-01-01", "revenue": 2}]
    out = filter_statements_by_date(rows, "2025-06-01")
    assert [r["date"] for r in out] == ["2025-01-01"]


def test_filter_noop_without_curr_date():
    rows = _statements()
    assert filter_statements_by_date(rows, None) == rows


# --------------------------------------------------------------------------
# fundamentals endpoints
# --------------------------------------------------------------------------

def test_get_fundamentals_combines_profile_metrics_ratios(monkeypatch):
    payloads = {
        "profile/AAPL": [{"companyName": "Apple Inc.", "sector": "Technology"}],
        "key-metrics-ttm/AAPL": [{"peRatioTTM": 30.0}],
        "ratios-ttm/AAPL": [{"currentRatioTTM": 1.1}],
    }

    def fake_request(path, params=None, **kwargs):
        return payloads[path]

    monkeypatch.setattr(fmp_fundamentals, "fmp_request", fake_request)
    out = json.loads(fmp_fundamentals.get_fundamentals("AAPL", curr_date="2026-06-01"))
    assert out["symbol"] == "AAPL"
    assert out["as_of_date"] == "2026-06-01"
    assert out["profile"]["companyName"] == "Apple Inc."
    assert out["key_metrics_ttm"]["peRatioTTM"] == 30.0
    assert out["ratios_ttm"]["currentRatioTTM"] == 1.1


def test_get_income_statement_pit_filtered(monkeypatch):
    monkeypatch.setattr(fmp_fundamentals, "fmp_request", lambda *a, **k: _statements())
    out = json.loads(
        fmp_fundamentals.get_income_statement("AAPL", "quarterly", "2026-02-01")
    )
    assert [r["date"] for r in out] == ["2025-12-31", "2025-09-30"]


def test_freq_maps_to_period(monkeypatch):
    captured = {}

    def fake_request(path, params=None, **kwargs):
        captured["params"] = params
        return []

    monkeypatch.setattr(fmp_fundamentals, "fmp_request", fake_request)
    fmp_fundamentals.get_balance_sheet("AAPL", "annual", "2026-02-01")
    assert captured["params"]["period"] == "annual"
    fmp_fundamentals.get_balance_sheet("AAPL", "quarterly", "2026-02-01")
    assert captured["params"]["period"] == "quarter"


# --------------------------------------------------------------------------
# news endpoint
# --------------------------------------------------------------------------

def test_get_news_passes_date_range(monkeypatch):
    captured = {}

    def fake_request(path, params=None, **kwargs):
        captured["path"] = path
        captured["params"] = params
        return [{"title": "headline", "publishedDate": "2026-05-15"}]

    monkeypatch.setattr(fmp_news, "fmp_request", fake_request)
    out = json.loads(fmp_news.get_news("AAPL", "2026-05-01", "2026-05-31"))
    assert captured["path"] == "stock_news"
    assert captured["params"]["tickers"] == "AAPL"
    assert captured["params"]["from"] == "2026-05-01"
    assert captured["params"]["to"] == "2026-05-31"
    assert out[0]["title"] == "headline"


# --------------------------------------------------------------------------
# vendor registry wiring
# --------------------------------------------------------------------------

def test_fmp_registered_in_vendor_methods():
    from tradingagents.dataflows import interface

    assert "fmp" in interface.VENDOR_LIST
    for method in (
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
    ):
        assert "fmp" in interface.VENDOR_METHODS[method]


def test_route_to_vendor_falls_back_on_fmp_rate_limit(monkeypatch):
    """A primary FMP rate-limit must fall through to the next vendor."""
    from tradingagents.dataflows import interface

    def boom(*a, **k):
        raise FMPRateLimitError("limit")

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_fundamentals"], "fmp", boom
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_fundamentals"],
        "yfinance",
        lambda *a, **k: "yf-fallback",
    )
    monkeypatch.setattr(
        interface, "get_vendor", lambda category, method=None: "fmp,yfinance"
    )
    out = interface.route_to_vendor("get_fundamentals", "AAPL", "2026-06-01")
    assert out == "yf-fallback"
