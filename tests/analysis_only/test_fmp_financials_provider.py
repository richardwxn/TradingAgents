"""Tests for FMPFinancialsProvider (analysis pipeline fundamentals source).

Network is stubbed; no FMP_API_KEY required.
"""

import pytest

from tradingagents.analysis_only.providers import (
    FMPFinancialsProvider,
    PolygonFinancialsProvider,
    reset_fmp_financials_cache,
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


class _StatementSession:
    """Routes by statement path so one session serves all three calls."""

    def __init__(self, income, balance, cashflow):
        self._by_statement = {
            "income-statement": income,
            "balance-sheet-statement": balance,
            "cash-flow-statement": cashflow,
        }
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        for key, payload in self._by_statement.items():
            if key in url:
                return _StubResponse(payload)
        return _StubResponse([])


def _income():
    return [
        {"date": "2026-03-31", "fillingDate": "2026-04-20", "revenue": 400,
         "netIncome": 80, "operatingIncome": 100, "grossProfit": 200},
        {"date": "2025-12-31", "fillingDate": "2026-01-25", "revenue": 380,
         "netIncome": 70, "operatingIncome": 95, "grossProfit": 190},
        {"date": "2025-09-30", "fillingDate": "2025-10-22", "revenue": 360,
         "netIncome": 60, "operatingIncome": 90, "grossProfit": 180},
    ]


def _balance():
    return [
        {"date": "2026-03-31", "fillingDate": "2026-04-20", "totalAssets": 1000,
         "totalLiabilities": 600, "totalStockholdersEquity": 400,
         "totalCurrentAssets": 300, "totalCurrentLiabilities": 150,
         "inventory": 50},
        {"date": "2025-12-31", "fillingDate": "2026-01-25", "totalAssets": 950,
         "totalStockholdersEquity": 380},
    ]


def _cashflow():
    return [
        {"date": "2026-03-31", "fillingDate": "2026-04-20",
         "netCashProvidedByOperatingActivities": 120,
         "netCashUsedForInvestingActivites": -40},
    ]


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_fmp_financials_cache()
    yield
    reset_fmp_financials_cache()


def _provider(session):
    return FMPFinancialsProvider(api_key="test", session=session)


def test_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    provider = FMPFinancialsProvider(api_key="", session=_StatementSession([], [], []))
    assert provider.fetch_quarterly("ABCD", "2026-05-01") == []


def test_fetch_quarterly_merges_into_polygon_shape():
    session = _StatementSession(_income(), _balance(), _cashflow())
    out = _provider(session).fetch_quarterly("ABCD", "2026-05-01")
    assert len(out) == 3
    rec = out[0]
    assert rec["filing_date"] == "2026-04-20"
    assert rec["end_date"] == "2026-03-31"
    assert rec["period_of_report_date"] == "2026-03-31"
    fin = rec["financials"]
    # Polygon-style nested shape + value_of compatibility.
    assert PolygonFinancialsProvider.value_of(fin["income_statement"]["revenues"]) == 400
    assert PolygonFinancialsProvider.value_of(fin["income_statement"]["net_income_loss"]) == 80
    assert PolygonFinancialsProvider.value_of(fin["balance_sheet"]["assets"]) == 1000
    assert PolygonFinancialsProvider.value_of(fin["balance_sheet"]["equity"]) == 400
    assert PolygonFinancialsProvider.value_of(
        fin["cash_flow_statement"]["net_cash_flow_from_operating_activities"]
    ) == 120


def test_pit_filter_uses_filling_date():
    session = _StatementSession(_income(), _balance(), _cashflow())
    # As-of between the 2025-12-31 period end and its 2026-01-25 filing.
    out = _provider(session).fetch_quarterly("ABCD", "2026-01-10")
    assert [r["end_date"] for r in out] == ["2025-09-30"]


def test_pit_filter_drops_future_filings():
    session = _StatementSession(_income(), _balance(), _cashflow())
    out = _provider(session).fetch_quarterly("ABCD", "2026-02-01")
    assert [r["end_date"] for r in out] == ["2025-12-31", "2025-09-30"]


def test_limit_caps_results():
    session = _StatementSession(_income(), _balance(), _cashflow())
    out = _provider(session).fetch_quarterly("ABCD", "2026-05-01", limit=1)
    assert len(out) == 1
    assert out[0]["end_date"] == "2026-03-31"


def test_missing_balance_section_is_empty_not_error():
    # 2025-09-30 has no balance/cashflow rows -> empty sections, no crash.
    session = _StatementSession(_income(), _balance(), _cashflow())
    out = _provider(session).fetch_quarterly("ABCD", "2026-05-01")
    oldest = out[-1]
    assert oldest["end_date"] == "2025-09-30"
    assert oldest["financials"]["balance_sheet"] == {}
    assert oldest["financials"]["income_statement"]["revenues"]["value"] == 360


def test_caching_dedupes_repeat_calls():
    session = _StatementSession(_income(), _balance(), _cashflow())
    provider = _provider(session)
    provider.fetch_quarterly("ABCD", "2026-05-01")
    provider.fetch_quarterly("ABCD", "2026-01-10", limit=2)
    # Three statement calls total (one per statement), not six.
    assert len(session.calls) == 3


def test_quarterly_and_annual_use_distinct_periods():
    session = _StatementSession(_income(), _balance(), _cashflow())
    provider = _provider(session)
    provider.fetch_quarterly("ABCD", "2026-05-01")
    provider.fetch_annual("ABCD", "2026-05-01")
    periods = {c["params"]["period"] for c in session.calls}
    assert periods == {"quarter", "annual"}
    assert len(session.calls) == 6  # 3 per timeframe, separate cache keys


def test_fetch_ttm_returns_empty():
    session = _StatementSession(_income(), _balance(), _cashflow())
    assert _provider(session).fetch_ttm("ABCD", "2026-05-01") == []


def test_http_error_negative_caches_empty():
    class _FailSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=None):
            self.calls += 1
            return _StubResponse({}, status_code=500)

    session = _FailSession()
    provider = FMPFinancialsProvider(api_key="test", session=session)
    assert provider.fetch_quarterly("ABCD", "2026-05-01") == []
    # Negative-cached: a second call does not re-hit the network.
    assert provider.fetch_quarterly("ABCD", "2026-05-01") == []
    assert session.calls == 1
